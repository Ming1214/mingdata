import os
import fire
import json
import jsonlines
from tqdm import tqdm
from colorama import init, Fore

init()

from src.llms import LLM


def get_instructions_step1(category, instruction_type, requirement, example, num_samples = 10):
    schema = json.dumps({
        "type": "object", 
        "properties": {
            "instructions": {
                "type": "array", 
                "items": {
                    "type": "string"
                }
            }
        }, 
        "required": ["instructions"]
    }, ensure_ascii = False, indent = 2)
    system = (
        f"你是一个大语言模型指令生成器。\n"
        f"你的任务是生成一些可以用于测试大语言模型各方面能力的指令。\n"
        f"指令的要求如下：{requirement}\n"
        f"注意：\n"
        f"1. 当指令涉及到语言时，请以中文为主；当涉及到文化内容时，请以中华文化为主。\n"
        f"2. 指令必须是需要大模型回答的，指令中不能透露最终的答案。"
    )
    user = (
        f"这是一个关于大语言模型 *{category}* 能力的一个测试。\n"
        f"请你生成 {num_samples} 个 {instruction_type} 类型的指令。\n"
        f"指令的一个示例如下：{example}\n"
        f"为了保证测试的多样性，生成的指令请尽可能涉及不同的主题、不同的领域、尽可能覆盖 {instruction_type} 所有可能的场景。\n\n"
        f"请你按照如下 Json Schema 所要求的格式进行输出：\n"
        f"{schema}\n"
        f"注意：是按照要求输出而不是直接输出 Json Schema！"
    )
    messages = [
        {"role": "system", "content": system}, 
        {"role": "user", "content": user}
    ]
    for _ in range(10):
        try:
            response = llm(messages, return_schema = schema, temperature = 1.0, desc = user)
            instructions = response.get("instructions")
            assert isinstance(instructions, list) and isinstance(instructions[0], str)
            return instructions
        except: continue


def get_instructions_step2(category, instruction_type, requirement, instruction):
    schema = json.dumps({
        "type": "object", 
        "properties": {
            "analysis": {
                "type": "string", 
                "description": "该字段用于分析指令可以从哪些方面进行优化（包括不合理的、需要修正的地方也需要指出来并提供修改方法）"
            }, 
            "optimized_instruction": {
                "type": "string", 
                "description": "该字段为最终优化后的指令"
            }
        }, 
        "required": ["analysis", "optimized_instruction"]
    }, ensure_ascii = False, indent = 2)
    system = (
        f"你是一个大语言模型指令优化器。\n"
        f"你的任务是对给定的指令进行优化，包括添加上下文信息、添加输出格式要求（输出格式的要求只能用抽象表达、禁止举例）等，使得指令更合规、更具体、更清晰、更具挑战性。\n"
        f"指令的要求如下：{requirement}\n"
        f"注意：\n"
        f"1. 当指令涉及到语言时，请以中文为主；当涉及到文化内容时，请以中华文化为主。\n"
        f"2. 指令必须是需要大模型回答的，指令中（尤其是对输出格式的描述中）不能透露最终的答案。\n"
        f"3. 请使用 Markdown 格式表达指令、使之更具有结构性。\n"
        f"4. 多个同样格式的输出只需要描述一个输出模板。"
    )
    user = (
        f"这是一个关于大语言模型 *{category}* 能力的一个测试指令。\n"
        f"指令的类型是：{instruction_type}\n"
        f"指令的内容是：'''\n"
        f"{instruction}\n"
        f"'''\n\n"
        f"请你分析该指令还可以从哪些方面进行优化、从而提高对大语言模型的挑战性，并给出优化后的指令。\n"
        f"如果当前指令中有不合理的部分，例如缺少作答所必要的信息、或者已经透露了最终的答案等，你也需要对这些不合理的内容进行修改，这也是优化的一部分。\n\n"
        f"请你按照如下 Json Schema 所要求的格式进行输出：\n"
        f"{schema}\n"
        f"注意：是按照要求输出而不是直接输出 Json Schema！"
    )
    messages = [
        {"role": "system", "content": system}, 
        {"role": "user", "content": user}
    ]
    for _ in range(10):
        try:
            response = llm(messages, return_schema = schema, temperature = 0.7, desc = user)
            optimized_instruction = response.get("optimized_instruction")
            assert isinstance(optimized_instruction, str)
            return optimized_instruction.strip("'")
        except: continue


def get_instructions(data_path, save_path, num_samples, optimize_steps, show = False):
    with jsonlines.open(data_path, "r") as f:
        data = [x for x in f]
    for x in tqdm(data):
        category, instruction_type, requirement, example = x["category"], x["instruction_type"], x["requirement"], x["example"]
        instructions = get_instructions_step1(category, instruction_type, requirement, example, num_samples)
        if instructions is not None:
            if show:
                print("+"*100 + Fore.YELLOW)
                print(f"Category: {category}, Type: {instruction_type}")
                print(f"Requirement: {requirement}")
                print(f"Example: {example}")
                print(Fore.RESET + "-"*100 + Fore.YELLOW)
                print(("\n"+'-'*100+"\n").join(instructions))
                print(Fore.RESET + "+"*100)
            for instruction in instructions:
                if show:
                    print("*"*100 + Fore.BLUE)
                    print(instruction)
                    print(Fore.RESET + "*"*100)
                opt_instructions = [instruction]
                to_be_optimized_instruction = instruction
                for step in range(optimize_steps):
                    optimized_instruction = get_instructions_step2(category, instruction_type, requirement, to_be_optimized_instruction)
                    if optimized_instruction is not None:
                        if show:
                            print("*"*100 + Fore.RED)
                            print(optimized_instruction)
                            print(Fore.RESET + "*"*100)
                        opt_instructions.append(optimized_instruction)
                        to_be_optimized_instruction = optimized_instruction
                for i, ins in enumerate(opt_instructions):
                    with jsonlines.open(save_path, "a") as f:
                        f.write({
                            "category": category, 
                            "instruction_type": instruction_type, 
                            "requirement": requirement, 
                            "instruction": ins, 
                            "optimized_level": i
                        })


def get_responses(data_path, save_path, optimized_level = None, show = False):
    with jsonlines.open(data_path, "r") as f:
        data = [x for x in f if optimized_level is None or x["optimized_level"] == optimized_level]
    for x in tqdm(data):
        system = (
            f"你是 {x['category']} 领域的专家，十分擅长解决 {x['instruction_type']} 的问题。\n"
            f"用户提出指令后，你需要详细、清晰、正确、完整地给出回复。"
        )
        user = x["instruction"]
        messages = [
            {"role": "system", "content": system}, 
            {"role": "user", "content": user}, 
        ]
        response = llm(messages)
        x["response"] = response
        with jsonlines.open(save_path, "a") as f:
            f.write(x)
        if show:
            print("*"*100 + Fore.YELLOW)
            print(f"Category: {x['category']}, Type: {x['instruction_type']}")
            print(Fore.RESET + "-"*100 + Fore.BLUE)
            print(user)
            print(Fore.RESET + "-"*100 + Fore.RED)
            print(response)
            print(Fore.RESET + "*"*100)


llm = LLM({
    "model": "gpt-4o", 
    "base_url": "http://172.18.160.39:8168/v1", 
    "api_key": "zhuiyi", 
    "save_path": "./log/llm.jsonl", 
    "print_response": False, 
})


if __name__ == "__main__":
    fire.Fire()


