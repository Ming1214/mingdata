import os
import json
import jsonlines
import asyncio
from colorama import init, Fore

init()

from src.llms import LLM


async def get_response(llm, messages):
    return llm(messages = messages,  temperature = 0)


async def run_mode(mode, version):
    path = f"./data/{version}/instructions.{mode}.jsonl"
    save_path = f"./data/{version}/" + "{model}" + f".{mode}.jsonl"
    with jsonlines.open(path, "r") as f:
        data = [x for x in f]
    model_responses = {}
    for llm in llms:
        model_responses[llm.model] = []
    for x in data:
        for instruction in x["instructions"]:
            messages = [{"role": "user", "content": instruction}]
            tasks = []
            for llm in llms:
                tasks.append(get_response(llm, messages))
            responses = await asyncio.gather(*tasks)
            print("*"*100 + Fore.RED)
            print(instruction)
            print(Fore.RESET + "*"*100)
            for llm, response in zip(llms, responses):
                print("*"*100 + Fore.YELLOW)
                print(llm.model)
                print(Fore.RESET + "-"*100 + Fore.BLUE)
                print(response)
                print(Fore.RESET + "*"*100)
                model_responses[llm.model].append({
                    "category": x["category"], 
                    "instruction_type": x["instruction_type"], 
                    "requirement": x["requirement"], 
                    "example": x["example"], 
                    "instruction": instruction, 
                    "response": response, 
                })
                with jsonlines.open(save_path.format(model = llm.model), "w") as f:
                    for xx in model_responses[llm.model]:
                        f.write(xx)


async def main(modes = None, version = "latest"):
    allowed_modes = ["junior", "intermediate", "advanced", "nlp", "discipline", "application.discipline"]
    if modes is not None:
        for mode in modes:
            assert mode in allowed_modes, f"Undefined mode: {mode}"
    tasks = []
    for mode in modes:
        tasks.append(run_mode(mode, version))
    await asyncio.gather(*tasks)


if __name__ == "__main__":

    llm_list = [
        #{"model": "gpt-4o", "base_url": "http://172.18.160.39:8168/v1", "api_key": "zhuiyi", "save_path": "./log/llm/jsonl"}, 
        {"model": "Qwen1.5-110B-Chat-AWQ", "base_url": "http://localhost:7001/v1", "api_key": "zhuiyi", "save_path": "./log/llm.jsonl"}, 
        {"model": "Qwen1.5-72B-Chat-AWQ", "base_url": "http://180.76.158.66:8000/v1", "api_key": "zhuiyi", "save_path": "./log/llm.jsonl"}, 
    ]
    llms = [LLM(config) for config in llm_list]
    print("*"*100)
    messages = [{"role": "user", "content": "hello"}]
    for llm in llms:
        print(llm.model, llm(messages = [{"role": "user", "content": "hello"}]))
    print("*"*100)
    asyncio.run(main(modes = ["intermediate", "advanced"], version = "v1"))



