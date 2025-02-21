import os
import re
import json
import jsonlines
from colorama import Fore
from openai import OpenAI


class LLM:

    def __init__(self, config):
        self.model = config.get("model")
        self.client = OpenAI(base_url = config.get("base_url"), api_key = config.get("api_key"))
        self.print_response = config.get("print_response")
        self.save_path = config.get("save_path")
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(os.path.dirname(self.save_path), exist_ok = True)

    def __call__(self, messages, temperature = 0, return_schema = None, desc = None):
        response_type = "json_object" if return_schema is not None else "text"
        exception = None
        max_retry = 3
        for retry in range(max_retry+1):
            try:
                data = dict(
                    model = self.model, 
                    messages = messages, 
                    temperature = temperature+retry*(1.-temperature)/max_retry, 
                    response_format = {"type": response_type}, 
                    #max_tokens = 2048, 
                    #timeout = 60, 
                )
                if return_schema is not None and not self.model.startswith("gpt"):
                    data.update({"extra_body": {"guided_json": return_schema}})
                response = self.client.chat.completions.create(**data)
                response = response.choices[0].message.content
                data.update({"response": response})
                if self.save_path is not None:
                    with jsonlines.open(self.save_path, "a") as f:
                        f.write(data)
                if return_schema is not None:
                    try:
                        response = json.loads(response)
                    except:
                        response = json.loads(re.findall(r"```json([\s\S]*)```", response)[0].strip())
                if self.print_response:
                    print("~"*100)
                    if desc is not None:
                        print(Fore.YELLOW + f"{desc}\n" + Fore.RESET)
                    if return_schema is not None:
                        print(json.dumps(response, ensure_ascii = False, indent = 2))
                    else:
                        print(response)
                    print("~"*100)
                return response
            except Exception as e:
                exception = e
                continue
        raise Exception(exception)
