import openai
import os
import time
import json
import anthropic
from ollama import chat
from ollama import ChatResponse
from groq import Groq


from dotenv import load_dotenv

load_dotenv()

def use_openai_api(assistant_contents, user_contents, system, model, temperature):
    num_assistant_mes = len(assistant_contents)
    messages = []

    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx]})
        messages.append({"role": "assistant", "content": assistant_contents[idx]})
    messages.append({"role": "user", "content": user_contents[-1]})

    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def use_anthropic_api(assistant_contents, user_contents, system, model, temperature):
    num_assistant_mes = len(assistant_contents)
    messages = []

    # messages.append({"role": "system", "content": [{"type": "text", "text": "{}".format(system)}]})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": [{"type": "text", "text": user_contents[idx]}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_contents[idx]}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": user_contents[-1]}]})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=1,
        system=system,
        messages=messages
    )

    result = ''
    for choice in message.content:
        result += choice.text

    return result


def use_groq_api(assistant_contents, user_contents, system, model, temperature=0.7):
    num_assistant_mes = len(assistant_contents)
    messages = []
    
    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx]})
        messages.append({"role": "assistant", "content": assistant_contents[idx]})
    messages.append({"role": "user", "content": user_contents[-1]})

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    result = ''
    result += response.choices[0].message.content
    
    return result


def use_ollama_api(assistant_contents, user_contents, system, model, temperature=0.7):
    num_assistant_mes = len(assistant_contents)
    messages = []
    
    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx]})
        messages.append({"role": "assistant", "content": assistant_contents[idx]})
    messages.append({"role": "user", "content": user_contents[-1]})

    response: ChatResponse = chat(model=model, messages=messages)
        
    result = ''
    # TODO assume there is only a single choice
    result += response.message.content

    return result


def query(system, user_contents, assistant_contents, model=None, save_path=None, temperature=1, debug=False):
    # Use MODEL environment variable if provided, otherwise use the passed model parameter or defaults
    if model is None:
        model = os.getenv('MODEL')
        if model is None:
            # Set default models based on provider
            if os.getenv("TARGET_MODEL_PROVIDER") == "openai":
                model = 'gpt-4'
            elif os.getenv("TARGET_MODEL_PROVIDER") == "anthropic":
                model = 'claude-opus-4-20250514'
            elif os.getenv("TARGET_MODEL_PROVIDER") == "groq":
                model = 'llama-3.3-70b-versatile'
            elif os.getenv("TARGET_MODEL_PROVIDER") == "ollama":
                model = 'llama3.2:3b'
            else:
                model = 'gpt-4'  # fallback default

    for user_content, assistant_content in zip(user_contents, assistant_contents):
        user_content = user_content.split("\n")
        assistant_content = assistant_content.split("\n")

        for u in user_content:
            print(u)
        print("=====================================")
        for a in assistant_content:
            print(a)
        print("=====================================")

    for u in user_contents[-1].split("\n"):
        print(u)

    if debug:
        import pdb; pdb.set_trace()
        return None

    print("=====================================")

    start = time.time()

    if os.getenv("TARGET_MODEL_PROVIDER") == "openai":
        result = use_openai_api(assistant_contents, user_contents, system, model, temperature)
    elif os.getenv("TARGET_MODEL_PROVIDER") == "anthropic":
        result = use_anthropic_api(assistant_contents, user_contents, system, model, temperature)
    elif os.getenv("TARGET_MODEL_PROVIDER") == "groq":
        result = use_groq_api(assistant_contents, user_contents, system, model, temperature)
    elif os.getenv("TARGET_MODEL_PROVIDER") == "ollama":
        result = use_ollama_api(assistant_contents, user_contents, system, model, temperature)
    else:
        raise ValueError("Invalid target model provider. Please set the environment variable TARGET_MODEL_PROVIDER to 'openai', 'anthropic', 'groq', or 'ollama'.")

    end = time.time()
    used_time = end - start

    print(result)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump({"used_time": used_time, "res": result, "system": system, "user": user_contents, "assistant": assistant_contents}, f, indent=4)

    return result
