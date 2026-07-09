import os
import sys
from langchain_openai import ChatOpenAI

key = "tgp_v1_4ZrbdE0dFCK6RnuZxiTtog6wesCBrX3ApQLpUiN3Bxg"
prompt = "Write an extremely detailed research section (at least 1500 words) about the demographic aging trends in Japan from 2020 to 2050. Explain cohorts (65-74, 75-84, 85+), fertility rates, and population decline. Use detailed analytical prose and expand on mechanisms."

for model_name in ["google/gemma-4-31B-it", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=key,
            base_url="https://api.together.xyz/v1",
            max_tokens=4000
        )
        print(f"Calling {model_name}...")
        res = model.invoke(prompt)
        print(f"{model_name} response word count:", len(res.content.split()))
    except Exception as e:
        print(f"{model_name} failed:", e)
