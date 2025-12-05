import asyncio
from loguru import logger
import time 
from typing import Optional
import os 
from dotenv import load_dotenv
import requests
from config import POLLINATIONS_ENDPOINT
import random

load_dotenv()


async def generate_reply(prompt: str, max_tokens: Optional[int] = 5000) -> str:
    logger.info(f"Generating reply for prompt: {prompt} with max tokens: {max_tokens}")
    start_time = time.time()
    payload = {
        
        "model": os.getenv("MODEL"),
        "messages": [
            {
                "role": "system",
                "content": """You are a friendly, natural conversational AI. 
                Generate short, casual replies unless the user explicitly requests longer content.
                No scripts, narration, essays, or long paragraphs unless the user directly asks for them.
                Never use emojis or special characters—only plain text (letters, numbers, punctuation).
                Token-length guide:
                1 minute of spoken audio ≈ 150–180 tokens.
                To estimate needed tokens: tokens = minutes * 160 (approx).
                Never exceed the max_tokens for the request; scale down if needed.
                If the user requests X minutes, generate only the amount of text that fits: 
                min(X * 160, max_tokens).
                If no duration is requested, keep replies short.
                Always sound natural and human. 
                Don't include any overhead in the response just the raw response text.
                """
            },
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ],
        "temperature": 1,
        "stream": False,
        "private": True,
        "referrer": "elixpoart",
        "max_tokens": max_tokens,
        "seed": random.randint(1000, 1000000)
    }

    header = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {os.getenv('POLLI_TOKEN')}"
    }

    try:
        response = requests.post(POLLINATIONS_ENDPOINT, headers=header, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")

        data = response.json()
        try:
            reply: str = data["choices"][0]["message"]["content"]
            if "---" in reply and "**Sponsor**" in reply:
                sponsor_start = reply.find("---")
                if sponsor_start != -1:
                    sponsor_section = reply[sponsor_start:]
                    if "**Sponsor**" in sponsor_section:
                        reply = reply[:sponsor_start].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected response format: {data}") from e
        elapsed_time = time.time() - start_time
        logger.info(f"Reply generated in {elapsed_time:.2f} seconds")
        return reply.strip()
    except requests.exceptions.Timeout:
        logger.warning("Timeout occurred in generate_reply, returning generic system instruction.")
        return f"{prompt}"
    
async def generate_ttt(text: str, requestID:str, system: str = None):
    replyText = await generate_reply(f"Prompt: {text} & System: {system}", 300)
    print(f"The generated reply text is: {replyText}")
    return replyText

if __name__ == "__main__":
    async def main():
        text = "Make me a story of the lion king"
        requestID = "request123"
        system = "This is a very tense story"
        await generate_ttt(text, requestID, system)

    asyncio.run(main())