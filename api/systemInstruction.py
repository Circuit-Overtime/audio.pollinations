import requests
from dotenv import load_dotenv
import os
from loguru import logger
import asyncio
import random
from config import POLLINATIONS_ENDPOINT
import time 
load_dotenv()

async def generate_higgs_system_instruction(text: str) -> str:
    logger.info(f"Generating Higgs system instruction for text: {text}")
    start_time = time.time()
    base_instruction = """You create scene descriptions for the Higgs 
    speech model. Describe only how the text should be spoken: 
    voice texture, tone, emotion, pacing, rhythm, environment, and vocal 
    character. No dialogue or content. Keep language plain and natural. 
    Fit the entire output within 120 tokens.
    RESPONSE FORMAT STRICTLY -- INCLUDE ():
    (
    "You are a masterful voice performer adding human emotion and nuance."
    "Use expressive tone, natural pacing, and subtle imperfections."
    "Generate audio following instruction."
    "<|scene_desc_start|>"
    [SCENE DESCRIPTION HERE]
    "<|scene_desc_end|>"
    )
""" 

    header = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {os.getenv('POLLI_TOKEN')}"
    }

    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": f"Text to analyze for vocal style: {text}"}
        ],
        "temperature": 1,
        "stream": False,
        "private": True,
        "referrer": "elixpoart",
        "max_tokens": 200,
        "seed": random.randint(1000, 1000000)
    }
    
    try:
        response = requests.post(POLLINATIONS_ENDPOINT, headers=header, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        try:
            system_instruction: str = data["choices"][0]["message"]["content"]
            if "---" in system_instruction and "**Sponsor**" in system_instruction:
                sponsor_start = system_instruction.find("---")
                system_instruction = system_instruction[:sponsor_start].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected response format: {data}") from e
        elapsed_time = time.time() - start_time
        logger.info(f"Higgs system instruction generated in {elapsed_time:.2f} seconds")
        return system_instruction.strip()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return await _get_fallback_instruction(text)
    except requests.exceptions.Timeout:
        logger.warning("Request timed out, using fallback")
        return await _get_fallback_instruction(text)




async def _get_fallback_instruction(text: str) -> str:
    text_lower = text.lower()
    if any(word in text_lower for word in ["exciting", "amazing", "wow", "awesome", "incredible"]):
        scene = "Speak with vibrant enthusiasm and bright energy. Voice should sparkle with genuine excitement, using animated tones and lively pacing. Create an atmosphere of joy and wonder, as if sharing thrilling news with a close friend."
    elif any(word in text_lower for word in ["calm", "peaceful", "gentle", "soft", "quiet"]):
        scene = "Use a soft, soothing voice with gentle warmth. Speak with measured, peaceful pacing in an intimate, comforting atmosphere. Voice should feel like a warm embrace, creating a sense of tranquility and safety."
    elif any(word in text_lower for word in ["serious", "important", "urgent", "critical"]):
        scene = "Adopt a clear, authoritative tone with focused intensity. Speak with measured gravity and purposeful pacing. Create an atmosphere of importance and attention, like addressing a meaningful gathering."
    elif any(word in text_lower for word in ["funny", "joke", "laugh", "humor", "silly"]):
        scene = "Use a playful, warm voice with natural chuckles and light-hearted energy. Speak with bouncy rhythm and mischievous undertones. Create a jovial atmosphere filled with warmth and good humor."
    else:
        scene = "Speak with natural conversational warmth and genuine human connection. Use a balanced, expressive voice with organic pacing and authentic emotional undertones. Create a comfortable, engaging atmosphere like chatting with a trusted friend."
    
    return f"""
    (
    "You are a masterful voice performer bringing text to life with authentic human artistry."
    "Channel the energy of a skilled actor - make every word breathe with genuine emotion and personality."
    "Use natural vocal textures, micro-pauses, emotional inflections, and dynamic pacing to create a captivating performance."
    "Avoid robotic delivery - embrace the beautiful imperfections and nuances of human speech."
    "Generate audio following instruction."
    "<|scene_desc_start|>"
    {scene}
    "<|scene_desc_end|>"
    )
        """

if __name__ == "__main__":
    async def main():
        user_prompt = "generate me a story about a brave little toaster for about 4 minutes of speech time"

        system_instruction = await generate_higgs_system_instruction(user_prompt)
        print("\n--- Generated Higgs System Instruction ---\n")
        print(system_instruction)
    asyncio.run(main())