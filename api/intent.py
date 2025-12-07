from loguru import logger
import requests
from typing import Optional
from dotenv import load_dotenv
import os
import asyncio
import loggerConfig

load_dotenv()


async def getContentRefined(text: str, system: Optional[str] = None, max_tokens: Optional[int] = 3000) -> dict:
    logger.info(f"Classifying intent and extracting content for prompt: {text} with max tokens: {max_tokens}")

    system_instruction_content = ""
    if not system:
        system_instruction_content = """
            Generate system instructions describing how the provided text should be spoken. 
            Do not repeat or reference the actual text. Your job is to describe the vocal performance style only.
            Focus on:
            Voice texture and tone (warm, crisp, breathy, rich, smooth, raspy, etc.)
            Emotional atmosphere (intimate, energetic, contemplative, dramatic, playful, etc.)
            Speaking pace and rhythm (leisurely, urgent, measured, flowing, slow-moderate, etc.)
            Physical environment feel (cozy room, grand hall, quiet library, nighttime outdoors, etc.)
            Vocal character (gentle storyteller, confident narrator, wise mentor, excited friend, etc.)
            Human qualities (slight breathiness, micro-pauses, natural inflection, soft chuckles, etc.)
            Example System Instruction -- SPEAKER0: slow-moderate pace;storytelling cadence;warm expressive tone;emotional nuance;dynamic prosody;subtle breaths;smooth inflection shifts;gentle emphasis;present and human;balanced pitch control
            USE multiple SPEAKER0: SPEAKER1: SPEAKER2: tags if different voices or styles are implied.
            Also apply the same speaker notation in the script for TTS synthesis.
            """
        
    payload = {
        "model": "mistral",
        "messages": [
            {
                "role": "system",
                "content": """
                    You are an intent-classification and speech-content extractor. Output ONLY a JSON object:
                    { \"intent\": \"DIRECT\" or \"REPLY\", \"content\": \"...\", \"system_instruction\": \"...\" }
                    Rules:
                    1. intent=\"DIRECT\" when the user wants text spoken exactly as given (quotes, verbs like say/speak/read, verbatim/exact wording). Extract only the text to be spoken, remove command words, keep meaning unchanged, add light punctuation for natural speech.
                    2. intent=\"REPLY\" when the user expects a conversational answer. Generate a short, natural, human-sounding reply.
                    3. For both: optimize for TTS with clear punctuation, natural pauses, simple speakable phrasing.
                    4. Infer intent by context, not keywords alone.
                    5. Output ONLY the JSON object. No extra text, no emojis or formatting.
                    6. If it's a REPLY don't send back the exact user prompt - generate a new natural response.
                    \n
                    """+
                    f"{system_instruction_content}"
                
            },
            {
                "role": "user",
                "content": f"Prompt: {text}\nSystem: {system if system else 'None - generate system instruction'}"
            }
        ],
        "temperature": 0.7,
        "stream": False,
        "private": True,
        "token": os.getenv("POLLI_TOKEN"),
        "referrer": "elixpoart",
        "max_tokens": max_tokens,
        "json": True,
    }
    header = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {os.getenv('POLLI_TOKEN')}"
    }

    try:
        response = requests.post("https://enter.pollinations.ai/api/generate/v1/chat/completions", json=payload, headers=header, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")

        data = response.json()
        try:
            reply = data["choices"][0]["message"]["content"]
            import json as pyjson
            result = pyjson.loads(reply)
            required_fields = ["intent", "content"]
            if not system:
                required_fields.append("system_instruction")

            for field in required_fields:
                assert field in result
        except Exception as e:
            raise RuntimeError(f"Unexpected response format: {data}") from e

        logger.info(f"Intent and content: {result}")
        return result

    except requests.exceptions.Timeout:
        logger.warning("Timeout occurred in getContentRefined, returning default DIRECT.")
        default_result = {"intent": "DIRECT", "content": text}
        if not system:
            default_result["system_instruction"] = (
                "SPEAKER0: slow-moderate pace;storytelling cadence;warm expressive tone;emotional nuance;dynamic prosody;subtle breaths;smooth inflection shifts;gentle emphasis;present and human;balanced pitch control"
            )
        return default_result


if __name__ == "__main__":
    async def main():
        test_text = "Wow, that was an amazing performance! How did you manage to pull that off?"
        print(f"\nTesting: {test_text}")
        result = await getContentRefined(test_text, None)
        print(f"Intent: {result.get('intent')}")
        print(f"Content: {result.get('content')}")
        print(f"System Instruction: {result.get('system_instruction')}")
        print("-" * 50)

    asyncio.run(main())
