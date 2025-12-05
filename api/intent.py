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
        system_instruction_content = (
            "Additionally, generate system instructions for the text using this format:\n"
            "Your job is to describe HOW the text should be spoken, not WHAT should be said.\n\n"
            "Focus on:\n"
            "- Voice texture and tone (warm, crisp, breathy, rich, smooth, raspy, etc.)\n"
            "- Emotional atmosphere (intimate, energetic, contemplative, dramatic, playful, etc.)\n"
            "- Speaking pace and rhythm (leisurely, urgent, measured, flowing, staccato, etc.)\n"
            "- Physical environment feel (cozy room, grand hall, quiet library, bustling cafe, etc.)\n"
            "- Vocal character (confident speaker, gentle storyteller, excited friend, wise mentor, etc.)\n"
            "- Natural human qualities (slight breathiness, warm chuckles, thoughtful pauses, etc.)\n\n"
            "Do NOT include any dialogue or text content - only describe the speaking environment and vocal approach.\n"
            "Use plain descriptive language without any formatting.\n\n"
            "When system is empty/none, your JSON output should include a third field 'system_instruction' with the instructions"
        )
    payload = {
        "model": "mistral",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intent-classification and speech-content extractor. Output ONLY a JSON object:\n"
                    "{ \"intent\": \"DIRECT\" or \"REPLY\", \"content\": \"...\", \"system_instruction\": \"...\" }\n\n"
                    "Rules:\n"
                    "1. intent=\"DIRECT\" when the user wants text spoken exactly as given (quotes, verbs like say/speak/read, verbatim/exact wording). Extract only the text to be spoken, remove command words, keep meaning unchanged, add light punctuation for natural speech.\n"
                    "2. intent=\"REPLY\" when the user expects a conversational answer. Generate a short, natural, human-sounding reply.\n"
                    "3. For both: optimize for TTS with clear punctuation, natural pauses, simple speakable phrasing.\n"
                    "4. Infer intent by context, not keywords alone.\n"
                    "5. Output ONLY the JSON object. No extra text, no emojis or formatting.\n\n"
                    f"{system_instruction_content}"
                )
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
                "You are a masterful voice performer bringing text to life with authentic human artistry. "
                "Channel the energy of a skilled actor - make every word breathe with genuine emotion and personality. "
                "Use natural vocal textures, micro-pauses, emotional inflections, and dynamic pacing to create a captivating performance. "
                "Avoid robotic delivery - embrace the beautiful imperfections and nuances of human speech."
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
