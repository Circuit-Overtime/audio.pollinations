from templates import create_speaker_chat
from utility import cleanup_temp_file, validate_and_decode_base64_audio
from voiceMap import VOICE_BASE64_MAP
import asyncio
from typing import Optional
from multiprocessing.managers import BaseManager
import os
import threading
import time
import torch
import torchaudio
import io 
import numpy as np
import time
from timing_stat import TimingStats

class ModelManager(BaseManager): pass
ModelManager.register("Service")
manager = ModelManager(address=("localhost", 6000), authkey=b"secret")
manager.connect()
service = manager.Service()

async def generate_tts(text: str, requestID: str, system: Optional[str] = None, clone_text: Optional[str] = None, voice_1: Optional[str] = "alloy", voice_2: Optional[str] = None, clone_text_2: Optional[str] = None, timing_stat: Optional[object] = None) -> tuple:
    if timing_stat is None:
        timing_stat = TimingStats(requestID)

    if voice_1 and not VOICE_BASE64_MAP.get(voice_1):
        with open(voice_1, "r") as f:
            audio_data = f.read()
            if validate_and_decode_base64_audio(audio_data):
                clone_path = voice_1
    elif voice_1 and VOICE_BASE64_MAP.get(voice_1):
        clone_path = VOICE_BASE64_MAP.get(voice_1)
    else:
        clone_path = VOICE_BASE64_MAP.get("alloy")

    clone_path_2 = None
    if voice_2:
        if not VOICE_BASE64_MAP.get(voice_2):
            with open(voice_2, "r") as f:
                audio_data = f.read()
                if validate_and_decode_base64_audio(audio_data):
                    clone_path_2 = voice_2
        elif VOICE_BASE64_MAP.get(voice_2):
            clone_path_2 = VOICE_BASE64_MAP.get(voice_2)

    if system:
        system = f"""
        (
        "Generate audio following instruction\n\n."
        "<|scene_desc_start|>\n"
        "{system} \n"
        "<|scene_desc_end|>"
        )
        """
    if not system:
        if clone_path_2:
            system = """ 
            (
            Generate audio following instruction with two speakers.
            <|scene_desc_start|>
            SPEAKER0: slow-moderate pace;storytelling cadence;warm expressive tone;emotional nuance;dynamic prosody;subtle breaths;smooth inflection shifts;gentle emphasis;present and human;balanced pitch control
            SPEAKER1: distinct voice characteristics;different tone from SPEAKER0;maintain conversation flow;natural dialogue exchange
            <|scene_desc_end|>
            )
            """
        else:
            system = """ 
            (
            Generate audio following instruction.
            <|scene_desc_start|>
            SPEAKER0: slow-moderate pace;storytelling cadence;warm expressive tone;emotional nuance;dynamic prosody;subtle breaths;smooth inflection shifts;gentle emphasis;present and human;balanced pitch control
            <|scene_desc_end|>
            )
            """
        
    prepareChatTemplate = create_speaker_chat(
        text=text,
        requestID=requestID,
        system=system,
        clone_audio_path=clone_path,
        clone_audio_path_2=clone_path_2,
        clone_audio_transcript=clone_text,
        clone_audio_transcript_2=clone_text_2
    )

    print(f"Generating Audio for {requestID}")
    timing_stat.start_timer("TTS_AUDIO_GENERATION")
    audio_numpy, audio_sample = service.speechSynthesis(chatTemplate=prepareChatTemplate)
    timing_stat.end_timer("TTS_AUDIO_GENERATION")
    audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, audio_sample, format="wav")
    return audio_numpy, audio_sample

if __name__ == "__main__":
    class ModelManager(BaseManager): pass
    ModelManager.register("Service")
    manager = ModelManager(address=("localhost", 6000), authkey=b"secret")
    manager.connect()
    service = manager.Service()

    async def main():
        text = """
        <|generation_instruction_start|> 
        Use the male voice for this segment. Maintain natural pacing and conversational tone.
        <|generation_instruction_end|>
        Hey, do you have any plans for the weekend?

        <|generation_instruction_start|> 
        Use the female voice for this segment. Maintain warm, relaxed, conversational tone.
        <|generation_instruction_end|>
        I was thinking we could go hiking if the weather is nice.

        <|generation_instruction_start|> 
        Use the male voice for this segment. Keep an upbeat, engaged tone.
        <|generation_instruction_end|>
        That sounds great. Should we invite the kids?

        <|generation_instruction_start|> 
        Use the female voice for this segment. Gentle, cheerful delivery.
        <|generation_instruction_end|>
        Absolutely, they’ll love it. Let’s pack a picnic too.

        <|generation_instruction_start|> 
        Use the male voice for this segment. Calm, confident tone.
        <|generation_instruction_end|>
        Perfect. I’ll check the forecast and get everything ready.

        """
        requestID = "request123"
        system = None
        voice = "alloy"
        voice_2 = "ash"
        clone_text = None
        clone_text_2 = None
        
        def cleanup_cache():
            while True:
                try:
                    service.cleanup_old_cache_files()
                except Exception as e:
                    print(f"Cleanup error: {e}")

                time.sleep(600)

        cleanup_thread = threading.Thread(target=cleanup_cache, daemon=True)
        cleanup_thread.start()
        cache_name = service.cacheName(text)
        # if os.path.exists(f"genAudio/{cache_name}.wav"):
        #     print(f"Cache hit: genAudio/{cache_name}.wav already exists.")
        #     return
        
        audio_numpy, audio_sample = await generate_tts(text, requestID, system, clone_text, voice, voice_2, clone_text_2)
        audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0)
        torchaudio.save(f"{cache_name}.wav", audio_tensor, audio_sample)
        torchaudio.save(f"genAudio/{cache_name}.wav", audio_tensor, audio_sample)
        print(f"Audio saved as {cache_name}.wav")

    asyncio.run(main())