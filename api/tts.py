from templates import create_speaker_chat
from utility import encode_audio_base64, validate_and_decode_base64_audio, save_temp_audio
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

async def generate_tts(text: str, requestID: str, system: Optional[str] = None, clone_text: Optional[str] = None, voice: Optional[str] = "alloy") -> tuple:
    if timing_stat is None:
        timing_stat = TimingStats(requestID)

    if voice and not VOICE_BASE64_MAP.get(voice):
        with open(voice, "r") as f:
            audio_data = f.read()
            if validate_and_decode_base64_audio(audio_data):
                clone_path = voice
    else:
        load_audio_path = VOICE_BASE64_MAP.get("alloy")
        base64_data = encode_audio_base64(load_audio_path)    
        clone_path = save_temp_audio(base64_data, requestID, "clone")

    if not system:
        system = "Neutral tone, clear articulation, natural pacing."
        
    
    prepareChatTemplate = create_speaker_chat(
        text=text,
        requestID=requestID,
        system=system,
        clone_audio_path=clone_path,
        clone_audio_transcript=clone_text
    )
        
    # print(f"Chat Template PrepareChatTemplate: \n {prepareChatTemplate}")

    print(f"Generating Audio for {requestID}")
    timing_stat.start_timer("TTS_AUDIO_GENERATION")
    audio_numpy, audio_sample = service.speechSynthesis(chatTemplate=prepareChatTemplate)
    timing_stat.end_timer("TTS_AUDIO_GENERATION")
    audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, audio_sample, format="wav")
    audio_bytes = buffer.getvalue()
    
    return audio_numpy, audio_sample

if __name__ == "__main__":
    class ModelManager(BaseManager): pass
    ModelManager.register("Service")
    manager = ModelManager(address=("localhost", 6000), authkey=b"secret")
    manager.connect()
    service = manager.Service()

    async def main():
        text = "Such a beautiful day to start with!! What's on your mind?"
        requestID = "request123"
        system = None
        voice = "alloy"
        clone_text = None
        
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
        
        audio_numpy, audio_sample = await generate_tts(text, requestID, system, clone_text, voice)
        audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0)
        torchaudio.save(f"{cache_name}.wav", audio_tensor, audio_sample)
        torchaudio.save(f"genAudio/{cache_name}.wav", audio_tensor, audio_sample)
        print(f"Audio saved as {cache_name}.wav")

    asyncio.run(main())