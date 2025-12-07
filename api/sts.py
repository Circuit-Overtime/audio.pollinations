from utility import encode_audio_base64, save_temp_audio, validate_and_decode_base64_audio, convertToAudio
from templates import create_speaker_chat
from multiprocessing.managers import BaseManager
from voiceMap import VOICE_BASE64_MAP
from typing import Optional
import asyncio
import loggerConfig
import torchaudio
import torch 
from intent import getContentRefined
import io
import time
from timing_stat import TimingStats


class ModelManager(BaseManager): pass
ModelManager.register("Service")
manager = ModelManager(address=("localhost", 6000), authkey=b"secret")
manager.connect()
service = manager.Service()

async def generate_sts(text: str, audio_base64_path: str, requestID: str, system: Optional[str] = None, clone_text: Optional[str] = None, voice: Optional[str] = "alloy", timing_stats: Optional[object] = None) -> bytes:
    if timing_stats is None:
        timing_stats = TimingStats(requestID)
    
    if voice and not VOICE_BASE64_MAP.get(voice):
        with open(voice, "r") as f:
            audio_data = f.read()
            if validate_and_decode_base64_audio(audio_data):
                clone_path = voice
    elif voice and VOICE_BASE64_MAP.get(voice):
        clone_path = VOICE_BASE64_MAP.get(voice)
    else:
        clone_path = VOICE_BASE64_MAP.get("alloy")

    
    
    timing_stats.start_timer("STS_TRANSCRIPTION")
    transcription = service.transcribe(audio_base64_path, requestID)
    timing_stats.end_timer("STS_TRANSCRIPTION")
    
    print(f"Transcription result: {transcription[:100]}...")
    
    timing_stats.start_timer("STS_INTENT_DETECTION")
    intention_detection = await getContentRefined(f"This is the prompt and {text} and this is the audio transcript {transcription}", system)
    timing_stats.end_timer("STS_INTENT_DETECTION")
    
    intention = intention_detection.get("intent")
    content = intention_detection.get("content")
    system = intention_detection.get("system_instruction")
    
    prepareChatTemplate = create_speaker_chat(
        text=content,
        requestID=requestID,
        system=system,
        clone_audio_path=clone_path,
    )
    
    timing_stats.start_timer("STS_SYNTHESIS")
    audio_numpy, audio_sample = service.speechSynthesis(chatTemplate=prepareChatTemplate)
    timing_stats.end_timer("STS_SYNTHESIS")
    
    audio_tensor = torch.from_numpy(audio_numpy).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, audio_sample, format="wav")
    audio_bytes = buffer.getvalue()
    return audio_bytes, audio_sample


if __name__ == "__main__":
    async def main():
        text = "How do you feel about this?"
        audio = "testing/W8i19O5P6L.wav"
        base64_audio = encode_audio_base64(audio)
        saved_audio_path = save_temp_audio(base64_audio, "request224", "speech")
        audio_conv = convertToAudio(saved_audio_path, "request224")
        requestID = "request123"
        system = None
        clone_text = None
        voice = None

        audio_bytes, sample_rate = await generate_sts(text, audio_conv, requestID, system, clone_text, voice)
        with open(f"{requestID}.wav", "wb") as f:
            f.write(audio_bytes)
        print(f"Generated audio saved as {requestID}.wav with sample rate {sample_rate}")
    
    asyncio.run(main())