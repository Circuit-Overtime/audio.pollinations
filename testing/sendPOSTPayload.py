import requests
import base64
from pydub import AudioSegment
import os

input_audio_path = "testing/W8i19O5P6L.wav"
wav_audio_path = "voice.wav"

# Convert input audio (any format) to wav
audio = AudioSegment.from_file(input_audio_path)
audio.export(wav_audio_path, format="wav")

# Read the wav file and encode as base64
with open(wav_audio_path, "rb") as f:
    audio_bytes = f.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")


url = "http://localhost:8000/audio"
payload = {
    "messages": [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "An extremely excited person talking about a new technology product."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Verbtim: Omg Omg omg!! This is amazing! I can't believe it actually works."},
                {
                    "type": "voice",
                    "voice": {
                        "data": audio_b64,   
                        "format": "wav"
                    }
                }
            ]
        }
    ],
    "seed": 43
}

response = requests.post(url, json=payload)

if response.headers.get("Content-Type") == "audio/wav":
    with open("output2.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved as output2.wav")
    os.remove("voice.wav")
else:
    print(response.json())