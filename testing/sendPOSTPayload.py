import requests
import base64
from pydub import AudioSegment
import os

audio = AudioSegment.from_file("testing/W8i19O5P6L.wav", format="wav")
audio.export("voice.wav", format="wav")
with open("voice.wav", "rb") as f:
    audio_bytes = f.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
url = "http://localhost:8000/audio"
payload = {
    "messages": [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You generate audio from the users input"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me a cool reply of the transcribed audio"},
                {
                    "type": "voice",
                    "voice": {
                        "name": "ash",
                        "format": "wav"
                    }
                },
                {
                    "type": "speech_audio",
                    "audio": {"data": f"{audio_b64}", "format": "wav"}
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