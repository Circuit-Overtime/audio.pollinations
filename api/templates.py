import json
from typing import Optional
import os
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from loguru import logger
from utility import normalize_text
import sys
from config import TEMP_SAVE_DIR


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_speaker_chat(
    text: str,
    requestID: str,
    system: Optional[str] = None,
    clone_audio_path: Optional[str] = None,
    clone_audio_path_2: Optional[str] = None,
    clone_audio_transcript: Optional[str] = None,
    clone_audio_transcript_2: Optional[str] = None
) -> ChatMLSample:
    logger.info(f"Creating chat template for request {requestID} with text: {text}")
    messages = []
    if system:
        if "<|scene_desc_start|>" not in system or "<|scene_desc_end|>" not in system:
            systemPromptWrapper: str = f"""
                (
                Generate audio following instruction.\n
                <|scene_desc_start|>\n
                "{system}"
                <|scene_desc_end|>
                )
            """
        else:
            systemPromptWrapper: str = system

        
        messages.append(
            Message(
                role="system",
                content=systemPromptWrapper,
            )
        )


    if clone_audio_path:
        with open(clone_audio_path, "r") as f:
            reference_audio_data = f.read()
        messages.append(
            Message(
                role="user",
                content="Please use this voice as a female voice",
            )
        )
        
        if clone_audio_transcript:
            messages.append(
                Message(
                    role="user",
                    content=normalize_text(clone_audio_transcript),
                )
            )
        else:
            messages.append(
                Message(
                    role="user",
                    content="Please clone this voice.",
                )
            )
        
        audio_content_list = [AudioContent(raw_audio=reference_audio_data, audio_url="")]
        
        if clone_audio_path_2:
            with open(clone_audio_path_2, "r") as f:
                reference_audio_data_2 = f.read()
            messages.append(
                Message(
                    role="assistant",  
                    content=audio_content_list,
                )
            )
            messages.append(
                Message(
                    role="user",
                    content="Please use this voice as a male voice",
                )
            )
            
            if clone_audio_transcript_2:
                messages.append(
                    Message(
                        role="user",
                        content=normalize_text(clone_audio_transcript_2),
                    )
                )
            
            audio_content_list = [AudioContent(raw_audio=reference_audio_data_2, audio_url="")]
            messages.append(
                Message(
                    role="assistant",  
                    content=audio_content_list,
                )
            )
        else:
            messages.append(
                Message(
                    role="assistant",  
                    content=audio_content_list,
                )
            )

    messages.append(
        Message(
            role="user",
            content=text
        )
    )
    return ChatMLSample(messages=messages)


if __name__ == "__main__":
    template = create_speaker_chat(
        "Woohooo!! This is super awesomeeeee!!!",
        "request12",
        "An energetic ambience! Don't add any overhead sounds or tokens, just say as much",
    )
    # print(template)
    print(f"Chat template saved to {template}")