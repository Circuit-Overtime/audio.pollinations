import requests
import json
import os
import random
import logging
import asyncio
import shutil
from typing import Optional
import torch
import torchaudio
from tools import tools
from config import TEMP_SAVE_DIR, POLLINATIONS_ENDPOINT, TRIAL_MODE
from utility import encode_audio_base64, save_temp_audio, cleanup_temp_file, validate_and_decode_base64_audio
from requestID import reqID
from voiceMap import VOICE_BASE64_MAP
from main_instruction import inst, user_inst
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elixpo-audio")

POLLINATIONS_TOKEN = os.getenv("POLLI_TOKEN")
MODEL = os.getenv("MODEL")
REFERRER = os.getenv("REFERRER")
print(f"Token: {POLLINATIONS_TOKEN} Model: {MODEL} Referrer: {REFERRER}")



async def run_audio_pipeline(
    reqID: str = None,
    text: str = None,
    voice: str = None,
    synthesis_audio_path: Optional[str] = None, 
    clone_audio_transcript: Optional[str] = None,
    system_instruction: Optional[str] = None 
):
    text = text.strip()
    print(f"Recieved the parameters: {reqID}, {text}, {voice}, {synthesis_audio_path}, {clone_audio_transcript}, {system_instruction}")
    logger.info(f" [{reqID}] Starting Audio Pipeline")
    logger.info(f"Synthesis audio {synthesis_audio_path} | Clone Audio {voice}")
    higgs_dir = f"/tmp/higgs/{reqID}"
    os.makedirs(higgs_dir, exist_ok=True)
    logger.info(f"[{reqID}] Created higgs directory: {higgs_dir}")    
    logger.info(f"[{reqID}] Saved base64 for the required media")
    try:
        messages = [
        {
        "role": "system",
        "content": f"{inst}"
        },

        {
        "role": "user",
        "content": user_inst(reqID, text, synthesis_audio_path, system_instruction, voice, clone_audio_transcript)
        }
    ]
        
        max_iterations = 1
        current_iteration = 0
        while current_iteration < max_iterations:
            current_iteration += 1
            logger.info(f"Iteration {current_iteration} for reqID={reqID}")

            payload = {
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "n": 1,
                "seed": random.randint(1000, 9999),
                "max_tokens": 3500,
                "temperature": 1,
                "top_p": 1,
                "stream": False,
                "retry": {}
            }

            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {POLLINATIONS_TOKEN}"}
            try:
                response = requests.post(POLLINATIONS_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
            except requests.exceptions.RequestException as e:
                error_text = getattr(e.response, "text", "[No error text]")
                logger.error(f"Pollinations API call failed: {e}\n{error_text}")
                break
            assistant_message = response_data["choices"][0]["message"]
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls")
            print(f"Tool calls {tool_calls} received for reqID={reqID}")

            if not tool_calls:
                final_content = assistant_message.get("content")
                if final_content:
                    logger.info(f"Final response: {final_content}")
                    return {
                        "type": "error",
                        "message": "No pipeline was executed",
                        "reqID": reqID
                    }
                break

            tool_outputs = []
            for tool_call in tool_calls:
                print(f"[{reqID}] Processing tool call: {tool_call['id']}")
                fn_name = tool_call["function"]["name"]
                fn_args = json.loads(tool_call["function"]["arguments"])
                logger.info(f"[reqID={reqID}] Executing pipeline: {fn_name} with args: {fn_args}")
                if not TRIAL_MODE:
                    try:
                        if fn_name == "generate_tts":
                            from tts import generate_tts 
                            logger.info(f"[{reqID}] Calling TTS pipeline")
                            audio_bytes,sample_rate = await generate_tts(
                                text=fn_args.get("text"),
                                requestID=fn_args.get("requestID"),
                                system=fn_args.get("system"),
                                clone_text=fn_args.get("clone_text"),
                                voice=fn_args.get("voice"),
                            )

                            os.makedirs("genAudio", exist_ok=True)
                            gen_audio_path = f"genAudio/{reqID}.wav"
                            with open(gen_audio_path, "wb") as f:
                                f.write(audio_bytes)
                            logger.info(f"[{reqID}] TTS audio saved to: {gen_audio_path}")

                            return {
                                "type": "audio",
                                "data": audio_bytes,
                                "file_path": gen_audio_path,
                                "reqID": reqID
                            }

                        elif fn_name == "generate_ttt":
                            from ttt import generate_ttt
                            logger.info(f"[{reqID}] Calling TTT pipeline")
                            text_result = await generate_ttt(
                                text=fn_args.get("text"),
                                requestID=fn_args.get("requestID"),
                                system=fn_args.get("system")
                            )
                            
                            text_path = os.path.join(higgs_dir, f"{reqID}.txt")
                            with open(text_path, "w", encoding="utf-8") as f:
                                f.write(text_result)
                            
                            logger.info(f"[{reqID}] TTT text saved to: {text_path}")
                            
                            return {
                                "type": "text",
                                "data": text_result,
                                "file_path": text_path,
                                "reqID": reqID
                            }

                        elif fn_name == "generate_sts":
                            from sts import generate_sts
                            logger.info(f"[{reqID}] Calling STS pipeline")
                            audio_bytes, sample_rate = await generate_sts(
                                text=fn_args.get("text"),
                                audio_base64_path=fn_args.get("synthesis_audio_path"),
                                requestID=fn_args.get("requestID"),
                                system=fn_args.get("system"),
                                clone_text=fn_args.get("clone_text"),
                                voice=fn_args.get("voice", "alloy")
                            )
                            
                            
                            os.makedirs("genAudio", exist_ok=True)
                            gen_audio_path = f"genAudio/{reqID}.wav"
                            with open(gen_audio_path, "wb") as f:
                                f.write(audio_bytes)
                            logger.info(f"[{reqID}] TTS audio saved to: {gen_audio_path}")

                            return {
                                "type": "audio",
                                "data": audio_bytes,
                                "file_path": gen_audio_path,
                                "reqID": reqID
                            }

                        elif fn_name == "generate_stt":
                            from stt import generate_stt
                            logger.info(f"[{reqID}] Calling STT pipeline")
                            text_result = await generate_stt(
                                text=fn_args.get("text"),
                                audio_base64_path=fn_args.get("synthesis_audio_path"),
                                requestID=fn_args.get("requestID"),
                                system=fn_args.get("system")
                            )

                            text_path = os.path.join(higgs_dir, f"{reqID}.txt")
                            with open(text_path, "w", encoding="utf-8") as f:
                                f.write(text_result)
                            
                            logger.info(f"[{reqID}] STT text saved to: {text_path}")
                            
                            return {
                                "type": "text",
                                "data": text_result,
                                "file_path": text_path,
                                "reqID": reqID
                            }

                        else:
                            tool_result = f"Unknown pipeline: {fn_name}"

                    except Exception as e:
                        logger.error(f"Error executing pipeline {fn_name}: {e}", exc_info=True)
                        return {
                            "type": "error",
                            "message": f"Pipeline {fn_name} failed: {str(e)}",
                            "reqID": reqID
                        }

                tool_outputs.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": "Pipeline executed successfully"
                })

            messages.extend(tool_outputs)
            

        return {
            "type": "error",
            "message": f"Pipeline execution failed after {max_iterations} iterations",
            "reqID": reqID
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return {
            "type": "error",
            "message": f"Pipeline error: {str(e)}",
            "reqID": reqID
        }
    finally:
        try:
            if os.path.exists(higgs_dir):
                shutil.rmtree(higgs_dir)
                logger.info(f"[{reqID}] Cleaned up higgs directory: {higgs_dir}")
        except Exception as cleanup_error:
            logger.error(f"[{reqID}] Failed to cleanup higgs directory: {cleanup_error}")
        
        cleanup_temp_file(f"{TEMP_SAVE_DIR}{reqID}")
        logger.info(f"Audio Pipeline Completed for reqID={reqID}")
        


if __name__ == "__main__":
    async def main():
        text = """ 
        Hello, Good morning everyone! Welcome to a fresh new day what are you upto
        """
        synthesis_audio_path = None
        requestID = reqID()
        voice = "alloy"
        synthesis_audio_path=None
        clone_audio_transcript = None
        saved_base64_path_clone = None
        saved_base64_path_speech = None
        result = None

        if (VOICE_BASE64_MAP.get(voice)):
            print(f"[INFO] Using named voice: {voice}")
            named_voice_audio_path = VOICE_BASE64_MAP.get(voice)
            named_voice_audio_base64 = encode_audio_base64(named_voice_audio_path)
            saved_base64_path_clone = save_temp_audio(named_voice_audio_base64, requestID, "clone")
        else:
            if(validate_and_decode_base64_audio(voice)):
                saved_base64_path_clone = save_temp_audio(voice, reqID, "clone")
            else:
                named_voice_audio = VOICE_BASE64_MAP.get("alloy")
                named_voice_audio_base64 = encode_audio_base64(named_voice_audio)
                saved_base64_path_clone = save_temp_audio(named_voice_audio_base64, requestID, "clone")
    
        if synthesis_audio_path:
            base64_synthesis_audio = encode_audio_base64(synthesis_audio_path)
            saved_base64_path_speech = save_temp_audio(base64_synthesis_audio, reqID, "speech")

        result = await run_audio_pipeline(reqID=requestID, text=text, voice=saved_base64_path_clone, synthesis_audio_path=saved_base64_path_speech, clone_audio_transcript=clone_audio_transcript)
        

        if not result:
            print("[ERROR] Pipeline returned None")
            return

        if result["type"] == "text":
            print(f"[Pipeline Result | Text] Content: {result['data']}")
            print(f"[Pipeline Result | Text] File saved at: {result.get('file_path', 'N/A')}")

        elif result["type"] == "audio":
            print(f"[Pipeline Result | Audio] Audio bytes length: {len(result['data'])} bytes")
            print(f"[Pipeline Result | Audio] File saved at: {result.get('file_path', 'N/A')}")

        elif result["type"] == "error":
            print(f"[Pipeline Error] {result['message']}")

    asyncio.run(main())