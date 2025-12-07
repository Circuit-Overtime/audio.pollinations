from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from loguru import logger
from utility import save_temp_audio, validate_and_decode_base64_audio, encode_audio_base64, convertToAudio
from requestID import reqID
from voiceMap import VOICE_BASE64_MAP
from server import run_audio_pipeline
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import traceback
from wittyMessages import get_validation_error, get_witty_error
import time
import asyncio
import os
import traceback
from config import WORKERS, THREADS
from gunicorn.app.base import BaseApplication
from gunicorn.workers.sync import SyncWorker

app = Flask(__name__)
CORS(app)
class ModelManager(BaseManager): pass
ModelManager.register("Service")
manager = ModelManager(address=("localhost", 6000), authkey=b"secret")
manager.connect()
service = manager.Service()

class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                     if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

@app.before_request
def before_request():
    g.request_id = reqID()
    g.start_time = time.time()

@app.after_request
def after_request(response):
    process_time = time.time() - g.get('start_time', time.time())
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "endpoints": {
            "GET": "/audio?text=your_text_here&system=optional_system_prompt&voice=optional_voice",
            "POST": "/audio"
        },
        "message": "All systems operational! 🚀"
    })

@app.route("/audio", methods=["GET", "POST"])
def audio_endpoint():
    if request.method == "GET":
        try:
            text = request.args.get("text")
            system = request.args.get("system")
            voice = request.args.get("voice")
            seed = request.args.get("seed")
            voice_path = None
            request_id = None
            generateHashValue = service.cacheName(f"{text}{system if system else ''}{voice if voice else ''}{str(seed) if seed else 42}")
            request_id = generateHashValue
            gen_audio_folder = os.path.join(os.path.dirname(__file__), "..", "genAudio")
            cached_audio_path = os.path.join(gen_audio_folder, f"{generateHashValue}.wav")
            cached_text_path = os.path.join(gen_audio_folder, f"{generateHashValue}.txt")
            
            if os.path.isfile(cached_audio_path) or os.path.isfile(cached_text_path):
                if os.path.isfile(cached_text_path):
                    with open(cached_text_path, "r") as f:
                        cached_text = f.read()
                    return jsonify({"text": cached_text, "request_id": request_id})
                else:
                    with open(cached_audio_path, "rb") as f:
                        audio_data = f.read()
                    return Response(
                        audio_data,
                        mimetype="audio/wav",
                        headers={
                            "Content-Disposition": f"inline; filename={request_id}.wav",
                            "Content-Length": str(len(audio_data))
                        }
                    )

            if voice and not VOICE_BASE64_MAP.get(voice):
                with open(voice, "r") as f:
                    audio_data = f.read()
                    if validate_and_decode_base64_audio(audio_data):
                        voice_path = voice
            elif voice and VOICE_BASE64_MAP.get(voice):
                voice_path = VOICE_BASE64_MAP.get(voice)
            else:
                voice_path = VOICE_BASE64_MAP.get("alloy")

            if not text or not isinstance(text, str) or not text.strip():
                return jsonify({"error": {"message": "Missing required 'text' parameter.", "code": 400}}), 400

            result = asyncio.run(run_audio_pipeline(
                reqID=request_id,
                text=text,
                system_instruction=system,
                voice=voice_path
            ))

            if result["type"] == "audio":
                return Response(
                    result["data"],
                    mimetype="audio/wav",
                    headers={
                        "Content-Disposition": f"inline; filename={request_id}.wav",
                        "Content-Length": str(len(result["data"]))
                    }
                )
            elif result["type"] == "text":
                return jsonify({"text": result["data"], "request_id": request_id})
            else:
                return jsonify({"error": {"message": result.get("message", "Unknown error"), "code": 500}}), 500

        except Exception as e:
            logger.error(f"GET error: {traceback.format_exc()}")
            return jsonify({"error": {"message": str(e), "code": 500}}), 500

    elif request.method == "POST":
        try:
            body = request.get_json(force=True)
            messages = body.get("messages", [])
            seed = body.get("seed", 42)
            
            if not messages or not isinstance(messages, list):
                return jsonify({"error": {"message": "Missing or invalid 'messages' in payload.", "code": 400}}), 400
            request_id = None
            system_instruction = None
            user_content = None
            for msg in messages:
                if msg.get("role") == "system":
                    for item in msg.get("content", []):
                        if item.get("type") == "text":
                            system_instruction = item.get("text")
                elif msg.get("role") == "user":
                    user_content = msg.get("content", [])
            if not user_content or not isinstance(user_content, list):
                return jsonify({"error": {"message": "Missing or invalid 'content' in user message.", "code": 400}}), 400
            text = None
            voice_name = "alloy"
            voice_b64 = None
            clone_audio_transcript = None
            speech_audio_b64 = None
            for item in user_content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text")
                    elif item.get("type") == "voice":
                        v = item.get("voice", {})
                        if isinstance(v, dict):
                            voice_name = v.get("name", "alloy")
                            if v.get("data"):
                                voice_b64 = v.get("data")
                    elif item.get("type") == "clone_audio_transcript":
                        clone_audio_transcript = item.get("audio_text")
                    elif item.get("type") == "speech_audio":
                        speech_audio_b64 = item.get("audio", {}).get("data")

            if not text or not isinstance(text, str) or not text.strip():
                return jsonify({"error": {"message": "Missing required 'text' in content.", "code": 400}}), 400
            if voice_b64 and voice_name and voice_name != "alloy":
                return jsonify({"error": {"message": "Provide either 'voice.data' (base64) or 'voice.name', not both.", "code": 400}}), 400
            generateHashValue = service.cacheName(f"{text}{system_instruction if system_instruction else ''}{voice_name if voice_name else ''}{str(seed) if seed else 42}")
            request_id = generateHashValue
            gen_audio_folder = os.path.join(os.path.dirname(__file__), "..", "genAudio")
            cached_audio_path = os.path.join(gen_audio_folder, f"{generateHashValue}.wav")
            cached_text_path = os.path.join(gen_audio_folder, f"{generateHashValue}.txt")
            if os.path.isfile(cached_audio_path) or os.path.isfile(cached_text_path):
                if os.path.isfile(cached_text_path):
                    with open(cached_text_path, "r") as f:
                        cached_text = f.read()
                    return jsonify({"text": cached_text, "request_id": request_id})
                else:
                    with open(cached_audio_path, "rb") as f:
                        audio_data = f.read()
                    return Response(
                        audio_data,
                        mimetype="audio/wav",
                        headers={
                            "Content-Disposition": f"inline; filename={request_id}.wav",
                            "Content-Length": str(len(audio_data))
                        }
                    )
            voice_path = None
            speech_audio_path = None
            if voice_b64:
                try:
                    decoded = validate_and_decode_base64_audio(voice_b64, max_duration_sec=15)
                    voice_path = save_temp_audio(decoded, request_id, "clone")
                except Exception as e:
                    return jsonify({"error": {"message": f"Invalid voice audio: {e}", "code": 400}}), 400
            elif voice_name and not voice_b64:
                try:
                    if voice and not VOICE_BASE64_MAP.get(voice):
                        with open(voice, "r") as f:
                            audio_data = f.read()
                            if validate_and_decode_base64_audio(audio_data):
                                voice_path = voice
                    elif voice and VOICE_BASE64_MAP.get(voice):
                        voice_path = VOICE_BASE64_MAP.get(voice)
                    else:
                        voice_path = VOICE_BASE64_MAP.get("alloy")
                except Exception as e:
                    return jsonify({"error": {"message": f"Invalid voice name: {e}", "code": 400}}), 400
            else:
                voice_path = VOICE_BASE64_MAP.get("alloy")

            if speech_audio_b64:
                try:
                    decoded = validate_and_decode_base64_audio(speech_audio_b64, max_duration_sec=60)
                    saved_audio_path = save_temp_audio(decoded, request_id, "speech")
                    speech_audio_path = convertToAudio(saved_audio_path, request_id)
                except Exception as e:
                    return jsonify({"error": {"message": f"Invalid speech_audio: {e}", "code": 400}}), 400


            result = asyncio.run(run_audio_pipeline(
                reqID=request_id,
                text=text,
                voice=voice_path,
                synthesis_audio_path=speech_audio_path,
                clone_audio_transcript=clone_audio_transcript,
                system_instruction=system_instruction,
            ))

            if result["type"] == "audio":
                return Response(
                    result["data"],
                    mimetype="audio/wav",
                    headers={
                        "Content-Disposition": f"inline; filename={request_id}.wav",
                        "Content-Length": str(len(result["data"]))
                    }
                )
            elif result["type"] == "text":
                return jsonify({
                    "id": request_id,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result["data"]
                            },
                            "finish_reason": "stop"
                        }
                    ]
                })
            else:
                return jsonify({
                    "id": request_id,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result.get("message", "Unknown error")
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }), 500

        except Exception as e:
            logger.error(f"POST error: {traceback.format_exc()}")
            return jsonify({"error": {"message": str(e), "code": 500}}), 500
            
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive", "message": "Still breathing! 💨"}), 200

@app.errorhandler(400)
def bad_request(e):
    logger.warning(f"400 error: {str(e)}")
    return jsonify({"error": {"message": get_validation_error(), "code": 400}}), 400

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled 500 exception: {traceback.format_exc()}")
    return jsonify({"error": {"message": get_witty_error(), "code": 500}}), 500

@app.errorhandler(404)
def not_found(e):
    logger.info(f"404 error: {request.url}")
    return jsonify({"error": {"message": "This endpoint went on vacation and forgot to leave a forwarding address! 🏖️", "code": 404}}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    logger.info(f"405 error: {request.method} on {request.url}")
    return jsonify({"error": {"message": "That HTTP method is not invited to this party! Try a different one! 🎉", "code": 405}}), 405

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    workers = WORKERS
    logger.info(f"Starting Elixpo Audio API Server at {host}:{port} with {workers} workers")
    options = {
        "bind": f"{host}:{port}",
        "workers": workers,
        "worker_class": "gthread",
        "threads" : THREADS,
        "timeout": 120,
        "accesslog": "-",
        "errorlog": "-",
        "loglevel": "info",
        "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
        "max_requests": 1000,
        "max_requests_jitter": 50,
    }
    try:
        gunicorn_app = GunicornApp(app, options)
        gunicorn_app.run()
    except Exception as e:
        logger.error(f"Failed to start Gunicorn: {e}")
        raise