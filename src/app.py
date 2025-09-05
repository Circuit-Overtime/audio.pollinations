from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import aiofiles
import uvloop
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import queue
import time
import uuid
import io
import traceback
import tempfile
import os
from loguru import logger
import atexit
from utility import validate_and_decode_base64_audio, encode_audio_base64
from voiceMap import VOICE_BASE64_MAP
from server import run_audio_pipeline
from wittyMessages import get_validation_error, get_witty_error

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    voice: Optional[Dict[str, Any]] = None
    audio_text: Optional[str] = None
    audio: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: List[ContentItem]

class AudioRequest(BaseModel):
    messages: List[Message]

class WorkerPool:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or min(32, (os.cpu_count() or 1) + 4)
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers * 2)
        self.request_queues = []
        self.response_queues = []
        self.workers = []
        self._initialize_workers()
        atexit.register(self.cleanup)

    def _initialize_workers(self):
        try:
            from model_server import model_worker
            
            for i in range(self.num_workers):
                req_queue = mp.Queue(maxsize=100)
                resp_queue = mp.Queue(maxsize=100)
                
                worker = mp.Process(
                    target=model_worker,
                    args=(req_queue, resp_queue),
                    daemon=True
                )
                worker.start()
                
                self.request_queues.append(req_queue)
                self.response_queues.append(resp_queue)
                self.workers.append(worker)
                
            logger.info(f"Initialized {self.num_workers} workers")
        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")
            raise

    def get_least_busy_worker(self):
        min_size = float('inf')
        best_idx = 0
        
        for i, queue in enumerate(self.request_queues):
            size = queue.qsize()
            if size < min_size:
                min_size = size
                best_idx = i
                if size == 0:
                    break
                    
        return best_idx

    async def process_request(self, request_data):
        worker_idx = self.get_least_busy_worker()
        req_queue = self.request_queues[worker_idx]
        resp_queue = self.response_queues[worker_idx]
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, req_queue.put, request_data
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, resp_queue.get
            )
            
            return result
        except Exception as e:
            logger.error(f"Worker {worker_idx} error: {e}")
            raise

    def cleanup(self):
        try:
            for queue in self.request_queues:
                try:
                    queue.put("STOP", timeout=1)
                except:
                    pass
            
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=2)
                    if worker.is_alive():
                        worker.terminate()
            
            self.process_pool.shutdown(wait=False)
            self.thread_pool.shutdown(wait=False)
            
            logger.info("Worker pool cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

class TempFileManager:
    def __init__(self):
        self.temp_files = {}
        self.lock = threading.Lock()
        self.cleanup_task = None

    async def save_audio(self, audio_data: bytes, request_id: str, audio_type: str) -> str:
        loop = asyncio.get_event_loop()
        
        def _save():
            fd, path = tempfile.mkstemp(suffix=f"_{audio_type}.wav", prefix=f"{request_id}_")
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_data)
            return path
        
        path = await loop.run_in_executor(None, _save)
        
        with self.lock:
            if request_id not in self.temp_files:
                self.temp_files[request_id] = []
            self.temp_files[request_id].append(path)
        
        return path

    async def cleanup_request_files(self, request_id: str):
        with self.lock:
            files = self.temp_files.pop(request_id, [])
        
        if files:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._cleanup_files, files)

    def _cleanup_files(self, files):
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

worker_pool = None
temp_manager = TempFileManager()

app = FastAPI(
    title="Elixpo Audio API",
    description="High-performance audio processing API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global worker_pool
    worker_pool = WorkerPool()
    logger.info("API server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    if worker_pool:
        worker_pool.cleanup()

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    return response

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "workers": worker_pool.num_workers if worker_pool else 0,
        "endpoints": {
            "GET": "/audio?text=your_text_here&system=optional_system_prompt&voice=optional_voice",
            "POST": "/audio"
        },
        "message": "All systems operational! 🚀"
    }

@app.get("/health")
async def health():
    return {"status": "alive", "message": "Still breathing! 💨"}

@app.get("/audio")
async def audio_get_endpoint(
    text: str,
    system: Optional[str] = None,
    voice: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    request_id = str(uuid.uuid4())
    
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Missing required 'text' parameter")

        voice_name = voice or "alloy"
        voice_path = None
        
        if VOICE_BASE64_MAP.get(voice_name):
            named_voice_path = VOICE_BASE64_MAP.get(voice_name)
            coded = encode_audio_base64(named_voice_path)
            voice_path = await temp_manager.save_audio(coded, request_id, "clone")
        else:
            named_voice_path = VOICE_BASE64_MAP.get("alloy")
            coded = encode_audio_base64(named_voice_path)
            voice_path = await temp_manager.save_audio(coded, request_id, "clone")

        request_data = {
            "reqID": request_id,
            "text": text,
            "system_instruction": system,
            "voice": voice_path
        }

        result = await worker_pool.process_request(request_data)
        
        background_tasks.add_task(temp_manager.cleanup_request_files, request_id)

        if result["type"] == "audio":
            return StreamingResponse(
                io.BytesIO(result["data"]),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"inline; filename={request_id}.wav",
                    "Content-Length": str(len(result["data"]))
                }
            )
        elif result["type"] == "text":
            return {"text": result["data"], "request_id": request_id}
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET error for {request_id}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio")
async def audio_post_endpoint(
    request: AudioRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    request_id = str(uuid.uuid4())
    
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Missing or invalid 'messages' in payload")

        system_instruction = None
        user_content = None
        
        for msg in request.messages:
            if msg.role == "system":
                for item in msg.content:
                    if item.type == "text":
                        system_instruction = item.text
            elif msg.role == "user":
                user_content = msg.content

        if not user_content:
            raise HTTPException(status_code=400, detail="Missing or invalid 'content' in user message")

        text = None
        voice_name = "alloy"
        voice_b64 = None
        clone_audio_transcript = None
        speech_audio_b64 = None

        for item in user_content:
            if item.type == "text":
                text = item.text
            elif item.type == "voice" and item.voice:
                voice_name = item.voice.get("name", "alloy")
                voice_b64 = item.voice.get("data")
            elif item.type == "clone_audio_transcript":
                clone_audio_transcript = item.audio_text
            elif item.type == "speech_audio" and item.audio:
                speech_audio_b64 = item.audio.get("data")

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Missing required 'text' in content")

        if voice_b64 and voice_name and voice_name != "alloy":
            raise HTTPException(status_code=400, detail="Provide either 'voice.data' (base64) or 'voice.name', not both")

        voice_path = None
        speech_audio_path = None

        if voice_b64:
            try:
                decoded = validate_and_decode_base64_audio(voice_b64, max_duration_sec=5)
                voice_path = await temp_manager.save_audio(decoded, request_id, "clone")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid voice audio: {e}")
        else:
            voice_file = VOICE_BASE64_MAP.get(voice_name) or VOICE_BASE64_MAP.get("alloy")
            coded = encode_audio_base64(voice_file)
            voice_path = await temp_manager.save_audio(coded, request_id, "clone")

        if speech_audio_b64:
            try:
                decoded = validate_and_decode_base64_audio(speech_audio_b64, max_duration_sec=60)
                speech_audio_path = await temp_manager.save_audio(decoded, request_id, "speech")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid speech_audio: {e}")

        request_data = {
            "reqID": request_id,
            "text": text,
            "voice": voice_path,
            "synthesis_audio_path": speech_audio_path,
            "clone_audio_transcript": clone_audio_transcript,
            "system_instruction": system_instruction,
        }

        result = await worker_pool.process_request(request_data)
        
        background_tasks.add_task(temp_manager.cleanup_request_files, request_id)

        if result["type"] == "audio":
            return StreamingResponse(
                io.BytesIO(result["data"]),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"inline; filename={request_id}.wav",
                    "Content-Length": str(len(result["data"]))
                }
            )
        elif result["type"] == "text":
            return {"text": result["data"], "request_id": request_id}
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST error for {request_id}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    mp.set_start_method('spawn', force=True)
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop",
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False
    )
    
    server = uvicorn.Server(config)
    server.run()