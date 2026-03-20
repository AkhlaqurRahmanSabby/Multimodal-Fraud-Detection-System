from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import numpy as np
import time
import asyncio
from contextlib import asynccontextmanager
from backend.batch_worker import batch_processor, batch_queue, QueueItem

ml_models = {}

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Loads models into VRAM and starts the background worker on boot."""

    print("Loading V2 multimodal models into VRAM...")
    
    from src.features.audio_extractor import StreamingAudioExtractor
    from src.features.text_extractor import StreamingTextExtractor
    from src.features.transcriber import StreamingTranscriber
    from src.inference.pipeline import InferencePipeline

    ml_models["audio_extractor"] = StreamingAudioExtractor()
    ml_models["text_extractor"] = StreamingTextExtractor()
    ml_models["transcriber"] = StreamingTranscriber()
    ml_models["pipeline"] = InferencePipeline(model_path="/root/models/pytorch_fraud_model.pth")
    
    print("Models loaded. Starting dynamic batch worker...")
    worker_task = asyncio.create_task(batch_processor(ml_models["pipeline"]))

    yield

    print("Shutting down...")
    worker_task.cancel()
    ml_models.clear()


web_app = FastAPI(title="Multimodal Fraud Detection System API", lifespan=lifespan)

@web_app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Isolated WebSocket instance for a single caller."""
    
    await websocket.accept()
    session_hidden_state = None 

    try:
        while True: 
            payload = await websocket.receive_text()
            data = json.loads(payload)
            audio_array = np.array(data.get("audio_16k_chunk"), dtype=np.float32)

            if len(audio_array) == 0:
                continue

            inference_start = time.time()

            # Background Extraction
            transcript_chunk = await asyncio.to_thread(ml_models["transcriber"].transcribe_chunk, audio_array)
            audio_features = await asyncio.to_thread(ml_models["audio_extractor"].extract_features, audio_array)
            text_features = await asyncio.to_thread(ml_models["text_extractor"].extract_features, transcript_chunk)
            
            # Queue for batched inference
            request_item = QueueItem(audio_features, text_features, session_hidden_state)
            await batch_queue.put(request_item)

            # Wait for the GPU to process the batch
            scam_prob, session_hidden_state = await request_item.future

            backend_latency_ms = round((time.time() - inference_start) * 1000, 2)

            await websocket.send_json({
                "status": "success",
                "transcript": transcript_chunk, 
                "scam_probability": scam_prob,
                "alert": "TRIGGERED" if scam_prob > 0.85 else "SAFE",
                "model_latency_ms": backend_latency_ms
            })

    except WebSocketDisconnect:
        print("Call disconnected cleanly.")
    except Exception as e:
        print(f"WebSocket error: {e}")