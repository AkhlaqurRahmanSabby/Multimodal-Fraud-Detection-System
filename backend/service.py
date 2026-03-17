import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import numpy as np
import time
import torch
import asyncio

# Persistent Cloud Volume for Hugging Face Cache
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Reroute HF downloads to the Volume via Environment Variable
env_image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "torch", "transformers", "sentence-transformers", "numpy", "librosa")
    .env({"HF_HUB_CACHE": "/root/.cache/huggingface"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("models", remote_path="/root/models")
)

app = modal.App("multimodal-fraud-api") 
web_app = FastAPI(title="Multimodal Fraud Detection System API")

# Global variables to hold models in memory
audio_extractor = None
text_extractor = None
transcriber = None
pipeline = None

@web_app.on_event("startup")
def load_models():
    """Runs once when the Modal container boots up."""
    global audio_extractor, text_extractor, transcriber, pipeline
    
    from src.features.audio_extractor import StreamingAudioExtractor
    from src.features.text_extractor import StreamingTextExtractor
    from src.features.transcriber import StreamingTranscriber
    from src.inference.pipeline import InferencePipeline

    print("Loading V2 multimodal models into memory...")
    audio_extractor = StreamingAudioExtractor()
    text_extractor = StreamingTextExtractor()
    transcriber = StreamingTranscriber()
    
    pipeline = InferencePipeline(model_path="/root/models/pytorch_fraud_model.pth")
    print("All V2 models loaded successfully.")


@web_app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("New phone call connected.")

    # This specific phone call gets its own private, isolated memory
    session_hidden_state = None 

    try:
        while True: 
            payload = await websocket.receive_text()
            data = json.loads(payload)

            audio_array = np.array(data.get("audio_16k_chunk"), dtype=np.float32)

            if len(audio_array) == 0:
                await websocket.send_json({"error": "Empty audio chunk"})
                continue

            inference_start = time.time()

            # RUN HEAVY MATH IN BACKGROUND THREADS TO UNBLOCK FASTAPI
            transcript_chunk = await asyncio.to_thread(
                transcriber.transcribe_chunk, audio_array
            )
            
            audio_features = await asyncio.to_thread(
                audio_extractor.extract_features, audio_array
            )
            
            text_features = await asyncio.to_thread(
                text_extractor.extract_features, transcript_chunk
            )
            
            # FUSE & UPDATE MEMORY
            scam_prob, session_hidden_state = await asyncio.to_thread(
                pipeline.predict_chunk, audio_features, text_features, session_hidden_state
            )

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

# Apply the new architecture settings to the Cloud Function
@app.function(
    image=env_image,
    cpu=1,
    gpu="T4",
    timeout=300,
    scaledown_window=120,
    volumes={"/root/.cache/huggingface": hf_cache_vol}
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    return web_app