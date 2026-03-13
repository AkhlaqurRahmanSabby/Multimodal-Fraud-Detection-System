import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import numpy as np
import time
import torch

env_image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "torch", "transformers", "sentence-transformers", "numpy")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("models", remote_path="/root/models")
)

app = modal.App("multimodal-fraud-api") 
web_app = FastAPI(title="Multimodal Fraud Detection System API")

audio_extractor = None
text_extractor = None
pipeline = None
asr_model = None

@web_app.on_event("startup")
def load_models():
    """Runs once when the Modal container boots up."""
    global audio_extractor, text_extractor, pipeline, asr_model
    
    from src.audio.extractor import StreamingAudioExtractor
    from src.text.extractor import StreamingTextExtractor
    from src.inference.pipeline import InferencePipeline
    from transformers import pipeline as hf_pipeline

    print("Loading multimodal models into memory...")
    audio_extractor = StreamingAudioExtractor()
    text_extractor = StreamingTextExtractor()
    pipeline = InferencePipeline(model_path="/root/models/pytorch_fraud_model.pth")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading Whisper ASR onto device {device}...")
    asr_model = hf_pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=device)
    print("All models loaded successfully.")


@web_app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("New connection established.")

    # --- MEMORY STATE ---
    call_history = ""
    # Initialize an empty numpy array for the audio memory
    cumulative_audio = np.array([], dtype=np.float32)

    try:
        while True: 
            payload = await websocket.receive_text()
            data = json.loads(payload)

            audio_array = np.array(data.get("audio_16k_chunk"), dtype=np.float32)

            if len(audio_array) == 0:
                await websocket.send_json({"error": "Empty audio chunk"})
                continue

            inference_start = time.time()

            # TRANSCRIBE (Whisper only needs the current 5-second chunk)
            transcription_result = asr_model({"sampling_rate": 16000, "raw": audio_array})
            transcript_chunk = transcription_result["text"].strip()

            # UPDATE MEMORY
            call_history += " " + transcript_chunk
            current_text_context = call_history.strip()
            
            # Concatenate the new 5 seconds of audio to the historical audio
            cumulative_audio = np.concatenate((cumulative_audio, audio_array))

            # EXTRACT & FUSE (Both models now evaluate the FULL history)
            audio_features = audio_extractor.extract_features(cumulative_audio)
            text_features = text_extractor.extract_features(current_text_context)
            scam_prob = pipeline.predict(audio_features, text_features)

            backend_latency_ms = round((time.time() - inference_start) * 1000, 2)

            # RESPOND
            await websocket.send_json({
                "status": "success",
                "transcript": transcript_chunk, 
                "scam_probability": scam_prob,
                "alert": "TRIGGERED" if scam_prob > 0.85 else "SAFE",
                "model_latency_ms": backend_latency_ms
            })

    except WebSocketDisconnect:
        print("Client disconnected cleanly")
    except Exception as e:
        print("WebSocket error:", e)


@app.function(
    image=env_image,
    cpu=1,
    gpu="T4",
    timeout=300,
    container_idle_timeout=120
)
@modal.asgi_app()
def serve():
    return web_app