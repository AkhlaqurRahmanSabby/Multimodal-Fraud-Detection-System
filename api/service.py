import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import numpy as np


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


@web_app.on_event("startup")
def load_models():
    """Runs once when the Modal container boots up."""
    
    global audio_extractor, text_extractor, pipeline
    
    from src.audio.extractor import StreamingAudioExtractor
    from src.text.extractor import StreamingTextExtractor
    from src.inference.pipeline import InferencePipeline

    audio_extractor = StreamingAudioExtractor()
    text_extractor = StreamingTextExtractor()
    pipeline = InferencePipeline(model_path="/root/models/pytorch_fraud_model.pth")


@web_app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("New connection established.")
    
    try:
        while True:
            payload = await websocket.receive_text()
            data = json.loads(payload)
            
            audio_array = np.array(data.get("audio_16k_chunk"), dtype=np.float32)
            transcript = data.get("transcript_chunk", "")
            
            audio_features = audio_extractor.extract_features(audio_array)
            text_features = text_extractor.extract_features(transcript)
            scam_prob = pipeline.predict(audio_features, text_features)
            
            await websocket.send_json({
                "status": "success",
                "scam_probability": scam_prob,
                "alert": "TRIGGERED" if scam_prob > 0.85 else "SAFE"
            })
            
    except WebSocketDisconnect:
        print("Call Disconnected.")


# Bind to Modal using the updated image variable
@app.function(image=env_image)
@modal.asgi_app()
def serve():
    return web_app