import modal

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

env_image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install("fastapi", "torch", "transformers", "sentence-transformers", "numpy", "librosa", "websockets")
    .env({"HF_HUB_CACHE": "/root/.cache/huggingface"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("backend", remote_path="/root/backend") 
)

app = modal.App("multimodal-fraud-api") 

@app.function(
    image=env_image,
    cpu=4,
    gpu="T4",
    timeout=300,
    scaledown_window=120,
    volumes={"/root/.cache/huggingface": hf_cache_vol}
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def serve():
    from backend.service import web_app
    return web_app