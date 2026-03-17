import torch
import numpy as np
from transformers import pipeline

class StreamingTranscriber:
    def __init__(self, model_name: str = "openai/whisper-tiny.en"):
        """
        Loads the Whisper ASR model into VRAM once for live transcription.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper Transcriber ({model_name}) onto {self.device}...")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )


    def transcribe_chunk(self, audio_chunk_16k: np.ndarray) -> str:
        """
        Transcribes a 5-second audio chunk into text.
        """
        
        # Safety check for absolute silence
        if np.max(np.abs(audio_chunk_16k)) < 0.001:
            return ""

        result = self.pipe({"sampling_rate": 16000, "raw": audio_chunk_16k}, generate_kwargs={"task": "transcribe"})
        text = result["text"].strip()
        
        return text