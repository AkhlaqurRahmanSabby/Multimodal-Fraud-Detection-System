import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class StreamingTranscriber:
    def __init__(self, model_name: str = "openai/whisper-tiny.en"):
        """
        Loads the Whisper ASR processor and model into VRAM directly for manual control.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper Transcriber ({model_name}) onto {self.device}...")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()


    def transcribe_chunk(self, audio_chunk_16k: np.ndarray) -> str:
        """
        Transcribes a 5-second audio chunk directly using model.generate().
        """
        # Safety check for absolute silence
        if np.max(np.abs(audio_chunk_16k)) < 0.001:
            return ""

        # Convert raw audio into Whisper's required input format
        input_features = self.processor(
            audio_chunk_16k, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)

        # Generate the transcription IDs without tracking gradients
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # Decode the IDs back into English text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()