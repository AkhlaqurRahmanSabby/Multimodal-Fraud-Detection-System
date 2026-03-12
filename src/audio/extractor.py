import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class StreamingAudioExtractor:
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        """
        Initializes the model once into memory to prevent latency during live calls.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Wav2Vec2 onto {self.device} for streaming inference...")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval() 


    def extract_features(self, audio_chunk_16k: np.ndarray) -> np.ndarray:
        """
        Processes a live chunk of audio directly from RAM.
        Expected input: 1D numpy array of float32, sampled at exactly 16kHz.
        """

        # Convert raw audio into PyTorch format
        inputs = self.processor(
            audio_chunk_16k, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features without tracking gradients (saves memory/time)
        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state

        # Create the 2304-dimensional Fusion Vector (768 * 3)
        mean_pool = torch.mean(hidden_states, dim=1)         # (1, 768)
        max_pool = torch.max(hidden_states, dim=1).values    # (1, 768)
        std_pool = torch.std(hidden_states, dim=1)           # (1, 768)

        combined_features = torch.cat([mean_pool, max_pool, std_pool], dim=1)

        return combined_features.cpu().numpy().flatten()