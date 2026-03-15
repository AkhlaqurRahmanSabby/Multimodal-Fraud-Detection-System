import numpy as np
from sentence_transformers import SentenceTransformer


class StreamingTextExtractor:
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5'):
        """
        Initializes the BGE model once into the server's RAM/VRAM.
        This prevents latency during live calls.
        """

        print(f"Loading Text Model ({model_name}) for streaming inference...")
        self.model = SentenceTransformer(model_name)


    def extract_features(self, transcript_chunk: str) -> np.ndarray:
        """
        Processes a live string of text (e.g., a 5-second transcript).
        Returns a flat 1D numpy array of exactly 768 dimensions.
        """

        # Safety check: Live audio often has moments of pure silence.
        # If the ASR returns an empty string, we return a blank feature vector.
        if not transcript_chunk or transcript_chunk.strip() == "":
            return np.zeros(768, dtype=np.float32)

        embedding = self.model.encode([transcript_chunk])
        
        return embedding.flatten()