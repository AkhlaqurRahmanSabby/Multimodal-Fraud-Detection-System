import torch
import torch.nn as nn
import numpy as np


class MultimodalFusionClassifier(nn.Module):
    def __init__(self, input_dim=3072):
        super(MultimodalFusionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        return self.network(x)


class InferencePipeline:
    def __init__(self, model_path: str = "models/pytorch_fraud_model.pth"):
        """
        Loads the PyTorch model architecture and injects the trained weights.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Multimodal Pipeline onto {self.device}...")
        
        self.model = MultimodalFusionClassifier(input_dim=3072)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 


    def predict(self, audio_features: np.ndarray, text_features: np.ndarray) -> float:
        """
        Fuses the audio and text vectors and returns a scam probability.
        """
        
        combined_features = np.concatenate((audio_features, text_features))
        
        # Convert to PyTorch tensor and add a batch dimension -> Shape: (1, 3072)
        input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item() 
            
        return probability