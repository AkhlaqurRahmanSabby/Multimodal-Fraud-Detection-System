import torch
import torch.nn as nn
import numpy as np


class LSTMStatefulClassifier(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, num_layers=1):
        super(LSTMStatefulClassifier, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )


    # We just pass the chunk and the previous memory state.
    def forward(self, x, hidden_state=None):
        out, hidden_state = self.lstm(x, hidden_state)
        last_hidden = out[:, -1, :] 
        logits = self.classifier(last_hidden)

        return logits, hidden_state


class InferencePipeline:
    def __init__(self, model_path: str = "../../models/pytorch_v2_lstm_fusion.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Stateful Multimodal Pipeline onto {self.device}...")
        
        self.model = LSTMStatefulClassifier(input_dim=3072)
        # Load the weights we trained in Notebook 6
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        
        # This will hold the (hn, cn) memory tuples during a live call
        self.hidden_state = None 


    def reset_memory(self):
        """Must be called when a new phone call begins to wipe the LSTM's memory."""
        self.hidden_state = None


    def predict_chunk(self, audio_features: np.ndarray, text_features: np.ndarray, hidden_state=None):
        """
        Takes the chunk AND the specific caller's memory, returns the prob AND the updated memory.
        """
        
        combined_features = np.concatenate((audio_features, text_features))
        input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass the specific call's memory in
            logits, new_hidden_state = self.model(input_tensor, hidden_state)
            probability = torch.sigmoid(logits).item() 
            
        # Hand the updated memory back to the web server
        return probability, new_hidden_state