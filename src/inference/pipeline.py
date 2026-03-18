import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class LSTMStatefulClassifier(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, num_layers=1):
        super(LSTMStatefulClassifier, self).__init__()
        
        # batch_first=True means the model natively expects [Batch, Sequence, Features]
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        

    def predict_batch(
        self, 
        audio_features_list: List[np.ndarray], 
        text_features_list: List[np.ndarray], 
        hidden_states_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[List[float], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Takes a BATCH of inputs from the queue. Stacks them, processes them
        simultaneously on the GPU, and splits them back up.
        """

        batch_size = len(audio_features_list)
        
        # Convert lists of 1D arrays into 2D arrays: [Batch, 2304] and [Batch, 768]
        audio_batch = np.stack(audio_features_list)
        text_batch = np.stack(text_features_list)
        combined_batch = np.concatenate((audio_batch, text_batch), axis=1)

        input_tensor = torch.FloatTensor(combined_batch).unsqueeze(1).to(self.device)
        
        h_list, c_list = [], []
        
        for state in hidden_states_list:
            if state is None:
                # If this is a new caller, create a blank memory slate: [1, 1, 512]
                h_list.append(torch.zeros(1, 1, 512, device=self.device))
                c_list.append(torch.zeros(1, 1, 512, device=self.device))
            else:
                # If they are an existing caller, grab their specific memory tensor
                h_list.append(state[0])
                c_list.append(state[1])
                
        # Stack the memory tensors along the Batch dimension (dim=1), resulting shape: [1, Batch, 512]
        batched_h = torch.cat(h_list, dim=1)
        batched_c = torch.cat(c_list, dim=1)
        batched_hidden = (batched_h, batched_c)
        
        with torch.no_grad():
            logits, new_batched_hidden = self.model(input_tensor, batched_hidden)
            probabilities = torch.sigmoid(logits).squeeze(-1).tolist()
            
            # Safety catch: If batch size is 1, .tolist() returns a single float, not a list.
            if type(probabilities) is float:
                probabilities = [probabilities]

        new_h, new_c = new_batched_hidden  # Shapes are [1, Batch, 512]
        new_states_list = []
        
        for i in range(batch_size):
            # Slice the massive tensor to extract just Caller i's specific memory
            # Keep shape as [1, 1, 512] so it's ready for their next chunk
            caller_h = new_h[:, i:i+1, :].contiguous()
            caller_c = new_c[:, i:i+1, :].contiguous()
            new_states_list.append((caller_h, caller_c))
            
        return probabilities, new_states_list