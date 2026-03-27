import os
import json
import uuid
import datetime
import numpy as np
from typing import Dict, Any, Optional

class UncertaintyRouter:
    def __init__(self, lower_bound: float = 0.4, upper_bound: float = 0.6, queue_dir: str = "data/feedback/hitl_queue"):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.queue_dir = queue_dir
        
        os.makedirs(self.queue_dir, exist_ok=True)


    def _save_to_queue(
        self, 
        probability: float, 
        transcript: str, 
        audio_features: np.ndarray, 
        text_features: np.ndarray,
        session_id: Optional[str]
    ):
        segment_id = str(uuid.uuid4())
        call_id = session_id if session_id else "unknown_session"
        timestamp = datetime.datetime.now().isoformat()
        
        feature_filename = f"{segment_id}_features.npz"
        feature_path = os.path.join(self.queue_dir, feature_filename)
        
        np.savez_compressed(
            feature_path, 
            audio=audio_features, 
            text=text_features
        )
        
        metadata = {
            "segment_id": segment_id,
            "session_id": call_id,
            "timestamp": timestamp,
            "predicted_probability": probability,
            "transcript_snippet": transcript,
            "feature_file": feature_filename,
            "review_status": "pending",
            "human_label": None
        }
        
        json_filename = f"{segment_id}_metadata.json"
        json_path = os.path.join(self.queue_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)