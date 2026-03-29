import numpy as np
from collections import deque
from scipy.stats import wasserstein_distance
from typing import Dict, Any, List


class DriftDetector:
    def __init__(self, reference_probabilities: List[float], window_size: int = 1000, drift_threshold: float = 0.15):
        """
        Initializes the Drift Detector to monitor model output distributions.
        
        Args:
            reference_probabilities: A list of predictions from your validation set to act as the "healthy" baseline.
            window_size: The number of recent live predictions to keep in memory.
            drift_threshold: The statistical distance required to trigger a drift alert.
        """

        if not reference_probabilities:
            raise ValueError("A baseline of reference probabilities must be provided.")
            
        self.reference_dist = np.array(reference_probabilities)
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # deque automatically drops the oldest item when maxlen is reached
        self.current_window = deque(maxlen=window_size)
        
        print(f"Drift Detector initialized. Baseline size: {len(self.reference_dist)}, Window size: {self.window_size}.")