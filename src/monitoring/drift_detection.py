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


def update_and_check(self, live_probabilities: List[float]) -> Dict[str, Any]:
        """
        Updates the rolling window with new predictions and checks for data drift.
        
        Args:
            live_probabilities: A list of new fraud probabilities from the inference pipeline.
            
        Returns:
            A dictionary containing drift metrics and an alert flag.
        """
        # Append the new probabilities. The deque automatically drops the oldest items.
        self.current_window.extend(live_probabilities)
        
        # Ensure we have a statistically significant sample before calculating drift (e.g., at least 100 calls)
        if len(self.current_window) < min(100, self.window_size):
            return {
                "is_drifting": False,
                "distance": 0.0,
                "status": "warming_up",
                "current_samples": len(self.current_window)
            }
            
        # Convert the live window into an array for comparison
        live_dist = np.array(self.current_window)
        
        # Calculate the Wasserstein distance (Earth Mover's Distance)
        distance = wasserstein_distance(self.reference_dist, live_dist)
        
        # Flag if the distribution has drifted beyond the allowed threshold
        is_drifting = distance > self.drift_threshold
        
        if is_drifting:
            print(f"DATA DRIFT ALERT! Distribution shifted by {distance:.4f} (Threshold: {self.drift_threshold})")
            
        return {
            "is_drifting": bool(is_drifting),
            "distance": float(distance),
            "status": "drifting" if is_drifting else "healthy",
            "current_samples": len(self.current_window)
        }