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