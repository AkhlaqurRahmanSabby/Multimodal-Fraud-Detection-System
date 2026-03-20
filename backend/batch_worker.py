import asyncio
from typing import List, Tuple, Optional
import numpy as np
import torch


# Shared queue used to collect incoming audio chunks from all WebSocket connections.
batch_queue = asyncio.Queue()


class QueueItem:
    """Container representing a single inference request."""

    def __init__(
        self, 
        audio_features: np.ndarray, 
        text_features: np.ndarray, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ):
        self.audio_features = audio_features
        self.text_features = text_features
        self.hidden_state = hidden_state
        self.future = asyncio.Future()  # Future used to return inference results asynchronously


async def batch_processor(pipeline):
    """
    Continuously monitors the queue, aggregates items over a short time window,
    and processes them as a batch for efficient inference.
    """
    
    print("Dynamic batching worker started. Monitoring queue...")
    
    while True:
        # Wait for at least one item to be available
        first_item = await batch_queue.get()
        batch: List[QueueItem] = [first_item]

        # Allow additional items to accumulate for batching
        await asyncio.sleep(0.05)  # 50 ms batching window
        
        # Drain remaining items currently in the queue
        while not batch_queue.empty():
            try:
                batch.append(batch_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
                
        # Prepare inputs for batch inference
        audio_list = [item.audio_features for item in batch]
        text_list = [item.text_features for item in batch]
        state_list = [item.hidden_state for item in batch]
        
        try:
            # Execute batch inference in a separate thread to avoid blocking the event loop
            probabilities, new_states = await asyncio.to_thread(
                pipeline.predict_batch, audio_list, text_list, state_list
            )
            
            # Resolve futures with corresponding results
            for i, item in enumerate(batch):
                if not item.future.done():
                    item.future.set_result((probabilities[i], new_states[i]))
                    
        except Exception as e:
            # Propagate errors to all pending requests in the batch
            print(f"GPU batching error: {e}")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)