# Architecture & Scaling Evolution

This document tracks the engineering evolution of our real-time streaming fraud detection system. It details the initial baseline architecture, the scaling bottlenecks encountered during live concurrent streaming, and the architectural pivot required to achieve production-grade latency.

---

## V1: The Baseline Architecture (Stateless Full-Context Processing)

### How We Built It
The V1 system was designed as a stateless pipeline optimized for maximum accuracy. 
* **Data Generation:** We generated 800 synthetic audio files using Google TTS (gTTS) based on real scam and legitimate call transcripts.
* **Feature Extraction:** We passed the *entire* raw audio file through `facebook/wav2vec2-base`, producing a 2304-dimensional embedding and the transcript through `BAAI/bge-base-en-v1.5` , producing a 768-dimensional embedding.
* **Classification:** We concatenated these into a 3072-dimensional multimodal feature vector and trained a standard PyTorch Feed-Forward Neural Network (`MultimodalFusionClassifier`) and an XGBoost baseline.

### The Win
The model performed extremely well in isolated offline evaluation. The PyTorch MLP achieved **98.7% accuracy**.

To achieve this in a streaming environment, the backend utilized a `cumulative_audio` array. As the WebSocket received new 5-second chunks, they were appended to the previous chunks. The model re-processed the entire conversation history from the beginning, giving the stateless model simulated "memory." During early streaming tests, the system successfully terminated scam calls right around the 15-second mark. 

---

### The Breakage at Scale (The 50-Call Load Test)

The V1 architecture broke down immediately under heavy concurrent load. We simulated 50 concurrent WebSocket streams targeting a single NVIDIA T4 GPU container.

**Visualizing the Bottleneck:**
Below is the telemetry from the 50-stream concurrent load test. 

![End-to-End System Latency Spike](../assets/v1_system_latency_spike.png)

*Figure 1: System latency queues to critical levels (15s+) as the single GPU struggles to clear the synchronous backlog.*

![Raw GPU Inference Creep](../assets/v1_gpu_inference_creep.png)

*Figure 2: The O(N²) trap in action. As the `cumulative_audio` array grows from 5s to 30s, the raw mathematical execution time on the GPU steadily increases, choking the pipeline.*

#### What Broke
* **Raw GPU Speed:** The models were fully optimized. Raw PyTorch inference time initially hovered around ~200 ms per call.
* **System Latency:** The end-to-end system latency spiked massively, queueing up to **10,000ms – 15,000ms+**.

#### The Root Cause: The $O(N^2)$ Trap
The `cumulative_audio` array was a fatal infrastructure flaw at scale. As a call progressed from 5 seconds to 30 seconds, the array grew. Because Transformers rely on Self-Attention (which scales quadratically at O(N²)), the compute and VRAM requirements exploded as the calls got longer.

A single T4 GPU could not process 50 massive, growing audio arrays sequentially. Calls were forced to wait in the event loop for up to 15 seconds, defeating the purpose of a real-time monitoring system.

---

### The Batching Roadblock

To fix the queue, the immediate thought was to implement dynamic batching (passing all 50 calls to the GPU simultaneously). This exposed another flaw in the `cumulative_audio` approach: **jagged input tensors.**

In a live environment, concurrent calls are at different stages. Caller A might be at 5 seconds, while Caller B is at 30 seconds. To batch these into a single PyTorch tensor, we would have to pad the 5-second call with 25 seconds of zeroes. This wastes massive amounts of GPU compute and VRAM on dead silence, leading directly to imminent Out-Of-Memory (OOM) crashes.

---

# Architecture & Scaling Evolution

This document tracks the engineering evolution of our real-time streaming fraud detection system. It outlines the initial design, the scaling issues observed under concurrent load, and the architectural changes made to achieve stable, production-ready latency.

---

## V1: The Baseline Architecture (Stateless Full-Context Processing)

### How We Built It
The V1 system was designed as a stateless pipeline focused on maximizing accuracy.

* **Data Generation:** We generated 800 synthetic audio files using Google TTS (gTTS) based on real scam and legitimate call transcripts.
* **Feature Extraction:** We passed the *entire* raw audio file through `facebook/wav2vec2-base`, producing a 2304-dimensional embedding, and the transcript through `BAAI/bge-base-en-v1.5`, producing a 768-dimensional embedding.
* **Classification:** We concatenated these into a 3072-dimensional multimodal feature vector and trained a PyTorch Feed-Forward Neural Network (`MultimodalFusionClassifier`) along with an XGBoost baseline.

### The Win
The model performed well in offline evaluation. The PyTorch MLP achieved **98.7% accuracy**.

To support streaming, the backend used a `cumulative_audio` array. As new 5-second chunks arrived over WebSocket, they were appended to previous chunks. The model reprocessed the full conversation each time, giving the stateless model simulated memory.

In early tests, the system was able to terminate scam calls around the 15-second mark.

---

### The Breakage at Scale (The 50-Call Load Test)

The V1 system failed under concurrent load. We simulated 50 simultaneous WebSocket streams targeting a single NVIDIA T4 GPU.

**What We Observed**
* **Raw GPU Inference:** Initially fast (~200ms per call).
* **System Latency:** Increased significantly, reaching **10,000ms – 15,000ms+**.

#### Root Cause: The O(N²) Growth
The `cumulative_audio` array grew as calls progressed. Because transformer-based models scale roughly with O(N²), both compute time and memory usage increased with input length.

As a result:
- Each request became more expensive over time  
- The GPU could not keep up with multiple growing inputs  
- Requests queued in the event loop for up to 15 seconds  

This made the system unsuitable for real-time use.

---

### The Batching Roadblock

To address queueing, we attempted batching. This exposed another issue: **jagged inputs**.

Different calls were at different lengths (e.g., 5s vs 30s). To batch them, shorter inputs had to be padded heavily, leading to:
- Wasted compute on padding  
- Increased VRAM usage  
- High risk of OOM errors  

---

## V2: Fixed-Window Streaming & Horizontal Scaling

### The Architectural Shift
To address memory issues and latency spikes, we moved to a distributed system that processes fixed-size audio chunks independently.

### Core Optimizations
* **Fixed-Window Streaming (Removing the O(N²) slowdown):** We removed the `cumulative_audio` array. The frontend now sends strict 5-second chunks that are processed independently. This keeps processing time consistent and avoids issues with growing inputs and uneven tensor sizes.
* **Strict Hardware Limits (Horizontal Scaling):** We enforced a limit of 10 active streams per GPU (`max_inputs=10`). When this limit is reached, additional containers are started. This prevents GPU overload and reduces queueing under high traffic.

### The Tradeoff
These changes removed queueing delays and stabilized the system. End-to-end system latency is now consistently under **6000ms**, compared to **15,000ms+** in V1.

However, inference latency per request increased to **4000ms–4500ms** due to batching. This reflects a tradeoff: improved system stability and throughput at the cost of higher per-request compute time.

**Visualizing the Improvement:**

![V2 System Latency Steady State](../assets/v2_system_latency_steady.png)

*Figure 3: End-to-end system latency remains stable under load, with no queue buildup.*

![V2 GPU Inference Steady State](../assets/v2_gpu_inference_steady.png)

*Figure 4: GPU inference latency remains consistent due to fixed-size inputs and controlled batching.*

### Hypotheses
* **GPU Bottleneck:** Since inference accounts for a large portion of total latency (~4.5s out of <6s), the system may be compute-bound. Using more powerful GPUs or optimizing inference could reduce latency, but this needs to be validated through benchmarking.

* **Frontend & Network Overhead:** There is still a gap between inference time and total system latency. This suggests additional overhead in networking or request handling. The exact source has not yet been isolated and requires further profiling.
