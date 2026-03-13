import streamlit as st
import asyncio
import websockets
import json
import numpy as np
import librosa
import io
import time
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(page_title="Fraud Detection API", layout="wide")
MODAL_WS_URL = "wss://sabby-rahman--multimodal-fraud-api-serve.modal.run/stream"

# --- CURRENT CLOUD COMPUTE COSTS (USD/sec) ---
# GPU (Nvidia T4): $0.0001640
# CPU (1 Core):    $0.0000131
# RAM (1 GiB):     $0.0000022
# TOTAL:           $0.0001793 / sec
COMPUTE_COST_PER_SEC = 0.0001793

# --- Helper: HTML Cost Box ---
def format_cost_box(cost):
    return f"""
    <div style="background-color: #fff3cd; color: #856404; padding: 10px 20px; border-radius: 8px; 
                text-align: center; border: 2px solid #ffeeba; font-size: 20px; font-weight: bold; 
                margin: 0 auto 40px auto; max-width: 400px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        💸 Compute Cost: ${cost:.6f}
    </div>
    """

# ==========================================
# 1. HEADER & COST
# ==========================================
st.title("Multimodal Fraud Detection System")
st.markdown("True real-time continuous streaming simulation via Serverless WebSockets.")

st.write("") 

cost_display = st.empty()
cost_display.markdown(format_cost_box(0.0), unsafe_allow_html=True)

# ==========================================
# 2. LIVE GRAPHS (Side-by-Side)
# ==========================================
def plot_confidence(times, conf_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=conf_scores, mode='lines+markers', 
                             line=dict(color='#d62728', width=4), marker=dict(size=8)))
    fig.update_layout(
        title="Scam Confidence Score Over Time",
        xaxis_title="Time (Seconds)",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(dtick=5),  
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_latency(times, total_lats, model_lats):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=total_lats, mode='lines+markers', name='Total System Latency', line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=times, y=model_lats, mode='lines+markers', name='Model Inference Time', line=dict(color='#ff7f0e', width=3)))
    fig.update_layout(
        title="Processing Latency Per Segment",
        xaxis_title="Time (Seconds)",
        yaxis_title="Milliseconds (ms)",
        xaxis=dict(dtick=5),  
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

col_graph1, col_graph2 = st.columns(2, gap="small")
conf_graph_display = col_graph1.empty()
lat_graph_display = col_graph2.empty()

# Initialize empty graphs 
conf_graph_display.plotly_chart(plot_confidence([0], [0]), width="stretch")
lat_graph_display.plotly_chart(plot_latency([0], [0], [0]), width="stretch")

st.write("")
st.divider()
st.write("")
st.write("")

# ==========================================
# 3. SPLIT VIEW: INPUT (Left) & OUTPUT (Right)
# ==========================================
col_input, col_output = st.columns(2, gap="large")

with col_input:
    # Wrap in a bordered container for a clean "card" look
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>System Input</h3>", unsafe_allow_html=True)
        st.write("") 
        
        input_method = st.radio("Select Audio Source:", ["📁 Upload File", "🎙️ Record Call", "🎵 Sample Audio"], horizontal=True)
        
        audio_source = None
        selected_sample_path = None
        
        if input_method == "📁 Upload File":
            audio_source = st.file_uploader("Upload PCM Audio (.wav / .mp3):", type=["wav", "mp3"])
        elif input_method == "🎵 Sample Audio":
            st.info("💡 **Demo Testing:** These pre-loaded samples match the model's training distribution. You can play the audio to hear the script before initiating the stream.")
            
            samples_data = {
                "🟢 SAFE | Sample 0: Friend Follow-up": "samples/sample_0.wav",
                "🟢 SAFE | Sample 1: Board Game Invite": "samples/sample_1.wav",
                "🔴 SCAM | Sample 2: Customs Package Intercept": "samples/sample_2.wav",
                "🟢 SAFE | Sample 3: Utility Bill Callback": "samples/sample_3.wav",
                "🔴 SCAM | Sample 4: Customs Package Variation": "samples/sample_4.wav"
            }
            
            sample_name = st.selectbox("Choose a test sample:", list(samples_data.keys()))
            selected_sample_path = samples_data[sample_name]
            st.write("**Listen to the selected sample:**")
            st.audio(selected_sample_path, format="audio/wav")
        else:
            audio_source = st.audio_input("Record directly from your microphone:")
        
        st.write("") 
        start_button = st.button("Initiate Live Stream", type="primary", use_container_width=True)
        cold_start_indicator = st.empty()

with col_output:
    # Wrap in a matching bordered container
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Live ASR Output & Alerts</h3>", unsafe_allow_html=True)
        st.write("") 
        st.write("") 
        
        transcript_box = st.empty()
        
        # --- NEW: Empty State Placeholder ---
        transcript_box.markdown(
            """
            <div style='padding: 40px 20px; text-align: center; color: #6c757d; background-color: #f8f9fa; border-radius: 10px; border: 2px dashed #dee2e6;'>
                <span style='font-size: 24px;'>⏳</span><br><br>
                <strong>Awaiting Stream</strong><br>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.write("") 
        status_indicator = st.empty()
        alert_box = st.empty()

# --- The Continuous Streaming Loop ---
async def run_live_stream(audio_bytes):
    wake_up_start = time.time()
    y, sr = librosa.load(audio_bytes, sr=16000)
    chunk_size = 16000 * 5 
    
    full_transcript = ""
    cumulative_cost = 0.0
    
    time_seconds = []  
    confidences = []
    total_latencies = []
    model_latencies = []
    current_chunk = 0
    
    try:
        async with websockets.connect(MODAL_WS_URL, open_timeout=300, close_timeout=10, max_size=5_000_000) as websocket:
            wake_up_time = time.time() - wake_up_start
            cold_start_indicator.success(f"🟢 Server Awake ({wake_up_time:.1f}s). Streaming...")
            
            for i in range(0, len(y), chunk_size):
                chunk = y[i:i + chunk_size]
                if len(chunk) < 16000 * 2: 
                    continue
                
                current_chunk += 1
                start_time = time.time()
                
                chunk = np.round(chunk, 4)
                await websocket.send(json.dumps({"audio_16k_chunk": chunk.tolist()}))
                
                response_json = await websocket.recv()
                result = json.loads(response_json)
                
                total_latency = time.time() - start_time
                cumulative_cost += (total_latency * COMPUTE_COST_PER_SEC)
                prob = result.get("scam_probability", 0)
                
                time_seconds.append(current_chunk * 5)
                confidences.append(prob * 100) 
                total_latencies.append(round(total_latency * 1000, 2))
                model_latencies.append(result.get('model_latency_ms', 0))
                
                cost_display.markdown(format_cost_box(cumulative_cost), unsafe_allow_html=True)
                
                if result.get('transcript'):
                    full_transcript += result['transcript'] + " "
                    transcript_box.markdown(f"**🎙️ Transcript:**\n> {full_transcript}")
                
                conf_graph_display.plotly_chart(plot_confidence(time_seconds, confidences), width="stretch")
                lat_graph_display.plotly_chart(plot_latency(time_seconds, total_latencies, model_latencies), width="stretch")
                
                if result.get("alert") == "TRIGGERED":
                    status_indicator.error("🔴 Stream closed due to security violation.")
                    alert_box.error(f"🚨 **SCAM DETECTED** | Confidence: {prob:.1%} | **CALL TERMINATED.**")
                    cold_start_indicator.empty() 
                    break 
                else:
                    alert_box.success(f"✅ Call segment {current_chunk} safe. (Confidence: {prob:.1%})")
                
                await asyncio.sleep(2) 
                
            if result.get("alert") != "TRIGGERED":
                cold_start_indicator.info("End of audio stream reached.")

    except Exception as e:
        status_indicator.error(f"System Error: {str(e)}")

# Trigger Execution
if start_button:
    if input_method == "🎵 Sample Audio" and selected_sample_path:
        cold_start_indicator.info("⏳ Waking up Modal Cloud Server...")
        with open(selected_sample_path, "rb") as f:
            asyncio.run(run_live_stream(io.BytesIO(f.read())))
    elif audio_source:
        cold_start_indicator.info("⏳ Waking up Modal Cloud Server...")
        asyncio.run(run_live_stream(io.BytesIO(audio_source.read())))
    else:
        cold_start_indicator.error("⚠️ Please provide an audio input before starting the stream.")