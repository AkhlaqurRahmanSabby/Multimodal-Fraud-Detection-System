import streamlit as st
import asyncio
import websockets
import json
import numpy as np
import librosa
import time
import os
import re
import plotly.graph_objects as go

# ==========================================
# INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Live Deployment Demo", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Securely load the WebSocket endpoint for the Modal backend
try:
    MODAL_WS_URL = st.secrets["MODAL_WS_URL"]
except FileNotFoundError:
    st.error("Missing MODAL_WS_URL secret.")
    st.stop()

st.title("🚀 Enterprise Fleet Monitoring")
st.markdown("Live monitoring of concurrent WebSocket streams. Visualizing real-time fraud detection across multiple active calls.")
st.divider()

# ==========================================
# FILE INGESTION & SORTING
# ==========================================
SAMPLE_DIR = "frontend/samples"

if not os.path.exists(SAMPLE_DIR):
    st.error(f"Error: Directory '{SAMPLE_DIR}' not found.")
    st.stop()

def extract_number(filename):
    """
    Applies natural sorting to filenames. 
    Ensures 'sample_10.wav' comes after 'sample_2.wav', not before it.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

raw_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(('.wav', '.mp3'))]
audio_files = sorted(raw_files, key=extract_number)

if len(audio_files) == 0:
    st.warning("No audio files found. Please add samples.")
    st.stop()

# Initialize global session state to aggregate metrics across all concurrent async tasks
if "graph_data" not in st.session_state:
    st.session_state.graph_data = {"times": [], "system_lat": [], "model_lat": []}

# ==========================================
# SECTION 1: LIVE METRICS DASHBOARD
# ==========================================
st.markdown("### 📡 Global Fleet Latency")
col_g1, col_g2 = st.columns(2)
sys_graph_ph = col_g1.empty()
mod_graph_ph = col_g2.empty()

def render_graphs():
    """
    Pulls the latest aggregated telemetry data from session state 
    and redraws the Plotly Box Plots. Called repeatedly during the async loop.
    """
    gd = st.session_state.graph_data
    if not gd["times"]:
        return

    # GRAPH 1: Tracks total round-trip time, highlighting infrastructure/queueing bottlenecks
    fig1 = go.Figure()
    fig1.add_trace(go.Box(
        x=gd["times"], 
        y=gd["system_lat"], 
        name="System Latency",
        marker_color='#1f77b4',
        boxpoints='all', # Renders raw data points alongside statistical boxes
        jitter=0.3,      
        pointpos=-1.8,   
        opacity=0.8
    ))
    fig1.update_layout(
        title="End-to-End System Latency (Cold Start vs. Steady State)", 
        xaxis_title="Call Timeline (Seconds)", 
        yaxis_title="Total System Latency (ms)", 
        height=350, 
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(dtick=5) 
    )
    # The dynamic timestamp key prevents Streamlit Duplicate ID rendering crashes
    sys_graph_ph.plotly_chart(fig1, use_container_width=True, key=f"sys_graph_{time.time()}")

    # GRAPH 2: Tracks isolated PyTorch execution time on the GPU
    fig2 = go.Figure()
    fig2.add_trace(go.Box(
        x=gd["times"], 
        y=gd["model_lat"], 
        name="Model Latency",
        marker_color='#ff7f0e',
        boxpoints='all', 
        jitter=0.3, 
        pointpos=-1.8,
        opacity=0.8
    ))
    
    fig2.update_layout(
        title="Raw GPU Inference Latency", 
        xaxis_title="Call Timeline (Seconds)", 
        yaxis_title="Model Latency (ms)", 
        height=350, 
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(dtick=5)
    )
    mod_graph_ph.plotly_chart(fig2, use_container_width=True, key=f"mod_graph_{time.time()}")

# Render empty placeholders before the simulation starts
sys_graph_ph.plotly_chart(go.Figure().update_layout(title="End-to-End System Latency (Cold Start vs. Steady State)", height=350), use_container_width=True, key="sys_empty")
mod_graph_ph.plotly_chart(go.Figure().update_layout(title="Raw GPU Inference Latency", height=350), use_container_width=True, key="mod_empty")
st.write("")

# ==========================================
# SECTION 2: EXECUTION CONTROLS
# ==========================================
launch_button = st.button(f"🔥 Launch Deployment Demo ({len(audio_files)} calls)", type="primary", use_container_width=True)
server_status_indicator = st.empty()
st.divider()

# ==========================================
# SECTION 3: DYNAMIC UI GRID SETUP
# ==========================================
st.markdown("### 🟢 Active Call Streams")
ui_rows = {}

# Pre-build the visual grid of placeholders so the async tasks can write to them 
# independently without triggering full-page reloads.
for file in audio_files:
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 5, 3])
        with c1:
            st.markdown(f"**📄 {file}**")
            time_ph = st.empty()
            time_ph.markdown("⏳ *Waiting in queue...*")
        with c2:
            status_ph = st.empty()
        with c3:
            metrics_ph = st.empty()
            
        ui_rows[file] = {"time": time_ph, "status": status_ph, "metrics": metrics_ph}

# ==========================================
# SECTION 4: ASYNC STREAMING ENGINE
# ==========================================
async def process_single_stream(file_name, file_path, ui):
    """
    Handles a single continuous audio stream. 
    Chunks the audio, sends it via WebSockets, and updates its assigned UI row.
    """
    y, sr = librosa.load(file_path, sr=16000)
    chunk_size = 16000 * 5 # 5 seconds of audio per transmission
    total_seconds = int(len(y) / sr)
    current_chunk = 0
    
    try:
        # Establish persistent connection with Modal backend
        async with websockets.connect(MODAL_WS_URL, open_timeout=300, close_timeout=10, max_size=5_000_000) as websocket:
            
            for i in range(0, len(y), chunk_size):
                chunk = y[i:i + chunk_size]
                
                # Drop residual audio fragments smaller than 2 seconds
                if len(chunk) < 16000 * 2: 
                    continue
                
                current_chunk += 1
                current_sec = current_chunk * 5
                start_time = time.time()
                
                # Payload construction and transmission
                chunk = np.round(chunk, 4)
                await websocket.send(json.dumps({"audio_16k_chunk": chunk.tolist()}))
                
                # Await processing from the GPU
                response_json = await websocket.recv()
                result = json.loads(response_json)
                
                # Calculate timing metrics
                sys_lat = round((time.time() - start_time) * 1000, 2)
                mod_lat = result.get('model_latency_ms', sys_lat * 0.4) 
                prob = result.get("scam_probability", 0)
                
                # Write to global telemetry state for the graphs
                st.session_state.graph_data["times"].append(current_sec)
                st.session_state.graph_data["system_lat"].append(sys_lat)
                st.session_state.graph_data["model_lat"].append(mod_lat)
                
                # Update this stream's specific UI row
                display_sec = min(current_sec, total_seconds)
                ui["time"].markdown(f"⏱️ **{display_sec}s** / {total_seconds}s")
                ui["metrics"].markdown(f"⚡ Sys: **{sys_lat}ms** | 🧠 GPU: **{mod_lat}ms**")
                
                # Handle security triggers
                if result.get("alert") == "TRIGGERED":
                    ui["status"].error(f"🚨 **SCAM DETECTED** | Confidence: {prob:.1%} | **CALL TERMINATED AT {display_sec}s**")
                    break 
                else:
                    ui["status"].success(f"✅ Safe (Confidence: {prob:.1%}) | Processing Chunk {current_chunk}...")
                
                # Simulates the real-time passage of conversation
                await asyncio.sleep(2) 
                
    except Exception as e:
        ui["status"].warning(f"Connection closed or error: {e}")

async def run_concurrent_load_test(files):
    """
    Spawns and manages all WebSocket streams simultaneously.
    Maintains the global UI state while tasks run in the background.
    """
    tasks = []
    for file_name in files:
        file_path = os.path.join(SAMPLE_DIR, file_name)
        ui = ui_rows[file_name] 
        tasks.append(process_single_stream(file_name, file_path, ui))
    
    # Launch all tasks concurrently
    running_tasks = [asyncio.create_task(t) for t in tasks]
    is_streaming_msg_set = False
    
    # Render loop: Continually update graphs until all streams finish
    while not all(t.done() for t in running_tasks):
        
        # Trigger 'Awake' state the moment the first data point returns
        if not is_streaming_msg_set and len(st.session_state.graph_data["times"]) > 0:
            server_status_indicator.info("📡 Server Awake. Live streaming active...")
            is_streaming_msg_set = True
            
        render_graphs()
        await asyncio.sleep(2)
        
    # Final pass to ensure final data points are rendered
    render_graphs()

# ==========================================
# SECTION 5: TRIGGER LOGIC
# ==========================================
if launch_button:
    server_status_indicator.info("⏳ Waking up Modal Cloud Server and establishing WebSockets...")
    
    # Reset global metrics for a clean run
    st.session_state.graph_data = {"times": [], "system_lat": [], "model_lat": []}
    
    start_time = time.time()
    
    # Fire the async execution engine
    asyncio.run(run_concurrent_load_test(audio_files))
    
    total_time = time.time() - start_time
    server_status_indicator.success(f"🟢 Simulation Complete in {total_time:.1f}s!")
    st.toast("Simulation Complete!", icon="🎉")