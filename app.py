import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase, AudioProcessorBase
import av
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
import queue
import time
import cv2
import requests
import os
import json

# --- VOSK IMPORT (Safe Fallback) ---
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Campus Safety Beacon", page_icon="🚨", layout="wide")

# --- CONFIGURATION ---
BEACON_ID = "ab907856-3412-3412-3412-341278563412"
DEVICE_ID = "AI-AUDIO-MONITORING-CLOUD"
BACKEND_URL = "https://resq-server.onrender.com/api/scream-detected/"

# STUN Servers (Required for Cloud Connectivity to connect browser to server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Load YAMNet
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # 2. Load Vosk (Speech Recognition)
    vosk_model = None
    if VOSK_AVAILABLE and os.path.exists("model"):
        try:
            vosk_model = Model("model")
            print("✅ Vosk Model Loaded")
        except Exception as e:
            print(f"❌ Vosk Error: {e}")
    return yamnet, vosk_model

try:
    with st.spinner("Loading AI Models..."):
        YAMNET_MODEL, VOSK_MODEL = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- SHARED STATE ---
lock = threading.Lock()
shared_state = {
    "latest_frame": None,
    "photos_taken_session": 0,
    "audio_buffer": queue.Queue(),
}

if "data_queue" not in st.session_state:
    st.session_state.data_queue = queue.Queue()

# --- BACKGROUND AI WORKER ---
def ai_worker():
    print("🚀 AI Worker Started")
    rec = None
    if VOSK_MODEL:
        rec = KaldiRecognizer(VOSK_MODEL, 16000, '["help", "save me", "stop", "danger", "bachao", "scream"]')

    yamnet_buffer = np.array([], dtype=np.float32)

    while True:
        try:
            chunk_bytes, chunk_float = shared_state["audio_buffer"].get(timeout=1.0)
        except queue.Empty:
            continue

        # 1. SPEECH RECOGNITION
        detected_text = None
        if rec:
            if rec.AcceptWaveform(chunk_bytes):
                result = json.loads(rec.Result())
                if result['text']:
                    detected_text = result['text']

        # 2. SCREAM DETECTION
        yamnet_buffer = np.append(yamnet_buffer, chunk_float)
        
        if len(yamnet_buffer) >= 16000:
            analysis_chunk = yamnet_buffer[:16000]
            yamnet_buffer = yamnet_buffer[16000:] 

            vol = np.sqrt(np.mean(analysis_chunk ** 2))
            scream_score = 0.0
            
            if vol > 0.005:
                try:
                    scores, _, _ = YAMNET_MODEL(analysis_chunk)
                    scream_score = float(np.max(scores.numpy()))
                except: pass

            # 3. DECISION LOGIC
            final_score = (0.4 * scream_score)
            if detected_text: 
                final_score = 0.9 

            alert = False
            if final_score > 0.5:
                alert = True
                with lock:
                    if shared_state["latest_frame"] is not None and shared_state["photos_taken_session"] < 1:
                        t = threading.Thread(
                            target=send_alert_worker, 
                            args=(shared_state["latest_frame"].copy(), final_score, f"Alert: {detected_text if detected_text else 'Scream'}")
                        )
                        t.start()
                        shared_state["photos_taken_session"] = 1
            else:
                if vol < 0.002: 
                    with lock: shared_state["photos_taken_session"] = 0

            # Update UI
            try:
                st.session_state.data_queue.put_nowait({
                    "vol": vol,
                    "score": final_score,
                    "text": detected_text,
                    "alert": alert
                })
            except: pass

# --- UPLOAD WORKER ---
def send_alert_worker(frame, confidence, description):
    try:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: return

        timestamp = int(time.time())
        unique_filename = f"cloud_alert_{timestamp}.jpg"

        data = {
            'beacon_id': BEACON_ID,
            'confidence_score': f"{confidence:.2f}",
            'description': description,
            'device_id': DEVICE_ID
        }
        files = {'images': (unique_filename, buffer.tobytes(), 'image/jpeg')}

        requests.post(BACKEND_URL, data=data, files=files, timeout=10)
    except Exception as e:
        print(f"Upload Error: {e}")

# --- WEBRTC PROCESSORS ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with lock:
            shared_state["latest_frame"] = img.copy()
        return frame

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        sound_data = frame.to_ndarray()
        
        if sound_data.ndim > 1:
            sound_data_mono = np.mean(sound_data, axis=1)
        else:
            sound_data_mono = sound_data
            
        if sound_data_mono.dtype != np.float32:
            sound_data_float = sound_data_mono.astype(np.float32) / 32768.0
        else:
            sound_data_float = sound_data_mono

        sound_data_float = sound_data_float[::3]
        sound_data_int16 = (sound_data_float * 32767).astype(np.int16)
        sound_bytes = sound_data_int16.tobytes()

        shared_state["audio_buffer"].put((sound_bytes, sound_data_float))
        return frame

# --- UI LAYOUT ---
st.title("🚨 Campus Safety (Cloud Edition)")
st.caption("Status: AI Running in Background Thread")

if "ai_thread_started" not in st.session_state:
    t = threading.Thread(target=ai_worker, daemon=True)
    t.start()
    st.session_state.ai_thread_started = True

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Sensor Stream")
    ctx = webrtc_streamer(
        key="safety-beacon",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": True},
        video_processor_factory=VideoProcessor,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

with col2:
    st.subheader("Telemetry")
    vol_metric = st.empty()
    score_metric = st.empty()
    text_metric = st.empty()
    alert_box = st.empty()

    if ctx.state.playing:
        while True:
            try:
                data = st.session_state.data_queue.get(timeout=0.1)
                
                vol_metric.metric("Volume", f"{data['vol']:.4f}")
                score_metric.metric("Confidence", f"{data['score']:.2f}")
                
                if data['text']:
                    text_metric.info(f"🗣️ Heard: {data['text']}")
                
                if data['alert']:
                    alert_box.error("🚨 SCREAM DETECTED!")
                else:
                    alert_box.success("Monitoring...")
            except queue.Empty:
                time.sleep(0.01)