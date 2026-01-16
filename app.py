import streamlit as st
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import speech_recognition as sr
import io
import soundfile as sf
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Campus Safety Beacon", page_icon="🚨", layout="wide")

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 3 # Reduced to 3s for faster UI feedback in Streamlit
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# --- LOAD MODELS (Cached for Speed) ---
@st.cache_resource
def load_yamnet_model():
    print("Loading YAMNet...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    return model

try:
    with st.spinner("Loading AI Models..."):
        YAMNET_MODEL = load_yamnet_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

SPEECH_KEYWORDS = ["help", "save me", "stop", "please", "danger", "bachao"]

# --- PROCESSING FUNCTIONS ---
def yamnet_scream_confidence(audio):
    try:
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = YAMNET_MODEL(audio_tensor)
        scores_np = scores.numpy()
        return float(np.max(scores_np)) 
    except:
        return 0.0

def speech_keyword_confidence(audio):
    recognizer = sr.Recognizer()
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype='PCM_16')
    buffer.seek(0)
    
    try:
        with sr.AudioFile(buffer) as source:
            audio_data = recognizer.record(source)
        
        # Note: recognize_google is blocking and needs internet
        try:
            text = recognizer.recognize_google(audio_data, show_all=False).lower()
        except:
            return 0.0, None
            
        conf = 0.8 if any(k in text for k in SPEECH_KEYWORDS) else 0.0
        return conf, text
    except:
        return 0.0, None

# --- UI LAYOUT ---
st.title("🚨 AI Campus Safety Beacon")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    run_detection = st.checkbox("Start Monitoring", value=False)
    
    st.divider()
    st.subheader("System Status")
    status_indicator = st.empty()
    
    st.divider()
    st.info("Ensure your microphone is active.")

with col2:
    st.subheader("Live Analysis")
    
    # Metrics containers
    m1, m2, m3 = st.columns(3)
    vol_metric = m1.empty()
    yamnet_metric = m2.empty()
    speech_metric = m3.empty()
    
    st.markdown("### 🔍 Detected Keywords")
    log_area = st.empty()
    
    st.markdown("### ⚠️ Alert Status")
    alert_area = st.empty()

# --- MAIN LOOP ---
if run_detection:
    status_indicator.markdown("🟢 **Listening...**")
    
    # Initialize Log List in Session State if not exists
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []

    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as stream:
            while run_detection:
                # 1. Read Audio
                indata, overflowed = stream.read(BLOCK_SIZE)
                audio_chunk = indata.flatten()
                
                # 2. Process Volume
                vol = np.sqrt(np.mean(audio_chunk ** 2))
                
                # Update Metrics
                vol_metric.metric("Volume", f"{vol:.4f}")

                # 3. AI Processing (Only if loud enough to save API calls)
                yamnet_score = 0.0
                speech_score = 0.0
                detected_text = None
                
                if vol > 0.003:
                    status_indicator.markdown("🟠 **Analyzing...**")
                    
                    yamnet_score = yamnet_scream_confidence(audio_chunk)
                    speech_score, detected_text = speech_keyword_confidence(audio_chunk)
                    
                    # Fusion Logic
                    final_score = (0.4 * yamnet_score) + (0.6 * speech_score)
                    if speech_score > 0: final_score = 0.9

                    # Update UI with Results
                    yamnet_metric.metric("Scream Score", f"{yamnet_score:.2f}")
                    speech_metric.metric("Speech Score", f"{speech_score:.2f}")
                    
                    # Handle Keywords
                    if detected_text:
                        st.session_state['logs'].insert(0, f"🗣️ Heard: '{detected_text}'")
                        # Keep only last 5 logs
                        st.session_state['logs'] = st.session_state['logs'][:5]
                        log_area.code("\n".join(st.session_state['logs']))

                    # Trigger Alarm
                    if final_score > 0.5:
                        alert_area.error(f"🚨 ALARM TRIGGERED! (Conf: {final_score:.2f})")
                        # Optional: Add sound or API call here
                    else:
                        alert_area.success("✅ Safe")
                        
                    status_indicator.markdown("🟢 **Listening...**")
                else:
                    # Silence
                    yamnet_metric.metric("Scream Score", "0.00")
                    speech_metric.metric("Speech Score", "0.00")
                    alert_area.success("✅ Safe")

                # Small sleep to allow UI to refresh if needed (Streamlit quirk)
                time.sleep(0.01)
                
    except Exception as e:
        st.error(f"Microphone Error: {e}")
else:
    status_indicator.markdown("🔴 **Stopped**")