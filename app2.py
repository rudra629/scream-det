import streamlit as st
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import speech_recognition as sr
import io
import soundfile as sf
import time
import cv2
import requests
import threading

# --- PAGE CONFIG ---
st.set_page_config(page_title="Campus Safety Beacon", page_icon="🚨", layout="wide")

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 3 
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Beacon Config
BEACON_ID = "ab907856-3412-3412-3412-341278563412"
DEVICE_ID = "AI-AUDIO-MONITORING-02"

# --- LOAD MODELS ---
@st.cache_resource
def load_yamnet_model():
    return hub.load("https://tfhub.dev/google/yamnet/1")

try:
    with st.spinner("Loading AI Models..."):
        YAMNET_MODEL = load_yamnet_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

SPEECH_KEYWORDS = ["help", "save me", "stop", "please", "danger", "bachao"]

# --- API FUNCTION ---
def send_alert_worker(frame, confidence, description, target_url):
    """Sends the alert to the specified target_url in a background thread."""
    if frame is None or frame.size == 0:
        return

    try:
        # Encode image to JPG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: return

        # ✅ FIX: Generate a UNIQUE filename
        timestamp = int(time.time())
        unique_filename = f"scream_alert_{timestamp}.jpg"

        # Prepare Data Payload
        data_payload = {
            'beacon_id': BEACON_ID,
            'confidence_score': f"{confidence:.2f}",
            'description': description,
            'device_id': DEVICE_ID
        }
        
        # Prepare File Payload
        files_payload = {
            'images': (unique_filename, buffer.tobytes(), 'image/jpeg')
        }

        # Send Request
        print(f"📤 Uploading {unique_filename} to {target_url}...")
        response = requests.post(target_url, data=data_payload, files=files_payload, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"✅ Upload Success: {response.status_code}")
        else:
            print(f"❌ Upload Failed: {response.status_code} | {response.text}")

    except Exception as e:
        print(f"❌ Connection Error: {e}")

# --- PROCESSING FUNCTIONS ---
def yamnet_scream_confidence(audio):
    try:
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = YAMNET_MODEL(audio_tensor)
        return float(np.max(scores.numpy())) 
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

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. API URL Config
    api_url = st.text_input(
        "Backend API URL", 
        value="https://resq-server.onrender.com/api/scream-detected/"
    )
    
    st.divider()
    
    # 2. Camera Switcher
    st.subheader("📷 Camera Source")
    # Added index 2 just in case
    camera_option = st.selectbox(
        "Select Camera",
        options=[0, 1, 2], 
        format_func=lambda x: f"Camera Index {x}"
    )
    st.info("Try Index 0 or 1. If 'Frame Dropped', try restarting the app.")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Controls")
    run_detection = st.checkbox("Start Monitoring", value=False)
    st.divider()
    st.subheader("Live Feed")
    camera_placeholder = st.empty() 
    status_indicator = st.empty()

with col2:
    st.subheader("Live Analysis")
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
    status_indicator.markdown("🟢 **System Active**")
    if 'logs' not in st.session_state: st.session_state['logs'] = []
    
    photos_taken_session = 0
    
    # ✅ FIX: Force DirectShow (cv2.CAP_DSHOW) for Windows
    # This usually fixes the "Camera frame dropped" error
    cap = cv2.VideoCapture(camera_option, cv2.CAP_DSHOW)
    
    # If DSHOW fails to open, fallback to default
    if not cap.isOpened():
        print(f"DirectShow failed for Index {camera_option}, trying default...")
        cap = cv2.VideoCapture(camera_option)

    if not cap.isOpened():
        st.error(f"❌ Could not open Camera {camera_option}. Check if another app is using it.")
    else:
        # Camera Warmup: Read one frame to initialize auto-exposure
        cap.read()
        time.sleep(0.5)
    
    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as stream:
            while run_detection:
                # 1. Audio
                indata, overflowed = stream.read(BLOCK_SIZE)
                audio_chunk = indata.flatten()
                vol = np.sqrt(np.mean(audio_chunk ** 2))
                
                # 2. Camera
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                else:
                    # This warning will show if the driver is busy or failing
                    st.warning(f"⚠️ Camera frame dropped (Index {camera_option})")

                # 3. AI Logic
                yamnet_score = 0.0
                speech_score = 0.0
                detected_text = None
                final_score = 0.0
                
                vol_metric.metric("Volume", f"{vol:.4f}")

                # Silence Filter
                if vol > 0.003:
                    yamnet_score = yamnet_scream_confidence(audio_chunk)
                    speech_score, detected_text = speech_keyword_confidence(audio_chunk)
                    
                    final_score = (0.4 * yamnet_score) + (0.6 * speech_score)
                    if speech_score > 0: final_score = 0.9

                    # --- ALERT TRIGGER ---
                    if final_score > 0.5:
                        alert_area.error(f"🚨 ALARM! Conf: {final_score:.2f}")
                        
                        # Only 1 Photo per Scream Event
                        if ret and frame is not None and photos_taken_session < 1:
                            
                            desc_text = f"Scream detected. Keyword: {detected_text if detected_text else 'None'}"
                            
                            # Use frame.copy() + unique filename
                            t = threading.Thread(
                                target=send_alert_worker, 
                                args=(frame.copy(), final_score, desc_text, api_url)
                            )
                            t.start()
                            
                            st.toast(f"📸 Evidence Photo Uploaded!")
                            
                            photos_taken_session = 1 

                        if detected_text:
                            st.session_state['logs'].insert(0, f"🗣️ '{detected_text}'")
                            st.session_state['logs'] = st.session_state['logs'][:5]
                            log_area.code("\n".join(st.session_state['logs']))     
                    else:
                        alert_area.success("✅ Safe")
                        if vol < 0.002: photos_taken_session = 0

                    yamnet_metric.metric("Scream Score", f"{yamnet_score:.2f}")
                    speech_metric.metric("Speech Score", f"{speech_score:.2f}")
                else:
                    photos_taken_session = 0
                    alert_area.success("✅ Safe")
                    yamnet_metric.metric("Scream Score", "0.00")
                    speech_metric.metric("Speech Score", "0.00")

                time.sleep(0.01)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cap.release() 
else:
    status_indicator.markdown("🔴 **Stopped**")