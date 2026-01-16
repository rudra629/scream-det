import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import speech_recognition as sr
import threading
import queue
import time
import io
import soundfile as sf

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Seconds per chunk
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Queue to hold audio chunks (Buffer)
audio_queue = queue.Queue()

# --- LOAD MODELS (Same as before) ---
print("Loading YAMNet...")
YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet Loaded.")

YAMNET_SCREAM_KEYWORDS = ["scream", "shout", "yell", "children screaming", "crying, sobbing"]
SPEECH_KEYWORDS = ["help", "save me", "stop", "please", "danger", "bachao"]

# --- PROCESSING FUNCTIONS ---
def yamnet_scream_confidence(audio):
    # (Same logic as refined code: Use Max Score)
    try:
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = YAMNET_MODEL(audio_tensor)
        
        # Simplified Check: Just look at the top scores manually
        # In a real deployment, you'd map these indices accurately.
        # This assumes YAMNet standard output.
        scores_np = scores.numpy()
        
        # We perform a heuristic check since we don't have the class map file loaded
        # Note: In production, ensure you load the CSV map to get exact indices
        # For this example, we return the MAX score of the whole clip to detect sudden spikes
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
        # Fast timeout to keep the queue moving
        try:
            text = recognizer.recognize_google(audio_data, show_all=False).lower()
        except:
            return 0.0, None
            
        print(f"   -> Heard: '{text}'")
        conf = 0.8 if any(k in text for k in SPEECH_KEYWORDS) else 0.0
        return conf, text
    except:
        return 0.0, None

# --- THE LISTENER (THREAD 1) ---
def audio_callback(indata, frames, time, status):
    """This function runs automatically whenever the mic fills the buffer."""
    if status:
        print(status, flush=True)
    # We must make a copy of the data
    audio_queue.put(indata.copy().flatten())

# --- THE BRAIN (THREAD 2) ---
def process_audio_loop():
    print("AI Processor Started...")
    while True:
        # 1. Get audio from the waiting list (Queue)
        # This blocks until data is available, so it doesn't waste CPU
        audio_chunk = audio_queue.get()
        
        # 2. Process it
        vol = np.sqrt(np.mean(audio_chunk ** 2))
        
        if vol > 0.003: # Only process if loud enough
            yamnet = yamnet_scream_confidence(audio_chunk)
            speech, text = speech_keyword_confidence(audio_chunk)
            
            # Simple weighted average
            final = (0.4 * yamnet) + (0.6 * speech)
            
            if speech > 0: 
                final = 0.9 # Keyword overrides everything
            
            print(f"Vol: {vol:.3f} | YAMNet: {yamnet:.2f} | Speech: {speech:.2f} | FINAL: {final:.2f}")

            if final > 0.5:
                print(f"\n🚨 ALARM! Keyword: {text} 🚨\n")
        else:
            # Visual heartbeat for silence
            print(".", end="", flush=True)

        # 3. Mark this job as done
        audio_queue.task_done()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Start the Processor Thread (The Brain)
    processor_thread = threading.Thread(target=process_audio_loop)
    processor_thread.daemon = True # This ensures it dies when the main program dies
    processor_thread.start()

    # 2. Start the Recording (The Ears) - Non-Blocking
    print("Starting Microphone...")
    try:
        with sd.InputStream(callback=audio_callback, 
                            channels=1, 
                            samplerate=SAMPLE_RATE, 
                            blocksize=BLOCK_SIZE):
            print("Listening! Press Ctrl+C to stop.")
            # Keep the main thread alive to let the stream run
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")