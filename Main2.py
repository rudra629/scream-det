import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json

# --- CONFIGURATION ---
# Download model from: https://alphacephei.com/vosk/models
# Extract the zip and rename folder to "model"
model = Model("model") 

# This list forces the AI to ONLY recognize these words.
# It effectively ignores background chatter.
KEYWORDS = '["help", "save me", "stop", "danger", "bachao", "scream"]'

# Create the recognizer with the specific grammar
rec = KaldiRecognizer(model, 16000, KEYWORDS)

q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called by the microphone driver every fraction of a second"""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def listen_for_keywords():
    print("Security System Active. Listening...")
    
    # Open Microphone
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            
            # Vosk processes the chunk immediately
            if rec.AcceptWaveform(data):
                # We got a full sentence/phrase
                result = json.loads(rec.Result())
                text = result.get("text", "")
                
                if text:
                    print(f"\n🚨 KEYWORD DETECTED: {text.upper()} 🚨")
                    # TRIGGER YOUR ALARM FUNCTION HERE
                    
            else:
                # Partial result (updates while you are speaking)
                partial = json.loads(rec.PartialResult())
                # print(partial["partial"]) # Optional: see what it thinks as you speak

if __name__ == '__main__':
    try:
        listen_for_keywords()
    except KeyboardInterrupt:
        print("\nStopping...")