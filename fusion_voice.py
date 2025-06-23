# ==============================
# 📦 Imports & Configurations
# ==============================
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from datetime import datetime
from keras.models import load_model
from transformers import pipeline
import whisper
import pyttsx3
import pygame
import sounddevice as sd
import wavio
import csv

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================
# 📁 Paths & Globals
# ==============================
FACIAL_MODEL_PATH = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\facial_emotion_training.h5"
BEHAVIOR_MODEL_PATH = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\behavior_drift_xgb_model.pkl"
BEHAVIOR_LOG_PATH = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\behavior_log.csv"
ALERT_SOUND = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\drift_alert.mp3"
LOG_FILE = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\ai_dhyaan_log.csv"

WHISPER_MODEL_SIZE = "base"
AUDIO_FILENAME = "realtime_audio.wav"
SAMPLE_RATE = 16000
RECORD_DURATION = 5

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
drift_phrases = [ "i'm tired", "i can't focus", "i feel lost", "i'm bored", "this is hard", "i give up", 
                  "i'm not okay", "i feel anxious", "i'm stressed", "i want to quit", "no motivation", 
                  "i hate this", "i'm frustrated", "i can't think straight", "i feel blank", "i'm zoning out",
                  "i lost track", "my mind is wandering", "i feel overloaded", "too much on my plate" ]

expected_features = ['idle_time_sec', 'mouse_movement_rate', 'keyboard_activity_rate',
                     'app_switch_count', 'session_length', 'scroll_activity']

# ==============================
# 📦 Model Loading
# ==============================
print("Loading models...")
facial_model = load_model(FACIAL_MODEL_PATH)
behavior_model = joblib.load(BEHAVIOR_MODEL_PATH)
stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
print("Models loaded successfully.")

# ==============================
# 🧠 Utilities
# ==============================
def speak(text):
    tts = pyttsx3.init()
    tts.say(text)
    tts.runAndWait()

def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    return resized.reshape(1, 48, 48, 1)

def detect_facial_emotion(frame, face_mesh):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        for lm in results.multi_face_landmarks:
            x1 = int(min([pt.x for pt in lm.landmark]) * w)
            y1 = int(min([pt.y for pt in lm.landmark]) * h)
            x2 = int(max([pt.x for pt in lm.landmark]) * w)
            y2 = int(max([pt.y for pt in lm.landmark]) * h)

            face_img = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if face_img.size > 0:
                tensor = preprocess_face(face_img)
                pred = facial_model.predict(tensor)[0]
                return emotion_labels[np.argmax(pred)]
    return "No Face"

def get_latest_behavior():
    if not os.path.exists(BEHAVIOR_LOG_PATH):
        return None
    df = pd.read_csv(BEHAVIOR_LOG_PATH)
    if df.empty or any(f not in df.columns for f in expected_features):
        return None
    return df.tail(1)[expected_features]

def fuse_insight(emotion, behavior_drift):
    mood_type = "Negative" if emotion in ['Angry', 'Sad', 'Disgust', 'Fear'] else "Positive"
    attention_status = "Distracted" if behavior_drift == 1 else "Focused"
    if mood_type == "Negative" and attention_status == "Distracted":
        state = "⚠️ Critical: Emotionally unstable + Attention lost"
    elif mood_type == "Negative":
        state = "⚠️ Emotionally Negative, but attentive"
    elif attention_status == "Distracted":
        state = "⚠️ Distracted despite positive emotion"
    else:
        state = "✅ Emotionally Stable and Focused"
    return f"[{datetime.now().strftime('%H:%M:%S')}] Emotion: {emotion} ({mood_type}) | Behavior: {attention_status} → {state}"

def play_drift_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
    except Exception as e:
        print("🔊 Error playing alert:", str(e))

def record_audio():
    print("Recording audio for {} seconds...".format(RECORD_DURATION))
    audio = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wavio.write(AUDIO_FILENAME, audio, SAMPLE_RATE, sampwidth=2)

def transcribe_audio():
    print("Transcribing audio...")
    result = stt_model.transcribe(AUDIO_FILENAME)
    return result['text']

def detect_mental_drift(text):
    return any(phrase in text.lower() for phrase in drift_phrases)

def get_main_emotion(emotion_scores):
    sorted_scores = sorted(emotion_scores[0], key=lambda x: x['score'], reverse=True)
    return sorted_scores[0]["label"], sorted_scores[0]["score"]

def suggest_intervention(emotion):
    return {
        "sadness": "Take a deep breath and try a 2-minute break.",
        "anger": "Let’s cool off. Want a calming audio?",
        "fear": "You’re not alone. Take a pause.",
        "joy": "Great! You're focused and doing awesome.",
        "neutral": "Keep up the steady pace.",
        "surprise": "Reflect on the moment. Take a short break if needed."
    }.get(emotion.lower(), "You're doing fine. Let's keep going.")

def log_to_csv(source, text, emotion, confidence, drift):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Source", "Text", "Emotion", "Confidence", "Drift"])
        writer.writerow([datetime.now(), source, text, emotion, f"{confidence:.2f}", "Yes" if drift else "No"])

# ==============================
# 🎤 Voice Model Integration Stub
# ==============================
def process_voice_file(filepath):
    """
    Placeholder function to process an audio file using your voice model.
    Replace this stub with your actual voice model inference code.
    """
    print(f"Processing voice file: {filepath}")
    # Example dummy output -- replace with real model inference:
    # 1) Load audio
    # 2) Run voice model inference
    # 3) Extract speaker ID, voice emotion, focus/confidence scores etc.

    # Dummy results for demonstration:
    results = {
        "speaker_id": "Speaker_1",
        "voice_emotion": "Calm",
        "focus_score": 0.85
    }
    return results

def voice_live_mode():
    print("[Voice Mode] Recording live audio and analyzing with voice model...")
    record_audio()  # saves to AUDIO_FILENAME
    results = process_voice_file(AUDIO_FILENAME)
    print(f"Voice Model Results:\nSpeaker: {results['speaker_id']}\nEmotion: {results['voice_emotion']}\nFocus Score: {results['focus_score']:.2f}")

# ==============================
# 🔄 Main Loop: Camera Fusion Mode
# ==============================
def ai_dhyaan_loop():
    print("[AI-Dhyaan] Fusion Mode Activated. Press 'q' to quit camera mode.")
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_facial_emotion(frame, face_mesh)
        behavior = get_latest_behavior()

        if emotion != "No Face" and behavior is not None:
            drift_pred = behavior_model.predict(behavior)[0]
            insight = fuse_insight(emotion, drift_pred)
            print(insight)
            if drift_pred == 1:
                play_drift_alert()
        else:
            print("No face detected or behavior data unavailable.")

        cv2.imshow('AI-Dhyaan Camera Fusion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# 📝 Speech / Typed Input Mode
# ==============================
def input_mode():
    print("[AI-Dhyaan] Speech / Text Input Mode")
    choice = input("Enter (1) to record audio OR (2) to type text: ").strip()
    text = ""

    if choice == '1':
        record_audio()
        text = transcribe_audio()
        print(f"Transcribed Text: {text}")
    elif choice == '2':
        text = input("Type your thoughts here: ").strip()
    else:
        print("Invalid choice. Returning to main menu.")
        return

    # Emotion detection on text
    emotion_scores = emotion_model(text)
    main_emotion, confidence = get_main_emotion(emotion_scores)

    # Mental drift detection
    drift = detect_mental_drift(text)
    print(f"Detected Emotion: {main_emotion} (Confidence: {confidence:.2f})")
    print(f"Mental Drift Detected: {'Yes' if drift else 'No'}")

    if drift:
        play_drift_alert()
        intervention = suggest_intervention(main_emotion)
        print("Suggested Intervention:", intervention)
        speak(intervention)

    log_to_csv("Speech/Text Input", text, main_emotion, confidence, drift)

# ==============================
# 🎛️ Main Program Loop with Menu
# ==============================
def main():
    while True:
        print("\n===== AI-Dhyaan Mental Drift & Emotion Fusion =====")
        print("1. Camera Fusion Mode (Face + Behavior)")
        print("2. Speech / Typed Emotion Input")
        print("3. Voice Model Audio File Processing")
        print("4. Voice Model Live Audio Recording")
        print("5. Exit")
        mode = input("Select Mode: ").strip()

        if mode == '1':
            ai_dhyaan_loop()
        elif mode == '2':
            input_mode()
        elif mode == '3':
            audio_path = input("Enter path to WAV audio file: ").strip()
            if os.path.exists(audio_path):
                results = process_voice_file(audio_path)
                print(f"Voice Model Results:\nSpeaker: {results['speaker_id']}\nEmotion: {results['voice_emotion']}\nFocus Score: {results['focus_score']:.2f}")
            else:
                print("❌ File not found.")
        elif mode == '4':
            voice_live_mode()
        elif mode == '5':
            print("Exiting. Stay focused! 🙏")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
