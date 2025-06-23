import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import sounddevice as sd
import soundfile as sf
import pyttsx3
import speech_recognition as sr
import pygame
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------- Paths --------
facial_model_path = r"C:\Path\to\facial_emotion_model.h5"
behavior_model_path = r"C:\Path\to\behavior_model.pkl"
voice_model_path = r"C:\Path\to\voice_model.pkl"
voice_encoder_path = r"C:\Path\to\voice_label_encoder.pkl"
behavior_log_path = r"C:\Path\to\behavior_log.csv"
fusion_log_path = r"C:\Path\to\fusion_log.csv"
intervention_sound = r"C:\Path\to\alert.wav"

# -------- Load Models --------
facial_model = load_model(facial_model_path)
behavior_model = joblib.load(behavior_model_path)
voice_model = joblib.load(voice_model_path)
voice_encoder = joblib.load(voice_encoder_path)

# -------- Emotion Labels --------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------- Mediapipe Setup --------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------- Preprocessing --------
def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    return resized.reshape(1, 48, 48, 1)

def extract_voice_features(audio_path):
    import librosa
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# -------- Detect Emotions --------
def detect_facial_emotion(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            x1 = int(min([lm.x for lm in face_landmarks.landmark]) * w)
            y1 = int(min([lm.y for lm in face_landmarks.landmark]) * h)
            x2 = int(max([lm.x for lm in face_landmarks.landmark]) * w)
            y2 = int(max([lm.y for lm in face_landmarks.landmark]) * h)

            face_img = frame[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]
            if face_img.size == 0:
                continue

            tensor = preprocess_face(face_img)
            pred = facial_model.predict(tensor)[0]
            return emotion_labels[np.argmax(pred)]
    return "No Face"

def detect_voice_emotion():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Say something for voice emotion detection...")
        audio = recognizer.listen(source, phrase_time_limit=4)
        file_path = "temp_voice.wav"
        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())

    try:
        features = extract_voice_features(file_path)
        pred = voice_model.predict(features)[0]
        return voice_encoder.inverse_transform([pred])[0]
    except:
        return "Unknown"

# -------- Fusion Logic --------
def fuse_insight(face_emotion, voice_emotion, behavior_drift):
    mood_type = "Negative" if face_emotion in ['Angry', 'Sad', 'Disgust', 'Fear'] else "Positive"
    voice_type = "Negative" if voice_emotion in ['Angry', 'Sad', 'Disgust', 'Fear'] else "Positive"
    attention_status = "Distracted" if behavior_drift == 1 else "Focused"

    # Logic rules
    if mood_type == "Negative" and voice_type == "Negative" and attention_status == "Distracted":
        state = "🛑 CRITICAL: All channels show distress"
        intervene()
    elif mood_type == "Negative" or voice_type == "Negative":
        state = "⚠️ Emotional disturbance detected"
    elif attention_status == "Distracted":
        state = "⚠️ Behavioral distraction"
    else:
        state = "✅ Stable and focused"

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data = {
        'timestamp': timestamp,
        'facial_emotion': face_emotion,
        'voice_emotion': voice_emotion,
        'behavior_status': attention_status,
        'final_state': state
    }

    log_fusion(log_data)
    return f"[{timestamp}] Face: {face_emotion} | Voice: {voice_emotion} | Behavior: {attention_status} → {state}"

# -------- Logging --------
def log_fusion(log_data):
    df = pd.DataFrame([log_data])
    if not os.path.exists(fusion_log_path):
        df.to_csv(fusion_log_path, index=False)
    else:
        df.to_csv(fusion_log_path, mode='a', header=False, index=False)

# -------- Interventions --------
def intervene():
    pygame.mixer.init()
    pygame.mixer.music.load(intervention_sound)
    pygame.mixer.music.play()
    print("🚨 Intervention triggered! Stay focused!")
    os.system('powershell -Command "Add-Type –AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show(\'Your focus seems low. Take a short break or refocus.\')"')

# -------- Main Loop --------
def main():
    if not os.path.exists(behavior_log_path):
        print("❌ Behavior CSV not found!")
        return

    df_behavior = pd.read_csv(behavior_log_path)
    expected_features = ['idle_time_sec', 'mouse_movement_rate', 'keyboard_activity_rate',
                         'app_switch_count', 'session_length', 'scroll_activity']

    missing = [f for f in expected_features if f not in df_behavior.columns]
    if missing:
        print(f"❌ Missing features in behavior log: {missing}")
        return

    latest_behavior = df_behavior.tail(1)[expected_features]
    cap = cv2.VideoCapture(0)

    print("[AI-Dhyaan] Running Fusion Engine... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_emotion = detect_facial_emotion(frame)
        voice_emotion = detect_voice_emotion() if face_emotion != "No Face" else "Unknown"
        behavior_pred = behavior_model.predict(latest_behavior)[0]

        if face_emotion == "No Face":
            cv2.putText(frame, "[No Face Detected]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            fused = fuse_insight(face_emotion, voice_emotion, behavior_pred)
            print(fused)
            cv2.putText(frame, fused, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow("AI-Dhyaan Fusion Monitor", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- Run --------
if __name__ == "__main__":
    main()
