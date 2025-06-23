import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from datetime import datetime
from keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --------- Facial Emotion Setup ---------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
facial_model_path = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\facial_emotion_training.h5"

# Load the Keras facial emotion model
facial_model = load_model(facial_model_path)

# --------- Behavior Drift Setup ---------
behavior_model_path = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\behavior_drift_xgb_model.pkl"
behavior_model = joblib.load(behavior_model_path)

# --------- Behavior CSV Load ---------
behavior_log_path = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\behavior_log.csv"
if not os.path.exists(behavior_log_path):
    print("❌ Behavior log CSV not found!")
    exit()

df_behavior = pd.read_csv(behavior_log_path)
if df_behavior.empty:
    print("❌ Behavior log is empty!")
    exit()

# --------- Use exact expected features for the behavior model ---------
expected_features = [
    'idle_time_sec',
    'mouse_movement_rate',
    'keyboard_activity_rate',
    'app_switch_count',
    'session_length',
    'scroll_activity'
]

missing_features = [feat for feat in expected_features if feat not in df_behavior.columns]
if missing_features:
    print(f"❌ Missing features in behavior log CSV: {missing_features}")
    exit()

latest_behavior = df_behavior.tail(1)[expected_features]


# Check if all expected features exist in CSV
missing_features = [feat for feat in expected_features if feat not in df_behavior.columns]
if missing_features:
    print(f"❌ Missing features in behavior log CSV: {missing_features}")
    exit()

# Select only the expected features (latest row)
latest_behavior = df_behavior.tail(1)[expected_features]

# --------- Mediapipe Setup ---------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --------- Preprocessing ---------
def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    reshaped = resized.reshape(1, 48, 48, 1)  # Keras expects NHWC format
    return reshaped

# --------- Emotion Detection ---------
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
            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue

            tensor = preprocess_face(face_img)
            pred = facial_model.predict(tensor)[0]
            predicted_idx = np.argmax(pred)
            return emotion_labels[predicted_idx]
    return "No Face"

# --------- Fusion Logic ---------
def fuse_insight(emotion, behavior_drift):
    mood_type = "Negative" if emotion in ['Angry', 'Sad', 'Disgust', 'Fear'] else "Positive"
    attention_status = "Distracted" if behavior_drift == 1 else "Focused"

    if mood_type == "Negative" and attention_status == "Distracted":
        state = "⚠️ Critical: Emotionally unstable + Attention lost"
    elif mood_type == "Negative":
        state = "⚠️ Emotionally Negative, but still attentive"
    elif attention_status == "Distracted":
        state = "⚠️ Distracted despite positive emotion"
    else:
        state = "✅ Emotionally Stable and Focused"

    return f"[{datetime.now().strftime('%H:%M:%S')}] Emotion: {emotion} ({mood_type}) | Behavior: {attention_status} → {state}"

# --------- Real-Time Fusion Loop ---------
cap = cv2.VideoCapture(0)
print("[AI-Dhyaan] Running Fusion in Real-time... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion = detect_facial_emotion(frame)
    if emotion == "No Face":
        cv2.putText(frame, "[No Face Detected]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("AI-Dhyaan Fusion Monitor", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue

    behavior_pred = behavior_model.predict(latest_behavior)[0]
    fused_result = fuse_insight(emotion, behavior_pred)

    print(fused_result)
    cv2.putText(frame, fused_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imshow("AI-Dhyaan Fusion Monitor", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
