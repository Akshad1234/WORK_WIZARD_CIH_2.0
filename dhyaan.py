# ==============================
#  Imports & Configurations
# ==============================
import os, cv2, numpy as np, pandas as pd, joblib, re, csv, threading
import mediapipe as mp
from datetime import datetime
from keras.models import load_model
from transformers import pipeline
import whisper, pyttsx3, pygame, sounddevice as sd, wavio
from collections import deque, Counter
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = "microsoft/phi-2"

# Load tokenizer and add pad token fix
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Required for padding support

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
# ==============================
#  Paths & Globals
# ==============================
BASE_PATH = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code"
FACIAL_MODEL_PATH = os.path.join(BASE_PATH, "facial_emotion_training.h5")
BEHAVIOR_MODEL_PATH = os.path.join(BASE_PATH, "behavior_drift_xgb_model.pkl")
BEHAVIOR_LOG_PATH = os.path.join(BASE_PATH, "behavior_log.csv")
ALERT_SOUND = os.path.join(BASE_PATH, "drift_alert.mp3")
LOG_FILE = os.path.join(BASE_PATH, "ai_dhyaan_log.csv")
INSIGHT_TXT_FILE = os.path.join(BASE_PATH, "insight_logs.txt")
DRIFT_SNAPSHOT_DIR = os.path.join(BASE_PATH, "drift_snapshots")

WHISPER_MODEL_SIZE = "base"
AUDIO_FILENAME = "realtime_audio.wav"
SAMPLE_RATE = 16000
RECORD_DURATION = 5

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
drift_phrases = [
   "i'm tired", "i can't focus", "i feel lost", "i'm bored", "this is hard", "i give up",
    "i'm not okay", "i feel anxious", "i'm stressed", "i want to quit", "no motivation", "i hate this",
    "i'm frustrated", "i can't think straight", "i feel blank", "i'm zoning out", "i lost track",
    "my mind is wandering", "i feel overloaded", "too much on my plate", "i don't understand",
    "i want to sleep", "why is this happening", "this is overwhelming", "i need a break", "it's too much",
    "what's the point", "nothing's working", "i'm burnt out", "i’m burned out", "i’m done", "i feel heavy",
    "i can’t deal with this", "i'm exhausted", "i can't take this anymore", "i’m spacing out", 
    "i’m just not in the mood", "i can’t think", "i need rest", "this is draining", "i hate this task",
    "i’ve lost my spark", "i can't handle this", "i feel like giving up", "i’m not interested",
    "i just want to lie down", "i’ve hit a wall", "everything’s a blur", "my brain’s fried", 
    "i can’t concentrate", "i can’t even", "this feels pointless", "i’ve had enough", "i want to escape",
    "i just don’t care anymore", "nothing makes sense", "i feel empty", "i feel stuck", "i feel numb",
    "i’m freaking out", "i’m mentally drained", "i feel crushed", "my head’s all over the place",
    "i feel like i’m drowning", "why bother", "i’m spiraling", "i can't stay focused",
    "i hate everything right now", "i feel pressure", "i’m under pressure", "my head hurts", 
    "i feel hopeless", "i just can’t", "i’m so done", "i want this to stop", "i need to disappear",
    "i’m sick of this", "i’ve had it", "leave me alone", "my mind won’t stop racing", 
    "i’m overwhelmed", "i want to scream", "i'm so behind", "everything is messed up",
    "i’m tired of trying", "i feel like crying", "i feel like a failure", "i messed up", 
    "why am i like this", "i can't do anything right", "i’m losing my mind", "i feel like quitting", 
    "nothing works out", "why can’t i focus", "i’m running on empty", "i can’t go on", "i’m done with this",
    "i feel so low", "i'm emotionally drained", "i feel broken", "i’m panicking", "i can't stop worrying",
    "my anxiety is bad", "i can’t breathe", "i feel like I’m collapsing", "i feel insecure",
    "i’m losing control", "i feel worthless", "i don't feel good", "i feel defeated", "i’m not fine",
    "i’m failing", "i hate myself", "i don’t know what to do", "i can’t remember anything",
    "i don’t feel like myself", "i just want to sleep forever", "i can’t focus on anything",
    "i feel like i’m falling apart", "i feel paralyzed", "i can't find the energy", "i’m so unmotivated",
    "i feel lazy", "i don’t feel like doing anything", "everything’s too much", "i can’t deal right now",
    "i just need everything to stop", "i’m so distracted", "i keep getting sidetracked",
    "i’m procrastinating again", "i feel dull", "i feel like time is slipping", "i keep forgetting things",
    "i feel restless", "i can’t sit still", "i can’t stop scrolling", "i’m mentally elsewhere",
    "i feel like a mess", "i need silence", "i wish i could disappear", "i feel like running away",
    "i can’t escape my thoughts", "i feel unsafe", "i’m having a hard time", "i need help", 
    "i’m barely holding on", "i can’t believe this is happening", "this isn’t me", 
    "i feel misunderstood", "i feel like screaming", "i feel invisible", "i don’t matter",
    "no one gets it", "i’m scared", "i just want peace", "i hate how i feel", "i feel trapped"
]
expected_features = ['idle_time_sec', 'mouse_movement_rate', 'keyboard_activity_rate',
                     'app_switch_count', 'session_length', 'scroll_activity']

emotion_buffer = deque(maxlen=15)

# ==============================
#  Model Loading
# ==============================
def safe_load_model():
    try:
        facial_model = load_model(FACIAL_MODEL_PATH)
        behavior_model = joblib.load(BEHAVIOR_MODEL_PATH)
        stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
        emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        return facial_model, behavior_model, stt_model, emotion_model
    except Exception as e:
        print(f"Model loading failed: {e}")
        exit()

facial_model, behavior_model, stt_model, emotion_model = safe_load_model()

# ==============================
#  Utilities
# ==============================
def speak(text):
    try:
        tts = pyttsx3.init()
        tts.say(text)
        tts.runAndWait()
    except:
        pass

def play_drift_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
    except:
        pass

def manage_log_file():
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 5_000_000:
        new_name = LOG_FILE.replace(".csv", f"_{datetime.now().strftime('%Y%m%d%H%M')}.csv")
        os.rename(LOG_FILE, new_name)

# ==============================
#  Facial Emotion Detection
# ==============================

# ==============================
# MediaPipe Face Mesh Setup
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks for gaze detection
LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]

# ==============================
# Global Variables & Thresholds
# ==============================
focus_start_time = None
emotion_drift_start_time = None
gaze_drift_start_time = None
drift_detected = False

focus_threshold_seconds = 2.5
gaze_deviation_threshold = 0.15
emotion_drift_threshold = 5     # seconds
gaze_drift_threshold = 60       # seconds

# ==============================
# Preprocess Face Image
# ==============================
def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    return resized.reshape(1, 48, 48, 1)

# ==============================
# Emotion and Drift Detection
# ==============================
def detect_facial_emotion(frame, face_mesh):
    global focus_start_time, emotion_drift_start_time, gaze_drift_start_time, drift_detected

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_time = time.time()

    emotion_drift = False
    gaze_drift = False

    if results.multi_face_landmarks:
        h, w, _ = frame.shape

        for lm in results.multi_face_landmarks:
            # Eye position for gaze
            left_eye = lm.landmark[LEFT_EYE_LANDMARKS[0]]
            right_eye = lm.landmark[RIGHT_EYE_LANDMARKS[0]]
            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_center_y = (left_eye.y + right_eye.y) / 2

            deviation_x = abs(eye_center_x - 0.5)
            deviation_y = abs(eye_center_y - 0.5)

            # Gaze Drift Detection
            if deviation_x > gaze_deviation_threshold or deviation_y > gaze_deviation_threshold:
                if gaze_drift_start_time is None:
                    gaze_drift_start_time = current_time
                elif current_time - gaze_drift_start_time >= gaze_drift_threshold:
                    gaze_drift = True
            else:
                gaze_drift_start_time = None

            # Extract Face ROI
            x1 = max(0, int(min(pt.x for pt in lm.landmark) * w))
            y1 = max(0, int(min(pt.y for pt in lm.landmark) * h))
            x2 = min(w, int(max(pt.x for pt in lm.landmark) * w))
            y2 = min(h, int(max(pt.y for pt in lm.landmark) * h))
            face_img = frame[y1:y2, x1:x2]

            # Emotion Detection
            if face_img.size > 0:
                tensor = preprocess_face(face_img)
                pred = facial_model.predict(tensor, verbose=0)[0]
                confidence = np.max(pred)
                top_emotion = emotion_labels[np.argmax(pred)]

                if confidence < 0.4:
                    emotion_status = "Emotion unclear"
                else:
                    emotion_status = top_emotion

                    if top_emotion in ["angry", "sad"]:
                        if emotion_drift_start_time is None:
                            emotion_drift_start_time = current_time
                        elif current_time - emotion_drift_start_time >= emotion_drift_threshold:
                            emotion_drift = True
                    else:
                        emotion_drift_start_time = None

                drift_detected = emotion_drift or gaze_drift
                focus_status = "Focus lost" if gaze_drift else "Focus ok"
            return top_emotion, focus_status, drift_detected, emotion_drift, gaze_drift

    # If no face detected
    if gaze_drift_start_time and time.time() - gaze_drift_start_time >= gaze_drift_threshold:
        drift_detected = True
    else:
        drift_detected = False

    return "Face not visible", "Focus lost", drift_detected, False, False


def update_dashboard_data(emotion, focus_score):
    try:
        # Set your desired full file path
        file_path = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\models\code\shared\live_data.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare the data
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "emotion": str(emotion),
            "focus": round(float(focus_score), 3)
        }

        # Write data to JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"[ERROR] Failed to write dashboard data: {e}")




# ==============================
#  Behavior Drift Detection
# ==============================
def get_latest_behavior():
    if not os.path.exists(BEHAVIOR_LOG_PATH):
        return None
    df = pd.read_csv(BEHAVIOR_LOG_PATH)
    if df.empty or any(f not in df.columns for f in expected_features):
        return None
    row = df[expected_features].tail(1)
    return None if row.isnull().values.any() else row

# ==============================
#  Voice Drift Detection
# ==============================
def record_audio_thread():
    print(" Recording... Please speak.")
    audio = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    audio_path = os.path.abspath(AUDIO_FILENAME)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True) if os.path.dirname(audio_path) else None
    wavio.write(AUDIO_FILENAME, audio, SAMPLE_RATE, sampwidth=2)
    print(" Recording complete.")

def detect_mental_drift(text):
    return any(re.search(rf"\b{re.escape(p)}\b", text.lower()) for p in drift_phrases)

def get_main_emotion(emotion_scores):
    if isinstance(emotion_scores, list) and len(emotion_scores) > 0 and isinstance(emotion_scores[0], list):
        sorted_scores = sorted(emotion_scores[0], key=lambda x: x['score'], reverse=True)
        return sorted_scores[0]["label"], sorted_scores[0]["score"]
    return "neutral", 0.5

def suggest_intervention(emotion):
     intervention_library = {
       "angry": [
        "Let’s take a calming breath together. You're safe.",
        "Anger is valid. Want to take a quick break to reset?",
        "Try clenching your fists, then slowly releasing — it helps ease tension.",
        "Let’s pause. Maybe a short walk or music can help.",
        "You're not alone. Deep breathing might help bring balance back."
    ],
    "disgust": [
        "That reaction makes sense. Let’s gently shift focus.",
        "Take a step back and center yourself — you're in control.",
        "Discomfort is a signal. Let’s acknowledge and breathe through it.",
        "You're allowed to dislike things. Let’s reset and move forward.",
        "Let’s try focusing on something you do enjoy right now."
    ],
    "fear": [
        "It’s okay to feel scared. You’re not alone.",
        "Let’s do 4-7-8 breathing together to calm the mind.",
        "Try grounding yourself — look around and name 3 things you see.",
        "Fear passes. You're stronger than this moment.",
        "Imagine a peaceful place. You’re safe and supported."
    ],
    "happy": [
        "You're shining! Let’s use that energy to move ahead.",
        "That joy is beautiful. Let’s keep the momentum going.",
        "Let’s celebrate this focus — maybe set a small challenge next?",
        "Your positive energy is powerful. Keep riding the wave!",
        "Keep that smile going — you’re doing awesome."
    ],
    "sad": [
        "It’s okay to feel sad. You’re not alone here.",
        "Let’s breathe together for a moment and reset gently.",
        "Try placing your hand on your heart — offer yourself kindness.",
        "You’re stronger than you know. Let’s take it one step at a time.",
        "Small steps are powerful. You're doing just fine."
    ],
    "surprise": [
        "That was unexpected! Let’s take a pause to process.",
        "Let’s slow things down and breathe for a moment.",
        "Try reflecting — what felt surprising or unusual?",
        "Surprises happen. You're handling it well.",
        "Let’s bring your focus back gently and steadily."
    ],
    "neutral": [
        "You're doing great — stay in the flow.",
        "Let’s keep the momentum going. Maybe a small stretch?",
        "This calm is valuable. Let’s use it to build focus.",
        "Steady and balanced — you’ve got this.",
        "Would you like to try a mini focus challenge?"
    ],
    "drift": [
        "Let's take a short break and stretch a little.",
        "It’s okay to feel off — let’s pause and breathe together.",
        "Let’s try a quick grounding exercise to bring you back.",
        "How about taking a moment to re-focus — you’ve got this.",
        "Maybe a mental reset will help — look around and name 3 things you see.",
        "Try sipping some water and closing your eyes for 10 seconds.",
        "Let’s reconnect — your thoughts matter.",
        "One step at a time — you’re not behind, you’re human.",
        "Distractions are normal — gently return your attention.",
        "You're not alone in this. A small pause can work wonders."
    ]
}
     if drift_detected:
        return random.choice(intervention_library["drift"])
     emotion = emotion.lower()
     return random.choice(intervention_library.get(emotion, [
        "You're doing fine. Let's keep going."
    ]))
     
    
   

# ==============================
#  Smart Insight
# ==============================
def fuse_insight(emotion, drift, score=None):
    mood_type = "Negative" if emotion in ['Angry', 'Sad', 'Disgust', 'Fear'] else "Positive"
    attention_status = "Distracted" if drift else "Focused"
    status = (
        "Critical: Emotionally unstable + Attention lost" if (mood_type == "Negative" and drift) else
        "Emotionally Negative, but attentive" if mood_type == "Negative" else
        "Distracted despite positive emotion" if drift else
        "Emotionally Stable and Focused"
    )
    drift_text = f"{drift} ({score:.2f})" if score is not None else f"{drift}"
    insight_line = f"[{datetime.now().strftime('%H:%M:%S')}] Emotion: {emotion} ({mood_type}) | Behavior: {attention_status} [{drift_text}] -> {status}"
    if not hasattr(fuse_insight, "prev_emotion"):
        fuse_insight.prev_emotion = None
        fuse_insight.prev_drift = None
        fuse_insight.prev_status = None
    if (fuse_insight.prev_emotion != emotion or
        fuse_insight.prev_drift != drift or
        fuse_insight.prev_status != status):
        with open(INSIGHT_TXT_FILE, 'a', encoding='utf-8') as f:
            f.write(insight_line + "\n")
        fuse_insight.prev_emotion = emotion
        fuse_insight.prev_drift = drift
        fuse_insight.prev_status = status
        return insight_line
    return fuse_insight.prev_status

# ==============================
#  Logging
# ==============================
LOG_FILE = "behavior_log.csv"  # Update if needed

def log_to_csv(source, text, emotion, confidence, drift):
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:  # Only try to make directory if one is specified
        os.makedirs(log_dir, exist_ok=True)

    # Continue with rest of the logging logic
    manage_log_file()  # (if implemented)

    file_exists = os.path.exists(LOG_FILE)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        conf_str = f"{float(confidence):.2f}"
    except (TypeError, ValueError):
        conf_str = "N/A"

    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Source", "Text", "Emotion", "Confidence", "Drift"])
        writer.writerow([
            timestamp,
            str(source).strip(),
            str(text).strip(),
            emotion,
            conf_str,
            "Yes" if drift else "No"
        ])
        
    # =====================================
# Emotion + Intention Detection (LLM)
# =====================================
def llm_analyze(text):
    prompt = (
        "What emotion is being expressed in the following sentence? "
        "Only reply with one of these: happy, sad, angry, neutral.\n\n"
        f"Sentence: \"{text}\"\nEmotion:"
    )

    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate output with proper configuration
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id  # avoids warning
    )

    # Decode and clean up the result
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extract emotion
    if "emotion:" in response.lower():
        emotion = response.split("Emotion:")[-1].strip().lower()
    else:
        emotion = response.strip().lower()

    # Fallback to ensure valid emotion
    valid_emotions = ["happy", "sad", "angry", "neutral"]
    if emotion not in valid_emotions:
        emotion = "neutral"

    # Placeholder confidence (can be refined later)
    confidence = 0.8

    return emotion, confidence

 
    


# ============================================
# Personalized Drift Detection from Chat Text
# ============================================
def detect_personalized_drift(text):
    try:
        # Load memory if it exists
        memory = pd.read_csv("user_memory.csv")
        
        # Filter previous texts labeled as drifted
        drift_texts = memory[memory["drift"] == True]["text"].tolist()
        
        # If no history, fallback to basic keyword check
        if not drift_texts:
            return any(word in text.lower() for word in ["tired", "lost", "unfocused", "mind wandering"])
        
        # Learn most common drift-related words
        keywords = []
        for txt in drift_texts:
            keywords.extend(txt.lower().split())
        
        frequent = list(pd.Series(keywords).value_counts().head(10).index)
        
        # Detect drift if current input contains common drift words
        return any(word in text.lower() for word in frequent)
    
    except:
        # Fallback in case of any error (like file not found)
        return any(word in text.lower() for word in ["tired", "lost", "unfocused", "mind wandering"])
    


# =====================================
# Update Memory Function
# =====================================
def update_user_memory(text, emotion, confidence, drift):
    """
    Updates the user memory CSV file with the latest text input,
    detected emotion, confidence score, and drift status.
    """
    fname = "user_memory.csv"
    new_row = pd.DataFrame([[text, emotion, confidence, drift]],
                           columns=["text", "emotion", "confidence", "drift"])

    if os.path.exists(fname):
        old = pd.read_csv(fname)
        updated = pd.concat([old, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(fname, index=False)
    

# ==============================
#  Main AI Dhyaan Loop
# ==============================
def ai_dhyaan_loop(mode="webcam"):
    cap = None
    face_mesh = None
    status = "Neutral"

    if mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    facial_drift = False  # Initialize
    while True:
        drift_flag = False
        behavior_score = None

        # Webcam Facial Emotion Detection
        if mode == "webcam":
            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            emotion, status, drift_flag, emotion_drift, gaze_drift = detect_facial_emotion(frame, face_mesh)
            emotion_buffer.append(emotion)
            stable_emotion = Counter(emotion_buffer).most_common(1)[0][0] if len(emotion_buffer) == emotion_buffer.maxlen else "Neutral"
        else:
            stable_emotion = "Neutral"
            frame = np.zeros((300, 300, 3), dtype=np.uint8)

        # Behavior Model
        behavior_row = get_latest_behavior()
        behavior_drift = False
        if behavior_row is not None:
            behavior_pred = behavior_model.predict_proba(behavior_row)[0]
            behavior_score = behavior_pred[1]
            behavior_drift = behavior_score > 0.6

        drift_flag = facial_drift or behavior_drift
        focus_score = 1.0 if status.lower() == "focus ok" else 0.0
        update_dashboard_data(emotion=stable_emotion, focus_score=focus_score)

        # Voice Mode
        if mode in ["voice", "both"]:
            speak("Welcome to AI Dhyaan — your dedicated cognitive and emotional support system.")
            speak("I'm now listening. How may I assist you today?")

            audio_thread = threading.Thread(target=record_audio_thread)
            audio_thread.start()
            audio_thread.join()

            

            transcription = stt_model.transcribe(AUDIO_FILENAME)["text"].strip()
            print(f"🎤 You said: {transcription}")
            speak(f"You said: {transcription}")

            emotion_scores = emotion_model(transcription)
            voice_emotion, voice_conf = get_main_emotion(emotion_scores)
            text_drift = detect_mental_drift(transcription)
            drift_flag = drift_flag or text_drift
            log_to_csv("Voice", transcription, voice_emotion, voice_conf, text_drift)

            suggested_labels = []
            for emo in emotion_scores[0]:
                if emo["score"] > 0.3:
                        label = emo["label"].lower()
                        print(f" Emotion Detected: {label} ({emo['score']:.2f})")
                        if label not in suggested_labels:
                            suggested_labels.append(label)

                if suggested_labels:
                    for label in suggested_labels:
                        speak(suggest_intervention(label))
                else:
                    speak("Hmm, I couldn't clearly detect an emotion, but you're doing okay!")
                    
                if text_drift:
                    speak("It seems like your focus is drifting. Let’s refocus.")
                    break

                status = fuse_insight(voice_emotion, drift_flag, behavior_score)


        # Chat Mode
        elif mode == "chat":
           typed_input = input("\n🧠 Enter how you're feeling or thinking: ")

    # 1. Emotion detection using local LLM
           typed_emotion, typed_conf = llm_analyze(typed_input)

    # 2. Drift detection (personalized)
           text_drift = detect_personalized_drift(typed_input)
           drift_flag = drift_flag or text_drift

    # 3. Log & update user memory
           log_to_csv("Chat", typed_input, typed_emotion, typed_conf, text_drift)
           update_user_memory(typed_input, typed_emotion, typed_conf, text_drift)

    # 4. Fuse insights
           status = fuse_insight(typed_emotion, drift_flag, behavior_score)

    # 5. Suggestion & speech
           suggestion = suggest_intervention(typed_emotion.lower())
           print(f" Emotion: {typed_emotion} | Drift: {text_drift}")
           print(f" Status: {status}")
           print(f" Suggestion: {suggestion}\n")

    # 6. Handle critical/distracted states
           if "Critical" in status or "Distracted" in status:
               play_drift_alert()
               speak(suggestion)

               if frame is not None:
                  os.makedirs(DRIFT_SNAPSHOT_DIR, exist_ok=True)
                  snapshot_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                  cv2.imwrite(os.path.join(DRIFT_SNAPSHOT_DIR, snapshot_name), frame)
           else:
             speak(suggestion)


        # Webcam UI Overlay
        if mode == "webcam" and frame is not None:
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (300, 80), (0, 0, 0), -1)  # background box
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, overlay)

    # Add emotion and status text
            cv2.putText(overlay, f"Emotion: {stable_emotion.capitalize()}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"Status: {status}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # Drift Info (emotion/gaze)
        if drift_flag:
            if gaze_drift:
                cv2.putText(overlay, "Gaze Drift Detected", (15, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if emotion_drift:
                cv2.putText(overlay, f"Emotional Drift: {stable_emotion.capitalize()}", (15, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    # Optional: Add bottom hint text
        cv2.putText(overlay, "Press 'Q' to quit", (15, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

    # Show the overlay
        cv2.imshow("AI Dhyaan", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Print Status if not webcam
        if mode == "voice":
            print(f" Voice Status: {status}")
        elif mode == "chat":
            print(f" Chat Status: {status}")

        if mode != "webcam":
              # Chat & voice modes are single-turn, not continuous
             if cap:
                 cap.release()
             cv2.destroyAllWindows() 



# ==============================
#  Entry Point
# ==============================
def input_mode():
    print("\n===  AI Dhyaan Input Mode ===")
    print("1. Webcam Mode (Facial Emotion + Behavior Detection)")
    print("2. Voice Mode (Mic Input + Emotion & Drift Detection)")
    print("3. Chatbox Mode (Typed Input + Emotion & Drift Detection)")
    print("4. Exit\n")
    choice = input(" Select input mode (1/2/3/4): ").strip()
    if choice == "1":
        ai_dhyaan_loop("webcam")
    elif choice == "2":
        ai_dhyaan_loop("voice")
    elif choice == "3":
        ai_dhyaan_loop("chat")
    elif choice == "4":
        print(" Exiting AI Dhyaan. Stay focused!")
    else:
        print(" Invalid input. Please try again.")
        input_mode()


if __name__ == "__main__":
    input_mode()
