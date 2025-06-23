from pymongo import MongoClient
from datetime import datetime

# ==============================
# MongoDB Configuration
# ==============================
MONGO_URL = "mongodb://localhost:27017"
client = MongoClient(MONGO_URL)
db = client["ai-dhyaan"]

# ==============================
# Collections
# ==============================
behavior_logs = db["behavior_logs"]
user_memory = db["user_memory"]
insight_logs = db["insight_logs"]
emotion_logs = db["emotion_logs"]
intervention_logs = db["intervention_logs"]
session_metrics = db["session_metrics"]
device_logs = db["device_logs"]
feedback = db["feedback"]
user_profiles = db["user_profiles"]
users_auth = db["users_auth"]

# ==============================
# Common Insert Methods
# ==============================

def insert_behavior_log(source, text, emotion, confidence, drift):
    behavior_logs.insert_one({
        "timestamp": datetime.now(),
        "source": source,
        "text": text,
        "emotion": emotion,
        "confidence": float(confidence),
        "drift": drift
    })

def update_user_memory(text, emotion, confidence, drift):
    user_memory.insert_one({
        "timestamp": datetime.now(),
        "text": text,
        "emotion": emotion,
        "confidence": float(confidence),
        "drift": drift
    })

def log_insight(emotion, drift, status, score=None):
    insight_logs.insert_one({
        "timestamp": datetime.now(),
        "emotion": emotion,
        "drift": drift,
        "status": status,
        "score": float(score) if score else None
    })

def insert_emotion_log(source, emotion, confidence):
    emotion_logs.insert_one({
        "timestamp": datetime.now(),
        "source": source,
        "emotion": emotion,
        "confidence": float(confidence)
    })

def insert_intervention(emotion, suggestion, result=None):
    intervention_logs.insert_one({
        "timestamp": datetime.now(),
        "emotion": emotion,
        "suggestion": suggestion,
        "result": result
    })

def insert_session_metric(user_id, duration, avg_focus, session_start=None):
    session_metrics.insert_one({
        "user_id": user_id,
        "session_start": session_start or datetime.now(),
        "duration_minutes": duration,
        "avg_focus": avg_focus
    })

def log_device_usage(user_id, os_info, location=None):
    device_logs.insert_one({
        "user_id": user_id,
        "timestamp": datetime.now(),
        "os_info": os_info,
        "location": location
    })

def log_feedback(user_id, message, rating=None):
    feedback.insert_one({
        "timestamp": datetime.now(),
        "user_id": user_id,
        "message": message,
        "rating": rating
    })

def create_user_profile(user_id, name, age=None, preferences=None):
    user_profiles.insert_one({
        "user_id": user_id,
        "name": name,
        "age": age,
        "preferences": preferences or {},
        "created_at": datetime.now()
    })

def register_user_auth(email, password_hash):
    users_auth.insert_one({
        "email": email,
        "password_hash": password_hash,
        "registered_at": datetime.now()
    })

# ==============================
# Utility Functions
# ==============================

def get_user_memory_drift_texts():
    try:
        docs = user_memory.find({"drift": True})
        return [doc["text"] for doc in docs]
    except:
        return []

def get_recent_insights(limit=10):
    return list(insight_logs.find().sort("timestamp", -1).limit(limit))

def get_user_profile(user_id):
    return user_profiles.find_one({"user_id": user_id})

def get_feedback_messages(limit=10):
    return list(feedback.find().sort("timestamp", -1).limit(limit))

# ==============================
# Connection Test
# ==============================
def connect_db():
    try:
        client.admin.command("ping")
        print("✅ MongoDB connected successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
