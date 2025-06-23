# Unified AI Dhyaan App (FastAPI + Dash)
# ======================================

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from starlette.middleware.wsgi import WSGIMiddleware
import numpy as np
import uvicorn
import cv2
import threading
import asyncio
import whisper
import json
import os
import tempfile
import nest_asyncio
from collections import deque
import time
from collections import Counter
# Dash imports
import dash
from dash import dcc, html as dash_html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Local imports
from app.services.dhyaan import (
    stt_model, emotion_model, detect_mental_drift, get_main_emotion,
    get_latest_behavior, behavior_model,
    detect_facial_emotion, face_mesh
)
from app.database import (
    insert_behavior_log, update_user_memory, log_insight,
    insert_emotion_log, insert_intervention,
    get_user_memory_drift_texts, connect_db
)

# =========================
# Setup
# =========================
nest_asyncio.apply()
fastapi_app = FastAPI(title="AI Dhyaan Unified API", version="3.0")
templates = Jinja2Templates(directory="templates")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")
LIVE_DATA_PATH = "C:\\Users\\akalo\\OneDrive\\Desktop\\Dhyaan\\models\\code\\shared\\live_data.json"

# =========================
# Exception Handler
# =========================
@fastapi_app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    import traceback
    print("🔥 Internal Server Error:")
    traceback.print_exc()
    return PlainTextResponse(str(exc), status_code=500)

# =========================
# Pydantic Models
# =========================
class ChatInput(BaseModel):
    text: str

class ChatOutput(BaseModel):
    emotion: str
    drift: bool
    status: str
    suggestion: str

# =========================
# Globals & Helpers
# =========================
camera_active = False
latest_result = {}

# Emotion smoothing
emotion_smooth_queue = deque(maxlen=10)
last_emotion = "neutral"
last_emotion_time = 0
emotion_update_gap_sec = 2  # Minimum 2 sec between emotion updates


def focus_to_score(focus_status: str) -> float:
    mapping = {
        "Focus ok": 1.0,
        "Focus lost": 0.0,
        "Face not visible": -1.0
    }
    return mapping.get(focus_status, -1.0)

def fuse_insight(emotion, drift, score=None):
    mood_type = "Negative" if emotion in ['angry', 'sad', 'disgust', 'fear'] else "Positive"
    status = (
        "Critical: Emotionally unstable + Attention lost" if (mood_type == "Negative" and drift) else
        "Emotionally Negative, but attentive" if mood_type == "Negative" else
        "Distracted despite positive emotion" if drift else
        "Emotionally Stable and Focused"
    )
    log_insight(emotion, drift, status, score)
    return status

def detect_personalized_drift(text):
    drift_texts = get_user_memory_drift_texts()
    keywords = [word for txt in drift_texts for word in txt.lower().split()] if drift_texts else []
    frequent = list(set(keywords)) if keywords else ["tired", "lost", "unfocused", "mind wandering"]
    return any(word in text.lower() for word in frequent)

def llm_analyze(text):
    for keyword in ["happy", "sad", "angry"]:
        if keyword in text.lower():
            return keyword, 0.9
    return "neutral", 0.7

def suggest_intervention(emotion):
    suggestions = {
        "happy": "Keep it up! Stay positive and productive.",
        "sad": "Take a short break and talk to someone you trust.",
        "angry": "Pause. Take a few deep breaths.",
        "neutral": "You're doing okay. Maybe reflect or engage with something mindful.",
        "drift": "It seems like you're drifting. Try a quick refresh activity."
    }
    return suggestions.get(emotion.lower(), "Stay balanced and mindful.")

# =========================
# Webcam Logic
# =========================
def webcam_loop(debug_mode=False):
    global camera_active, latest_result, last_emotion, last_emotion_time

    print("🎥 Webcam loop started...")

    if not debug_mode:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            camera_active = False
            return
        print("✅ Camera accessed successfully")

    while camera_active:
        if debug_mode:
            import random
            emotion = random.choice(["happy", "sad", "angry", "neutral", "emotion unclear", "face not visible"])
            focus = round(random.uniform(0.3, 1.0), 2)
            drift = focus < 0.5
            emo_drift = random.choice([True, False])
            gaze_drift = random.choice([True, False])
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not read properly")
                continue
            try:
                emotion, focus_status, drift, emo_drift, gaze_drift = detect_facial_emotion(frame, face_mesh)
                focus = focus_to_score(focus_status)
            except Exception as e:
                print("⚠️ Emotion detection failed:", str(e))
                continue

        # Smoothing
        emotion_smooth_queue.append(emotion)
        most_common_emotion = Counter(emotion_smooth_queue).most_common(1)[0][0]

        now = time.time()
        if most_common_emotion == last_emotion and (now - last_emotion_time) < emotion_update_gap_sec:
            print("⏳ Webcam loop: Emotion stable. Skipping update.")
            time.sleep(1)
            continue

        last_emotion = most_common_emotion
        last_emotion_time = now
        emotion = most_common_emotion

        suggestion = suggest_intervention(emotion if not drift else "drift")
        status = fuse_insight(emotion, drift, focus)

        latest_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": emotion,
            "focus": focus,
            "drift_detected": drift,
            "emotion_drift": emo_drift,
            "gaze_drift": gaze_drift,
            "suggestion": suggestion,
            "status": status
        }

        print("✅ latest_result updated:", latest_result)

        try:
            with open(LIVE_DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(latest_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("❌ Failed to write live_data.json:", str(e))

        insert_emotion_log("Facial" if not debug_mode else "Debug", emotion, focus)
        insert_intervention(emotion, suggestion)

        time.sleep(1.0)

    if not debug_mode:
        cap.release()


# =========================
# FastAPI Routes
# =========================
@fastapi_app.get("/")
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@fastapi_app.get("/model")
async def serve_model_page(request: Request):
    routes = [r for r in fastapi_app.routes if hasattr(r, "methods") and "GET" in r.methods]
    return templates.TemplateResponse("model.html", {"request": request, "routes": routes})

@fastapi_app.get("/dashboard")
async def serve_dashboard(request: Request):
    routes = [r for r in fastapi_app.routes if hasattr(r, "methods") and "GET" in r.methods]
    return templates.TemplateResponse("dashboard.html", {"request": request, "routes": routes})

@fastapi_app.get("/start-webcam")
async def start_webcam(debug: bool = False):
    global camera_active
    if not camera_active:
        camera_active = True
        threading.Thread(target=lambda: webcam_loop(debug_mode=debug), daemon=True).start()
    return {"message": f"Webcam started in {'debug' if debug else 'real'} mode"}

@fastapi_app.get("/stop-webcam")
async def stop_webcam():
    global camera_active
    camera_active = False
    return {"message": "Webcam stopped"}

@fastapi_app.get("/live-status")
async def get_live_status():
    if not latest_result:
        return {"message": "Waiting for data..."}
    return latest_result

@fastapi_app.post("/chat", response_model=ChatOutput)
async def analyze_chat(chat: ChatInput):
    text = chat.text.strip()
    emotion, conf = llm_analyze(text)
    drift = detect_personalized_drift(text)
    insert_behavior_log("Chat", text, emotion, conf, drift)
    update_user_memory(text, emotion, conf, drift)
    status = fuse_insight(emotion, drift)
    suggestion = suggest_intervention(emotion if not drift else "drift")
    insert_intervention(emotion, suggestion)
    return ChatOutput(emotion=emotion, drift=drift, status=status, suggestion=suggestion)

@fastapi_app.post("/analyze-voice")
async def analyze_voice(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        text = result["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper error: {str(e)}")
    finally:
        os.remove(tmp_path)

    emotion, conf = llm_analyze(text)
    drift = detect_personalized_drift(text)
    status = fuse_insight(emotion, drift)
    suggestion = suggest_intervention(emotion if not drift else "drift")

    return {
        "emotion": emotion,
        "drift": drift,
        "status": status,
        "suggestion": suggestion
    }

@fastapi_app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    global last_emotion, last_emotion_time

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Uploaded file could not be decoded into an image.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print("📥 Frame received, calling detect_facial_emotion...")
        emotion, focus_status, drift, emo_drift, gaze_drift = detect_facial_emotion(rgb_frame, face_mesh)

        # Smoothing logic
        emotion_smooth_queue.append(emotion)
        most_common_emotion = Counter(emotion_smooth_queue).most_common(1)[0][0]

        # Update only if different and enough time passed
        now = time.time()
        if most_common_emotion == last_emotion and (now - last_emotion_time) < emotion_update_gap_sec:
            print("⏳ Skipping update to avoid rapid fluctuation.")
            return JSONResponse(status_code=200, content={"message": "Emotion stable. No update."})

        last_emotion = most_common_emotion
        last_emotion_time = now
        emotion = most_common_emotion

        if emotion.lower() in ["face not visible", "emotion unclear"]:
            return JSONResponse(status_code=200, content={
                "message": "Face not visible or unclear emotion.",
                "result": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "emotion": emotion,
                    "focus": -1.0,
                    "drift_detected": True,
                    "emotion_drift": emo_drift,
                    "gaze_drift": gaze_drift,
                    "suggestion": "Please adjust your face position or lighting.",
                    "status": "Face not visible or unclear"
                }
            })

        focus_numeric = focus_to_score(focus_status)
        suggestion = suggest_intervention(emotion if not drift else "drift")
        status = fuse_insight(emotion, drift, focus_numeric)

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": emotion,
            "focus": focus_numeric,
            "drift_detected": drift,
            "emotion_drift": emo_drift,
            "gaze_drift": gaze_drift,
            "suggestion": suggestion,
            "status": status
        }

        insert_emotion_log("Frame", emotion, focus_numeric)

        with open(LIVE_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        insert_intervention(emotion, suggestion)

        return JSONResponse(content={"message": "Frame analyzed successfully", "result": result})

    except Exception as e:
        print("❌ Error in /analyze-frame:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})



@fastapi_app.get("/live-graph-data")
async def live_graph_data():
    try:
        with open(LIVE_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        emotion_mapping = {
            "happy": 2,
            "neutral": 0,
            "sad": -2,
            "angry": -3,
            "emotion unclear": -1,
            "face not visible": -1
        }

        return {
            "time": data.get("timestamp", datetime.now().strftime("%H:%M:%S")),
            "focus": float(data.get("focus", 0.5)),
            "emotion": emotion_mapping.get(data.get("emotion", "neutral").lower(), 0)
        }
    except Exception as e:
        print("❌ Error reading live data:", str(e))
        return {
            "time": datetime.now().strftime("%H:%M:%S"),
            "focus": 0.5,
            "emotion": 0
        }

# Dash Init
# =========================
dash_app = dash.Dash(
    __name__,
    routes_pathname_prefix="/live-dashboard/",
    requests_pathname_prefix="/live-dashboard/"
)
fastapi_app.mount("/live-dashboard", WSGIMiddleware(dash_app.server))

@fastapi_app.get("/test")
async def test():
    return {"msg": "FastAPI is running"}

max_length = 50
time_data = deque(maxlen=max_length)
focus_data = deque(maxlen=max_length)
emotion_data = deque(maxlen=max_length)

emotion_mapping = {
    "happy": 2,
    "neutral": 0,
    "sad": -2,
    "angry": -3,
    "emotion unclear": -1,
    "face not visible": -1
}

def read_live_data():
    try:
        with open(LIVE_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        emotion_label = data.get("emotion", "neutral").lower()
        focus_score = float(data.get("focus", 0.5))
        timestamp = data.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        emotion_val = emotion_mapping.get(emotion_label, 0)
        return timestamp, emotion_val, focus_score, emotion_label
    except Exception:
        return datetime.now().strftime('%H:%M:%S'), 0, 0.5, "neutral"

dash_app.layout = dash_html.Div([
    dash_html.H1("🧠 AI-Dhyaan - Real-Time Focus & Emotion Tracker", style={'textAlign': 'center'}),
    dcc.Graph(id='live-line-chart', animate=True),
    dash_html.Div(id="live-status-box", style={
        'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px',
        'padding': '10px', 'border': '1px solid #ccc',
        'borderRadius': '10px', 'backgroundColor': '#f0f8ff'
    }),
    dcc.Interval(id='update-interval', interval=2000, n_intervals=0)
])

@dash_app.callback(
    [Output('live-line-chart', 'figure'), Output('live-status-box', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_graph(n):
    now, emotion_val, focus_val, emotion_label = read_live_data()

    time_data.append(now)
    emotion_data.append(emotion_val)
    focus_data.append(focus_val)

    fig = go.Figure()

    # Focus Score Line + Area
    fig.add_trace(go.Scatter(
        x=list(time_data),
        y=list(focus_data),
        name='Focus Score',
        mode='lines+markers',
        line=dict(color='blue'),
        fill='tozeroy'
    ))

    # Emotion Score Line + Area
    fig.add_trace(go.Scatter(
        x=list(time_data),
        y=list(emotion_data),
        name='Emotion Score',
        mode='lines+markers',
        line=dict(color='red'),
        fill='tozeroy'
    ))

    fig.update_layout(
        title='Real-Time Emotion and Focus Monitoring (Line + Area)',
        xaxis_title='Time',
        yaxis_title='Score',
        yaxis=dict(range=[-4, 3]),  # Adjust if you expect wider range
        template='plotly_white',
        legend=dict(x=0, y=1)
    )

    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "neutral": "😐",
        "emotion unclear": "🤔",
        "face not visible": "🚫"
    }

    emoji = emoji_map.get(emotion_label, "❓")
    status = f"{emoji} Current Emotion: {emotion_label.upper()} | 🎯 Focus Score: {focus_val:.2f}"

    return fig, status


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    print("✅ Dash layout ready:", dash_app.layout is not None)
    connect_db()
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, reload=True) 