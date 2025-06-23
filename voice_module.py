import os
import torch
import numpy as np
import scipy.signal as signal
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import queue
import threading
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf

class VoiceModule:
    def __init__(self, reference_path="reference_voice.npy", sr=16000, duration=2):
        """
        Initialize VoiceModule with model loading, reference voice embedding,
        and dummy ML models for emotion and focus detection.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "./saved_wav2vec2_model"
        self.processor_path = "./saved_wav2vec2_processor"
        self.reference_path = reference_path
        self.reference_voice = np.load(reference_path) if os.path.exists(reference_path) else None

        self.SPEAKER_SIMILARITY_THRESHOLD = 0.85
        self.sr = sr
        self.duration = duration
        self.blocksize = sr * duration
        self.audio_queue = queue.Queue(maxsize=10)

        self.load_model()

        # Initialize dummy models — replace with real training or loading
        self.emotion_model = LogisticRegression()
        self.focus_model = LogisticRegression()
        dummy_X = np.random.rand(10, 768 * 3)
        dummy_emotions = np.random.choice(['happy', 'sad', 'angry', 'neutral'], 10)
        dummy_focus = np.random.choice(['focused', 'drifting'], 10)
        self.emotion_model.fit(dummy_X, dummy_emotions)
        self.focus_model.fit(dummy_X, dummy_focus)

        self._stop_event = threading.Event()  # For graceful shutdown

    def load_model(self):
        """
        Load Wav2Vec2 model and processor from saved paths or download if not available.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.processor_path):
            self.processor = Wav2Vec2Processor.from_pretrained(self.processor_path)
            self.model = Wav2Vec2Model.from_pretrained(self.model_path).to(self.device)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.model.save_pretrained(self.model_path)
            self.processor.save_pretrained(self.processor_path)
        self.model.eval()

    def high_pass_filter(self, audio, cutoff=100):
        """
        Apply a high-pass Butterworth filter to remove low-frequency noise.
        """
        b, a = signal.butter(1, cutoff / (self.sr / 2), btype='high', analog=False)
        return signal.lfilter(b, a, audio)

    def is_speech_energy(self, audio, threshold=0.01):
        """
        Check if audio has enough energy to be considered speech.
        """
        return np.mean(audio**2) > threshold

    def extract_embedding(self, audio):
        """
        Extract Wav2Vec2 embeddings (mean, std, max) from audio segment.
        """
        if len(audio) < self.sr * 2:
            return None
        input_values = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state.cpu().squeeze()
        mean = hidden_states.mean(dim=0)
        std = hidden_states.std(dim=0)
        max_val, _ = hidden_states.max(dim=0)
        return torch.cat([mean, std, max_val]).numpy()

    def log_event(self, data):
        """
        Append JSON event data to a log file.
        """
        with open("ai_dhyaan_logs.txt", "a") as f:
            f.write(json.dumps(data) + "\n")

    def process_audio(self):
        """
        Thread target: process audio chunks from the queue and perform emotion,
        focus, and speaker similarity analysis.
        """
        blink_state = False
        last_blink_time = time.time()

        while not self._stop_event.is_set():
            try:
                raw_audio = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            filtered = self.high_pass_filter(raw_audio)

            if not self.is_speech_energy(filtered):
                print(" " * 30, end="\r")  # Clear line if no speech
                continue

            # Blink animation for mic activity
            if time.time() - last_blink_time > 0.5:
                blink_state = not blink_state
                last_blink_time = time.time()
            print("🎤 Voice detected " + ("🔴" if blink_state else "⚪"), end="\r")

            embed = self.extract_embedding(filtered)
            if embed is None:
                continue

            emotion = self.emotion_model.predict([embed])[0]
            focus = self.focus_model.predict([embed])[0]

            if self.reference_voice is None:
                self.reference_voice = embed
                np.save(self.reference_path, self.reference_voice)
                speaker_match = "Reference voice set ✅"
                similarity = None
            else:
                similarity = cosine_similarity([embed], [self.reference_voice])[0][0]
                speaker_match = "Same Speaker ✅" if similarity > self.SPEAKER_SIMILARITY_THRESHOLD else "Different Speaker ❌"

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            summary = {
                "timestamp": timestamp,
                "emotion": emotion,
                "focus_state": focus,
                "speaker_match": speaker_match,
                "similarity": float(similarity) if similarity is not None else None
            }

            print(f"\n🕒 {timestamp}")
            print(f"🎭 Emotion: {emotion}")
            print(f"🧠 Focus: {focus}")
            print(f"👤 Speaker Match: {speaker_match}")
            print("-" * 40)
            self.log_event(summary)

    def simulate_streaming_audio(self, wav_path):
        """
        Simulate real-time streaming by reading the wav file in chunks
        and putting them into the audio queue.
        """
        data, file_sr = sf.read(wav_path)
        if file_sr != self.sr:
            raise ValueError(f"Sample rate mismatch: {file_sr} != {self.sr}")
        total_samples = len(data)
        pos = 0
        while pos + self.blocksize <= total_samples and not self._stop_event.is_set():
            chunk = data[pos:pos + self.blocksize]
            try:
                self.audio_queue.put(chunk, timeout=1)
            except queue.Full:
                print("Warning: audio queue is full, dropping audio chunk")
            pos += self.blocksize
            time.sleep(self.duration)

    def start_voice_monitoring(self, audio_path):
        """
        Start the audio processing thread and simulate audio streaming.
        """
        processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        processing_thread.start()
        self.simulate_streaming_audio(audio_path)

    def stop(self):
        """
        Signal all running threads to stop.
        """
        self._stop_event.set()
