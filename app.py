from flask import Flask, render_template, Response, redirect, url_for
import cv2, numpy as np, sqlite3, os
from datetime import datetime
from tensorflow.keras.models import load_model
from collections import deque

app = Flask(__name__)

# --- Load Model ---
model = load_model("face_emotionModel.h5", compile=False)
try:
    model.build((None, 48, 48, 1))
except:
    pass

# --- Emotion Labels ---
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# --- Create Database ---
conn = sqlite3.connect("database.db")
conn.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT,
    timestamp TEXT
)
""")
conn.close()

# --- Initialize Camera ---
camera = cv2.VideoCapture(0)

# --- Emotion Smoothing Settings ---
emotion_history = deque(maxlen=25)
stable_emotion = None
last_stable_emotion = None
detecting = True
consistency_threshold = 18 # number of similar frames before switching

def get_stable_emotion(history):
    """Return the most consistent emotion recently detected."""
    if not history:
        return None
    most_common = max(set(history), key=history.count)
    if history.count(most_common) >= consistency_threshold:
        return most_common
    return None

def gen_frames():
    """Generate frames for video streaming and emotion prediction."""
    global stable_emotion, detecting, last_stable_emotion

    while detecting:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to grayscale and preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        face = resized.astype("float") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict emotion
        preds = model.predict(face, verbose=0)[0]
        label = EMOTIONS[np.argmax(preds)]

        # Update smoothing history
        emotion_history.append(label)
        new_stable = get_stable_emotion(list(emotion_history))

        # Only update emotion if it's consistent for a while
        if new_stable and new_stable != last_stable_emotion:
            stable_emotion = new_stable
            last_stable_emotion = new_stable

        # Display emotion on video
        if stable_emotion:
            cv2.putText(frame, f"Emotion: {stable_emotion}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 0), 3)

        # Encode frame to bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream frame to web
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main webpage."""
    return render_template('index.html', emotion=stable_emotion)

@app.route('/start')
def start():
    """Start detection."""
    global detecting
    detecting = True
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    """Stop detection and log last result."""
    global detecting
    detecting = False

    if stable_emotion:
        conn = sqlite3.connect("database.db")
        conn.execute(
            "INSERT INTO results (emotion, timestamp) VALUES (?, ?)",
            (stable_emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    """Live video feed route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


