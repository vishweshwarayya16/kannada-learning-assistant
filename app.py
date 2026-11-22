from pathlib import Path
import json
import sqlite3
import random
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modules import preprocess_image, extract_features

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "kannada_hog_svm.joblib"
AUDIO_DIR = ROOT / "dataset" / "audio"
DB_PATH = ROOT / "data" / "learning_history.sqlite"

st.set_page_config(page_title="Kannada Audio Handwriting Assistant",
                   page_icon="ಕ", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def load_character_info():
    with open(ROOT / "characters.json", encoding="utf-8") as f:
        return json.load(f)["characters"]

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner TEXT NOT NULL,
                expected TEXT NOT NULL,
                predicted TEXT NOT NULL,
                confidence REAL NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def save_attempt(learner, expected, predicted, confidence, result):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO attempts
            (learner, expected, predicted, confidence, result, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (learner, expected, predicted, float(confidence), result,
              datetime.now().isoformat(timespec="seconds")))

def read_history(learner):
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT expected, predicted, confidence, result, created_at "
            "FROM attempts WHERE learner = ? ORDER BY id DESC",
            con, params=(learner,)
        )

def decode_upload(uploaded_file):
    raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return image

def available_quiz_characters(info, model):
    model_classes = set(model.classes_.tolist())
    return [
        item for item in info
        if item["label"] in model_classes
        and (AUDIO_DIR / f'{item["label"]}.wav').exists()
    ]

init_db()
info = load_character_info()
model = load_model()

st.title("ಕನ್ನಡ ಕಲಿಕೆ")
st.subheader("Audio-Based Kannada Handwriting Learning Assistant")
st.write("Listen to a Kannada sound, write the character on paper, "
         "then upload a clear photograph.")

learner = st.text_input("Learner name or nickname", value="Learner").strip()
if not learner:
    st.warning("Enter a name or nickname to save your learning progress.")
    st.stop()

if model is None:
    st.error("The trained model was not found.")
    st.info("First collect labelled handwriting images, then run "
            "`python train_model.py`.")
    st.stop()

quiz_items = available_quiz_characters(info, model)

tab_practice, tab_history, tab_help = st.tabs(
    ["🎧 Practice", "📊 My progress", "ℹ️ Instructions"]
)

with tab_practice:
    st.markdown("### Listening challenge")

    if not quiz_items:
        st.warning(
            "No playable quiz questions are ready. Add a WAV recording for "
            "each character in dataset/audio using the exact filename, "
            "for example dataset/audio/ಕ.wav. Also train the model with "
            "images for that character."
        )
    else:
        labels = [item["label"] for item in quiz_items]
        if "expected_character" not in st.session_state or \
           st.session_state.expected_character not in labels:
            st.session_state.expected_character = random.choice(labels)
            st.session_state.last_result = None

        selected = next(
            item for item in quiz_items
            if item["label"] == st.session_state.expected_character
        )
        expected = selected["label"]
        audio_path = AUDIO_DIR / f"{expected}.wav"

        st.write("Listen carefully. The character is hidden during the test.")
        st.audio(str(audio_path), format="audio/wav")
        st.caption("Listen to the recording as many times as needed. "
                   "Do not look up the answer before submitting.")

        uploaded = st.file_uploader(
            "Upload a photograph of ONE handwritten character",
            type=["png", "jpg", "jpeg"],
            key=f"upload_{expected}"
        )

        if uploaded:
            raw_image = decode_upload(uploaded)
            if raw_image is None:
                st.error("The image could not be opened. Try another photo.")
            else:
                st.image(raw_image, caption="Your uploaded photograph",
                         use_container_width=True)

                if st.button("Recognize my handwriting", type="primary"):
                    processed = preprocess_image(raw_image)
                    if processed is None:
                        st.error("No clear dark character was detected. "
                                 "Use dark ink on plain light paper.")
                    else:
                        features = extract_features(processed).reshape(1, -1)
                        prediction = model.predict(features)[0]
                        probabilities = model.predict_proba(features)[0]
                        confidence = float(np.max(probabilities))

                        # A conservative reject option; this is not a
                        # calibrated guarantee of correctness.
                        if confidence < 0.45:
                            st.warning(
                                "The model is uncertain. Please take a "
                                "clearer photo and try again."
                            )
                            result = "Uncertain"
                        else:
                            st.write(f"Recognized character: **{prediction}**")
                            if prediction == expected:
                                st.success("Correct! You identified the sound.")
                                result = "Correct"
                            else:
                                st.error(
                                    "Not quite. The recognized character was "
                                    f"**{prediction}**. Listen again and retry."
                                )
                                result = "Incorrect"

                        save_attempt(learner, expected, str(prediction),
                                     confidence, result)
                        st.session_state.last_result = result
                        st.caption(
                            f"Model confidence: {confidence*100:.1f}% "
                            "(confidence is not the same as accuracy)."
                        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New audio question"):
                st.session_state.expected_character = random.choice(labels)
                st.session_state.last_result = None
                st.rerun()
        with col2:
            if st.button("Reveal answer"):
                st.info(
                    f"Answer: {expected} — {selected['roman']}. "
                    "Example: {0} ({1}).".format(
                        selected["example"], selected["meaning"]
                    )
                )

with tab_history:
    st.markdown("### Learning history")
    history = read_history(learner)
    if history.empty:
        st.info("No attempts recorded yet. Complete a practice question first.")
    else:
        total = len(history)
        correct = int((history["result"] == "Correct").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Attempts", total)
        c2.metric("Correct", correct)
        c3.metric("Success rate", f"{correct / total * 100:.1f}%")
        st.dataframe(history, use_container_width=True, hide_index=True)

with tab_help:
    st.markdown("### How to use the application")
    st.markdown("""
    1. Enter a learner name or nickname.
    2. Open the Practice tab.
    3. Play the Kannada audio recording.
    4. Write the sound you heard on plain white paper.
    5. Write only ONE character in dark ink.
    6. Photograph the character from directly above, with good lighting.
    7. Upload the image and select **Recognize my handwriting**.
    8. Read the feedback and check your progress.

    **Photo tips:** Avoid shadows, tilted photos, ruled lines crossing the
    character, multiple characters, and very small writing.

    **Scope:** This starter version recognizes one isolated character at a
    time. It does not assess stroke order, handwriting beauty, or full words.
    """)
    st.markdown("### Supported starter characters")
    for item in info:
        st.write(f"**{item['label']}** — {item['roman']} — "
                 f"{item['example']} ({item['meaning']})")
