import os
import pickle
from numbers import Number

import streamlit as st
import truststore
from sentence_transformers import SentenceTransformer


truststore.inject_into_ssl()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
EMBEDDER_PATH = os.path.join(BASE_DIR, "embedder.txt")


def normalize_label(prediction) -> str:
    if isinstance(prediction, Number):
        return "REAL" if int(prediction) == 1 else "FAKE"

    text = str(prediction).strip().upper()
    if text in {"1", "REAL"}:
        return "REAL"
    if text in {"0", "FAKE"}:
        return "FAKE"
    return text


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(EMBEDDER_PATH) as embedder_file:
        embedder_name = embedder_file.read().strip()
    embedder = SentenceTransformer(embedder_name)
    return model, embedder


def render_confidence(confidence: float):
    st.subheader("Trust Score")
    st.metric("Confidence", f"{confidence:.2f}%")
    st.progress(max(0, min(100, int(round(confidence)))) / 100)


def status_color(label: str) -> str:
    normalized = str(label).strip().upper()
    if normalized == "REAL":
        return "#22c55e"
    if normalized == "FAKE":
        return "#dc2626"
    return "#f59e0b"


def decide_label(prob_real: float, prob_fake: float, input_word_count: int) -> str:
    max_probability = max(prob_real, prob_fake)

    if input_word_count < 20:
        if max_probability < 0.75:
            return "UNCERTAIN"
        return "REAL" if prob_real >= prob_fake else "FAKE"

    if prob_fake > 0.75:
        return "FAKE"
    if prob_real > 0.6:
        return "REAL"
    return "UNCERTAIN"


st.set_page_config(page_title="Fake News Detector", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1f2937 0%, #0b1020 55%, #020617 100%);
        color: #f8fafc;
    }
    h1 {
        text-align: center;
        letter-spacing: 0.5px;
    }
    .stTextArea textarea {
        background-color: #111827;
        color: #f8fafc;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fake News Credibility Checker")

if not os.path.exists(EMBEDDER_PATH):
    st.error("Run train.py first")
    st.stop()

model, embedder = load_artifacts()

article_text = st.text_area("Paste a news paragraph", height=220, placeholder="Enter news text here...")
verify_clicked = st.button("Verify Credibility", width="stretch")

left_col, right_col = st.columns(2)

label_result = "Waiting for input"
confidence = 0.0
prob_real = 0.0
prob_fake = 0.0
show_uncertain = False

if verify_clicked:
    if not article_text.strip():
        st.warning("Please enter some text before verifying.")
        st.stop()

    embedding = embedder.encode([article_text])

    pred = model.predict(embedding)[0]
    proba = model.predict_proba(embedding)[0]

    prob_fake = float(proba[0])
    prob_real = float(proba[1])

    label_result = "REAL" if pred == 1 else "FAKE"
    confidence = max(prob_real, prob_fake) * 100
    show_uncertain = confidence < 60

with left_col:
    st.subheader("Result")
    st.markdown(
        f"<div style='font-size:1.1rem; font-weight:700; color:{status_color(label_result)};'>Label: {label_result}</div>",
        unsafe_allow_html=True,
    )
    st.write(f"Confidence: {confidence:.2f}%")
    if verify_clicked and show_uncertain:
        st.warning("Low confidence / ambiguous input")

with right_col:
    render_confidence(confidence)