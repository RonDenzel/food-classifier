import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time
import io
import gdown

# ── Auto-download models from Google Drive ─────────────────────────────────────
MODEL_FILES = {
    "resnet50_meal_classifier.keras":    "1gFOvMtI07tjmJOlC6oMwYkfRcBI0wx1v",
    "mobilenetv2_meal_classifier.keras": "1rTesCL4ondRuUBviBplk6X0jdmBXTTRa",
}

def download_models():
    for filename, file_id in MODEL_FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename} from Google Drive…"):
                gdown.download(id=file_id, output=filename, quiet=False)

download_models()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["BEVERAGE", "SNACK", "STAPLE"]   # alphabetical – matches training order
CLASS_EMOJI   = {"BEVERAGE": "🥤", "SNACK": "🍿", "STAPLE": "🍚"}
CLASS_COLOR   = {"BEVERAGE": "#4FC3F7", "SNACK": "#FFB74D", "STAPLE": "#81C784"}

MODEL_CONFIG = {
    "ResNet50": {
        "path":        "resnet50_meal_classifier.keras",
        "preprocess":  tf.keras.applications.resnet50.preprocess_input,
        "description": "Deep residual network – high accuracy, larger model",
    },
    "MobileNetV2": {
        "path":        "mobilenetv2_meal_classifier.keras",
        "preprocess":  tf.keras.applications.mobilenet_v2.preprocess_input,
        "description": "Lightweight depthwise-separable CNN – fast inference",
    },
}

IMG_SIZE = (224, 224)

# ── Food gate constants ────────────────────────────────────────────────────────
# ImageNet synsets starting with "n07" cover most food/produce classes.
# _FOOD_KEYWORDS catches food-related classes outside that prefix.
_FOOD_SYNSET_PREFIX = "n07"
_FOOD_KEYWORDS = {
    "soup", "bowl", "plate", "pizza", "burger", "hotdog", "hot_dog",
    "egg", "bread", "meat", "fish", "steak", "salad", "sandwich",
    "ice_cream", "cake", "cookie", "donut", "chocolate", "candy",
    "coffee", "espresso", "milk", "beer", "wine", "juice", "cup", "mug",
    "lobster", "crab", "shrimp", "clam", "oyster", "sushi", "dumpling",
    "taco", "burrito", "noodle", "pasta", "rice", "casserole",
    "pretzel", "popcorn", "cereal", "granola", "waffle", "pancake",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #a0a8c0 !important; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; }

/* ── Main background ── */
.stApp { background: #0d0d1a; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card .label { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7499; margin-bottom: 6px; }
.metric-card .value { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.metric-card .sub   { font-size: 0.8rem; color: #6b7499; margin-top: 4px; }

/* ── Prediction banner ── */
.pred-banner {
    border-radius: 20px;
    padding: 28px 32px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 20px;
}
.pred-banner .emoji { font-size: 3.2rem; line-height: 1; }
.pred-banner .pred-class { font-size: 2rem; font-weight: 700; letter-spacing: 0.04em; }
.pred-banner .pred-conf  { font-size: 1rem; opacity: 0.75; margin-top: 4px; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(120,120,200,0.35) !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Headings ── */
h1, h2, h3 { color: #e8e8f8 !important; font-family: 'Sora', sans-serif !important; }
p, li, span { color: #b0b8d0; }

/* ── Mono font for stats ── */
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── Comparison card ── */
.cmp-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 22px;
}
.cmp-card .cmp-title { font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; color: #555d80; margin-bottom: 12px; }

/* Pulse animation on new result */
@keyframes pulse-in {
    0%   { opacity: 0; transform: translateY(8px); }
    100% { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: pulse-in 0.45s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    path = MODEL_CONFIG[model_name]["path"]
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_gate_model():
    """MobileNetV2 with ImageNet weights — used only as a food/non-food gate."""
    return tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)


def is_food_image(image: Image.Image) -> bool:
    gate = load_gate_model()
    arr = np.expand_dims(np.array(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32), axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds = gate.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=10)[0]
    for synset_id, class_name, _ in decoded:
        if synset_id.startswith(_FOOD_SYNSET_PREFIX):
            return True
        if any(kw in class_name.lower() for kw in _FOOD_KEYWORDS):
            return True
    return False


# ── Image preprocessor ────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)          # (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)
    preprocess_fn = MODEL_CONFIG[model_name]["preprocess"]
    return preprocess_fn(arr)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model_name: str):
    model = load_model(model_name)
    x = preprocess_image(image, model_name)

    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]           # shape (3,)
    inference_ms = (time.perf_counter() - t0) * 1000

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return pred_class, confidence, probs, inference_ms


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Food Classifier")
    st.markdown("<p style='font-size:0.82rem;color:#555d80;margin-top:-8px;'>Meal category recognition</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<p style='font-size:0.75rem;letter-spacing:0.12em;text-transform:uppercase;color:#555d80;'>Select Model</p>", unsafe_allow_html=True)
    model_name = st.selectbox(
        label="model",
        options=list(MODEL_CONFIG.keys()),
        label_visibility="collapsed",
    )

    cfg = MODEL_CONFIG[model_name]
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 16px;margin-top:8px;'>
        <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:#555d80;margin-bottom:6px;'>Active Model</div>
        <div style='font-size:1.05rem;font-weight:600;color:#c8d0f0;'>{model_name}</div>
        <div style='font-size:0.78rem;color:#6b7499;margin-top:4px;'>{cfg["description"]}</div>
        <div style='font-size:0.72rem;color:#3d4460;margin-top:8px;font-family:monospace;'>{cfg["path"]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:0.75rem;color:#3d4460;'>Classes</p>", unsafe_allow_html=True)
    for cls in CLASS_NAMES:
        st.markdown(f"<span style='font-size:0.85rem;'>{CLASS_EMOJI[cls]} {cls.capitalize()}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:0.7rem;color:#3d4460;'>Input size: 224 × 224 · RGB</p>", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# 🍽️ Food Image Classifier")
st.markdown("<p style='color:#6b7499;margin-top:-12px;'>Upload a food photo and classify it as Beverage, Snack, or Staple.</p>", unsafe_allow_html=True)

col_upload, col_results = st.columns([1, 1.35], gap="large")

with col_upload:
    st.markdown("### Upload Image")
    uploaded = st.file_uploader(
        "Choose a JPG / PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, use_container_width=True, caption=uploaded.name)

with col_results:
    if uploaded:
        st.markdown("### Prediction")

        with st.spinner("Checking image…"):
            food_detected = is_food_image(image)

        if not food_detected:
            # ── Not-food error banner ──────────────────────────────────────────
            st.markdown("""
            <div class="pred-banner animate-in" style="background:linear-gradient(135deg,#FF525222,#FF525208);border:1px solid #FF525555;">
                <div class="emoji">🚫</div>
                <div>
                    <div class="pred-class" style="color:#FF5252;">NOT FOOD</div>
                    <div class="pred-conf">PLS UPLOAD FOOD</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Run classifier ────────────────────────────────────────────────
            with st.spinner(f"Running {model_name}…"):
                pred_class, confidence, probs, inference_ms = predict(image, model_name)

            color = CLASS_COLOR[pred_class]
            emoji = CLASS_EMOJI[pred_class]

            # ── Prediction banner ─────────────────────────────────────────────
            st.markdown(f"""
            <div class="pred-banner animate-in" style="background:linear-gradient(135deg,{color}22,{color}08);border:1px solid {color}55;">
                <div class="emoji">{emoji}</div>
                <div>
                    <div class="pred-class" style="color:{color};">{pred_class.capitalize()}</div>
                    <div class="pred-conf">{confidence:.1f}% confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Quick metrics ─────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card animate-in">
                    <div class="label">Model</div>
                    <div class="value" style="font-size:1.1rem;color:#c8d0f0;">{model_name}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card animate-in">
                    <div class="label">Confidence</div>
                    <div class="value" style="color:{color};">{confidence:.1f}<span style="font-size:1rem;">%</span></div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card animate-in">
                    <div class="label">Inference</div>
                    <div class="value mono" style="color:#b8c0e0;font-size:1.4rem;">{inference_ms:.0f}<span style="font-size:0.9rem;">ms</span></div>
                </div>""", unsafe_allow_html=True)

            # ── Bar chart ─────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Probability Distribution**")
            chart_data = pd.DataFrame({
                "Class":       [f"{CLASS_EMOJI[c]} {c.capitalize()}" for c in CLASS_NAMES],
                "Probability": [float(p) * 100 for p in probs],
            }).set_index("Class")
            st.bar_chart(chart_data, color="#4FC3F7", height=200)

            # ── Expandable detail ─────────────────────────────────────────────
            with st.expander("📊 Detailed Probabilities", expanded=False):
                df = pd.DataFrame({
                    "Class":       [f"{CLASS_EMOJI[c]} {c.capitalize()}" for c in CLASS_NAMES],
                    "Probability": [f"{float(p)*100:.4f}%" for p in probs],
                    "Raw Score":   [f"{float(p):.6f}" for p in probs],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Run summary ───────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔍 Run Summary")
            st.markdown(f"""
            <div class="cmp-card animate-in">
                <div class="cmp-title">Last Inference</div>
                <table style="width:100%;border-collapse:collapse;">
                    <tr>
                        <td style="color:#555d80;font-size:0.82rem;padding:5px 0;">Model</td>
                        <td style="color:#c8d0f0;font-size:0.88rem;font-weight:600;text-align:right;">{model_name}</td>
                    </tr>
                    <tr>
                        <td style="color:#555d80;font-size:0.82rem;padding:5px 0;">Prediction</td>
                        <td style="color:{color};font-size:0.88rem;font-weight:600;text-align:right;">{emoji} {pred_class.capitalize()}</td>
                    </tr>
                    <tr>
                        <td style="color:#555d80;font-size:0.82rem;padding:5px 0;">Confidence</td>
                        <td style="color:#c8d0f0;font-size:0.88rem;font-weight:600;text-align:right;">{confidence:.2f}%</td>
                    </tr>
                    <tr>
                        <td style="color:#555d80;font-size:0.82rem;padding:5px 0;">Inference Time</td>
                        <td style="color:#c8d0f0;font-size:0.88rem;font-weight:600;font-family:monospace;text-align:right;">{inference_ms:.1f} ms</td>
                    </tr>
                    <tr>
                        <td style="color:#555d80;font-size:0.82rem;padding:5px 0;">Image</td>
                        <td style="color:#6b7499;font-size:0.82rem;text-align:right;">{uploaded.name}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style='
            border:2px dashed rgba(100,110,180,0.2);
            border-radius:20px;
            padding:60px 32px;
            text-align:center;
            margin-top:24px;
        '>
            <div style='font-size:3rem;margin-bottom:12px;'>🍽️</div>
            <div style='font-size:1.1rem;color:#4a5270;font-weight:600;'>No image uploaded yet</div>
            <div style='font-size:0.85rem;color:#3a3f5a;margin-top:8px;'>Upload a photo on the left to classify it</div>
        </div>
        """, unsafe_allow_html=True)
