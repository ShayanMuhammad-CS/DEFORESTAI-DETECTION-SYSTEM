"""
╔══════════════════════════════════════════════════════════════════════════╗
║         DEFORESTATION DETECTION AI SYSTEM — Production App              ║
║         Built with Streamlit + TensorFlow + Plotly                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time
from io import BytesIO

# Plotly is optional — graceful fallback if not installed
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeforestAI — Deforestation Detection System",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "DeforestAI v1.0 — AI-powered deforestation detection system.",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "deforestation.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Deforestation", "No Deforestation"]
CLASS_EMOJIS = ["🪓", "🌲"]
CLASS_COLORS = ["#FF4B4B", "#22C55E"]
THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium dark-green environmental theme
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:      #0A0F0A;
    --bg-secondary:    #111811;
    --bg-card:         #161F16;
    --bg-card-hover:   #1C2A1C;
    --accent-green:    #22C55E;
    --accent-lime:     #84CC16;
    --accent-emerald:  #10B981;
    --accent-red:      #EF4444;
    --accent-amber:    #F59E0B;
    --text-primary:    #F0FDF4;
    --text-secondary:  #86EFAC;
    --text-muted:      #4ADE80;
    --border:          #1F3A1F;
    --border-bright:   #22C55E40;
    --glow-green:      0 0 30px #22C55E30;
    --glow-red:        0 0 30px #EF444430;
    --radius-md:       12px;
    --radius-lg:       20px;
    --radius-xl:       28px;
    --transition:      all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #050a05 0%, #0a140a 40%, #0d1a0d 100%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-green); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050a05 0%, #0a140a 60%, #050a05 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #86EFAC !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--accent-green) !important;
    border-radius: var(--radius-lg) !important;
    transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-lime) !important;
    background: var(--bg-card-hover) !important;
    box-shadow: var(--glow-green) !important;
}
[data-testid="stFileUploadDropzone"] > div {
    color: var(--text-secondary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 20px #22C55E40 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    box-shadow: 0 6px 30px #22C55E60 !important;
    transform: translateY(-2px) !important;
}

/* ── Metric containers ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-green) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-color: var(--accent-green) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
}

/* ── Info / Warning / Error boxes ── */
.stAlert {
    border-radius: var(--radius-md) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def hero_banner() -> str:
    return """
    <div style="
        background: linear-gradient(135deg, #052e16 0%, #14532d 40%, #166534 70%, #052e16 100%);
        border: 1px solid #22c55e30;
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 60px rgba(0,0,0,0.5), inset 0 1px 0 #22c55e20;
    ">
        <!-- animated blobs -->
        <div style="
            position: absolute; top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, #22c55e18 0%, transparent 70%);
            border-radius: 50%;
            animation: pulse 4s ease-in-out infinite;
        "></div>
        <div style="
            position: absolute; bottom: -40px; left: 10%;
            width: 160px; height: 160px;
            background: radial-gradient(circle, #10b98118 0%, transparent 70%);
            border-radius: 50%;
        "></div>

        <div style="position: relative; z-index: 1;">
            <div style="display:flex; align-items:center; gap:16px; margin-bottom:1rem;">
                <span style="font-size:3rem;">🛰️</span>
                <div>
                    <h1 style="
                        font-size: clamp(1.8rem, 4vw, 3rem);
                        font-weight: 800;
                        background: linear-gradient(90deg, #22c55e, #84cc16, #10b981);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin: 0; line-height: 1.1;
                        letter-spacing: -0.5px;
                    ">DeforestAI Detection System</h1>
                    <p style="
                        color: #86efac; margin: 0.4rem 0 0 0;
                        font-size: 1.05rem; font-weight: 400;
                        letter-spacing: 0.3px;
                    ">AI-Powered Satellite Image Analysis for Forest Conservation</p>
                </div>
            </div>
            <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:1.5rem;">
                <span style="
                    background:#22c55e20; border:1px solid #22c55e50;
                    color:#86efac; padding:0.3rem 0.9rem;
                    border-radius:50px; font-size:0.82rem; font-weight:500;
                ">🧠 Deep Learning</span>
                <span style="
                    background:#10b98120; border:1px solid #10b98150;
                    color:#6ee7b7; padding:0.3rem 0.9rem;
                    border-radius:50px; font-size:0.82rem; font-weight:500;
                ">🌍 Environmental AI</span>
                <span style="
                    background:#84cc1620; border:1px solid #84cc1650;
                    color:#bef264; padding:0.3rem 0.9rem;
                    border-radius:50px; font-size:0.82rem; font-weight:500;
                ">⚡ Real-Time Analysis</span>
                <span style="
                    background:#f59e0b20; border:1px solid #f59e0b50;
                    color:#fcd34d; padding:0.3rem 0.9rem;
                    border-radius:50px; font-size:0.82rem; font-weight:500;
                ">📡 Satellite Imagery</span>
            </div>
        </div>
    </div>
    <style>
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.15); opacity: 1; }
    }
    </style>
    """


def result_card(label: str, confidence: float, probs: list) -> str:
    is_deforest = label == "Deforestation"
    color = "#EF4444" if is_deforest else "#22C55E"
    bg_color = "#1c0a0a" if is_deforest else "#0a1c0a"
    border_color = "#ef444440" if is_deforest else "#22c55e40"
    glow = "0 0 40px #ef444425" if is_deforest else "0 0 40px #22c55e25"
    icon = "🪓" if is_deforest else "🌲"
    status_text = "THREAT DETECTED" if is_deforest else "FOREST HEALTHY"
    conf_pct = f"{confidence * 100:.1f}%"

    return f"""
    <div style="
        background: linear-gradient(135deg, {bg_color}, #111811);
        border: 1.5px solid {border_color};
        border-radius: 20px;
        padding: 2rem;
        box-shadow: {glow}, inset 0 1px 0 {border_color};
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position:absolute; top:-20px; right:-20px;
            width:120px; height:120px;
            background: radial-gradient(circle, {color}12, transparent 70%);
            border-radius:50%;
        "></div>
        <div style="position:relative; z-index:1;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <span style="font-size:2.5rem;">{icon}</span>
                <div>
                    <p style="
                        color:{color}; font-size:0.8rem; font-weight:700;
                        letter-spacing:2px; margin:0; text-transform:uppercase;
                    ">{status_text}</p>
                    <h2 style="
                        color:#f0fdf4; font-size:1.8rem; font-weight:800;
                        margin:0; letter-spacing:-0.5px;
                    ">{label}</h2>
                </div>
            </div>
            <div style="
                background:{color}15; border:1px solid {color}30;
                border-radius:12px; padding:1rem;
                display:flex; justify-content:space-between; align-items:center;
            ">
                <span style="color:#86efac; font-weight:500;">Confidence Score</span>
                <span style="
                    color:{color}; font-size:2rem; font-weight:800;
                    font-family:'JetBrains Mono', monospace;
                ">{conf_pct}</span>
            </div>
        </div>
    </div>
    """


def info_card(title: str, value: str, icon: str, color: str = "#22C55E") -> str:
    return f"""
    <div style="
        background: #161f16;
        border: 1px solid #1f3a1f;
        border-radius: 14px;
        padding: 1.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.6rem;
        transition: all 0.3s ease;
    ">
        <span style="font-size:1.6rem;">{icon}</span>
        <div>
            <p style="color:#86efac; font-size:0.75rem; margin:0; font-weight:500; letter-spacing:1px; text-transform:uppercase;">{title}</p>
            <p style="color:#f0fdf4; font-size:0.95rem; margin:0; font-weight:600;">{value}</p>
        </div>
    </div>
    """


def section_header(title: str, icon: str, subtitle: str = "") -> str:
    sub_html = f'<p style="color:#86efac; margin:0.2rem 0 0 0; font-size:0.9rem;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.2rem;">
        <span style="
            background: linear-gradient(135deg, #166534, #14532d);
            border:1px solid #22c55e40;
            border-radius:10px; padding:0.5rem 0.7rem;
            font-size:1.3rem;
        ">{icon}</span>
        <div>
            <h3 style="
                color:#f0fdf4; font-size:1.2rem; font-weight:700;
                margin:0; letter-spacing:-0.3px;
            ">{title}</h3>
            {sub_html}
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# CORE ML FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the Keras model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: '{MODEL_PATH}'. Please place it in the same directory as app.py."
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize image to 128×128, convert to RGB.
    Returns RAW pixel values (0–255) as float32 with shape (1, 128, 128, 3).
    NOTE: This model has a Rescaling(1./255) layer built-in as its first layer,
    so we must NOT pre-normalize — feed raw pixels directly.
    """
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # raw 0-255, NO /255 here
    return np.expand_dims(arr, axis=0)


def predict_image(model, image: Image.Image) -> dict:
    """
    Run inference on a PIL image.
    Returns dict with: label, confidence, probs, raw_output.
    """
    preprocessed = preprocess_image(image)
    raw_output = model.predict(preprocessed, verbose=0)

    # Handle both binary sigmoid and softmax outputs
    if raw_output.shape[-1] == 1:
        # Binary sigmoid: single neuron
        prob_deforest = float(raw_output[0][0])
        prob_no_deforest = 1.0 - prob_deforest
        probs = [prob_deforest, prob_no_deforest]
        predicted_idx = 0 if prob_deforest >= THRESHOLD else 1
        confidence = probs[predicted_idx]
    else:
        # Softmax: two neurons
        probs = [float(p) for p in raw_output[0]]
        predicted_idx = int(np.argmax(probs))
        confidence = probs[predicted_idx]

    return {
        "label": CLASS_NAMES[predicted_idx],
        "confidence": confidence,
        "probs": probs,
        "class_idx": predicted_idx,
        "raw_output": raw_output.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_grad_cam(model, image: Image.Image, class_idx: int, alpha: float = 0.4) -> Image.Image:
    """
    Generates a Grad-CAM heatmap overlaid on the original image.
    """
    preprocessed = preprocess_image(image)
    
    try:
        last_conv_output = model.get_layer("gap").input
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[last_conv_output, model.output]
        )
    except Exception as e:
        return None
        
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed)
        loss = predictions[:, class_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return image.convert("RGB")
    heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Create Jet colormap
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    jet_colors = np.uint8(255 * np.column_stack((r, g, b)))
    
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_img = Image.fromarray(jet_heatmap, mode="RGB")
    jet_img = jet_img.resize(image.size, Image.LANCZOS)
    
    jet_array = np.array(jet_img, dtype=np.float32)
    orig_array = np.array(image.convert("RGB"), dtype=np.float32)
    
    superimposed_img = jet_array * alpha + orig_array * (1 - alpha)
    return Image.fromarray(np.uint8(superimposed_img))


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_probability_chart(probs: list, predicted_idx: int):
    """Create a styled bar chart for class probabilities. Uses Plotly if available, else st fallback."""
    pct = [p * 100 for p in probs]

    if PLOTLY_AVAILABLE:
        bar_colors = ["#EF4444" if i == 0 else "#22C55E" for i in range(len(CLASS_NAMES))]
        highlight_opacity = [1.0 if i == predicted_idx else 0.55 for i in range(len(CLASS_NAMES))]
        fig = go.Figure()
        for i, (name, pct_val, color, opacity) in enumerate(zip(CLASS_NAMES, pct, bar_colors, highlight_opacity)):
            fig.add_trace(go.Bar(
                x=[pct_val],
                y=[f"{CLASS_EMOJIS[i]} {name}"],
                orientation="h",
                marker=dict(color=color, opacity=opacity, line=dict(color=color, width=1.5)),
                text=[f"<b>{pct_val:.1f}%</b>"],
                textposition="outside",
                textfont=dict(color=color, size=15, family="Inter"),
                hovertemplate=f"<b>{name}</b><br>Probability: {pct_val:.2f}%<extra></extra>",
                name=name,
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,31,22,0.6)",
            font=dict(family="Inter", color="#86EFAC"),
            showlegend=False,
            height=220,
            margin=dict(l=10, r=60, t=20, b=10),
            xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#1F3A1F",
                       ticksuffix="%", zeroline=False),
            yaxis=dict(tickfont=dict(size=13), showgrid=False),
            bargap=0.35,
        )
        return fig, "plotly"
    else:
        # Fallback: return data dict for st.bar_chart
        import pandas as pd
        df = {name: [pct_val] for name, pct_val in zip(CLASS_NAMES, pct)}
        return df, "native"


def build_gauge_chart(confidence: float, is_deforest: bool):
    """Create a gauge chart. Uses Plotly if available, else returns None."""
    if not PLOTLY_AVAILABLE:
        return None, "none"

    color = "#EF4444" if is_deforest else "#22C55E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 32, "color": color, "family": "Inter"}},
        delta={"reference": 50, "valueformat": ".1f", "font": {"size": 14, "color": "#86EFAC"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#86EFAC",
                     "tickfont": {"color": "#86EFAC", "size": 10}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#1F3A1F"},
                {"range": [50, 75], "color": "#1A3A1A"},
                {"range": [75, 100], "color": "#0f2e0f"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": confidence * 100,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#86EFAC"),
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig, "plotly"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(model_loaded: bool):
    with st.sidebar:
        # Logo / Title
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🌲</div>
            <h2 style="
                background: linear-gradient(90deg, #22c55e, #84cc16);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size:1.4rem; font-weight:800; margin:0;
                letter-spacing:-0.5px;
            ">DeforestAI</h2>
            <p style="color:#4ade80; font-size:0.78rem; margin:0.3rem 0 0 0; letter-spacing:1px;">
                DETECTION SYSTEM v1.0
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown("""
        <p style="color:#4ade80; font-size:0.72rem; font-weight:700;
                  letter-spacing:2px; text-transform:uppercase; margin-bottom:0.6rem;">
            🗺️ Navigation
        </p>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["🔍 Analyze Image", "📊 Model Information", "ℹ️ About & Help"],
            label_visibility="collapsed",
        )

        st.divider()

        # System Status
        st.markdown("""
        <p style="color:#4ade80; font-size:0.72rem; font-weight:700;
                  letter-spacing:2px; text-transform:uppercase; margin-bottom:0.6rem;">
            ⚙️ System Status
        </p>
        """, unsafe_allow_html=True)

        if model_loaded:
            st.markdown("""
            <div style="
                background:#052e16; border:1px solid #22c55e50;
                border-radius:10px; padding:0.8rem;
            ">
                <p style="color:#22c55e; margin:0; font-size:0.85rem; font-weight:600;">
                    🟢 Model Online
                </p>
                <p style="color:#86efac; margin:0.2rem 0 0 0; font-size:0.75rem;">
                    Ready for inference
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background:#1c0505; border:1px solid #ef444450;
                border-radius:10px; padding:0.8rem;
            ">
                <p style="color:#ef4444; margin:0; font-size:0.85rem; font-weight:600;">
                    🔴 Model Offline
                </p>
                <p style="color:#fca5a5; margin:0.2rem 0 0 0; font-size:0.75rem;">
                    Check model path
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Quick Stats
        st.markdown("""
        <p style="color:#4ade80; font-size:0.72rem; font-weight:700;
                  letter-spacing:2px; text-transform:uppercase; margin-bottom:0.6rem;">
            📈 Model Specs
        </p>
        """, unsafe_allow_html=True)

        specs = [
            ("Architecture", "MobileNetV2", "🧠"),
            ("Input Size", "128 × 128 px", "📐"),
            ("Classes", "2 (Binary)", "🏷️"),
            ("Framework", "TensorFlow / Keras", "⚡"),
        ]
        for label, val, icon in specs:
            st.markdown(f"""
            <div style="
                display:flex; justify-content:space-between; align-items:center;
                padding:0.45rem 0.5rem; border-bottom:1px solid #1f3a1f;
            ">
                <span style="color:#86efac; font-size:0.78rem;">{icon} {label}</span>
                <span style="color:#f0fdf4; font-size:0.78rem; font-weight:600;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Developer info
        st.markdown("""
        <div style="
            background: #161f16; border:1px solid #1f3a1f;
            border-radius:12px; padding:1rem; text-align:center;
        ">
            <p style="color:#86efac; font-size:0.72rem; font-weight:700;
                      letter-spacing:1px; text-transform:uppercase; margin:0 0 0.6rem 0;">
                👨‍💻 Developer
            </p>
            <p style="color:#f0fdf4; font-size:0.8rem; font-weight:600; margin:0;">
                Shayan Muhammad (24MDBCS557)<br>
                Umar Farooq (24MDBCS561)<br>
                Owais Ghani Khan (24MDBCS587)
            </p>
            <p style="color:#4ade80; font-size:0.75rem; margin:0.2rem 0 0 0;">
                Final Year Project · 2026
            </p>
            <div style="margin-top:0.8rem;">
                <span style="
                    background:#22c55e20; border:1px solid #22c55e40;
                    color:#86efac; padding:0.2rem 0.7rem;
                    border-radius:50px; font-size:0.72rem;
                ">Environmental AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYZE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_analyze(model):
    st.markdown(hero_banner(), unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown(section_header("Upload Satellite Image", "📡",
                               "Supported formats: JPG, JPEG, PNG"), unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your satellite image here",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG satellite/aerial image for analysis.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        # Empty state
        st.markdown("""
        <div style="
            background: #161f16;
            border: 1px solid #1f3a1f;
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🛰️</div>
            <h3 style="color:#86efac; font-weight:600; margin:0 0 0.5rem 0;">
                Awaiting Image Upload
            </h3>
            <p style="color:#4ade80; font-size:0.9rem; margin:0;">
                Upload a satellite or aerial image to begin deforestation analysis.
            </p>
            <p style="color:#1f3a1f; font-size:0.8rem; margin:0.8rem 0 0 0;">
                ─── Supports JPG · JPEG · PNG ───
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Validate and display uploaded image ──
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Could not read image: {e}")
        return

    col_img, col_meta = st.columns([1.4, 1], gap="large")

    with col_img:
        st.markdown(section_header("Image Preview", "🖼️"), unsafe_allow_html=True)
        st.image(image, use_container_width=True,
                 caption=f"📂 {uploaded_file.name}")

    with col_meta:
        st.markdown(section_header("Image Details", "📋"), unsafe_allow_html=True)
        w, h = image.size
        mode = image.mode
        file_size = len(uploaded_file.getvalue())
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1_048_576 else f"{file_size / 1_048_576:.2f} MB"
        ext = uploaded_file.name.split(".")[-1].upper()

        for label, val, icon in [
            ("Filename", uploaded_file.name, "📄"),
            ("Format", ext, "🖼️"),
            ("Dimensions", f"{w} × {h} px", "📐"),
            ("Color Mode", mode, "🎨"),
            ("File Size", size_str, "💾"),
            ("Input After Resize", "128 × 128 px", "⚙️"),
        ]:
            st.markdown(info_card(label, val, icon), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)

    # ── Run prediction ──
    if analyze_btn:
        if model is None:
            st.error("❌ Model is not loaded. Check the model path.")
            return

        with st.spinner("🤖 Running deep learning inference..."):
            time.sleep(0.5)  # brief pause for UX drama
            try:
                result = predict_image(model, image)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return

        st.divider()

        # ── Result Section ──
        st.markdown(section_header("Analysis Results", "📊",
                                   "AI prediction with confidence scores"), unsafe_allow_html=True)

        res_col, chart_col = st.columns([1, 1.2], gap="large")

        with res_col:
            st.markdown(
                result_card(result["label"], result["confidence"], result["probs"]),
                unsafe_allow_html=True,
            )

            # Risk badge
            risk_level = "HIGH RISK" if result["label"] == "Deforestation" else "LOW RISK"
            risk_color = "#EF4444" if result["label"] == "Deforestation" else "#22C55E"
            risk_bg = "#1c0a0a" if result["label"] == "Deforestation" else "#0a1c0a"
            st.markdown(f"""
            <div style="
                background:{risk_bg}; border:1px solid {risk_color}40;
                border-radius:12px; padding:0.8rem;
                text-align:center; margin-top:0.5rem;
            ">
                <p style="color:{risk_color}; font-size:0.8rem; font-weight:700;
                          letter-spacing:2px; margin:0; text-transform:uppercase;">
                    ⚠️ Environmental Risk: {risk_level}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with chart_col:
            tab1, tab2 = st.tabs(["📊 Probabilities", "🎯 Confidence Gauge"])

            with tab1:
                st.markdown(section_header("Class Probabilities", "📊"), unsafe_allow_html=True)
                chart_data, chart_mode = build_probability_chart(result["probs"], result["class_idx"])
                if chart_mode == "plotly":
                    st.plotly_chart(chart_data, use_container_width=True, config={"displayModeBar": False})
                else:
                    import pandas as pd
                    st.bar_chart(pd.DataFrame(chart_data), color=["#EF4444", "#22C55E"])

                # Prob detail cards
                for i, (name, prob, emoji) in enumerate(zip(CLASS_NAMES, result["probs"], CLASS_EMOJIS)):
                    color = CLASS_COLORS[i]
                    st.markdown(f"""
                    <div style="
                        background:#161f16; border:1px solid #1f3a1f;
                        border-left: 3px solid {color};
                        border-radius:10px; padding:0.6rem 1rem;
                        display:flex; justify-content:space-between;
                        margin-bottom:0.4rem;
                    ">
                        <span style="color:#86efac; font-size:0.85rem;">{emoji} {name}</span>
                        <span style="color:{color}; font-weight:700; font-size:0.95rem;
                                     font-family:'JetBrains Mono', monospace;">
                            {prob*100:.2f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            with tab2:
                st.markdown(section_header("Confidence Gauge", "🎯"), unsafe_allow_html=True)
                gauge_data, gauge_mode = build_gauge_chart(
                    result["confidence"],
                    result["label"] == "Deforestation"
                )
                if gauge_mode == "plotly":
                    st.plotly_chart(gauge_data, use_container_width=True, config={"displayModeBar": False})
                else:
                    is_def = result["label"] == "Deforestation"
                    color = "#EF4444" if is_def else "#22C55E"
                    st.markdown(f"""
                    <div style="text-align:center; padding:2rem; background:#161f16;
                                border:1px solid #1f3a1f; border-radius:14px;">
                        <p style="color:#86efac; margin:0;">Confidence</p>
                        <h1 style="color:{color}; font-size:3rem; margin:0; font-family:monospace;">
                            {result['confidence']*100:.1f}%
                        </h1>
                        <p style="color:#4ade80; margin:0;">Install plotly for gauge chart</p>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Grad-CAM Explainability ──
        st.divider()
        with st.expander("🔬 Grad-CAM Explainability", expanded=True):
            st.markdown("""
            <div style="
                background:#161f16; border:1px solid #1f3a1f;
                border-radius:14px; padding:1.5rem; margin-bottom:1rem;
            ">
                <h4 style="color:#84cc16; margin:0 0 0.8rem 0;">
                    🧠 Gradient-weighted Class Activation Mapping
                </h4>
                <p style="color:#86efac; margin:0 0 0.6rem 0; font-size:0.9rem;">
                    This heatmap highlights which regions of the satellite image the model
                    focused on when making its prediction. Red areas indicate high importance.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Generating Grad-CAM visualization..."):
                cam_image = generate_grad_cam(model, image, result["class_idx"])
                if cam_image:
                    st.image(cam_image, use_container_width=True, caption="Grad-CAM Heatmap Overlay")
                else:
                    st.error("Could not generate Grad-CAM visualization for this model architecture.")


        # ── Raw debug info ──
        with st.expander("🔧 Raw Model Output (Debug)", expanded=False):
            st.json({
                "predicted_class": result["label"],
                "class_index": result["class_idx"],
                "confidence": f"{result['confidence']:.6f}",
                "probabilities": {
                    CLASS_NAMES[i]: f"{p:.6f}"
                    for i, p in enumerate(result["probs"])
                },
                "raw_model_output": result["raw_output"],
            })


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def page_model_info(model):
    st.markdown(section_header("Model Information", "🧠",
                               "Technical details of the DeforestAI detection system"), unsafe_allow_html=True)

    # Summary card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #052e16, #0a1c0a);
        border: 1px solid #22c55e30;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    ">
        <h3 style="
            color: #22c55e; font-size: 1.3rem; font-weight: 700;
            margin: 0 0 0.8rem 0;
        ">🌲 DeforestAI — Deforestation Detection Neural Network</h3>
        <p style="color: #86efac; margin: 0; line-height: 1.7; font-size:0.95rem;">
            DeforestAI uses a fine-tuned <strong style="color:#f0fdf4;">MobileNetV2</strong>
            convolutional neural network trained on satellite and aerial imagery to classify
            land areas as either deforested or healthy forest. The system is designed for
            <strong style="color:#f0fdf4;">real-time deployment</strong> in environmental
            monitoring pipelines, enabling rapid assessment of deforestation threats across
            large geographic regions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(section_header("Architecture Details", "⚙️"), unsafe_allow_html=True)
        details = [
            ("Base Model", "MobileNetV2", "🏗️"),
            ("Input Shape", "128 × 128 × 3 (RGB)", "📐"),
            ("Output Layer", "Dense + Sigmoid/Softmax", "🎯"),
            ("Framework", "TensorFlow 2.x / Keras", "⚡"),
            ("File Format", "HDF5 (.h5)", "💾"),
            ("Inference Device", "CPU / GPU Auto", "🖥️"),
        ]
        for label, val, icon in details:
            st.markdown(info_card(label, val, icon), unsafe_allow_html=True)

    with col2:
        st.markdown(section_header("Classification Classes", "🏷️"), unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background:#1c0a0a; border:1.5px solid #ef444440;
            border-radius:14px; padding:1.2rem; margin-bottom:0.8rem;
        ">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:2rem;">🪓</span>
                <div>
                    <h4 style="color:#ef4444; margin:0; font-size:1rem;">Class 0 — Deforestation</h4>
                    <p style="color:#fca5a5; margin:0.3rem 0 0 0; font-size:0.85rem;">
                        Regions showing cleared, degraded, or logged forest. Characterized
                        by bare soil, reduced canopy density, or logging roads.
                    </p>
                </div>
            </div>
        </div>
        <div style="
            background:#0a1c0a; border:1.5px solid #22c55e40;
            border-radius:14px; padding:1.2rem;
        ">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:2rem;">🌲</span>
                <div>
                    <h4 style="color:#22c55e; margin:0; font-size:1rem;">Class 1 — No Deforestation</h4>
                    <p style="color:#86efac; margin:0.3rem 0 0 0; font-size:0.85rem;">
                        Intact, dense forest cover. High vegetation index, continuous
                        canopy, and healthy biomass signatures.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Model layer summary (if model loaded)
    if model is not None:
        st.markdown(section_header("Model Summary", "📋",
                                   "Layer-by-layer architecture breakdown"), unsafe_allow_html=True)

        # Capture model summary as text
        import io
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        summary_str = buf.getvalue()

        total_params = model.count_params()
        trainable = sum(
            np.prod(v.shape) for v in model.trainable_variables
        )
        non_trainable = total_params - trainable

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Parameters", f"{total_params:,}")
        with m2:
            st.metric("Trainable Parameters", f"{trainable:,}")
        with m3:
            st.metric("Non-Trainable Params", f"{non_trainable:,}")

        with st.expander("📄 Full Layer Summary", expanded=False):
            st.code(summary_str, language="text")
    else:
        st.warning("⚠️ Model not loaded — detailed summary unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT & HELP
# ─────────────────────────────────────────────────────────────────────────────

def page_about():
    st.markdown(section_header("About DeforestAI", "ℹ️",
                               "Mission, methodology, and usage guide"), unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #052e16, #0a1c0a);
        border: 1px solid #22c55e30;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    ">
        <h3 style="color:#22c55e; margin:0 0 1rem 0;">🌍 Our Mission</h3>
        <p style="color:#86efac; line-height:1.8; margin:0; font-size:0.95rem;">
            Deforestation is one of the most pressing environmental crises of our time.
            <strong style="color:#f0fdf4;">DeforestAI</strong> leverages state-of-the-art
            deep learning to automatically detect deforestation from satellite and aerial
            imagery — enabling conservationists, governments, and NGOs to monitor forest
            health at unprecedented scale and speed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(section_header("How to Use", "📖"), unsafe_allow_html=True)
        steps = [
            ("1", "Navigate to **🔍 Analyze Image** in the sidebar.", "🗺️"),
            ("2", "Click the upload area or drag & drop your satellite image.", "📤"),
            ("3", "Supported formats: **JPG, JPEG, PNG**.", "🖼️"),
            ("4", "Click **Analyze Image** to run AI inference.", "🔍"),
            ("5", "View prediction, confidence score, and probability chart.", "📊"),
        ]
        for num, text, icon in steps:
            st.markdown(f"""
            <div style="
                display:flex; gap:12px; align-items:flex-start;
                margin-bottom:0.8rem;
            ">
                <div style="
                    background:linear-gradient(135deg,#166534,#14532d);
                    border:1px solid #22c55e40;
                    border-radius:50%; width:28px; height:28px;
                    display:flex; align-items:center; justify-content:center;
                    flex-shrink:0; font-size:0.75rem; font-weight:700; color:#f0fdf4;
                ">{num}</div>
                <p style="color:#86efac; margin:0; font-size:0.9rem; line-height:1.5;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(section_header("Technical Stack", "🔧"), unsafe_allow_html=True)
        stack = [
            ("Streamlit", "Web Application Framework", "🌐"),
            ("TensorFlow / Keras", "Deep Learning Inference", "🧠"),
            ("Pillow (PIL)", "Image Processing & Preprocessing", "🖼️"),
            ("NumPy", "Numerical Array Operations", "🔢"),
            ("Plotly", "Interactive Data Visualization", "📊"),
            ("Python 3.10+", "Core Programming Language", "🐍"),
        ]
        for name, desc, icon in stack:
            st.markdown(f"""
            <div style="
                background:#161f16; border:1px solid #1f3a1f;
                border-radius:10px; padding:0.7rem 1rem;
                display:flex; gap:10px; align-items:center;
                margin-bottom:0.5rem;
            ">
                <span style="font-size:1.3rem;">{icon}</span>
                <div>
                    <p style="color:#f0fdf4; margin:0; font-size:0.85rem; font-weight:600;">{name}</p>
                    <p style="color:#86efac; margin:0; font-size:0.75rem;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Tips
    st.markdown(section_header("Tips for Best Results", "💡"), unsafe_allow_html=True)
    tips = [
        ("Use high-resolution aerial/satellite imagery for more accurate results.", "🛰️"),
        ("Images should clearly show vegetation or bare land from a top-down perspective.", "🌿"),
        ("Avoid images with heavy cloud cover obscuring the terrain.", "☁️"),
        ("RGB color images work best — the model expects 3-channel (RGB) input.", "🎨"),
        ("Images will be automatically resized to 128×128 px for inference.", "⚙️"),
    ]
    tip_cols = st.columns(2)
    for i, (tip, icon) in enumerate(tips):
        with tip_cols[i % 2]:
            st.markdown(f"""
            <div style="
                background:#161f16; border:1px solid #1f3a1f;
                border-left:3px solid #22c55e;
                border-radius:10px; padding:0.8rem;
                margin-bottom:0.6rem;
                display:flex; gap:10px; align-items:flex-start;
            ">
                <span style="font-size:1.2rem;">{icon}</span>
                <p style="color:#86efac; margin:0; font-size:0.85rem; line-height:1.5;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Footer
    st.markdown("""
    <div style="
        text-align:center; padding:2rem;
        background:#161f16; border:1px solid #1f3a1f;
        border-radius:16px;
    ">
        <p style="color:#22c55e; font-size:1.2rem; margin:0 0 0.5rem 0;">🌲 DeforestAI v1.0</p>
        <p style="color:#86efac; font-size:0.85rem; margin:0;">
            Built for Final Year Exhibition · AI-Powered Environmental Conservation
        </p>
        <p style="color:#4ade80; font-size:0.75rem; margin:0.5rem 0 0 0;">
            Powered by TensorFlow · Streamlit · Plotly · Python
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load model (cached)
    with st.spinner("⚡ Loading AI model..."):
        model, model_error = load_model()

    # Render sidebar and get active page
    page = render_sidebar(model is not None)

    # Show model error in main area if needed
    if model_error and "Analyze" in page:
        st.error(f"🔴 {model_error}")

    # Route pages
    if "Analyze" in page:
        page_analyze(model)
    elif "Model" in page:
        page_model_info(model)
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
