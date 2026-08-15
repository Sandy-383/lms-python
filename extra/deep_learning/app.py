"""
DAiSEE LIVE EMOTION RECOGNITION APP
===================================
Uses RTX 4060 optimized C3D model.
Run with: streamlit run app.py
"""
# Check if running directly with python
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if not get_script_run_ctx():
        print("\n\n\033[93m⚠️  WARNING: IT LOOKS LIKE YOU ARE RUNNING THIS SCRIPT DIRECTLY WITH PYTHON. ⚠️")
        print("   To run this app correctly, use the following command in your terminal:")
        print("   streamlit run app.py\033[0m\n\n")
except:
    pass

import cv2
import torch
import streamlit as st
import time

from emotion_model import EmotionDetector
from focus_utils import FocusDetector

# 1. UI SETUP
st.set_page_config(page_title="DAiSEE Live Monitor", page_icon="📹", layout="wide")
st.title("📹 DAiSEE Live Emotion Monitor")

if torch.cuda.is_available():
    st.sidebar.success(f"Running on RTX 4060 (CUDA)")
else:
    st.sidebar.warning("Running on CPU within slower inference.")

# 2. LOAD MODEL + DETECTORS
@st.cache_resource
def load_emotion_detector():
    try:
        return EmotionDetector()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

emotion_detector = load_emotion_detector()

@st.cache_resource
def load_focus_detector():
    return FocusDetector()

focus_detector = load_focus_detector()

# 3. STREAMING LOGIC
col1, col2 = st.columns([2, 1])

with col1:
    st_frame = st.empty()

with col2:
    st.subheader("Real-Time Analysis")

    # Placeholders for dynamic UI
    warning_container = st.empty()
    engagement_container = st.empty()

    status_container = st.empty()

    st.divider()
    st.caption("Debug: raw model scores")
    debug_container = st.empty()

start_btn = st.button("Start Live Feed")
stop_btn = st.button("Stop")

if start_btn:
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.stop()

    while True:
        # Note: Clicking 'Stop' triggers a Rerun, which cleanly stops this loop.

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        # UI Display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(rgb_frame, channels="RGB", use_container_width=True)

        # --- FOCUS DETECTION (rule-based: face present + eyes open) ---
        is_focused, reason = focus_detector.process(frame)

        # --- EMOTION/ENGAGEMENT DETECTION (C3D model over a rolling frame buffer) ---
        result = emotion_detector.process(frame)
        scores = result["scores"]
        e_val, b_val, c_val, f_val = scores["engagement"], scores["boredom"], scores["confusion"], scores["frustration"]

        debug_container.markdown(
            f"Buffer: `{result['buffer_fill']}/{result['buffer_needed']}`  \n"
            f"Boredom: `{b_val:.3f}`  \n"
            f"Engagement: `{e_val:.3f}`  \n"
            f"Confusion: `{c_val:.3f}`  \n"
            f"Frustration: `{f_val:.3f}`"
        )

        # --- UPDATE UI BASED ON FOCUS ---
        if not is_focused:
            # WARNING MODE
            warning_container.warning(f"⚠️ **PLEASE FOCUS!** ({reason})", icon="⚠️")

            # Show ONLY Engagement
            with engagement_container.container():
                st.progress(e_val, text=f"Engagement: {e_val:.1%}")

            # Hide Status? Or show "NOT FOCUSED"
            status_container.metric("State", "🚫 DISTRACTED")

        else:
            # NORMAL MODE
            warning_container.empty()

            with engagement_container.container():
                st.progress(e_val, text=f"Engagement: {e_val:.1%}")

            state_icons = {
                "FOCUSED": "✅ FOCUSED",
                "CONFUSED": "❓ CONFUSED",
                "FRUSTRATED": "😤 FRUSTRATED",
                "BORED": "😴 BORED",
                "NEUTRAL": "😐 NEUTRAL",
            }
            status_container.metric("State", state_icons[result["state"]])

        time.sleep(0.05)  # Cap FPS slightly

    cap.release()
