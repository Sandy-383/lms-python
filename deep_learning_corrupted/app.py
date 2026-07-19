"""
DAiSEE LIVE EMOTION RECOGNITION APP
===================================
Uses RTX 4060 optimized C3D model.
Run with: streamlit run app.py
"""
from focus_utils import FocusDetector

# Check if running directly with python
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if not get_script_run_ctx():
        print("\n\n\033[93m⚠️  WARNING: IT LOOKS LIKE YOU ARE RUNNING THIS SCRIPT DIRECTLY WITH PYTHON. ⚠️")
        print("   To run this app correctly, use the following command in your terminal:")
        print("   streamlit run app.py\033[0m\n\n")
except:
    pass

import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from collections import deque
from pathlib import Path
import time

# 1. SETUP PATHS & IMPORTS
current_dir = Path(__file__).resolve().parent
original_scripts_dir = current_dir / 'original_scripts'
sys.path.insert(0, str(original_scripts_dir))

try:
    from model import C3D
except ImportError:
    st.error("Could not import model! Check 'original_scripts' folder.")
    st.stop()

# 2. CONFIGURATION
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = current_dir / 'checkpoints' / 'best_model.pth'
FRAMES_PER_CLIP = 16
IMG_SIZE = 112

# 3. UI SETUP
st.set_page_config(page_title="DAiSEE Live Monitor", page_icon="📹", layout="wide")
st.title("📹 DAiSEE Live Emotion Monitor")

if torch.cuda.is_available():
    st.sidebar.success(f"Running on RTX 4060 (CUDA)")
else:
    st.sidebar.warning("Running on CPU within slower inference.")

# 4. LOAD MODEL
@st.cache_resource
def load_model():
    model = C3D(num_classes=16) # 4 emotions * 4 intensities
    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        st.error(f"No checkpoint found at {MODEL_PATH}")
        st.stop()
    
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Load Focus Detector
from focus_utils import FocusDetector

@st.cache_resource
def load_detector():
    return FocusDetector()

detector = load_detector()

# 5. STREAMING LOGIC
col1, col2 = st.columns([2, 1])

with col1:
    st_frame = st.empty()

with col2:
    st.subheader("Real-Time Analysis")
    
    # Placeholders for dynamic UI
    warning_container = st.empty()
    engagement_container = st.empty()

    status_container = st.empty()

start_btn = st.button("Start Live Feed")
stop_btn = st.button("Stop")

if start_btn:
    cap = cv2.VideoCapture(0) # Open webcam
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.stop()

    # Frame Buffer
    buffer = deque(maxlen=FRAMES_PER_CLIP)
    
    stop_streaming = False
    
    # Initial vars
    b_val, e_val, c_val, f_val = 0.0, 0.0, 0.0, 0.0

    while True:
        # Note: Clicking 'Stop' triggers a Rerun, which cleanly stops this loop.

            
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break
            
        # UI Display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # --- FOCUS DETECTION ---
        is_focused, reason = detector.process(frame)
        
        # Preprocessing for Model
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        buffer.append(resized)
        
        # INFERENCE (When buffer is full)
        if len(buffer) == FRAMES_PER_CLIP:
            # (T, H, W, C) -> (C, T, H, W)
            input_clip = np.array(buffer)
            input_clip = np.transpose(input_clip, (3, 0, 1, 2))
            
            # Normalize (approx)
            input_clip = (input_clip - 110.0) / 58.0 
            
            input_tensor = torch.from_numpy(input_clip).float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_tensor) # (1, 16)
                logits = logits.view(4, 4)   # (4 emotions, 4 classes)
                
                # Get Probabilities (Softmax over 4 intensity classes)
                probs = torch.softmax(logits, dim=1) # (4, 4)
                
                # We want the probability of "High Intensity" (Class 2 or 3)
                # Or just weighted average? Let's assume Class 2+3 = Presence
                presence_probs = probs[:, 2:].sum(dim=1) # Sum of prob(2) + prob(3)
                
                # Update Values
                b_val = float(presence_probs[0])
                e_val = float(presence_probs[1])
                c_val = float(presence_probs[2])
                f_val = float(presence_probs[3])

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
                

            
            # "FOCUSED" Logic Logic (High Engagement > 50% and Low Boredom)
            if e_val > 0.5 and b_val < 0.5:
                status_container.metric("State", "✅ FOCUSED")
            elif c_val > 0.6:
                status_container.metric("State", "❓ CONFUSED")
            elif f_val > 0.6:
                status_container.metric("State", "😤 FRUSTRATED")
            elif b_val > 0.6:
                status_container.metric("State", "😴 BORED")
            else:
                status_container.metric("State", "😐 NEUTRAL")

        
        time.sleep(0.05) # Cap FPS slightly

    cap.release()
