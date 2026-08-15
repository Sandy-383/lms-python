FROM python:3.12-slim

# --- System dependencies -------------------------------------------------
# opencv-contrib-python (a non-headless build) and mediapipe both link
# against real GUI/graphics shared libraries at import time even though
# this container never opens a window:
#   libgl1          - libGL.so.1, referenced by OpenCV's highgui module
#   libglib2.0-0    - libglib/libgobject, referenced by OpenCV and mediapipe
#   libsm6          - libSM.so.6 (X11 session mgmt), linked by OpenCV's GUI code
#   libxext6        - libXext.so.6 (X11 extensions), same reason
#   libxrender1     - libXrender.so.1, same reason
#   libgomp1        - OpenMP runtime; mediapipe's C++ inference graph and
#                     TFLite's XNNPACK delegate use it for threading
# Without these, `import cv2` / `import mediapipe` fails with
# "ImportError: libGL.so.1: cannot open shared object file" (or similar)
# on a bare python:3.12-slim (Debian) image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---------------------------------------------------
COPY requirements.txt .

# GPU build: this host has an NVIDIA GPU (see docker-compose.yml's "deploy"
# GPU reservation on the app/ollama services), so install torch's default
# Linux wheel, which bundles CUDA support (cuda-toolkit, nvidia-cudnn,
# nvidia-nccl, triton, etc. as declared deps -- several GB, but that's the
# cost of GPU acceleration). deep_learning/emotion_model.py already does
# `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, so it
# picks up the GPU automatically once it's visible in the container.
#
# Deploying to a GPU-less host (e.g. most cloud VPSes) instead? Swap this
# back to the CPU-only wheel to avoid pulling several unusable GB:
#   pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.13.0 \
    && pip install --no-cache-dir -r requirements.txt

# --- App code ---------------------------------------------------------------
# Large binary assets (public/static/videos/, embeddings.joblib,
# deep_learning/checkpoints/best_model.pth) are excluded via .dockerignore
# and bind-mounted at runtime instead (see docker-compose.yml) so rebuilding
# the image after a code change doesn't re-copy/re-hash ~600MB of media
# and model weights every time.
COPY . .

EXPOSE 5000

# 1 gunicorn worker with 4 threads: gunicorn's default "sync" worker is a
# full separate PROCESS, not a thread, so each additional worker
# independently imports app.py -- which would (a) race on the
# db.create_all()/seed_all() calls that run at import time (risking an
# IntegrityError on first boot as two processes both try to insert the
# same seed rows), (b) fragment the module-level `_focus_sessions` dict
# (see app.py) across separate processes' memory instead of one coherent
# per-user map, making the per-user threading.Lock()s meaningless since
# each process would have its own disjoint copy, and (c) independently
# construct its own EmotionDetector, doubling the ~107MB C3D checkpoint
# (plus mediapipe graph instances) in RAM for no benefit. Using a single
# worker means app.py's module-level state (the C3D model singleton
# behind deep_learning/emotion_model.py's `_forward_lock`, and the
# `_focus_sessions` dict) is truly process-wide/coherent, create_all()/
# seed_all() runs exactly once, and the model loads exactly once. Threads
# (not additional workers) then provide the request concurrency this app
# actually needs, since Flask/gunicorn threads share one process's memory
# -- which is exactly what the per-user locks and the shared model's
# `_forward_lock` were designed to serialize safely. (Do NOT switch to
# `--workers 2 --preload` instead: --preload only fixes the seed race by
# loading app.py once before forking, but each forked worker still gets
# its own copy of `_focus_sessions` post-fork, so focus-buffer
# fragmentation across processes remains, and preforking + SQLAlchemy
# engines is its own can of worms around sharing DB connections across
# forked processes.) Raise --threads (not --workers) if more concurrency
# is needed later, after checking `docker stats` against available RAM.
# --timeout 120: the /chat endpoint's call_generate() gives Ollama up to
# 60s to respond (see app.py); gunicorn's default 30s worker timeout would
# SIGKILL a worker mid-generation, so it's raised above that ceiling.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
