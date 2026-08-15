"""
Shared C3D emotion/engagement inference used by both the Streamlit demo
(app.py) and the Flask backend (../app.py's /api/focus/analyze endpoint).
Preprocessing here matches training (original_scripts/preprocess.py):
faces are cropped before resizing, and pixels are normalized to [0, 1].

Concurrency model (multi-user Flask backend):
  - The C3D model (get_shared_model() below) is a big (~107MB) read-only
    eval-mode module. It is loaded ONCE per process and shared by every
    session -- forward passes only read parameters/running-stats and
    produce fresh local output tensors, so they don't need a private copy
    per user (see the thread-safety note above _forward_lock).
  - EmotionDetector is the lightweight PER-SESSION half: its rolling frame
    buffer, its running scores dict, and (crucially) its own MediaPipe
    FaceDetection instance are per-instance state. MediaPipe Solution
    objects are NOT thread-safe/stateful, so each concurrent user/session
    must get its own EmotionDetector instance -- never share one across
    users the way the old single global instance did.
"""
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

_current_dir = Path(__file__).resolve().parent
_original_scripts_dir = _current_dir / 'original_scripts'
sys.path.insert(0, str(_original_scripts_dir))
from model import C3D  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = _current_dir / 'checkpoints' / 'best_model.pth'
FRAMES_PER_CLIP = 16
IMG_SIZE = 112
FACE_PAD = 20


# ------------------------- shared, read-only C3D model -------------------------
_shared_model = None
_shared_model_init_lock = threading.Lock()

# Guards concurrent forward() calls through the single shared model instance.
#
# Reasoning on whether this is actually needed (see original_scripts/model.py):
# C3D is a plain feed-forward stack of Conv3d/BatchNorm3d/ReLU/MaxPool3d blocks
# followed by a Linear head -- no RNN-style hidden state threaded between
# calls, no shared mutable buffers written during forward. In eval() mode:
#   - BatchNorm3d reads running_mean/running_var but does NOT update them
#     (updates only happen when the module is in .train() mode), so
#     concurrent reads of those buffers are safe.
#   - Dropout is a no-op in eval mode (identity), so there's no RNG state
#     being mutated either.
#   - Every call builds its own fresh activation tensors from its own input;
#     the `inplace=True` ReLUs mutate only those per-call local tensors, not
#     anything shared across threads.
#   - We always call this under torch.no_grad(), so no autograd graph
#     (which _would_ be per-call state anyway) is being built or shared.
# This means concurrent CPU forward passes through the same eval-mode model
# instance are, by design, safe -- this is the same pattern used by most
# multi-threaded model-serving setups (e.g. TorchServe workers sharing one
# loaded model). We still wrap the call in a lock below anyway: the forward
# pass here only happens once every 16 frames per session (not per-frame),
# so serializing it costs almost nothing, and it removes any residual
# uncertainty (e.g. contention/edge-cases in PyTorch's internal intra-op
# thread pool under concurrent invocation) in exchange for that negligible
# cost. Correctness > the small perf hit of occasionally queuing a forward
# pass behind another session's.
_forward_lock = threading.Lock()


def get_shared_model():
    """
    Lazily loads and caches ONE C3D model instance for the whole process.
    The checkpoint is ~107MB; instantiating it per-session would be wasteful
    and unnecessary -- eval-mode, no-grad forward passes don't mutate any
    model state, only the per-call INPUT tensor differs between sessions.
    """
    global _shared_model
    if _shared_model is None:
        with _shared_model_init_lock:
            if _shared_model is None:  # double-checked locking
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"No checkpoint found at {MODEL_PATH}")
                model = C3D(num_classes=16)  # 4 emotions * 4 intensity classes
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                _shared_model = model
    return _shared_model


def crop_face(frame, face_detector, img_size=IMG_SIZE):
    """
    Args:
        face_detector: caller-owned mp.solutions.face_detection.FaceDetection
            instance. Passed in (rather than a module-level singleton) so
            each session/thread uses its own -- MediaPipe Solution objects
            are not thread-safe.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:
        box = results.detections[0].location_data.relative_bounding_box
        x, y = int(box.xmin * w), int(box.ymin * h)
        fw, fh = int(box.width * w), int(box.height * h)
        x1, y1 = max(0, x - FACE_PAD), max(0, y - FACE_PAD)
        x2, y2 = min(w, x + fw + FACE_PAD), min(h, y + fh + FACE_PAD)
        cropped = frame[y1:y2, x1:x2]
    else:
        crop_h = crop_w = min(h, w, img_size)
        cropped = frame[h // 2 - crop_h // 2: h // 2 + crop_h // 2, w // 2 - crop_w // 2: w // 2 + crop_w // 2]

    try:
        return cv2.resize(cropped, (img_size, img_size))
    except Exception:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)


class EmotionDetector:
    """
    Lightweight, PER-SESSION state: buffers incoming frames and runs the
    (shared, process-wide) C3D model once FRAMES_PER_CLIP frames have
    accumulated. Meant for polling-style callers (one frame per call, e.g.
    an HTTP request every couple seconds) as well as tight loops (e.g. a
    live webcam read loop) -- either way it just keeps a rolling window of
    the most recent frames.

    Cheap to construct: it does NOT load the 107MB C3D checkpoint (that's
    the module-level get_shared_model() singleton, loaded once per process
    and reused read-only by every session's forward passes). What IS
    instance-local here -- and must be, one per concurrent user/session --
    is the frame buffer, the running scores, and this detector's own
    MediaPipe FaceDetection instance (stateful / not thread-safe, so it
    can never be shared across concurrent sessions).
    """

    def __init__(self):
        # Cheap: this only allocates a small MediaPipe graph, not the
        # 107MB C3D checkpoint (which lives in the shared get_shared_model()
        # singleton instead). Safe/fast to create one of these per session.
        self._face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        self.buffer = deque(maxlen=FRAMES_PER_CLIP)
        self.scores = {"boredom": 0.0, "engagement": 0.0, "confusion": 0.0, "frustration": 0.0}

    def process(self, frame):
        """
        Args:
            frame: a single BGR frame (as read by cv2.VideoCapture / cv2.imdecode)
        Returns:
            dict with the rolling buffer fill state, latest scores, and a
            derived state label. Scores hold their last computed value until
            the buffer fills for the first time.
        """
        face_crop = crop_face(frame, self._face_detector)
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        self.buffer.append(rgb)

        if len(self.buffer) == FRAMES_PER_CLIP:
            clip = np.array(self.buffer)
            clip = np.transpose(clip, (3, 0, 1, 2))  # (T,H,W,C) -> (C,T,H,W)
            clip = clip.astype(np.float32) / 255.0

            tensor = torch.from_numpy(clip).float().unsqueeze(0).to(DEVICE)

            model = get_shared_model()
            # See _forward_lock's comment above for why this lock exists
            # despite eval-mode forward passes being safe to run
            # concurrently in principle.
            with _forward_lock:
                with torch.no_grad():
                    logits = model(tensor).view(4, 4)  # (4 emotions, 4 intensity classes)
                    probs = torch.softmax(logits, dim=1)
                    presence = probs[:, 2:].sum(dim=1)  # prob(high intensity) per emotion

            self.scores = {
                "boredom": float(presence[0]),
                "engagement": float(presence[1]),
                "confusion": float(presence[2]),
                "frustration": float(presence[3]),
            }

        b, e, c, f = (self.scores[k] for k in ("boredom", "engagement", "confusion", "frustration"))
        if e > 0.5 and b < 0.5:
            state = "FOCUSED"
        elif c > 0.6:
            state = "CONFUSED"
        elif f > 0.6:
            state = "FRUSTRATED"
        elif b > 0.6:
            state = "BORED"
        else:
            state = "NEUTRAL"

        return {
            "ready": len(self.buffer) == FRAMES_PER_CLIP,
            "buffer_fill": len(self.buffer),
            "buffer_needed": FRAMES_PER_CLIP,
            "scores": self.scores,
            "state": state,
        }
