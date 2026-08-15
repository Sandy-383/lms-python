# app.py
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
# ----------------- Chatbot integration imports -----------------
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# requests is already used by other code; if not, ensure it's imported
import requests
import os
import datetime
# ================== DEEP LEARNING (FOCUS) ==================
import base64
import cv2
import threading
import time
from deep_learning.focus_utils import FocusDetector
from deep_learning.emotion_model import EmotionDetector
# ===========================================================

from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

# Load a .env file (if present) into os.environ before SECRET_KEY /
# DATABASE_URL are read below.
load_dotenv()

# Real (SQL) database models -- replaces the old in-memory fake_db.py.
from models import db, User, LoginActivity, Todo, Announcement, Course, Module, ModuleCompletion, Task
from seed_courses import seed_all

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)  # allow frontend JS to call the API

# ----------------- Per-session focus/emotion detector state -----------------
# Each concurrent user gets their OWN FocusDetector + EmotionDetector, keyed
# by session["user_id"]. This used to be a single pair of GLOBAL instances
# guarded by one big lock (focus_lock) -- that lock stopped the MediaPipe
# thread-safety crash (Solution objects like FaceMesh/FaceDetection are not
# thread-safe) but did NOT stop different users' frames from all landing in
# the SAME rolling buffer, silently mixing their data together. Now that
# real per-user sessions exist, each session gets its own buffer + its own
# MediaPipe instances, so two students using focus-detection at the same
# time no longer contaminate each other's state. The (heavy, ~107MB) C3D
# model itself is still a single process-wide shared instance -- see
# deep_learning/emotion_model.py's get_shared_model() -- since eval-mode
# forward passes on it are read-only and don't need per-user copies.
_focus_sessions = {}  # user_id -> {"focus": FocusDetector, "emotion": EmotionDetector, "lock": Lock, "last_used": float}
_focus_sessions_lock = threading.Lock()  # guards _focus_sessions dict bookkeeping only (creation/eviction) -- NOT held during per-frame processing
_FOCUS_SESSION_TTL_SECONDS = 10 * 60  # evict a session's detector state after ~10 minutes of inactivity


def _get_focus_session(user_id):
    """
    Returns (creating if necessary) the calling user's own private
    FocusDetector/EmotionDetector pair + a lock scoped to just that user's
    state. Also opportunistically evicts any *other* sessions that have
    been idle 10+ minutes -- a lazy check on access is enough at this
    project's scale, no background reaper thread needed.
    """
    now = time.time()
    with _focus_sessions_lock:
        stale_ids = [
            uid for uid, st in _focus_sessions.items()
            if uid != user_id and (now - st["last_used"]) > _FOCUS_SESSION_TTL_SECONDS
        ]
        for uid in stale_ids:
            del _focus_sessions[uid]

        state = _focus_sessions.get(user_id)
        if state is None:
            state = {
                "focus": FocusDetector(),
                "emotion": EmotionDetector(),
                "lock": threading.Lock(),
                "last_used": now,
            }
            _focus_sessions[user_id] = state
        else:
            state["last_used"] = now
    return state


# ----------------- Security / database configuration -----------------
# SECRET_KEY signs the session cookie used for auth (see require_auth()
# below). The fallback string is ONLY for local development convenience --
# PRODUCTION DEPLOYMENTS MUST SET A REAL SECRET_KEY ENV VAR (a long random
# value), otherwise session cookies can be forged by anyone who reads this
# source file.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-key-CHANGE-ME-in-production")

# Sessions persist for a week so a logged-in user (whose browser keeps
# `lmsUser` in localStorage indefinitely) doesn't get silently logged out
# on every browser restart.
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=7)

# Session cookie hardening. Caddy (the reverse proxy in front of this app,
# see Caddyfile) always terminates HTTPS, so SESSION_COOKIE_SECURE=True is
# safe here -- but note it means the cookie will NOT be sent over plain
# HTTP: if you're testing this app directly (outside Docker/Caddy) against
# http://localhost:5000 without HTTPS, you'll need to temporarily set this
# back to False, or the session cookie won't round-trip.
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# DATABASE_URL is read from the environment so production can point at a
# real Postgres instance; falls back to a local SQLite file so a developer
# without Postgres running can still boot the app.
_database_url = os.environ.get("DATABASE_URL", "sqlite:///local.db")
# Some hosting providers hand out "postgres://" URLs; SQLAlchemy's
# psycopg2 dialect requires the "postgresql://" scheme.
if _database_url.startswith("postgres://"):
    _database_url = _database_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = _database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
with app.app_context():
    db.create_all()
    seed_all()


# ----------------------------------------------------------------
# ----------------- Chatbot configuration -----------------
# All values are environment-driven so Docker Compose can point the app at
# the "ollama" service by name instead of localhost, with the original
# hardcoded values kept as local-dev defaults (matches the SECRET_KEY /
# DATABASE_URL fallback pattern above).
EMBEDDINGS_FILE = os.environ.get("EMBEDDINGS_FILE", "embeddings.joblib")        # <-- path to your embeddings.joblib (put file in project root)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api")   # <-- Ollama API base URL ("http://ollama:11434/api" in Compose)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")                          # <-- embedding model name
GEN_MODEL = os.environ.get("GEN_MODEL", "llama3.2")                            # <-- generation model name
CHAT_TOP_K = int(os.environ.get("CHAT_TOP_K", "6"))                            # <-- how many transcript chunks to pass
VIDEO_URL_BASE = os.environ.get("VIDEO_URL_BASE", "")                          # <-- optional base url to your hosted videos (e.g. "/static/videos/")
# ---------------------------------------------------------------------
# Load embeddings DB (single time)
if not os.path.exists(EMBEDDINGS_FILE):
    app.logger.warning(f"Embeddings file {EMBEDDINGS_FILE} not found — chatbot /search will fail until you add it.")
    embeddings_df = None
else:
    try:
        embeddings_df = joblib.load(EMBEDDINGS_FILE)
        # Ensure we have ndarray stacked if needed later
    except Exception as e:
        app.logger.exception("Failed to load embeddings.joblib")
        embeddings_df = None
def create_embedding(text_list):
    """Return list of embeddings for the provided list of strings."""
    r = requests.post(f"{OLLAMA_BASE}/embed", json={"model": EMBED_MODEL, "input": text_list}, timeout=30)
    r.raise_for_status()
    return r.json().get("embeddings", [])

def call_generate(prompt):
    """Call the generate endpoint and return extracted text (best-effort)."""
    r = requests.post(f"{OLLAMA_BASE}/generate", json={"model": GEN_MODEL, "prompt": prompt, "stream": False}, timeout=60)
    r.raise_for_status()
    resp = r.json()
    if isinstance(resp, dict):
        return resp.get("response") or resp.get("text") or resp.get("output") or str(resp)
    return str(resp)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Chat endpoint used by frontend chatbot UI.
    Expects: { "message": "..." }
    Returns: { "answer": "...", "sources": [...] }
    """
    # Every other data-bearing route requires a valid session (via
    # require_auth() or a direct session.get("user_id") check, e.g.
    # /api/focus/analyze above) -- this endpoint triggers embedding + LLM
    # generation calls and must not be reachable anonymously either.
    if session.get("user_id") is None:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    if embeddings_df is None:
        return jsonify({"error": "Embeddings database not loaded on server."}), 500

    # 1) create embedding for incoming user message
    try:
        q_emb = create_embedding([message])[0]
    except Exception as e:
        app.logger.exception("Embedding failed")
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500

    # 2) cosine similarity vs stored embeddings
    try:
        all_emb = np.vstack(embeddings_df['embedding'].values)
        sims = cosine_similarity(all_emb, [q_emb]).flatten()
    except Exception as e:
        app.logger.exception("Similarity computation failed")
        return jsonify({"error": f"Similarity computation failed: {str(e)}"}), 500

    # 3) choose top-k chunks
    top_idx = sims.argsort()[::-1][:CHAT_TOP_K]
    new_df = embeddings_df.iloc[top_idx].copy()
    new_df['score'] = sims[top_idx]

    # 4) build prompt for LLM (same as your original script)
    extracted = new_df[["title", "number", "start", "end", "text", "score"]].to_json(orient="records")
    prompt = f"""They are teaching web development in their web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{extracted}
---------------------------------
\"{message}\"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
"""

    try:
        answer = call_generate(prompt)
    except Exception as e:
        app.logger.exception("Generation failed")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # 5) prepare sources to return
    sources = []
    for _, row in new_df.iterrows():
        video_link = None
        if VIDEO_URL_BASE and row.get("number") is not None:
            # adjust formatting if your video URLs differ
            video_link = f"{VIDEO_URL_BASE}{row['number']}#t={int(row['start'])}"
        sources.append({
            "title": str(row.get("title", "")),
            "number": str(row.get("number", "")),
            "start": float(row.get("start", 0.0)),
            "end": float(row.get("end", 0.0)),
            "snippet": str(row.get("text", ""))[:350],
            "score": float(row.get("score", 0.0)),
            "video_link": video_link
        })

    return jsonify({"answer": answer, "sources": sources})

# ---------- Serve frontend ----------
@app.route("/")
def index():
    # Serve index.html from public/
    return send_from_directory(app.static_folder, "index.html")


# ---------- DB-BACKED BUSINESS LOGIC (replaces fake_db.py's in-memory helpers) ----------
def find_user_by_email(email):
    """Case-insensitive lookup, matching fake_db.py's original semantics."""
    if not email:
        return None
    return User.query.filter(db.func.lower(User.email) == email.strip().lower()).first()


def record_login(user_id: int) -> None:
    db.session.add(LoginActivity(user_id=user_id))
    db.session.commit()


def mark_module_complete(user_id: int, course_id: int, module_id: int) -> None:
    existing = ModuleCompletion.query.filter_by(
        user_id=user_id, course_id=course_id, module_id=module_id
    ).first()
    if existing:
        return  # already completed -- unique constraint stands in for the old set() dedup
    db.session.add(ModuleCompletion(user_id=user_id, course_id=course_id, module_id=module_id))
    try:
        db.session.commit()
    except Exception:
        # concurrent duplicate insert racing past the check above -- the
        # unique constraint already guarantees no duplicate row exists.
        db.session.rollback()


def get_user_courses_overview(user_id: int):
    overview = []
    for c in Course.query.order_by(Course.id).all():
        total = len(c.modules) or 1
        completed = ModuleCompletion.query.filter_by(user_id=user_id, course_id=c.id).count()
        pct = round(completed * 100.0 / total)
        overview.append({
            "id": c.id,
            "title": c.title,
            "description": c.description or "",
            "progressPercent": pct,
            "lastViewedAt": None,
        })
    return overview


def estimate_weekly_hours(user_id: int) -> float:
    total_completed = ModuleCompletion.query.filter_by(user_id=user_id).count()
    return round(total_completed * 0.5, 1)


_DEFAULT_TODOS = [
    "Complete Module 2 of Python Basics",
    "Review Data Science Fundamentals",
    "Plan 30 minutes of study for tomorrow",
]


def get_todos_for_user(user_id: int):
    rows = Todo.query.filter_by(user_id=user_id).order_by(Todo.position).all()
    if not rows:
        return list(_DEFAULT_TODOS)
    return [r.text for r in rows]


# -------------------------
# TASKS / SCHEDULE helpers (ported from fake_db.py -- same logic, SQLAlchemy-backed)
# -------------------------
def get_tasks_for_user(user_id: int, days: int = 7):
    now = datetime.datetime.utcnow()
    end = now + datetime.timedelta(days=days)
    result = []
    for t in Task.query.filter_by(userId=user_id).all():
        try:
            due = datetime.datetime.fromisoformat(t.dueAt.replace("Z", ""))
        except Exception:
            continue
        if due <= end:
            result.append(t.to_dict())
    result.sort(key=lambda x: x.get("dueAt") or "")
    return result


def create_task_for_user(user_id: int, payload: dict):
    title = (payload.get("title") or "").strip()
    if not title:
        raise ValueError("title is required")

    due_at = payload.get("dueAt")
    if due_at:
        due_dt = datetime.datetime.fromisoformat(due_at.replace("Z", ""))
    else:
        due_dt = datetime.datetime.utcnow() + datetime.timedelta(hours=1)

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    task = Task(
        userId=int(user_id),
        title=title,
        notes=payload.get("notes", ""),
        dueAt=due_dt.replace(microsecond=0).isoformat() + "Z",
        durationMinutes=int(payload.get("durationMinutes") or 0),
        priority=payload.get("priority", "medium"),
        status=payload.get("status", "todo"),
        recurrence=payload.get("recurrence"),
        createdAt=now_iso,
        updatedAt=now_iso,
    )
    db.session.add(task)
    db.session.commit()
    return task.to_dict()


def _find_task(user_id: int, task_id: int):
    return Task.query.filter_by(id=int(task_id), userId=int(user_id)).first()


def update_task_for_user(user_id: int, task_id: int, payload: dict):
    task = _find_task(user_id, task_id)
    if task is None:
        return None

    for key in ["title", "notes", "priority", "status", "recurrence"]:
        if key in payload:
            setattr(task, key, payload[key])

    if "dueAt" in payload:
        task.dueAt = payload["dueAt"]

    if "durationMinutes" in payload:
        task.durationMinutes = int(payload["durationMinutes"] or 0)

    task.updatedAt = datetime.datetime.utcnow().isoformat() + "Z"
    db.session.commit()
    return task.to_dict()


def delete_task_for_user(user_id: int, task_id: int) -> bool:
    task = _find_task(user_id, task_id)
    if task is None:
        return False
    db.session.delete(task)
    db.session.commit()
    return True


def complete_task_for_user(user_id: int, task_id: int):
    task = _find_task(user_id, task_id)
    if task is None:
        return None
    task.status = "done"
    task.updatedAt = datetime.datetime.utcnow().isoformat() + "Z"
    db.session.commit()
    return task.to_dict()


def snooze_task_for_user(user_id: int, task_id: int, minutes: int):
    task = _find_task(user_id, task_id)
    if task is None:
        return None
    due = datetime.datetime.fromisoformat(task.dueAt.replace("Z", ""))
    new_due = due + datetime.timedelta(minutes=int(minutes))
    task.dueAt = new_due.replace(microsecond=0).isoformat() + "Z"
    task.status = "snoozed"
    task.updatedAt = datetime.datetime.utcnow().isoformat() + "Z"
    db.session.commit()
    return task.to_dict()


def schedule_stats_for_user(user_id: int):
    tasks = Task.query.filter_by(userId=user_id).all()
    pending = sum(1 for t in tasks if t.status != "done")
    completed = sum(1 for t in tasks if t.status == "done")
    return {"pending": pending, "completed": completed, "total": len(tasks)}


# ---------- AUTH / SESSION HELPERS ----------
def require_auth(url_user_id):
    """
    Enforce that the caller's signed session cookie belongs to url_user_id.
    Returns None if the check passes, or a Flask (response, status) tuple
    to return immediately if it fails:
      - 401 if there's no authenticated session at all
      - 403 if the session belongs to a *different* user than the URL's
        user_id (this is what closes the IDOR hole -- previously every
        route here trusted user_id from the URL path with zero
        verification the caller actually IS that user).
    """
    session_user_id = session.get("user_id")
    if session_user_id is None:
        return jsonify({"message": "Authentication required"}), 401
    try:
        if int(session_user_id) != int(url_user_id):
            return jsonify({"message": "Forbidden"}), 403
    except (TypeError, ValueError):
        return jsonify({"message": "Forbidden"}), 403
    return None


# ---------- AUTH ROUTES ----------
@app.post("/api/auth/signup")
def signup():
    data = request.get_json() or {}
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "student")


    if not all([name, email, password]):
        return jsonify({"message": "All fields are required"}), 400

    existing = find_user_by_email(email)
    if existing:
        return jsonify({"message": "User already exists"}), 400

    user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
        role=role,
        interests=[],
    )
    db.session.add(user)
    db.session.commit()

    session.permanent = True
    session["user_id"] = user.id

    return jsonify(
        {
            "message": "Signup successful",
            "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role},
        }
    )


@app.post("/api/auth/login")
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    user = find_user_by_email(email) if email else None
    if not user or not check_password_hash(user.password_hash, password or ""):
        return jsonify({"message": "Invalid credentials"}), 401

    record_login(user.id)

    session.permanent = True
    session["user_id"] = user.id

    return jsonify(
        {
            "message": "Login successful",
            "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role},
        }
    )


@app.post("/api/auth/logout")
def logout():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out"})


@app.post("/api/auth/forgot")
def forgot_password():
    data = request.get_json() or {}
    email = data.get("email")
    # For demo: always respond success
    user = find_user_by_email(email) if email else None
    if not user:
        return jsonify(
            {"message": "If this email exists, a reset link was sent (simulated)."}
        )
    return jsonify(
        {
            "message": "Password reset link simulated. Show this message in a success UI."
        }
    )


# ---------- DASHBOARD ROUTE ----------
@app.get("/api/dashboard/<int:user_id>")
def get_dashboard(user_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error

    user = User.query.get(user_id)
    if not user:
        return jsonify({"message": "User not found"}), 404

    courses_overview = get_user_courses_overview(user_id)
    weekly_hours = estimate_weekly_hours(user_id)
    activity = [
        la.timestamp.isoformat(timespec="seconds") + "Z" for la in user.login_activity
    ]

    status = (
        "You are on track. Keep it up!"
        if weekly_hours >= 2
        else "You are slightly behind. Try to study a bit more."
    )

    todos = get_todos_for_user(user_id)

    # simple recommendation: first two courses
    recommended = [
        {
            "id": c.id,
            "title": c.title,
            "category": c.category,
            "level": c.level,
        }
        for c in Course.query.order_by(Course.id).limit(2).all()
    ]

    announcements_list = [
        {"message": a.message, "date": a.date.isoformat()}
        for a in Announcement.query.all()
    ]

    return jsonify(
        {
            "user": {
                "id": user.id,
                "name": user.name,
                "role": user.role,
                "interests": user.interests or [],
            },
            "coursesOverview": courses_overview,
            "summary": {
                "weeklyHours": weekly_hours,
                "activity": activity,
                "status": status,
            },
            "announcements": announcements_list,
            "recommended": recommended,
            "todos": todos,
        }
    )


# ---------- COURSES ROUTES ----------
@app.get("/api/courses")
def get_courses():
    # light subset for list view
    result = []
    for c in Course.query.order_by(Course.id).all():
        result.append(
            {
                "id": c.id,
                "title": c.title,
                "category": c.category,
                "level": c.level,
                "isPaid": c.is_paid,
                "description": c.description,
            }
        )
    return jsonify(result)


@app.get("/api/courses/<int:course_id>")
def get_course_detail(course_id: int):
    course = Course.query.get(course_id)
    if not course:
        return jsonify({"message": "Course not found"}), 404
    return jsonify(course.to_dict())


@app.post("/api/courses/<int:course_id>/complete")
def complete_module(course_id: int):
    data = request.get_json() or {}
    user_id = data.get("userId")
    module_id = data.get("moduleId")

    if not user_id or not module_id:
        return jsonify({"message": "userId and moduleId required"}), 400

    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error

    course = Course.query.get(course_id)
    if not course:
        return jsonify({"message": "Course not found"}), 404

    mark_module_complete(int(user_id), course_id, int(module_id))
    overview_list = get_user_courses_overview(int(user_id))
    course_overview = next((c for c in overview_list if c["id"] == course_id), None)

    return jsonify(
        {
            "message": "Module marked as complete",
            "courseProgress": course_overview,
        }
    )


# ---------- TODOS ROUTE ----------
@app.post("/api/todos/<int:user_id>")
def update_todos(user_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error

    data = request.get_json() or {}
    todos = data.get("todos", [])

    # Whole-list replace: delete this user's existing rows, then bulk-insert
    # the new list in order (position = list index) -- same semantics as
    # the old todos_by_user[user_id] = todos in-memory overwrite.
    Todo.query.filter_by(user_id=user_id).delete()
    for idx, text in enumerate(todos):
        db.session.add(Todo(user_id=user_id, text=text, position=idx))
    db.session.commit()

    return jsonify({"message": "Todos updated"})


# ---------- SCHEDULE / TASKS ROUTES ----------
# Get tasks for next N days (default 7)
@app.get("/api/schedule/<int:user_id>")
def api_get_schedule(user_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    days = request.args.get("days", default=7, type=int)
    tasks = get_tasks_for_user(user_id=user_id, days=days)
    return jsonify({"tasks": tasks})


# Create a task for user
@app.post("/api/schedule/<int:user_id>")
def api_create_task(user_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    payload = request.get_json() or {}
    try:
        task = create_task_for_user(user_id=user_id, payload=payload)
    except ValueError as e:
        return jsonify({"message": str(e)}), 400
    return jsonify({"message": "Task created", "task": task})


# Update task
@app.put("/api/schedule/<int:user_id>/<int:task_id>")
def api_update_task(user_id: int, task_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    payload = request.get_json() or {}
    try:
        task = update_task_for_user(user_id=user_id, task_id=task_id, payload=payload)
    except ValueError as e:
        return jsonify({"message": str(e)}), 400
    if task is None:
        return jsonify({"message": "Task not found"}), 404
    return jsonify({"message": "Task updated", "task": task})


# Delete task
@app.delete("/api/schedule/<int:user_id>/<int:task_id>")
def api_delete_task(user_id: int, task_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    ok = delete_task_for_user(user_id=user_id, task_id=task_id)
    if not ok:
        return jsonify({"message": "Task not found"}), 404
    return jsonify({"message": "Task deleted"})


# Mark task complete (handles recurrence)
@app.post("/api/schedule/<int:user_id>/<int:task_id>/complete")
def api_complete_task(user_id: int, task_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    result = complete_task_for_user(user_id=user_id, task_id=task_id)
    if result is None:
        return jsonify({"message": "Task not found"}), 404
    return jsonify({"message": "Task completed", "task": result})


# Snooze task: body { minutes: 15 }
@app.post("/api/schedule/<int:user_id>/<int:task_id>/snooze")
def api_snooze_task(user_id: int, task_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    data = request.get_json() or {}
    minutes = data.get("minutes")
    if minutes is None:
        return jsonify({"message": "minutes required"}), 400
    try:
        task = snooze_task_for_user(user_id=user_id, task_id=task_id, minutes=int(minutes))
    except ValueError as e:
        return jsonify({"message": str(e)}), 400
    if task is None:
        return jsonify({"message": "Task not found"}), 404
    return jsonify({"message": "Task snoozed", "task": task})


# Simple stats for schedule (counts)
@app.get("/api/schedule/<int:user_id>/stats")
def api_schedule_stats(user_id: int):
    auth_error = require_auth(user_id)
    if auth_error:
        return auth_error
    stats = schedule_stats_for_user(user_id)
    return jsonify({"stats": stats})

# ================== FOCUS DETECTION API ==================
@app.post("/api/focus/analyze")
def analyze_focus():
    # This route processes per-user webcam frames into per-user rolling
    # buffers, so it needs to know WHO is calling -- there's no user_id in
    # the URL (unlike the other user-scoped routes above), so we read it
    # straight from the signed session cookie the same way require_auth()
    # does, and simply require a session to exist (401 if not).
    user_id = session.get("user_id")
    if user_id is None:
        return jsonify({"message": "Authentication required"}), 401

    data = request.get_json() or {}
    image = data.get("image")

    if not image:
        return jsonify({"error": "No image provided"}), 400

    try:
        # image = "data:image/jpeg;base64,....."
        header, encoded = image.split(",", 1)

        img_bytes = base64.b64decode(encoded)
        frame = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        focus_session = _get_focus_session(user_id)
        # Lock only THIS user's own FocusDetector/EmotionDetector instances
        # (MediaPipe Solution objects are not thread-safe, so concurrent
        # calls on the SAME instance must still be serialized -- e.g. two
        # browser tabs from the same logged-in user). This no longer
        # serializes OTHER users' requests against each other the way the
        # old single global focus_lock did.
        with focus_session["lock"]:
            is_focused, reason = focus_session["focus"].process(frame)
            emotion_result = focus_session["emotion"].process(frame)

        return jsonify({
            "focused": is_focused,
            "reason": reason,
            "state": emotion_result["state"],
            "scores": emotion_result["scores"],
            "buffer_fill": emotion_result["buffer_fill"],
            "buffer_needed": emotion_result["buffer_needed"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# =========================================================

# ---------- Run ----------
if __name__ == "__main__":
    # debug=True is convenient during development; remove or change for production
    # threaded=True avoids requests queuing behind the focus/emotion-detection
    # endpoint, which now does real per-request ML inference.
    # use_reloader=False: the default reloader watches every imported package's
    # source files (including deep dependencies inside venv/), and a spurious
    # "file changed" restart mid-request was observed corrupting an in-flight
    # MediaPipe computation. Restart the server manually after editing code.
    app.run(debug=True, threaded=True, use_reloader=False)
