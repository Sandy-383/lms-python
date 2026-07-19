# app.py
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# ----------------- Chatbot integration imports -----------------
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# requests is already used by other code; if not, ensure it's imported
import requests
import os
# ================== DEEP LEARNING (FOCUS) ==================
import base64
import cv2
from deep_learning.focus_utils import FocusDetector
# ===========================================================



# import existing fake DB objects + new schedule functions
from fake_db import (
    users,
    courses,
    announcements,
    find_user_by_email,
    create_user,
    record_login,
    mark_module_complete,
    enroll_if_needed,
    get_user_courses_overview,
    estimate_weekly_hours,
    login_activity_by_user,
    todos_by_user,
    # schedule functions added in fake_db.py:
    get_tasks_for_user,
    create_task_for_user,
    update_task_for_user,
    delete_task_for_user,
    complete_task_for_user,
    snooze_task_for_user,
    schedule_stats_for_user,
)

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)  # allow frontend JS to call the API
focus_detector = FocusDetector()


# ----------------------------------------------------------------
# ----------------- Chatbot configuration (EDIT these) -----------------
EMBEDDINGS_FILE = "embeddings.joblib"        # <-- path to your embeddings.joblib (put file in project root)
OLLAMA_BASE = "http://localhost:11434/api"   # <-- Ollama API base URL
EMBED_MODEL = "bge-m3"                       # <-- embedding model name
GEN_MODEL = "llama3.2"                       # <-- generation model name
CHAT_TOP_K = 6                               # <-- how many transcript chunks to pass
VIDEO_URL_BASE = ""                          # <-- optional base url to your hosted videos (e.g. "/static/videos/")
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

    user = create_user(name=name, email=email, password=password, role=role)
    return jsonify(
        {
            "message": "Signup successful",
            "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]},
        }
    )


@app.post("/api/auth/login")
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    user = find_user_by_email(email) if email else None
    if not user or user["password"] != password:
        return jsonify({"message": "Invalid credentials"}), 401

    record_login(user["id"])
    return jsonify(
        {
            "message": "Login successful",
            "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]},
        }
    )


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
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return jsonify({"message": "User not found"}), 404

    courses_overview = get_user_courses_overview(user_id)
    weekly_hours = estimate_weekly_hours(user_id)
    activity = login_activity_by_user.get(user_id, [])

    status = (
        "You are on track. Keep it up!"
        if weekly_hours >= 2
        else "You are slightly behind. Try to study a bit more."
    )

    todos = todos_by_user.get(
        user_id,
        [
            "Complete Module 2 of Python Basics",
            "Review Data Science Fundamentals",
            "Plan 30 minutes of study for tomorrow",
        ],
    )

    # simple recommendation: first two courses
    recommended = [
        {
            "id": c["id"],
            "title": c["title"],
            "category": c["category"],
            "level": c["level"],
        }
        for c in courses[:2]
    ]

    return jsonify(
        {
            "user": {
                "id": user["id"],
                "name": user["name"],
                "role": user["role"],
                "interests": user.get("interests", []),
            },
            "coursesOverview": courses_overview,
            "summary": {
                "weeklyHours": weekly_hours,
                "activity": activity,
                "status": status,
            },
            "announcements": announcements,
            "recommended": recommended,
            "todos": todos,
        }
    )


# ---------- COURSES ROUTES ----------
@app.get("/api/courses")
def get_courses():
    # light subset for list view
    result = []
    for c in courses:
        result.append(
            {
                "id": c["id"],
                "title": c["title"],
                "category": c["category"],
                "level": c["level"],
                "isPaid": c["isPaid"],
                "description": c["description"],
            }
        )
    return jsonify(result)


@app.get("/api/courses/<int:course_id>")
def get_course_detail(course_id: int):
    course = next((c for c in courses if c["id"] == course_id), None)
    if not course:
        return jsonify({"message": "Course not found"}), 404
    return jsonify(course)


@app.post("/api/courses/<int:course_id>/complete")
def complete_module(course_id: int):
    data = request.get_json() or {}
    user_id = data.get("userId")
    module_id = data.get("moduleId")

    if not user_id or not module_id:
        return jsonify({"message": "userId and moduleId required"}), 400

    course = next((c for c in courses if c["id"] == course_id), None)
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
    data = request.get_json() or {}
    todos = data.get("todos", [])
    todos_by_user[user_id] = todos
    return jsonify({"message": "Todos updated"})


# ---------- SCHEDULE / TASKS ROUTES ----------
# Get tasks for next N days (default 7)
@app.get("/api/schedule/<int:user_id>")
def api_get_schedule(user_id: int):
    days = request.args.get("days", default=7, type=int)
    tasks = get_tasks_for_user(user_id=user_id, days=days)
    return jsonify({"tasks": tasks})


# Create a task for user
@app.post("/api/schedule/<int:user_id>")
def api_create_task(user_id: int):
    payload = request.get_json() or {}
    try:
        task = create_task_for_user(user_id=user_id, payload=payload)
    except ValueError as e:
        return jsonify({"message": str(e)}), 400
    return jsonify({"message": "Task created", "task": task})


# Update task
@app.put("/api/schedule/<int:user_id>/<int:task_id>")
def api_update_task(user_id: int, task_id: int):
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
    ok = delete_task_for_user(user_id=user_id, task_id=task_id)
    if not ok:
        return jsonify({"message": "Task not found"}), 404
    return jsonify({"message": "Task deleted"})


# Mark task complete (handles recurrence)
@app.post("/api/schedule/<int:user_id>/<int:task_id>/complete")
def api_complete_task(user_id: int, task_id: int):
    result = complete_task_for_user(user_id=user_id, task_id=task_id)
    if result is None:
        return jsonify({"message": "Task not found"}), 404
    return jsonify({"message": "Task completed", "task": result})


# Snooze task: body { minutes: 15 }
@app.post("/api/schedule/<int:user_id>/<int:task_id>/snooze")
def api_snooze_task(user_id: int, task_id: int):
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
    stats = schedule_stats_for_user(user_id)
    return jsonify({"stats": stats})

# ================== FOCUS DETECTION API ==================
@app.post("/api/focus/analyze")
def analyze_focus():
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

        is_focused, reason = focus_detector.process(frame)
        return jsonify({"focused": is_focused, "reason": reason})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# =========================================================

# ---------- Run ----------
if __name__ == "__main__":
    # debug=True is convenient during development; remove or change for production
    app.run(debug=True)
