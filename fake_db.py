# fake_db.py
"""
In-memory demo dataset for Learnova app.py

Contains:
- users, courses, announcements (existing)
- simple login activity and todos
- TASKS system (in-memory) with helper functions:
  get_tasks_for_user, create_task_for_user, update_task_for_user,
  delete_task_for_user, complete_task_for_user, snooze_task_for_user,
  schedule_stats_for_user

Times returned are UTC ISO strings with trailing 'Z' (e.g. 2025-12-10T13:00:00Z).
This is a simple demo store for the college mini-project.
"""

from __future__ import annotations
import os
import datetime
from typing import List, Dict, Any, Optional, Tuple

# -------------------------
# Users (demo)
# -------------------------
users: List[Dict[str, Any]] = [
    {"id": 1, "name": "Student Demo", "email": "student@example.com", "password": "password", "role": "student", "interests": ["Web Dev", "Video"]},
    {"id": 2, "name": "Admin Demo", "email": "admin@example.com", "password": "admin", "role": "admin", "interests": ["Platform"]},
]
_next_user_id = 3

def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    email = (email or "").lower()
    for u in users:
        if u.get("email", "").lower() == email:
            return u
    return None

def create_user(name: str, email: str, password: str, role: str = "student") -> Dict[str, Any]:
    global _next_user_id
    new = {"id": _next_user_id, "name": name, "email": email, "password": password, "role": role, "interests": []}
    users.append(new)
    login_activity_by_user[new["id"]] = []
    todos_by_user[new["id"]] = []
    tasks_by_user.setdefault(new["id"], [])
    _next_user_id += 1
    return new

# -------------------------
# Login activity (heatmap)
# -------------------------
login_activity_by_user: Dict[int, List[str]] = {1: [], 2: []}

def record_login(user_id: int) -> None:
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    login_activity_by_user.setdefault(int(user_id), []).append(ts)

# -------------------------
# Todos (per user)
# -------------------------
todos_by_user: Dict[int, List[str]] = {
    1: [
        "Complete Module 1 of Sample Video 1",
        "Rewatch key concepts",
        "Plan 30 minutes tomorrow"
    ],
    2: ["Review platform analytics", "Plan next course"],
}

# -------------------------
# Announcements
# -------------------------
announcements: List[Dict[str, str]] = [
    {"message": "3 course videos are available in your Courses (demo).", "date": datetime.date.today().isoformat()},
]

# -------------------------
# Courses (video examples)
# -------------------------
video_filenames_provided = [
    "SEO_and_Core_Web_Vitals_in_HTML.webm",
    "Video_Audio_Media_in_HTML.webm",
    "Your_First_HTML_Website.webm",
]

def _videos_dir_candidates() -> List[str]:
    base = os.path.dirname(__file__)
    return [
        os.path.join(base, "public", "static", "videos"),
        os.path.join(os.getcwd(), "public", "static", "videos"),
        os.path.join("C:\\", "Users", "saiku", "Desktop", "lms-python", "public", "static", "videos"),
    ]

def _resolve_actual_names(provided_list: List[str]) -> List[str]:
    resolved = []
    candidates = _videos_dir_candidates()
    for name in provided_list:
        found = False
        for d in candidates:
            try:
                if os.path.isdir(d) and name in os.listdir(d):
                    resolved.append(name)
                    found = True
                    break
            except Exception:
                continue
        if not found:
            resolved.append(name)
    return resolved

resolved_video_filenames = _resolve_actual_names(video_filenames_provided)

courses: List[Dict[str, Any]] = []
_next_course_id = 1
for fname in resolved_video_filenames:
    basename = os.path.basename(fname)
    video_url = f"/static/videos/{basename}"
    courses.append({
        "id": _next_course_id,
        "title": f"Video Course {_next_course_id} — {os.path.splitext(basename)[0]}",
        "description": f"Imported video file: {basename}",
        "category": "Web Development / Video",
        "level": "Beginner",
        "isPaid": False,
        "estimatedHours": 0.5,
        "media_type": "video",
        "video_url": video_url,
        "modules": [
            {"id": _next_course_id * 100 + 1, "order": 1, "title": "Watch the video"},
            {"id": _next_course_id * 100 + 2, "order": 2, "title": "Quick quiz / notes"},
        ],
    })
    _next_course_id += 1

while len([c for c in courses if c.get("media_type") == "video"]) < 3:
    pid = _next_course_id
    placeholder_name = f"placeholder_{pid}.mp4"
    courses.append({
        "id": pid,
        "title": f"Sample Video Course {pid} (placeholder)",
        "description": "Placeholder course — place a real video file in public/static/videos/ with this name to enable playback.",
        "category": "Web Development / Video",
        "level": "Beginner",
        "isPaid": False,
        "estimatedHours": 0.5,
        "media_type": "video",
        "video_url": f"/static/videos/{placeholder_name}",
        "modules": [
            {"id": pid * 100 + 1, "order": 1, "title": "Watch the video (placeholder)"},
            {"id": pid * 100 + 2, "order": 2, "title": "Quick reflection / notes"},
        ],
    })
    _next_course_id += 1

courses.append({
    "id": _next_course_id,
    "title": "Python Basics for AI",
    "description": "Refresh core Python concepts frequently used in AI and data projects.",
    "category": "Programming",
    "level": "Beginner",
    "isPaid": False,
    "estimatedHours": 4,
    "media_type": "text",
    "video_url": None,
    "modules": [
        {"id": _next_course_id * 100 + 1, "order": 1, "title": "Variables & Data Types"},
        {"id": _next_course_id * 100 + 2, "order": 2, "title": "Control Flow"},
        {"id": _next_course_id * 100 + 3, "order": 3, "title": "Functions & Modules"},
    ],
})

_courses_by_id = {c["id"]: c for c in courses}

# -------------------------
# Progress tracking helpers
# -------------------------
_user_progress: Dict[tuple, set] = {}

def enroll_if_needed(user_id: int, course_id: int) -> None:
    return

def mark_module_complete(user_id: int, course_id: int, module_id: int) -> None:
    key = (int(user_id), int(course_id))
    _user_progress.setdefault(key, set()).add(int(module_id))

def get_user_courses_overview(user_id: int) -> List[Dict[str, Any]]:
    overview = []
    for c in courses:
        key = (int(user_id), int(c["id"]))
        completed = _user_progress.get(key, set())
        total = len(c.get("modules", [])) or 1
        pct = round(len(completed) * 100.0 / total)
        overview.append({
            "id": c["id"],
            "title": c["title"],
            "description": c.get("description", ""),
            "progressPercent": pct,
            "lastViewedAt": None
        })
    return overview

def estimate_weekly_hours(user_id: int) -> float:
    total_completed = 0
    for (uid, cid), mods in _user_progress.items():
        if int(uid) == int(user_id):
            total_completed += len(mods)
    return round(total_completed * 0.5, 1)

# -------------------------
# TASKS / SCHEDULE (new)
# -------------------------
tasks_by_user: Dict[int, List[Dict[str, Any]]] = {
    1: [
        {
            "id": 1001,
            "userId": 1,
            "title": "Complete Module 2 — RAG Intro",
            "notes": "Finish the video and quick quiz",
            "dueAt": (datetime.datetime.utcnow() + datetime.timedelta(minutes=15)).replace(microsecond=0).isoformat() + "Z",
            "durationMinutes": 30,
            "priority": "high",
            "status": "todo",
            "recurrence": None,
            "createdAt": datetime.datetime.utcnow().isoformat() + "Z",
            "updatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        },
        {
            "id": 1002,
            "userId": 1,
            "title": "Revise notes — Python basics",
            "notes": "",
            "dueAt": (datetime.datetime.utcnow() + datetime.timedelta(hours=6)).replace(microsecond=0).isoformat() + "Z",
            "durationMinutes": 20,
            "priority": "medium",
            "status": "todo",
            "recurrence": None,
            "createdAt": datetime.datetime.utcnow().isoformat() + "Z",
            "updatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        },
    ]
}
_next_task_id = 2000

# -------------------------
# Aliases exported for app.py
# -------------------------
login_activity_by_user = login_activity_by_user
todos_by_user = todos_by_user

# -------------------------
# TASKS / SCHEDULE helpers
# -------------------------

def get_tasks_for_user(user_id: int, days: int = 7) -> List[Dict[str, Any]]:
    now = datetime.datetime.utcnow()
    end = now + datetime.timedelta(days=days)
    tasks = tasks_by_user.get(int(user_id), [])
    result = []
    for t in tasks:
        try:
            due = datetime.datetime.fromisoformat(t["dueAt"].replace("Z", ""))
        except Exception:
            continue
        if due <= end:
            result.append(t)
    result.sort(key=lambda x: x.get("dueAt") or "")
    return result


def _find_task(user_id: int, task_id: int):
    for idx, t in enumerate(tasks_by_user.get(int(user_id), [])):
        if int(t.get("id")) == int(task_id):
            return idx, t
    return None, None


def create_task_for_user(user_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    global _next_task_id
    title = (payload.get("title") or "").strip()
    if not title:
        raise ValueError("title is required")

    due_at = payload.get("dueAt")
    if due_at:
        due_dt = datetime.datetime.fromisoformat(due_at.replace("Z", ""))
    else:
        due_dt = datetime.datetime.utcnow() + datetime.timedelta(hours=1)

    task = {
        "id": _next_task_id,
        "userId": int(user_id),
        "title": title,
        "notes": payload.get("notes", ""),
        "dueAt": due_dt.replace(microsecond=0).isoformat() + "Z",
        "durationMinutes": int(payload.get("durationMinutes") or 0),
        "priority": payload.get("priority", "medium"),
        "status": payload.get("status", "todo"),
        "recurrence": payload.get("recurrence"),
        "createdAt": datetime.datetime.utcnow().isoformat() + "Z",
        "updatedAt": datetime.datetime.utcnow().isoformat() + "Z",
    }

    tasks_by_user.setdefault(int(user_id), []).append(task)
    _next_task_id += 1
    return task


def update_task_for_user(user_id: int, task_id: int, payload: Dict[str, Any]):
    idx, task = _find_task(user_id, task_id)
    if task is None:
        return None

    for key in ["title", "notes", "priority", "status", "recurrence"]:
        if key in payload:
            task[key] = payload[key]

    if "dueAt" in payload:
        task["dueAt"] = payload["dueAt"]

    if "durationMinutes" in payload:
        task["durationMinutes"] = int(payload["durationMinutes"] or 0)

    task["updatedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
    tasks_by_user[int(user_id)][idx] = task
    return task


def delete_task_for_user(user_id: int, task_id: int) -> bool:
    idx, task = _find_task(user_id, task_id)
    if task is None:
        return False
    tasks_by_user[int(user_id)].pop(idx)
    return True


def complete_task_for_user(user_id: int, task_id: int):
    idx, task = _find_task(user_id, task_id)
    if task is None:
        return None
    task["status"] = "done"
    task["updatedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
    return task


def snooze_task_for_user(user_id: int, task_id: int, minutes: int):
    idx, task = _find_task(user_id, task_id)
    if task is None:
        return None
    due = datetime.datetime.fromisoformat(task["dueAt"].replace("Z", ""))
    new_due = due + datetime.timedelta(minutes=int(minutes))
    task["dueAt"] = new_due.replace(microsecond=0).isoformat() + "Z"
    task["status"] = "snoozed"
    task["updatedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
    return task


def schedule_stats_for_user(user_id: int) -> Dict[str, Any]:
    tasks = tasks_by_user.get(int(user_id), [])
    pending = sum(1 for t in tasks if t.get("status") != "done")
    completed = sum(1 for t in tasks if t.get("status") == "done")
    return {
        "pending": pending,
        "completed": completed,
        "total": len(tasks),
    }
