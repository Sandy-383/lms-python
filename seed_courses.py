# seed_courses.py
"""
One-time, idempotent database seeding for Learnova.

Ports the exact seed content fake_db.py used to build at import time into
real rows via the models in models.py:

- 2 demo users (student@example.com / admin@example.com), with passwords
  hashed via werkzeug.security.generate_password_hash -- using the SAME
  original demo passwords ("password" and "admin") so demo login still
  works once the next agent wires app.py's auth against password_hash.
- Read-only course/module reference content: 3 video courses + 1 text
  course (see _course_seed_data() below for exactly how this content was
  derived from fake_db.py).
- The single demo announcement.

Each of the three sections is independently idempotent (it checks whether
its rows already exist before inserting anything), so this is safe to call
on every app startup.

Intended usage (from app.py, after db.init_app(app) and db.create_all()):

    from models import db
    from seed_courses import seed_all

    app = Flask(__name__)
    ...
    db.init_app(app)
    with app.app_context():
        db.create_all()
        seed_all()
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

from werkzeug.security import generate_password_hash

from models import db, User, Course, Module, Announcement

# ---------------------------------------------------------------------------
# Demo users (verbatim from fake_db.py's `users` list, minus plaintext
# passwords -- those are hashed below instead of stored as-is).
# ---------------------------------------------------------------------------
_DEMO_USERS: List[Dict[str, Any]] = [
    {
        "name": "Student Demo",
        "email": "student@example.com",
        "password": "password",  # hashed before storing -- never persisted as plaintext
        "role": "student",
        "interests": ["Web Dev", "Video"],
    },
    {
        "name": "Admin Demo",
        "email": "admin@example.com",
        "password": "admin",  # hashed before storing -- never persisted as plaintext
        "role": "admin",
        "interests": ["Platform"],
    },
]


def _seed_users() -> None:
    for u in _DEMO_USERS:
        if User.query.filter_by(email=u["email"]).first() is not None:
            continue  # already seeded
        db.session.add(User(
            name=u["name"],
            email=u["email"],
            password_hash=generate_password_hash(u["password"]),
            role=u["role"],
            interests=u["interests"],
        ))
    try:
        db.session.commit()
    except Exception:
        # Defense-in-depth: if this ever runs concurrently (e.g. worker/run
        # count changes later), a racing duplicate insert must not crash
        # the caller -- mirrors app.py's mark_module_complete().
        db.session.rollback()


# ---------------------------------------------------------------------------
# Courses (read-only reference content).
#
# fake_db.py builds these dynamically at import time by scanning
# public/static/videos/ for the filenames in `video_filenames_provided`,
# via _videos_dir_candidates() / _resolve_actual_names(). Crucially,
# _resolve_actual_names() appends the *original provided filename*
# whether or not it actually finds that file on disk -- both its "found"
# and "not found" branches do `resolved.append(name)`. That means the
# resulting course content NEVER actually varies with what's present on
# disk: resolved_video_filenames == video_filenames_provided, always, in
# that order. Because of that, the 3-video-course loop below always runs
# to completion and fake_db.py's trailing placeholder-padding `while`
# loop (top up to 3 video courses) never fires in practice.
#
# So the following is not a re-implementation of the scanning logic --
# it is the fixed, deterministic *output* of that logic, reproduced
# verbatim (same ids, titles, descriptions, categories, module ids/titles)
# as a static seed, since courses are read-only reference content here,
# not something to recompute from the filesystem on every startup.
# ---------------------------------------------------------------------------

_VIDEO_FILENAMES = [
    "SEO_and_Core_Web_Vitals_in_HTML.webm",
    "Video_Audio_Media_in_HTML.webm",
    "Your_First_HTML_Website.webm",
]


def _course_seed_data() -> List[Dict[str, Any]]:
    courses: List[Dict[str, Any]] = []
    next_course_id = 1

    for fname in _VIDEO_FILENAMES:
        basename = fname
        stem = basename.rsplit(".", 1)[0]
        courses.append({
            "id": next_course_id,
            "title": f"Video Course {next_course_id} — {stem}",
            "description": f"Imported video file: {basename}",
            "category": "Web Development / Video",
            "level": "Beginner",
            "is_paid": False,
            "estimated_hours": 0.5,
            "media_type": "video",
            "video_url": f"/static/videos/{basename}",
            "modules": [
                {"id": next_course_id * 100 + 1, "order_index": 1, "title": "Watch the video"},
                {"id": next_course_id * 100 + 2, "order_index": 2, "title": "Quick quiz / notes"},
            ],
        })
        next_course_id += 1

    # 4th course: the one text course (fake_db.py's "Python Basics for AI").
    courses.append({
        "id": next_course_id,
        "title": "Python Basics for AI",
        "description": "Refresh core Python concepts frequently used in AI and data projects.",
        "category": "Programming",
        "level": "Beginner",
        "is_paid": False,
        "estimated_hours": 4,
        "media_type": "text",
        "video_url": None,
        "modules": [
            {"id": next_course_id * 100 + 1, "order_index": 1, "title": "Variables & Data Types"},
            {"id": next_course_id * 100 + 2, "order_index": 2, "title": "Control Flow"},
            {"id": next_course_id * 100 + 3, "order_index": 3, "title": "Functions & Modules"},
        ],
    })

    return courses


def _seed_courses() -> None:
    if Course.query.first() is not None:
        return  # already seeded

    for c in _course_seed_data():
        course = Course(
            id=c["id"],
            title=c["title"],
            description=c["description"],
            category=c["category"],
            level=c["level"],
            is_paid=c["is_paid"],
            estimated_hours=c["estimated_hours"],
            media_type=c["media_type"],
            video_url=c["video_url"],
        )
        for m in c["modules"]:
            course.modules.append(Module(
                id=m["id"],
                order_index=m["order_index"],
                title=m["title"],
            ))
        db.session.add(course)

    try:
        db.session.commit()
    except Exception:
        # Defense-in-depth: if this ever runs concurrently (e.g. worker/run
        # count changes later), a racing duplicate insert must not crash
        # the caller -- mirrors app.py's mark_module_complete().
        db.session.rollback()


# ---------------------------------------------------------------------------
# Announcements (verbatim from fake_db.py's `announcements` list).
# ---------------------------------------------------------------------------

def _seed_announcements() -> None:
    if Announcement.query.first() is not None:
        return  # already seeded

    db.session.add(Announcement(
        message="3 course videos are available in your Courses (demo).",
        date=datetime.date.today(),
    ))
    try:
        db.session.commit()
    except Exception:
        # Defense-in-depth: if this ever runs concurrently (e.g. worker/run
        # count changes later), a racing duplicate insert must not crash
        # the caller -- mirrors app.py's mark_module_complete().
        db.session.rollback()


def seed_all() -> None:
    """
    Populate demo users, reference courses/modules, and the demo
    announcement -- but only the pieces that are missing. Safe to call
    unconditionally on every app startup.
    """
    _seed_users()
    _seed_courses()
    _seed_announcements()


if __name__ == "__main__":
    print(
        "seed_courses.py defines seed_all(), meant to be called from "
        "app.py inside a Flask app context after db.init_app(app) and "
        "db.create_all() have run (see this module's docstring). It is "
        "not meant to be executed standalone."
    )
