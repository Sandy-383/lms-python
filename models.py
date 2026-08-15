# models.py
"""
Flask-SQLAlchemy models for Learnova, replacing the in-memory fake_db.py
module-level globals with real (PostgreSQL-backed) tables.

This module does NOT bind the SQLAlchemy instance to a Flask app itself --
`db` is created here unbound, on purpose, so that app.py can do:

    from models import db, User, LoginActivity, Todo, Announcement, Course, Module, ModuleCompletion, Task

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql+psycopg2://..."
    db.init_app(app)

    with app.app_context():
        db.create_all()

        from seed_courses import seed_all
        seed_all()

Design notes / decisions made while porting fake_db.py's shapes to tables
(documented here since field names matter to whoever wires app.py against
these models next):

- User.password_hash replaces fake_db.py's plaintext "password" field.
  Hashing/verifying is the caller's (app.py's) responsibility -- this
  module only defines the column. Demo passwords are hashed by
  seed_courses.py using werkzeug.security.generate_password_hash.
- User.interests is stored as JSON (a JSON array of strings), matching
  fake_db.py's `interests: List[str]` shape 1:1 -- e.g. ["Web Dev", "Video"].
  It is NOT a comma-separated string.
- LoginActivity.timestamp is a real DateTime (UTC, naive) column rather
  than fake_db.py's pre-formatted ISO string. To reproduce fake_db.py's
  "...Z" formatted strings, format with:
      login.timestamp.isoformat(timespec="seconds") + "Z"
- Todo no longer stores just a bare string list; `text` holds the todo's
  text and `position` holds its 0-based order within the user's list, so
  that "replace whole list" (POST /api/todos/<user_id>) can be implemented
  as: delete all Todo rows for that user_id, then bulk-insert the new
  list in order (position = list index).
- Announcement.date is a real Date column (fake_db.py stored
  date.today().isoformat() as a string); format with `.isoformat()` when
  serializing to JSON.
- Course/Module are read-only reference content seeded once by
  seed_courses.py (see that module for the exact content, ported
  verbatim from fake_db.py's course-building logic). There are
  intentionally no create/update/delete routes implied by this schema --
  same as fake_db.py.
- Module.order_index corresponds to fake_db.py's module "order" key.
  It is named order_index (not `order`) purely because ORDER is a SQL
  reserved word -- same field, renamed to avoid quoting headaches.
- Course.is_paid / Course.estimated_hours correspond to fake_db.py's
  "isPaid" / "estimatedHours" keys (snake_case column, same meaning).
- ModuleCompletion is the durable form of fake_db.py's
  `_user_progress: Dict[(user_id, course_id), Set[module_id]]`: one row
  per (user, course, module) the user has completed, with a
  uniqueness constraint standing in for the old Python `set()`
  deduplication (marking the same module complete twice is a no-op).
- Task mirrors fake_db.py's tasks_by_user task dicts field-for-field,
  including the camelCase attribute names (userId, dueAt,
  durationMinutes, createdAt, updatedAt) fake_db.py / the existing
  JSON API already use, to minimize remapping work in app.py. dueAt,
  createdAt and updatedAt are stored as strings (not native DateTime),
  matching fake_db.py's existing convention of pre-formatting these as
  ISO-8601 UTC strings with a trailing "Z" (e.g.
  "2025-12-10T13:00:00Z") at write time, and parsing them back with
  `datetime.fromisoformat(s.replace("Z", ""))` at read time. recurrence
  is stored as nullable JSON and is never read or acted upon anywhere in
  this layer (pre-existing gap in the product, out of scope here) --
  it is simply persisted as given.
"""

from __future__ import annotations

import datetime

from flask_sqlalchemy import SQLAlchemy

# Module-level, unbound SQLAlchemy instance. app.py is responsible for
# calling db.init_app(app) (and db.create_all() inside an app context).
db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    # Replaces fake_db.py's plaintext "password" column. Populate via
    # werkzeug.security.generate_password_hash(...); verify via
    # werkzeug.security.check_password_hash(...). Hashing is the next
    # agent's job -- this column just needs to exist.
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="student")
    # JSON array of strings, e.g. ["Web Dev", "Video"]. See module
    # docstring for why JSON (not a comma-separated string) was chosen.
    interests = db.Column(db.JSON, nullable=False, default=list)

    login_activity = db.relationship(
        "LoginActivity", backref="user", cascade="all, delete-orphan",
        order_by="LoginActivity.timestamp",
    )
    todos = db.relationship(
        "Todo", backref="user", cascade="all, delete-orphan",
        order_by="Todo.position",
    )
    tasks = db.relationship(
        "Task", backref="user", cascade="all, delete-orphan",
        foreign_keys="Task.userId",
    )
    completions = db.relationship(
        "ModuleCompletion", backref="user", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"<User id={self.id} email={self.email!r} role={self.role!r}>"


class LoginActivity(db.Model):
    """One row per recorded login, replacing login_activity_by_user[user_id] (List[str] of ISO timestamps)."""

    __tablename__ = "login_activity"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)


class Todo(db.Model):
    """Replaces todos_by_user[user_id] (List[str]). `position` preserves list order for whole-list replace."""

    __tablename__ = "todos"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    text = db.Column(db.Text, nullable=False)
    position = db.Column(db.Integer, nullable=False, default=0)


class Announcement(db.Model):
    """Read-only content, no CRUD routes -- just seeded once. Replaces the `announcements` list."""

    __tablename__ = "announcements"

    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.date.today)


class Course(db.Model):
    """Read-only reference content (NOT user-generated). Seeded once by seed_courses.py."""

    __tablename__ = "courses"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False, default="")
    category = db.Column(db.String(255), nullable=False, default="")
    level = db.Column(db.String(50), nullable=False, default="Beginner")
    # Corresponds to fake_db.py's "isPaid" key.
    is_paid = db.Column(db.Boolean, nullable=False, default=False)
    # Corresponds to fake_db.py's "estimatedHours" key.
    estimated_hours = db.Column(db.Float, nullable=False, default=0.0)
    # "video" or "text".
    media_type = db.Column(db.String(20), nullable=False, default="text")
    video_url = db.Column(db.String(500), nullable=True)

    modules = db.relationship(
        "Module", backref="course", cascade="all, delete-orphan",
        order_by="Module.order_index",
    )

    def to_dict(self) -> dict:
        """Serialize back to the same shape fake_db.py's course dicts used."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "level": self.level,
            "isPaid": self.is_paid,
            "estimatedHours": self.estimated_hours,
            "media_type": self.media_type,
            "video_url": self.video_url,
            "modules": [m.to_dict() for m in self.modules],
        }


class Module(db.Model):
    """A single module/step within a Course. Read-only reference content."""

    __tablename__ = "modules"

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    # Corresponds to fake_db.py's module "order" key; renamed to avoid the
    # SQL reserved word ORDER.
    order_index = db.Column(db.Integer, nullable=False, default=0)
    title = db.Column(db.String(255), nullable=False)

    def to_dict(self) -> dict:
        return {"id": self.id, "order": self.order_index, "title": self.title}


class ModuleCompletion(db.Model):
    """
    Durable replacement for fake_db.py's
    `_user_progress: Dict[(user_id, course_id), Set[module_id]]`.

    One row per module a user has completed. The unique constraint plays
    the role the old Python `set()` played: marking the same module
    complete twice does not create a duplicate row.
    """

    __tablename__ = "module_completions"
    __table_args__ = (
        db.UniqueConstraint("user_id", "course_id", "module_id", name="uq_module_completion"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    module_id = db.Column(db.Integer, db.ForeignKey("modules.id", ondelete="CASCADE"), nullable=False, index=True)
    completed_at = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)

    course = db.relationship("Course")
    module = db.relationship("Module")


class Task(db.Model):
    """
    Replaces tasks_by_user[user_id] (List[Dict]). Field names intentionally
    mirror fake_db.py's task dicts (camelCase) -- see module docstring.
    """

    __tablename__ = "tasks"

    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False)
    notes = db.Column(db.Text, nullable=False, default="")
    # Stored as an ISO-8601 UTC string with trailing "Z", e.g.
    # "2025-12-10T13:00:00Z" -- matching fake_db.py's existing convention
    # (pre-formatted at write time, parsed with
    # datetime.fromisoformat(s.replace("Z", "")) at read time).
    dueAt = db.Column(db.String(40), nullable=False)
    durationMinutes = db.Column(db.Integer, nullable=False, default=0)
    priority = db.Column(db.String(20), nullable=False, default="medium")
    # 'todo' | 'done' | 'snoozed'
    status = db.Column(db.String(20), nullable=False, default="todo")
    # Nullable JSON payload, persisted as-is; never read/acted upon by
    # this layer (pre-existing gap in the product -- out of scope here).
    recurrence = db.Column(db.JSON, nullable=True)
    createdAt = db.Column(db.String(40), nullable=False)
    updatedAt = db.Column(db.String(40), nullable=False)

    def to_dict(self) -> dict:
        """Serialize back to the same shape fake_db.py's task dicts used."""
        return {
            "id": self.id,
            "userId": self.userId,
            "title": self.title,
            "notes": self.notes,
            "dueAt": self.dueAt,
            "durationMinutes": self.durationMinutes,
            "priority": self.priority,
            "status": self.status,
            "recurrence": self.recurrence,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
        }
