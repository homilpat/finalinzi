import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime


DB_PATH = os.environ.get(
    "MOCA_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "moca_demo.sqlite3"),
)


EDUCATION_LEVELS = {
    "none": {"label": "무학", "years": 0},
    "elementary": {"label": "초등학교 졸업", "years": 6},
    "middle": {"label": "중학교 졸업", "years": 9},
    "high": {"label": "고등학교 졸업", "years": 12},
    "college": {"label": "전문대 졸업", "years": 14},
    "university": {"label": "대학교 졸업", "years": 16},
    "graduate": {"label": "대학원 이상", "years": 18},
}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def normalize_phone(phone):
    return re.sub(r"\D+", "", phone or "")


def phone_hash(phone):
    normalized = normalize_phone(phone)
    salt = os.environ.get("PHONE_HASH_SALT", "moca-demo-phone-salt")
    return hashlib.sha256(f"{salt}:{normalized}".encode("utf-8")).hexdigest()


def phone_last4(phone):
    normalized = normalize_phone(phone)
    return normalized[-4:] if len(normalized) >= 4 else normalized


def education_years_from_level(level):
    return EDUCATION_LEVELS.get(level, EDUCATION_LEVELS["high"])["years"]


def education_label(level):
    return EDUCATION_LEVELS.get(level, EDUCATION_LEVELS["high"])["label"]


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone_hash TEXT NOT NULL UNIQUE,
                phone_last4 TEXT NOT NULL,
                education_level TEXT NOT NULL,
                education_years INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL UNIQUE,
                member_id INTEGER NOT NULL,
                version TEXT NOT NULL,
                location TEXT,
                sigungu TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                raw_json TEXT,
                score_json TEXT,
                raw_score INTEGER,
                final_score INTEGER,
                education_correction INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(member_id) REFERENCES members(id)
            );

            CREATE INDEX IF NOT EXISTS idx_assessments_member_id
                ON assessments(member_id);

            CREATE TABLE IF NOT EXISTS physical_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER NOT NULL,
                assessment_id INTEGER,
                gait_type TEXT,
                gait_level TEXT,
                gait_score INTEGER,
                cognitive_score INTEGER,
                walking_speed REAL,
                step_count INTEGER,
                measured_at TEXT NOT NULL,
                raw_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(member_id, measured_at),
                FOREIGN KEY(member_id) REFERENCES members(id),
                FOREIGN KEY(assessment_id) REFERENCES assessments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_physical_results_member_id
                ON physical_results(member_id, measured_at);

            CREATE TABLE IF NOT EXISTS exercise_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER NOT NULL,
                assessment_id INTEGER,
                exercise_name TEXT NOT NULL,
                exercise_type TEXT,
                duration_min INTEGER NOT NULL DEFAULT 0,
                completed_date TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                raw_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(member_id, completed_date, exercise_name),
                FOREIGN KEY(member_id) REFERENCES members(id),
                FOREIGN KEY(assessment_id) REFERENCES assessments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_exercise_records_member_id
                ON exercise_records(member_id, completed_at);
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(members)").fetchall()
        }
        if "member_code" not in columns:
            conn.execute("ALTER TABLE members ADD COLUMN member_code TEXT")
            rows = conn.execute("SELECT id FROM members ORDER BY id").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE members SET member_code = ? WHERE id = ?",
                    (_format_member_code(row["id"]), row["id"]),
                )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_members_member_code ON members(member_code)"
        )


def _format_member_code(number):
    return f"ID_A{int(number):04d}"


def _next_member_code(conn):
    row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM members").fetchone()
    return _format_member_code(row["next_id"])


def get_or_create_member(phone, education_level):
    normalized = normalize_phone(phone)
    if len(normalized) < 9:
        raise ValueError("전화번호를 다시 확인해 주세요.")

    level = education_level if education_level in EDUCATION_LEVELS else "high"
    years = education_years_from_level(level)
    phash = phone_hash(normalized)
    last4 = phone_last4(normalized)
    stamp = now_iso()

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, member_code FROM members WHERE phone_hash = ?",
            (phash,),
        ).fetchone()

        if row:
            member_code = row["member_code"] or _format_member_code(row["id"])
            conn.execute(
                """
                UPDATE members
                   SET name = ?, member_code = ?, phone_last4 = ?, education_level = ?,
                       education_years = ?, updated_at = ?
                 WHERE id = ?
                """,
                (member_code, member_code, last4, level, years, stamp, row["id"]),
            )
            return row["id"], years, member_code, False

        member_code = _next_member_code(conn)
        cur = conn.execute(
            """
            INSERT INTO members (
                name, member_code, phone_hash, phone_last4, education_level,
                education_years, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (member_code, member_code, phash, last4, level, years, stamp, stamp),
        )
        return cur.lastrowid, years, member_code, True


def create_assessment(uid, member_id, version, location, sigungu):
    stamp = now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO assessments (
                uid, member_id, version, location, sigungu,
                started_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (uid, member_id, version, location, sigungu, stamp, stamp, stamp),
        )
        return cur.lastrowid


def complete_assessment(assessment_id, raw, score):
    stamp = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE assessments
               SET completed_at = ?, raw_json = ?, score_json = ?,
                   raw_score = ?, final_score = ?, education_correction = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (
                stamp,
                json.dumps(raw, ensure_ascii=False, default=str),
                json.dumps(score, ensure_ascii=False, default=str),
                int(score.get("raw_score", 0)),
                int(score.get("final_score", 0)),
                int(score.get("education_correction", 0)),
                stamp,
                assessment_id,
            ),
        )


def get_member(member_id):
    if not member_id:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM members WHERE id = ?",
            (member_id,),
        ).fetchone()
        return dict(row) if row else None


def get_latest_assessment(member_id):
    if not member_id:
        return None
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM assessments
             WHERE member_id = ? AND completed_at IS NOT NULL
             ORDER BY completed_at DESC, id DESC
             LIMIT 1
            """,
            (member_id,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        for key in ("raw_json", "score_json"):
            try:
                item[key] = json.loads(item[key]) if item.get(key) else {}
            except json.JSONDecodeError:
                item[key] = {}
        return item


def save_physical_result(member_id, assessment_id, result):
    if not member_id:
        return None
    payload = result or {}
    measured_at = payload.get("measuredAt") or payload.get("measured_at") or now_iso()
    stamp = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO physical_results (
                member_id, assessment_id, gait_type, gait_level, gait_score,
                cognitive_score, walking_speed, step_count, measured_at,
                raw_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(member_id, measured_at) DO UPDATE SET
                assessment_id = excluded.assessment_id,
                gait_type = excluded.gait_type,
                gait_level = excluded.gait_level,
                gait_score = excluded.gait_score,
                cognitive_score = excluded.cognitive_score,
                walking_speed = excluded.walking_speed,
                step_count = excluded.step_count,
                raw_json = excluded.raw_json,
                updated_at = excluded.updated_at
            """,
            (
                member_id,
                assessment_id,
                payload.get("gaitType") or payload.get("gait_type"),
                payload.get("gaitLevel") or payload.get("gait_level"),
                int(payload.get("gaitScore") or payload.get("gait_score") or 0),
                int(payload.get("cognitiveScore") or payload.get("cognitive_score") or 0),
                payload.get("walkingSpeed") or payload.get("walking_speed"),
                payload.get("stepCount") or payload.get("step_count"),
                measured_at,
                json.dumps(payload, ensure_ascii=False, default=str),
                stamp,
                stamp,
            ),
        )
    return True


def get_latest_physical_result(member_id):
    if not member_id:
        return None
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM physical_results
             WHERE member_id = ?
             ORDER BY measured_at DESC, id DESC
             LIMIT 1
            """,
            (member_id,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["raw_json"] = json.loads(item["raw_json"]) if item.get("raw_json") else {}
        except json.JSONDecodeError:
            item["raw_json"] = {}
        return item


def save_exercise_record(member_id, assessment_id, exercise):
    if not member_id:
        return None
    payload = exercise or {}
    completed_at = payload.get("completed_at") or now_iso()
    completed_date = completed_at[:10]
    stamp = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO exercise_records (
                member_id, assessment_id, exercise_name, exercise_type,
                duration_min, completed_date, completed_at, raw_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(member_id, completed_date, exercise_name) DO UPDATE SET
                assessment_id = excluded.assessment_id,
                exercise_type = excluded.exercise_type,
                duration_min = excluded.duration_min,
                completed_at = excluded.completed_at,
                raw_json = excluded.raw_json,
                updated_at = excluded.updated_at
            """,
            (
                member_id,
                assessment_id,
                payload.get("exercise_name") or "exercise",
                payload.get("type") or payload.get("exercise_type"),
                int(payload.get("duration_min") or 0),
                completed_date,
                completed_at,
                json.dumps(payload, ensure_ascii=False, default=str),
                stamp,
                stamp,
            ),
        )
    return True


def get_exercise_summary(member_id):
    if not member_id:
        return {"present_days": 0, "streak_days": 0, "total_minutes": 0, "latest": None}
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT completed_date, duration_min, completed_at, exercise_name, exercise_type
              FROM exercise_records
             WHERE member_id = ?
             ORDER BY completed_date DESC, completed_at DESC
            """,
            (member_id,),
        ).fetchall()
    if not rows:
        return {"present_days": 0, "streak_days": 0, "total_minutes": 0, "latest": None}

    dates = []
    seen = set()
    total = 0
    latest = dict(rows[0])
    for row in rows:
        total += int(row["duration_min"] or 0)
        completed_date = row["completed_date"]
        if completed_date not in seen:
            seen.add(completed_date)
            dates.append(completed_date)

    streak = 0
    today = datetime.now().date()
    date_set = set(dates)
    cursor = today
    while cursor.isoformat() in date_set:
        streak += 1
        cursor = cursor.fromordinal(cursor.toordinal() - 1)

    return {
        "present_days": len(date_set),
        "streak_days": streak,
        "total_minutes": total,
        "latest": latest,
    }


def get_recent_assessment_summaries(limit=5):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                a.id,
                a.uid,
                a.version,
                a.started_at,
                a.completed_at,
                a.raw_score,
                a.final_score,
                a.score_json,
                m.member_code,
                m.phone_last4,
                m.education_level
            FROM assessments a
            JOIN members m ON m.id = a.member_id
            ORDER BY COALESCE(a.completed_at, a.started_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    summaries = []
    for row in rows:
        score = {}
        if row["score_json"]:
            try:
                score = json.loads(row["score_json"])
            except json.JSONDecodeError:
                score = {}
        summaries.append({
            "id": row["id"],
            "uid": row["uid"],
            "version": row["version"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "raw_score": row["raw_score"],
            "final_score": row["final_score"],
            "score": score,
            "member_code": row["member_code"],
            "phone_last4": row["phone_last4"],
            "education_level": row["education_level"],
            "is_completed": row["completed_at"] is not None,
        })
    return summaries
