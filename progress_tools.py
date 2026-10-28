"""Progress store: SQLite persistence + weak-topic logic.

Every public function returns a JSON string so the same functions can be
exposed unchanged as MCP tools or called directly.
"""
import datetime as dt
import json
import os
import sqlite3


def _db_path() -> str:
    return os.environ.get("PROGRESS_DB", "data/progress.db")


def _conn() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
            attempt_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            topic       TEXT NOT NULL,
            question    TEXT NOT NULL DEFAULT '',
            user_answer TEXT NOT NULL DEFAULT '',
            score       REAL NOT NULL,
            max_score   REAL NOT NULL DEFAULT 10,
            feedback    TEXT NOT NULL DEFAULT '',
            source      TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            mode        TEXT NOT NULL DEFAULT '',
            difficulty  TEXT NOT NULL DEFAULT ''
        )"""
    )
    have = {r["name"] for r in conn.execute("PRAGMA table_info(attempts)")}
    for col in ("mode", "difficulty"):  # databases created before these columns existed
        if col not in have:
            conn.execute(f"ALTER TABLE attempts ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
    return conn


def log_answer(
    topic: str,
    score: float,
    max_score: float = 10,
    question: str = "",
    user_answer: str = "",
    feedback: str = "",
    source: str = "",
    mode: str = "",
    difficulty: str = "",
) -> str:
    """Record a completed question attempt and its score.

    Only topic and score are required; the other fields are optional
    metadata (question, answer, feedback, source reference, practice mode,
    difficulty).
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic must not be empty")
    if max_score <= 0 or not (0 <= score <= max_score):
        raise ValueError("score must be between 0 and max_score")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO attempts (topic, question, user_answer, score, max_score,"
            " feedback, source, created_at, mode, difficulty) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (topic, question, user_answer, score, max_score, feedback, source, now, mode, difficulty),
        )
        return json.dumps({"attempt_id": cur.lastrowid})


def log_answers(attempts: list[dict]) -> str:
    """Record several attempts at once (used by MCQ sets). Each item takes the same fields as
    log_answer: topic and score are required, the rest optional. All-or-nothing."""
    ids = []
    for a in attempts:
        if not (a.get("topic") or "").strip():
            raise ValueError("topic must not be empty")
        ms = a.get("max_score", 10)
        if ms <= 0 or not (0 <= a.get("score", -1) <= ms):
            raise ValueError("score must be between 0 and max_score")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with _conn() as c:
        for a in attempts:
            cur = c.execute(
                "INSERT INTO attempts (topic, question, user_answer, score, max_score,"
                " feedback, source, created_at, mode, difficulty) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (a["topic"].strip(), a.get("question", ""), a.get("user_answer", ""), a["score"],
                 a.get("max_score", 10), a.get("feedback", ""), a.get("source", ""), now,
                 a.get("mode", ""), a.get("difficulty", "")),
            )
            ids.append(cur.lastrowid)
    return json.dumps({"attempt_ids": ids})


_STATS_SQL = """
SELECT topic,
       COUNT(*)                         AS attempt_count,
       ROUND(AVG(score * 10.0 / max_score), 2) AS average_score,
       MAX(created_at)                  AS last_practiced_at
FROM attempts GROUP BY topic
"""


def get_topic_stats() -> str:
    """Per-topic attempt count and average score (normalised to /10)."""
    with _conn() as c:
        rows = c.execute(_STATS_SQL + " ORDER BY average_score ASC").fetchall()
    return json.dumps([dict(r) for r in rows])


def get_weak_topics(threshold: float = 7.0, min_attempts: int = 1, limit: int = 5) -> str:
    """Topics whose average score (/10) is below threshold, weakest first."""
    sql = (
        f"SELECT * FROM ({_STATS_SQL}) WHERE attempt_count >= ? AND average_score < ? "
        "ORDER BY average_score ASC, attempt_count DESC LIMIT ?"
    )
    with _conn() as c:
        rows = c.execute(sql, (min_attempts, threshold, limit)).fetchall()
    return json.dumps([dict(r) for r in rows])


def get_history(limit: int = 50, topic: str = "") -> str:
    """Most recent attempts, newest first, optionally for a single topic."""
    sql = "SELECT * FROM attempts"
    args: list = []
    if topic:
        sql += " WHERE topic = ?"
        args.append(topic)
    sql += " ORDER BY attempt_id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        rows = c.execute(sql, args).fetchall()
    return json.dumps([dict(r) for r in rows])
