# database.py — SQLite persistence for prediction history

import sqlite3
import os
from datetime import datetime

# Path to the SQLite database file (project root)
DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')


def _get_connection():
    """Open and return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # allows column access by name
    return conn


def init_db():
    """
    Create the predictions table if it does not already exist.
    Called once at application startup.
    """
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                job_description   TEXT,
                predicted_title   TEXT,
                category          TEXT,
                confidence        REAL,
                confidence_label  TEXT,
                detection_stage   TEXT,
                technical_skills  TEXT,
                soft_skills       TEXT,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _confidence_label(confidence: float) -> str:
    """Map a numeric confidence value to a human-readable label."""
    if confidence >= 75:
        return 'High'
    elif confidence >= 45:
        return 'Medium'
    else:
        return 'Low'


def save_prediction(data: dict) -> None:
    """
    Persist one prediction result to the database.

    Expected keys in *data*:
        description      (str)  — raw job description text
        job_title        (str)  — predicted title
        job_category     (str)  — resolved category
        confidence       (float)— confidence percentage
        stage            (int)  — 1 or 2
        technical_skills (list) — list of skill strings
        soft_skills      (list) — list of skill strings
    """
    conn = _get_connection()
    try:
        # Truncate description to 500 chars for storage efficiency
        desc_snippet = str(data.get('description', ''))[:500]

        confidence = float(data.get('confidence', 0.0))
        label      = _confidence_label(confidence)
        stage_int  = data.get('stage', 2)
        stage_str  = f'Stage {stage_int}'

        tech_skills = ', '.join(data.get('technical_skills', []))
        soft_skills = ', '.join(data.get('soft_skills', []))

        conn.execute("""
            INSERT INTO predictions
                (job_description, predicted_title, category, confidence,
                 confidence_label, detection_stage, technical_skills, soft_skills)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            desc_snippet,
            str(data.get('job_title', '')),
            str(data.get('job_category', '')),
            confidence,
            label,
            stage_str,
            tech_skills,
            soft_skills,
        ))
        conn.commit()
    finally:
        conn.close()


def get_history(limit: int = 20) -> list:
    """
    Return the *limit* most recent predictions, newest first.

    Returns a list of plain dicts (id, job_description, predicted_title,
    category, confidence, confidence_label, detection_stage,
    technical_skills, soft_skills, timestamp).
    """
    conn = _get_connection()
    try:
        cursor = conn.execute("""
            SELECT id, job_description, predicted_title, category,
                   confidence, confidence_label, detection_stage,
                   technical_skills, soft_skills, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def clear_history() -> None:
    """Delete all rows from the predictions table."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict:
    """
    Return aggregate statistics for the summary bar:
        total            (int)
        most_common_title (str)
        most_common_category (str)
        avg_confidence   (float)
    """
    conn = _get_connection()
    try:
        total_row = conn.execute("SELECT COUNT(*) AS cnt FROM predictions").fetchone()
        total = total_row['cnt'] if total_row else 0

        if total == 0:
            return {
                'total': 0,
                'most_common_title': '—',
                'most_common_category': '—',
                'avg_confidence': 0.0,
            }

        title_row = conn.execute("""
            SELECT predicted_title, COUNT(*) AS cnt
            FROM predictions
            GROUP BY predicted_title
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()

        cat_row = conn.execute("""
            SELECT category, COUNT(*) AS cnt
            FROM predictions
            GROUP BY category
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()

        avg_row = conn.execute(
            "SELECT ROUND(AVG(confidence), 1) AS avg_conf FROM predictions"
        ).fetchone()

        return {
            'total':                total,
            'most_common_title':    title_row['predicted_title'] if title_row else '—',
            'most_common_category': cat_row['category']          if cat_row   else '—',
            'avg_confidence':       float(avg_row['avg_conf'])   if avg_row and avg_row['avg_conf'] else 0.0,
        }
    finally:
        conn.close()
