"""
SQLite-backed persistence for long-term guest preferences.

The database holds a single row (id = 1) containing all known preferences
as a JSON blob. Each save merges new keys into the existing record so
preferences accumulate across sessions rather than replacing each other.

The table is created automatically on first use — no migration step needed.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "preferences.db")


def _connect() -> sqlite3.Connection:
    """Open a connection to the SQLite database, creating the preferences table if absent."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            data       TEXT    NOT NULL,
            updated_at TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def load_preferences() -> dict:
    """Return the stored preferences dict, or an empty dict if none have been saved yet."""
    with _connect() as conn:
        row = conn.execute("SELECT data FROM preferences WHERE id = 1").fetchone()
    if row is None:
        return {}
    return json.loads(row[0])


def save_preferences(prefs: dict) -> None:
    """Merge new preferences into the existing record and persist to SQLite.

    Keys present in both the stored record and prefs are overwritten with
    the new value; keys only in the stored record are preserved.
    """
    # Load first so we merge rather than replace existing preferences.
    merged = load_preferences()
    merged.update(prefs)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO preferences (id, data, updated_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE
                SET data       = excluded.data,
                    updated_at = excluded.updated_at
            """,
            (json.dumps(merged), datetime.utcnow().isoformat()),
        )
        conn.commit()
    print(f"[PREFERENCES] Saved: {merged}")
