"""SQLite-backed persistence for production app telemetry."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

DB_PATH = Path(
    os.getenv(
        "APP_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "owlban_app.db"),
    )
)


class AppStore:
    """Simple SQLite store for app events and usage traces."""

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        self.db_path.chmod(0o600)

    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events(event_name, payload) VALUES (?, ?)",
                (event_name, json.dumps(payload, default=str)),
            )
            conn.commit()

    def count_events(self, event_name: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM events"
        params: tuple[Any, ...] = ()
        if event_name is not None:
            query += " WHERE event_name = ?"
            params = (event_name,)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row else 0
