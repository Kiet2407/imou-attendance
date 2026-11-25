"""SQLite persistence for enrolled faces and confirmed entrance events.

Event timestamps are stored in UTC. Reports use local calendar-day boundaries,
so an event before midnight UTC can correctly belong to the next office day.
"""

import json
import math
import sqlite3
from datetime import date, datetime, time, timedelta, timezone as dt_timezone
from pathlib import Path
from zoneinfo import ZoneInfo


class Store:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY,
                    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                    vector TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entry_events (
                    id INTEGER PRIMARY KEY,
                    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                    occurred_at TEXT NOT NULL,
                    camera TEXT NOT NULL,
                    score REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS event_time_idx
                    ON entry_events(occurred_at, employee_id);
            """)

    def close(self):
        self.connection.close()

    @staticmethod
    def _text(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    def add_employee(self, employee_id, name):
        employee_id = self._text(employee_id, "employee_id")
        name = self._text(name, "name")
        with self.connection:
            self.connection.execute(
                "INSERT INTO employees(employee_id, name) VALUES (?, ?) "
                "ON CONFLICT(employee_id) DO UPDATE SET name=excluded.name",
                (employee_id, name),
            )

    def save_embedding(self, employee_id, vector):
        employee_id = self._text(employee_id, "employee_id")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("vector must contain finite numbers") from exc
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("vector must contain finite numbers")
        if not any(value != 0 for value in values):
            raise ValueError("vector must not be zero")
        with self.connection:
            self.connection.execute(
                "INSERT INTO face_embeddings(employee_id, vector) VALUES (?, ?)",
                (employee_id, json.dumps(values, allow_nan=False)),
            )

    def embeddings(self):
        rows = self.connection.execute(
            "SELECT e.employee_id, e.name, f.vector FROM face_embeddings f "
            "JOIN employees e ON e.employee_id=f.employee_id "
            "WHERE e.active=1 ORDER BY f.id"
        )
        return [(row["employee_id"], row["name"], json.loads(row["vector"])) for row in rows]

    @staticmethod
    def _utc_string(value):
        return value.astimezone(dt_timezone.utc).isoformat(timespec="microseconds")

    def record_entry(self, employee_id, occurred_at, camera="entrance", score=0.0):
        employee_id = self._text(employee_id, "employee_id")
        camera = self._text(camera, "camera")
        if not isinstance(occurred_at, datetime) or occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be a timezone-aware datetime")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        with self.connection:
            self.connection.execute(
                "INSERT INTO entry_events(employee_id, occurred_at, camera, score) "
                "VALUES (?, ?, ?, ?)",
                (employee_id, self._utc_string(occurred_at), camera, score),
            )

    def daily_report(self, day, timezone="Asia/Ho_Chi_Minh"):
        office_zone = ZoneInfo(timezone)
        report_date = date.fromisoformat(day)
        start = datetime.combine(report_date, time.min, office_zone)
        end = datetime.combine(report_date + timedelta(days=1), time.min, office_zone)
        # Rank complete event rows so the camera belongs to the earliest event,
        # with event ID as the deterministic tie-breaker for equal timestamps.
        rows = self.connection.execute(
            """
            WITH ranked AS (
                SELECT employee_id, occurred_at, camera,
                    ROW_NUMBER() OVER (
                        PARTITION BY employee_id ORDER BY occurred_at, id
                    ) AS ordinal
                FROM entry_events WHERE occurred_at >= ? AND occurred_at < ?
            )
            SELECT e.employee_id, e.name, r.occurred_at, r.camera
            FROM employees e LEFT JOIN ranked r
                ON r.employee_id=e.employee_id AND r.ordinal=1
            WHERE e.active=1 ORDER BY e.employee_id
            """,
            (self._utc_string(start), self._utc_string(end)),
        )
        result = []
        for row in rows:
            first_arrival = row["occurred_at"]
            if first_arrival is not None:
                first_arrival = datetime.fromisoformat(first_arrival).astimezone(office_zone).isoformat()
            result.append({
                "employee_id": row["employee_id"],
                "name": row["name"],
                "first_arrival": first_arrival,
                "camera": row["camera"],
            })
        return result
