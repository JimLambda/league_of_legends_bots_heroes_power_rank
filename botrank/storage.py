"""SQLite persistence: processed files and player records.

Everything is content-hash keyed, so re-running a scan only OCR's new
screenshots; renamed/moved files are still recognised.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_hash   TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ok',      -- ok | error
    error       TEXT,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash     TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    file_name     TEXT NOT NULL,
    raw_name      TEXT NOT NULL,
    champion_id   TEXT,
    champion_name TEXT,
    is_bot        INTEGER NOT NULL DEFAULT 0,
    name_match    TEXT NOT NULL DEFAULT 'none',
    team          INTEGER NOT NULL DEFAULT 1,
    level         INTEGER,
    kills         INTEGER NOT NULL DEFAULT 0,
    deaths        INTEGER NOT NULL DEFAULT 0,
    assists       INTEGER NOT NULL DEFAULT 0,
    damage        INTEGER,
    gold          INTEGER,
    gold_per_min  INTEGER,
    outcome       TEXT NOT NULL DEFAULT 'UNKNOWN',
    duration_sec  INTEGER,
    match_time    TEXT,
    confidence    REAL NOT NULL DEFAULT 0,
    source        TEXT NOT NULL DEFAULT 'ocr'
);
-- NB: no UNIQUE(file_hash, raw_name) — bot games legitimately field two
-- bots with the same champion name (e.g. two Veigars) in one match.
-- De-duplication happens per file in replace_records().

CREATE INDEX IF NOT EXISTS idx_records_champion ON records(champion_id);
CREATE INDEX IF NOT EXISTS idx_records_bot ON records(is_bot);
"""

_RECORD_COLUMNS = ("id", "file_hash", "file_path", "file_name", "raw_name",
                   "champion_id", "champion_name", "is_bot", "name_match",
                   "team", "level", "kills", "deaths", "assists", "damage",
                   "gold", "gold_per_min", "outcome", "duration_sec",
                   "match_time", "confidence", "source")


class Store:
    """Thin SQLite wrapper (single-writer; the Streamlit app is the only user)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path),
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # files
    # ------------------------------------------------------------------ #
    def file_status(self, file_hash: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT status FROM files WHERE file_hash = ?",
            (file_hash,)).fetchone()
        return row["status"] if row else None

    def mark_file(self, file_hash: str, file_path: str, status: str,
                  error: Optional[str] = None) -> None:
        self._conn.execute(
            "INSERT INTO files (file_hash, file_path, file_name, status, error,"
            " processed_at) VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(file_hash) DO UPDATE SET file_path = excluded.file_path,"
            " file_name = excluded.file_name, status = excluded.status,"
            " error = excluded.error, processed_at = excluded.processed_at",
            (file_hash, str(file_path), Path(file_path).name, status, error,
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # records
    # ------------------------------------------------------------------ #
    def replace_records(self, file_hash: str, file_path: str,
                        records: Iterable[dict]) -> None:
        """Delete previous rows for this file and insert the new ones."""
        self._conn.execute("DELETE FROM records WHERE file_hash = ?",
                           (file_hash,))
        self._conn.executemany(
            "INSERT INTO records (file_hash, file_path, file_name, raw_name,"
            " champion_id, champion_name, is_bot, name_match, team, level,"
            " kills, deaths, assists, damage, gold, gold_per_min, outcome,"
            " duration_sec, match_time, confidence, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [self._record_values(file_hash, file_path, r) for r in records])
        self._conn.commit()

    @staticmethod
    def _record_values(file_hash: str, file_path: str, record: dict) -> tuple:
        return (
            file_hash, str(file_path), Path(file_path).name,
            record.get("raw_name", ""), record.get("champion_id"),
            record.get("champion_name"), int(bool(record.get("is_bot"))),
            record.get("name_match", "none"), record.get("team", 1),
            record.get("level"), record.get("kills", 0),
            record.get("deaths", 0), record.get("assists", 0),
            record.get("damage"), record.get("gold"),
            record.get("gold_per_min"), record.get("outcome", "UNKNOWN"),
            record.get("duration_sec"), record.get("match_time"),
            record.get("confidence", 0.0), record.get("source", "ocr"),
        )

    def all_records(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT %s FROM records ORDER BY id" % ", ".join(_RECORD_COLUMNS)
        ).fetchall()
        return [dict(row) for row in rows]

    def update_record(self, record_id: int, changes: Dict[str, object]) -> None:
        """Apply manual corrections (used by the Records tab editor)."""
        allowed = ("kills", "deaths", "assists", "damage", "gold",
                   "outcome", "champion_name", "raw_name", "team", "level")
        assignments = []
        values: List[object] = []
        for column, value in changes.items():
            if column in allowed and value is not None:
                assignments.append("%s = ?" % column)
                values.append(value)
        if not assignments:
            return
        values.append(record_id)
        self._conn.execute(
            "UPDATE records SET %s WHERE id = ?" % ", ".join(assignments),
            tuple(values))
        self._conn.commit()

    def delete_records(self, record_ids: Iterable[int]) -> None:
        self._conn.executemany("DELETE FROM records WHERE id = ?",
                               [(int(i),) for i in record_ids])
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # stats / maintenance
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        files = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
        ).fetchall()
        records = self._conn.execute(
            "SELECT COUNT(*) AS n, SUM(is_bot) AS bots FROM records"
        ).fetchone()
        champions = self._conn.execute(
            "SELECT COUNT(DISTINCT champion_id) AS n FROM records WHERE is_bot = 1"
        ).fetchone()
        last = self._conn.execute(
            "SELECT MAX(processed_at) AS t FROM files WHERE status = 'ok'"
        ).fetchone()
        return {
            "files_ok": next((f["n"] for f in files
                              if f["status"] == "ok"), 0),
            "files_error": next((f["n"] for f in files
                                 if f["status"] == "error"), 0),
            "records": records["n"] or 0,
            "bot_records": records["bots"] or 0,
            "bot_champions": champions["n"] or 0,
            "last_scan": last["t"] if last else None,
        }

    def failed_files(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT file_hash, file_path, file_name, error FROM files"
            " WHERE status = 'error' ORDER BY processed_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        self._conn.executescript(
            "DELETE FROM files; DELETE FROM records;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
