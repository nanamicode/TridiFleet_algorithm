from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class AuditLog:
    """Append-only, hash-chained local experiment log."""

    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.RLock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                config_json TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sim_time TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq)"
        )
        self.conn.commit()
        self.run_id: str | None = None
        self.last_hash = "GENESIS"
        self.event_count = 0
        self.pending_rows: list[tuple[str, str, str, str, str, str]] = []
        self.flush_every = 100
        self.flush_seconds = 1.0
        self.last_flush = time.monotonic()
        self.chain_valid = True

    def start_run(self, started_at: str, config: Any) -> str:
        with self.lock:
            run_id = str(uuid.uuid4())
            payload = self._normalize(config)
            config_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            self.conn.execute(
                "INSERT INTO runs(run_id, started_at, config_json) VALUES(?,?,?)",
                (run_id, started_at, config_json),
            )
            self.conn.commit()
            self.run_id = run_id
            self.last_hash = "GENESIS"
            self.event_count = 0
            self.pending_rows.clear()
            self.chain_valid = True
            self.append("run_start", started_at, {"config": payload})
            return run_id

    def _normalize(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {str(k): self._normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        if isinstance(value, set):
            return sorted(self._normalize(v) for v in value)
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except TypeError:
                pass
        return value

    def append(self, kind: str, sim_time: str, payload: Any) -> str:
        with self.lock:
            if not self.run_id:
                raise RuntimeError("audit run not started")
            normalized = self._normalize(payload)
            payload_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
            material = (
                f"{self.run_id}|{sim_time}|{kind}|{self.last_hash}|{payload_json}"
            ).encode("utf-8")
            event_hash = hashlib.sha256(material).hexdigest()
            self.pending_rows.append(
                (
                    self.run_id,
                    sim_time,
                    kind,
                    payload_json,
                    self.last_hash,
                    event_hash,
                )
            )
            self.last_hash = event_hash
            self.event_count += 1
            now = time.monotonic()
            if (
                len(self.pending_rows) >= self.flush_every
                or now - self.last_flush >= self.flush_seconds
            ):
                self.flush()
            return event_hash

    def flush(self) -> None:
        with self.lock:
            if not self.pending_rows:
                return
            rows = list(self.pending_rows)
            self.pending_rows.clear()
            try:
                self.conn.executemany(
                    """
                    INSERT INTO events(
                        run_id, sim_time, kind, payload_json, prev_hash, event_hash
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    rows,
                )
                self.conn.commit()
                self.last_flush = time.monotonic()
            except Exception:
                self.conn.rollback()
                # Preserve order and retryability if storage is temporarily busy.
                self.pending_rows = rows + self.pending_rows
                raise

    def verify(self) -> tuple[bool, int, str]:
        with self.lock:
            if not self.run_id:
                return True, 0, "GENESIS"
            self.flush()
            rows = self.conn.execute(
                """
                SELECT sim_time, kind, payload_json, prev_hash, event_hash
                FROM events WHERE run_id=? ORDER BY seq
                """,
                (self.run_id,),
            )
            count = 0
            prev = "GENESIS"
            for sim_time, kind, payload_json, stored_prev, stored_hash in rows:
                count += 1
                if stored_prev != prev:
                    self.chain_valid = False
                    return False, count, prev
                material = (
                    f"{self.run_id}|{sim_time}|{kind}|{prev}|{payload_json}"
                ).encode("utf-8")
                expected = hashlib.sha256(material).hexdigest()
                if expected != stored_hash:
                    self.chain_valid = False
                    return False, count, prev
                prev = stored_hash
            self.chain_valid = True
            return True, count, prev

    def status(self) -> dict:
        # Fast path for the live dashboard. Full O(N) verification is explicit.
        return {
            "run_id": self.run_id,
            "path": self.path,
            "events": self.event_count,
            "chain_valid": self.chain_valid,
            "chain_tip": self.last_hash,
        }

    def full_verify_status(self) -> dict:
        ok, events, tip = self.verify()
        return {
            "run_id": self.run_id,
            "path": self.path,
            "events": events,
            "chain_valid": ok,
            "chain_tip": tip,
        }

    def close(self) -> None:
        with self.lock:
            self.flush()
            self.conn.close()
