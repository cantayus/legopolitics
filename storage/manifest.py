from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legopolitics.constants import DATABASE_SCHEMA_VERSION
from legopolitics.exceptions import ManifestError
from legopolitics.utils.concurrency import current_worker

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, configuration_hash TEXT NOT NULL, status TEXT NOT NULL,
 started_at TEXT NOT NULL, ended_at TEXT, worker_id TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
 video_id TEXT PRIMARY KEY, path TEXT NOT NULL, fingerprint TEXT NOT NULL,
 status TEXT NOT NULL, metadata_json TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stages (
 run_id TEXT NOT NULL, video_id TEXT NOT NULL, stage TEXT NOT NULL,
 status TEXT NOT NULL, cache_key TEXT, started_at TEXT, ended_at TEXT,
 duration_seconds REAL, error_message TEXT, metadata_json TEXT NOT NULL,
 PRIMARY KEY (run_id, video_id, stage)
);
CREATE TABLE IF NOT EXISTS artifacts (
 artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, video_id TEXT,
 stage TEXT, path TEXT NOT NULL, kind TEXT NOT NULL, sha256 TEXT, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
 error_id TEXT PRIMARY KEY, run_id TEXT, video_id TEXT, stage TEXT NOT NULL,
 item_id TEXT, exception_type TEXT NOT NULL, message TEXT NOT NULL, traceback TEXT,
 retry_count INTEGER NOT NULL, recoverable INTEGER NOT NULL, created_at TEXT NOT NULL,
 metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
 model_key TEXT PRIMARY KEY, adapter TEXT NOT NULL, model_id TEXT, revision TEXT,
 metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
 prompt_id TEXT PRIMARY KEY, version TEXT NOT NULL, prompt_hash TEXT NOT NULL,
 rendered_prompt TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configurations (
 configuration_hash TEXT PRIMARY KEY, schema_version TEXT NOT NULL,
 configuration_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS locks (
 lock_key TEXT PRIMARY KEY, worker_id TEXT NOT NULL, acquired_at REAL NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_actions (
 action_id TEXT PRIMARY KEY, video_id TEXT NOT NULL, unit_type TEXT NOT NULL,
 unit_id TEXT NOT NULL, action TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS manual_corrections (
 correction_id TEXT PRIMARY KEY, video_id TEXT NOT NULL, unit_type TEXT NOT NULL,
 unit_id TEXT NOT NULL, action TEXT NOT NULL, original_json TEXT, corrected_json TEXT,
 coder_id TEXT, notes TEXT, created_at TEXT NOT NULL
);
"""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManifest:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_info(key,value) VALUES('schema_version',?)",
            (DATABASE_SCHEMA_VERSION,),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RunManifest:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def begin_run(self, configuration_hash: str, metadata: dict[str, Any] | None = None) -> str:
        run_id = f"run_{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
            (
                run_id,
                configuration_hash,
                "running",
                utc(),
                None,
                current_worker().value,
                json.dumps(metadata or {}, default=str),
            ),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        self.connection.execute(
            "UPDATE runs SET status=?,ended_at=? WHERE run_id=?", (status, utc(), run_id)
        )
        self.connection.commit()

    def record_config(
        self, configuration_hash: str, schema_version: str, payload: dict[str, Any]
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO configurations VALUES(?,?,?)",
            (configuration_hash, schema_version, json.dumps(payload, default=str)),
        )
        self.connection.commit()

    def record_video(
        self,
        video_id: str,
        path: str,
        fingerprint: str,
        status: str = "discovered",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO videos VALUES(?,?,?,?,?,?)",
            (video_id, path, fingerprint, status, json.dumps(metadata or {}, default=str), utc()),
        )
        self.connection.commit()

    def stage_status(self, run_id: str, video_id: str, stage: str) -> str | None:
        row = self.connection.execute(
            "SELECT status FROM stages WHERE run_id=? AND video_id=? AND stage=?",
            (run_id, video_id, stage),
        ).fetchone()
        return str(row[0]) if row else None

    def begin_stage(
        self, run_id: str, video_id: str, stage: str, cache_key: str | None = None
    ) -> float:
        started = time.time()
        self.connection.execute(
            "INSERT OR REPLACE INTO stages VALUES(?,?,?,?,?,?,?,?,?,?)",
            (run_id, video_id, stage, "running", cache_key, utc(), None, None, None, "{}"),
        )
        self.connection.commit()
        return started

    def finish_stage(
        self,
        run_id: str,
        video_id: str,
        stage: str,
        started: float,
        status: str = "complete",
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE stages SET status=?,ended_at=?,duration_seconds=?,error_message=?,metadata_json=? WHERE run_id=? AND video_id=? AND stage=?",
            (
                status,
                utc(),
                time.time() - started,
                error_message,
                json.dumps(metadata or {}, default=str),
                run_id,
                video_id,
                stage,
            ),
        )
        self.connection.commit()

    def record_error(
        self,
        run_id: str | None,
        video_id: str | None,
        stage: str,
        item_id: str | None,
        exception: BaseException,
        traceback_text: str | None,
        retry_count: int = 0,
        recoverable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        error_id = f"error_{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            "INSERT INTO errors VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                error_id,
                run_id,
                video_id,
                stage,
                item_id,
                type(exception).__name__,
                str(exception),
                traceback_text,
                retry_count,
                int(recoverable),
                utc(),
                json.dumps(metadata or {}, default=str),
            ),
        )
        self.connection.commit()
        return error_id

    def register_artifact(
        self,
        run_id: str,
        video_id: str | None,
        stage: str | None,
        path: Path,
        kind: str,
        sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        artifact_id = f"artifact_{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
            (
                artifact_id,
                run_id,
                video_id,
                stage,
                str(path),
                kind,
                sha256,
                json.dumps(metadata or {}, default=str),
            ),
        )
        self.connection.commit()
        return artifact_id

    def inspect(self) -> dict[str, list[dict[str, Any]]]:
        return {
            table: [
                dict(row) for row in self.connection.execute(f"SELECT * FROM {table}").fetchall()
            ]
            for table in ["runs", "videos", "stages", "errors", "artifacts"]
        }

    @contextmanager
    def lock(
        self, key: str, timeout_seconds: float = 30, stale_after_seconds: float = 3600
    ) -> Iterator[None]:
        worker = current_worker().value
        deadline = time.time() + timeout_seconds
        while True:
            now = time.time()
            try:
                with self.connection:
                    self.connection.execute("DELETE FROM locks WHERE expires_at<?", (now,))
                    self.connection.execute(
                        "INSERT INTO locks VALUES(?,?,?,?)",
                        (key, worker, now, now + stale_after_seconds),
                    )
                break
            except sqlite3.IntegrityError:
                if time.time() >= deadline:
                    raise ManifestError(f"Timed out acquiring lock: {key}") from None
                time.sleep(0.25)
        try:
            yield
        finally:
            with self.connection:
                self.connection.execute(
                    "DELETE FROM locks WHERE lock_key=? AND worker_id=?", (key, worker)
                )
