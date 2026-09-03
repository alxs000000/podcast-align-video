from __future__ import annotations

import json
import time
from pathlib import Path

from .util import atomic_write_json, safe_exception_text, utc_now


class Manifest:
    def __init__(self, path: Path, value: dict) -> None:
        self.path = path
        self.value = value

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        job_id: str,
        fingerprint: str,
        source: dict[str, object],
        config: dict[str, object],
    ) -> "Manifest":
        if path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                old = {}
            if old.get("fingerprint") == fingerprint and old.get("schema_version") == 1:
                old["job_id"] = job_id
                old["status"] = "running"
                old.pop("error", None)
                old.pop("completed_at", None)
                old["source"] = source
                old["models"] = config["models"]
                old["effective_config"] = {
                    key: value for key, value in config.items() if key != "models"
                }
                old["warnings"] = []
                old["artifacts"] = {}
                old["speech_cut"] = {"status": "pending"}
                old.setdefault("stages", {})
                old["updated_at"] = utc_now()
                instance = cls(path, old)
                instance.save()
                return instance
        now = utc_now()
        value = {
            "schema_version": 1,
            "job_id": job_id,
            "fingerprint": fingerprint,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "source": source,
            "models": config["models"],
            "effective_config": {
                key: value for key, value in config.items() if key not in {"models"}
            },
            "stages": {},
            "warnings": [],
            "artifacts": {},
            "speech_cut": {"status": "pending"},
        }
        instance = cls(path, value)
        instance.save()
        return instance

    def save(self) -> None:
        self.value["updated_at"] = utc_now()
        atomic_write_json(self.path, self.value)

    def start_stage(self, name: str, fingerprint: str) -> float:
        self.value["stages"][name] = {
            "status": "running",
            "fingerprint": fingerprint,
            "started_at": utc_now(),
        }
        self.save()
        return time.monotonic()

    def finish_stage(self, name: str, checkpoint: dict, *, reused: bool = False) -> None:
        entry = dict(checkpoint)
        entry["reused"] = reused
        entry.pop("artifacts", None)
        self.value["stages"][name] = entry
        self.save()

    def fail_stage(self, name: str, error: BaseException) -> None:
        entry = self.value["stages"].setdefault(name, {})
        entry.update({"status": "failed", "error": safe_exception_text(error)})
        self.value["status"] = "failed"
        self.save()

    def warning(self, message: str) -> None:
        if message not in self.value["warnings"]:
            self.value["warnings"].append(message)
            self.save()

    def complete(self, artifacts: dict[str, object], speech_cut: dict[str, object]) -> None:
        self.value["artifacts"] = artifacts
        self.value["speech_cut"] = speech_cut
        self.value["status"] = "complete"
        self.value["completed_at"] = utc_now()
        self.save()
