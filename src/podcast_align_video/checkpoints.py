from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

from .util import atomic_write_json, sha256_file, stable_hash, utc_now


def stage_fingerprint(
    name: str,
    *,
    schema_version: int,
    inputs: dict[str, str],
    settings: object,
    models: object = None,
) -> str:
    return stable_hash(
        {
            "stage": name,
            "schema_version": schema_version,
            "inputs": inputs,
            "settings": settings,
            "models": models,
        }
    )


def valid_checkpoint(
    directory: Path,
    fingerprint: str,
    *,
    extra_validator: Callable[[Path], None] | None = None,
) -> dict | None:
    checkpoint = directory / "checkpoint.json"
    if not checkpoint.is_file():
        return None
    try:
        import json

        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != 1
            or value.get("fingerprint") != fingerprint
            or value.get("status") != "complete"
        ):
            return None
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return None
        for artifact in artifacts:
            relative = artifact.get("path")
            expected = artifact.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                return None
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file():
                return None
            if sha256_file(path) != expected:
                return None
        if extra_validator is not None:
            extra_validator(directory)
        return value
    except Exception:
        return None


def write_checkpoint(
    directory: Path,
    fingerprint: str,
    artifacts: Iterable[Path],
    *,
    started_monotonic: float,
    details: dict[str, object] | None = None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for path in artifacts:
        resolved = path.resolve()
        if not resolved.is_relative_to(directory.resolve()):
            raise ValueError(f"checkpoint artifact is outside stage directory: {path}")
        records.append(
            {
                "path": str(resolved.relative_to(directory.resolve())),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    value: dict[str, object] = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "status": "complete",
        "completed_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "artifacts": records,
    }
    if details:
        value["details"] = details
    atomic_write_json(directory / "checkpoint.json", value)
    return value
