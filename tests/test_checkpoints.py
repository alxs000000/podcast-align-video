import json
from pathlib import Path

from podcast_align_video.checkpoints import stage_fingerprint, valid_checkpoint, write_checkpoint
from podcast_align_video.manifest import Manifest
from podcast_align_video.util import safe_exception_text


def test_fingerprint_is_stable_and_sensitive() -> None:
    left = stage_fingerprint("x", schema_version=1, inputs={"a": "1"}, settings={"b": 2})
    right = stage_fingerprint("x", schema_version=1, inputs={"a": "1"}, settings={"b": 2})
    changed = stage_fingerprint("x", schema_version=1, inputs={"a": "2"}, settings={"b": 2})
    assert left == right
    assert left != changed


def test_corrupt_checkpoint_artifact_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "value.txt"
    artifact.write_text("valid", encoding="utf-8")
    write_checkpoint(tmp_path, "fingerprint", [artifact], started_monotonic=0)
    assert valid_checkpoint(tmp_path, "fingerprint") is not None
    artifact.write_text("corrupt", encoding="utf-8")
    assert valid_checkpoint(tmp_path, "fingerprint") is None


def test_unknown_checkpoint_schema_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "value.txt"
    artifact.write_text("valid", encoding="utf-8")
    write_checkpoint(tmp_path, "fingerprint", [artifact], started_monotonic=0)
    checkpoint = tmp_path / "checkpoint.json"
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    value["schema_version"] = 999
    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    assert valid_checkpoint(tmp_path, "fingerprint") is None


def test_manifest_error_sanitizer_removes_paths_queries_and_tokens(tmp_path: Path) -> None:
    token = "hf_" + "a" * 30
    error = RuntimeError(
        f'File "/opt/private-user/private.py" P:\\Secret\\audio.wav '
        f"https://example.test/watch?v=secret&utm=x {token}"
    )
    safe = safe_exception_text(error, tmp_path)
    assert "/opt/private-user" not in safe
    assert "P:\\Secret" not in safe
    assert "v=secret" not in safe
    assert token not in safe


def test_successful_manifest_resume_clears_old_top_level_error(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    first = Manifest.open(path, job_id="job", fingerprint="same", source={}, config={"models": {}})
    first.value["status"] = "failed"
    first.value["error"] = "old failure"
    first.value["warnings"] = ["old warning"]
    first.value["artifacts"] = {"old": {}}
    first.value["speech_cut"] = {"status": "failed"}
    first.value["completed_at"] = "yesterday"
    first.save()
    resumed = Manifest.open(path, job_id="job", fingerprint="same", source={}, config={"models": {}})
    assert resumed.value["status"] == "running"
    assert "error" not in resumed.value
    assert "completed_at" not in resumed.value
    assert resumed.value["warnings"] == []
    assert resumed.value["artifacts"] == {}
    assert resumed.value["speech_cut"] == {"status": "pending"}
