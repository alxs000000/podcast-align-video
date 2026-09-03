from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_align_video.cli import _clean, _positive_float
from podcast_align_video.config import RunConfig


def _completed_job(data_root: Path, job_id: str = "job-safe-123") -> tuple[Path, Path]:
    job = data_root / "jobs" / job_id
    work = job / "work"
    work.mkdir(parents=True)
    (work / "checkpoint.json").write_text("{}", encoding="utf-8")
    (job / "job.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (job / "run-manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return job, work


def test_clean_is_dry_run_then_removes_only_completed_job_work(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_root = tmp_path / "data"
    job, work = _completed_job(data_root)
    published = tmp_path / "published" / "video.mp4"
    model = data_root / "models" / "model" / "config.json"
    published.parent.mkdir()
    model.parent.mkdir(parents=True)
    published.write_bytes(b"video")
    model.write_text("{}", encoding="utf-8")
    config = RunConfig(data_root=data_root)

    _clean(job.name, config, confirmed=False)
    assert work.is_dir()
    assert "Dry run only" in capsys.readouterr().out

    _clean(job.name, config, confirmed=True)
    assert not work.exists()
    assert job.is_dir()
    assert published.read_bytes() == b"video"
    assert model.read_text(encoding="utf-8") == "{}"


def test_clean_rejects_incomplete_or_unsafe_job_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    job, _ = _completed_job(data_root)
    (job / "job.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    config = RunConfig(data_root=data_root)
    with pytest.raises(RuntimeError, match="completed jobs"):
        _clean(job.name, config, confirmed=True)
    with pytest.raises(ValueError):
        _clean("../outside", config, confirmed=True)


def test_positive_float_rejects_zero_and_negative_values() -> None:
    assert _positive_float("0.25") == 0.25
    with pytest.raises(Exception, match="positive"):
        _positive_float("0")
    with pytest.raises(Exception, match="positive"):
        _positive_float("-1")
    with pytest.raises(Exception, match="positive"):
        _positive_float("nan")
    with pytest.raises(Exception, match="positive"):
        _positive_float("inf")
