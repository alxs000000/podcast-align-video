from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from podcast_align_video.config import RendererConfig, RunConfig, RuntimeConfig, SpeechCutConfig
from podcast_align_video import models
from podcast_align_video.models import _model_inventory, _model_revision_matches


def test_mfa_environment_defaults_to_data_root(tmp_path: Path) -> None:
    config = RunConfig(data_root=tmp_path)
    assert config.mfa_environment == (tmp_path / "envs" / "mfa").resolve()


def test_mfa_environment_uses_setup_location_file(tmp_path: Path) -> None:
    runtime = tmp_path / "linux-runtime" / "mfa"
    location = tmp_path / "envs" / "mfa.location"
    location.parent.mkdir(parents=True)
    location.write_text(f"{runtime}\n", encoding="utf-8")
    config = RunConfig(data_root=tmp_path)
    assert config.mfa_environment == runtime.resolve()
    environment = config.mfa_subprocess_environment()
    assert environment["MFA_ROOT_DIR"] == str((tmp_path / "mfa-data").resolve())
    assert environment["PATH"].split(os.pathsep)[0] == str(runtime.resolve() / "bin")


def test_mfa_environment_rejects_relative_location(tmp_path: Path) -> None:
    location = tmp_path / "envs" / "mfa.location"
    location.parent.mkdir(parents=True)
    location.write_text("relative/mfa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be absolute"):
        _ = RunConfig(data_root=tmp_path).mfa_environment


def test_browser_cache_uses_setup_location_file(tmp_path: Path) -> None:
    runtime = tmp_path / "linux-runtime" / "browser-cache"
    location = tmp_path / "browser-cache.location"
    location.write_text(f"{runtime}\n", encoding="utf-8")
    assert RunConfig(data_root=tmp_path).browser_cache == runtime.resolve()


def test_model_revision_match_requires_config_and_exact_marker(tmp_path: Path) -> None:
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "config.json").write_text("{}", encoding="utf-8")
    (destination / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (destination / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
    (destination / "model.safetensors").write_bytes(b"weights")
    marker = destination / ".podcast-align-video-model.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "CohereLabs/model",
                "revision": "abc",
                "files": _model_inventory(destination),
            }
        ),
        encoding="utf-8",
    )
    assert _model_revision_matches(destination, "CohereLabs/model", "abc")
    assert not _model_revision_matches(destination, "CohereLabs/model", "different")
    (destination / "model.safetensors").write_bytes(b"truncated")
    assert not _model_revision_matches(destination, "CohereLabs/model", "abc")


def test_config_rejects_unknown_keys_and_invalid_runtime() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_mapping({"typo": True})
    with pytest.raises(ValueError, match="unknown models.cohere keys"):
        RunConfig.from_mapping({"models": {"cohere": {"typo": True}}})
    with pytest.raises(ValueError, match="explicit CUDA device"):
        RunConfig(runtime=RuntimeConfig(device="cpu"))
    with pytest.raises(ValueError, match="encoder"):
        RunConfig(renderer=RendererConfig(encoder="vp9"))
    with pytest.raises(ValueError, match="positive"):
        RunConfig(speech_cut=SpeechCutConfig(threshold_seconds=float("nan")))
    with pytest.raises(ValueError, match="segment_seconds"):
        RunConfig(renderer=RendererConfig(segment_seconds=float("inf")))
    assert RunConfig(runtime=RuntimeConfig(device="cuda:3")).fingerprint_dict()["runtime"] == {
        "device": "cuda:3"
    }


def test_cohere_fetch_failure_explains_gated_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(**_):
        raise PermissionError("authorization header contained private details")

    monkeypatch.setattr(models, "snapshot_download", denied)
    with pytest.raises(RuntimeError, match="accepted the model terms") as caught:
        models.fetch_models(RunConfig(data_root=tmp_path))
    assert "authorization header" not in str(caught.value)
