from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from podcast_align_video import pipeline, sources, util
from podcast_align_video.config import RunConfig


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))


def _timings() -> dict:
    return {
        "schema_version": 1,
        "metadata": {
            "audio_duration_seconds": 1.0,
            "mfa_word_coverage": 1.0,
            "mfa_effective_words": 2,
            "missing_textgrids": [],
            "invalid_textgrids": [],
            "source_counts": {"mfa_3.4.1_fine_tuned": 2},
            "median_qwen_mfa_boundary_delta_seconds": 0.01,
            "p95_qwen_mfa_boundary_delta_seconds": 0.02,
        },
        "words": [
            {
                "index": 0,
                "text": "Hello",
                "clean": "Hello",
                "sentence_index": 0,
                "chunk_index": 0,
                "start": 0.0,
                "end": 0.4,
                "focus_start": 0.0,
                "focus_end": 0.4,
                "source": "mfa_3.4.1_fine_tuned",
            },
            {
                "index": 1,
                "text": "world.",
                "clean": "world",
                "sentence_index": 0,
                "chunk_index": 0,
                "start": 0.4,
                "end": 0.9,
                "focus_start": 0.4,
                "focus_end": 0.9,
                "source": "mfa_3.4.1_fine_tuned",
            },
        ],
        "sentences": [{"index": 0, "text": "Hello world.", "word_start": 0, "word_end": 2, "start": 0.0, "end": 0.9}],
        "chunks": [{"index": 0, "word_start": 0, "word_end": 2, "start": 0.0, "end": 0.9}],
    }


@pytest.mark.parametrize("source_kind", ["local", "youtube"])
def test_fake_model_end_to_end_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    source_wav = tmp_path / "input.wav"
    _write_wav(source_wav)
    source: str | Path = source_wav
    youtube_calls: list[list[str]] = []
    if source_kind == "youtube":
        source = "https://www.youtube.com/watch?v=public123"

        def fake_ytdlp(args, *, capture=False, **_):
            command = [str(item) for item in args]
            youtube_calls.append(command)
            if "--dump-single-json" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    '{"id":"public123","title":"Public test","channel_id":"channel"}\n',
                    "",
                )
            template = Path(command[command.index("-o") + 1])
            downloaded = template.with_name("source.wav")
            shutil.copyfile(source_wav, downloaded)
            return subprocess.CompletedProcess(command, 0, f"{downloaded}\n", "")

        monkeypatch.setattr(sources, "run_command", fake_ytdlp)
    config = RunConfig(data_root=tmp_path / "data")
    output = tmp_path / "output"
    calls: list[str] = []

    monkeypatch.setattr(pipeline, "require_ready", lambda _: None)
    monkeypatch.setattr(pipeline, "validate_video", lambda *_, **__: {"duration_seconds": 1.0})

    def fake_command(args, **_):
        command = [str(item) for item in args]
        joined = " ".join(command)
        if "cohere_worker.py" in joined:
            calls.append("stt")
            target = Path(command[command.index("--output-dir") + 1])
            (target / "chunks").mkdir(parents=True, exist_ok=True)
            (target / "transcript.txt").write_text("Hello world.\n", encoding="utf-8")
            (target / "sentences.txt").write_text("Hello world.\n", encoding="utf-8")
            (target / "chunks" / "chunk_000.txt").write_text("Hello world.\n", encoding="utf-8")
            (target / "stt-result.json").write_text(
                json.dumps({"schema_version": 1, "quality": {"ok": True, "word_count": 2}, "window_count": 1, "speech_seconds": 0.9}),
                encoding="utf-8",
            )
            (target / "stt-windows.json").write_text(
                json.dumps({"schedule": [[0, 0, 1]], "windows": {"0": {"text": "Hello world."}}}),
                encoding="utf-8",
            )
            (target / "vad-regions.json").write_text(json.dumps({"regions": [[0, 0.9]]}), encoding="utf-8")
        elif "qwen_worker.py" in joined:
            calls.append("qwen")
            target = Path(command[command.index("--output-dir") + 1])
            corpus = target / "mfa_corpus" / "speaker1"
            corpus.mkdir(parents=True, exist_ok=True)
            qwen = _timings()
            for word in qwen["words"]:
                word.pop("focus_start")
                word.pop("focus_end")
            (target / "qwen_word_timings.json").write_text(json.dumps(qwen), encoding="utf-8")
            (target / "mfa_manifest.json").write_text(
                json.dumps([{"utterance_id": "utt_00000", "word_start": 0, "word_end": 2, "offset": 0.0}]),
                encoding="utf-8",
            )
            _write_wav(corpus / "utt_00000.wav")
            (corpus / "utt_00000.lab").write_text("Hello world.\n", encoding="utf-8")
        elif "mfa_merge.py" in joined:
            calls.append("merge")
            target = Path(command[command.index("--output-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "word_timings.json").write_text(json.dumps(_timings()), encoding="utf-8")
            (target / "alignment_report.json").write_text(json.dumps(_timings()["metadata"]), encoding="utf-8")
            (target / "sentences.vtt").write_text("WEBVTT\n", encoding="utf-8")
        elif len(command) > 1 and command[1] == "align":
            calls.append("mfa")
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(util, "run_command", fake_command)

    def fake_render(**kwargs):
        calls.append("render")
        kwargs["output"].write_bytes(b"fake-mp4")
        (kwargs["output_dir"] / "layout-meta.json").write_text("{}", encoding="utf-8")
        (kwargs["output_dir"] / "layout.jsonl").write_text("{}\n", encoding="utf-8")
        return {"segment_count": 1, "duration_seconds": 1.0}

    monkeypatch.setattr(pipeline, "render_video", fake_render)
    monkeypatch.setattr(
        pipeline,
        "create_speech_cut",
        lambda **_: {"status": "no_cuts", "threshold_seconds": 5.0, "cut_count": 0, "removed_seconds": 0.0},
    )

    first = pipeline.execute(source, output_dir=output, config=config)
    assert first.full_video.read_bytes() == b"fake-mp4"
    assert first.speech_cut_video is None
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert set(manifest["artifacts"]) == {"video", "source", "transcript", "word_timings", "run_log"}
    assert str(tmp_path) not in first.manifest.read_text(encoding="utf-8")
    assert calls == ["stt", "qwen", "mfa", "merge", "render"]

    calls.clear()
    second = pipeline.execute(source, output_dir=output, config=config)
    assert second.job_id == first.job_id
    assert calls == []
    resumed = json.loads(second.manifest.read_text(encoding="utf-8"))
    assert all(stage.get("reused") for stage in resumed["stages"].values())
    if source_kind == "youtube":
        assert len(youtube_calls) == 4
        assert all("--ignore-config" in command and "--no-playlist" in command for command in youtube_calls)


def test_explicit_output_refuses_different_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run-manifest.json").write_text('{"fingerprint":"other"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        pipeline._validate_destination(output, "new")


def test_preflight_fails_before_source_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingested = False

    def fail_preflight(_):
        raise RuntimeError("models missing")

    def unexpected_ingest(*_):
        nonlocal ingested
        ingested = True
        raise AssertionError("source ingest must not run")

    monkeypatch.setattr(pipeline, "require_ready", fail_preflight)
    monkeypatch.setattr(pipeline, "prepare_source", unexpected_ingest)
    with pytest.raises(RuntimeError, match="models missing"):
        pipeline.execute("https://youtu.be/example", output_dir=tmp_path / "output", config=RunConfig(data_root=tmp_path / "data"))
    assert not ingested
