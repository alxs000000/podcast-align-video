from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from podcast_align_video import sources


def test_url_and_windows_path_detection() -> None:
    assert sources.is_web_source("https://youtu.be/abc123")
    assert not sources.is_web_source(r"P:\\audio\\episode.wav")
    assert not sources.is_web_source("episode.wav")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/watch?v=abc&list=PL123",
        "https://www.youtube.com/watch?v=abc&LIST=",
        "https://user:password@www.youtube.com/watch?v=abc",
        "https://example.com/watch?v=abc",
        "file:///tmp/audio.wav",
    ],
)
def test_rejects_playlist_and_non_youtube_urls(url: str) -> None:
    with pytest.raises(ValueError):
        sources.validate_youtube_url(url)


def test_local_source_is_byte_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "Episode.FLAC"
    audio.write_bytes(b"fake-audio")
    monkeypatch.setattr(
        sources,
        "audio_info",
        lambda *_: {"duration_seconds": 10.0, "codec": "flac", "sample_rate": 48000, "channels": 2},
    )
    result = sources.prepare_source(audio, tmp_path / "data", "ffprobe")
    assert result.path == audio
    assert result.extension == ".flac"
    assert result.metadata["kind"] == "local"
    assert result.metadata["extension"] == ".flac"
    assert "name" not in result.metadata


def test_mocked_ytdlp_is_isolated_and_sanitized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, *, capture=False, **_):
        command = [str(item) for item in args]
        calls.append(command)
        if "--dump-single-json" in command:
            return subprocess.CompletedProcess(command, 0, '{"id":"abc","title":"Public title","channel_id":"chan"}\n', "")
        template = Path(command[command.index("-o") + 1])
        downloaded = template.with_name("source.opus")
        downloaded.write_bytes(b"opus")
        return subprocess.CompletedProcess(command, 0, str(downloaded) + "\n", "")

    monkeypatch.setattr(sources, "run_command", fake_run)
    monkeypatch.setattr(
        sources,
        "audio_info",
        lambda *_: {"duration_seconds": 12.5, "codec": "opus", "sample_rate": 48000, "channels": 2},
    )
    result = sources.prepare_source(
        "https://www.youtube.com/watch?v=abc&utm_source=private",
        tmp_path / "data",
        "ffprobe",
    )
    try:
        assert result.extension == ".opus"
        assert result.metadata["source_page"] == "https://www.youtube.com/watch"
        assert all("--ignore-config" in command and "--no-playlist" in command for command in calls)
        download_call = next(command for command in calls if "-f" in command)
        assert download_call[download_call.index("-f") + 1] == "bestaudio"
        assert not any("cookie" in token.casefold() for command in calls for token in command)
    finally:
        result.cleanup()
