from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

from .media import audio_info
from .util import run_command, sha256_file


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


@dataclass(frozen=True)
class PreparedSource:
    path: Path
    sha256: str
    title: str
    extension: str
    metadata: dict[str, object]
    temporary_root: Path | None = None

    def cleanup(self) -> None:
        if self.temporary_root is not None:
            shutil.rmtree(self.temporary_root, ignore_errors=True)


def is_web_source(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)


def validate_youtube_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("YouTube source must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials embedded in YouTube URLs are not supported")
    host = (parsed.hostname or "").casefold().rstrip(".")
    if host not in YOUTUBE_HOSTS:
        raise ValueError(f"unsupported URL host: {host or '<missing>'}")
    query = {key.casefold(): value for key, value in parse_qs(parsed.query, keep_blank_values=True).items()}
    if "list" in query or parsed.path.casefold().startswith("/playlist"):
        raise ValueError("playlists are not supported; pass one public video URL")
    if not parsed.path.strip("/"):
        raise ValueError("YouTube URL does not identify a video")
    return value


def prepare_source(source: str | Path, data_root: Path, ffprobe_binary: str) -> PreparedSource:
    value = str(source)
    if is_web_source(value):
        return _prepare_youtube(validate_youtube_url(value), data_root, ffprobe_binary)
    path = Path(source).expanduser().resolve()
    info = audio_info(path, ffprobe_binary)
    extension = path.suffix if path.suffix and len(path.suffix) <= 16 else ".audio"
    return PreparedSource(
        path=path,
        sha256=sha256_file(path),
        title=path.stem,
        extension=extension.casefold(),
        metadata={
            "kind": "local",
            "extension": extension.casefold(),
            "duration_seconds": round(float(info["duration_seconds"]), 6),
            "codec": info["codec"],
        },
    )


def _prepare_youtube(url: str, data_root: Path, ffprobe_binary: str) -> PreparedSource:
    ingest_root = data_root / "ingest"
    ingest_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="youtube-", dir=ingest_root))
    common = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-playlist",
        "--no-warnings",
        "--no-progress",
        "--quiet",
    ]
    try:
        metadata_result = run_command([*common, "--dump-single-json", "--skip-download", url], capture=True)
        metadata = _last_json_object(metadata_result.stdout)
        if metadata.get("_type") in {"playlist", "multi_video"} or metadata.get("entries"):
            raise ValueError("playlists and multi-video pages are not supported")
        if metadata.get("is_live") is True or metadata.get("live_status") in {"is_live", "is_upcoming"}:
            raise ValueError("live and upcoming YouTube videos are not supported")
        if metadata.get("availability") in {"private", "subscriber_only", "needs_auth"}:
            raise ValueError("authenticated YouTube videos are not supported")

        template = temporary / "source.%(ext)s"
        download = run_command(
            [
                *common,
                "-f",
                "bestaudio",
                "--print",
                "after_move:filepath",
                "-o",
                str(template),
                url,
            ],
            capture=True,
        )
        candidates = [Path(line.strip()) for line in download.stdout.splitlines() if line.strip()]
        candidates.extend(path for path in temporary.glob("source.*") if path.is_file())
        downloaded = next((path.resolve() for path in reversed(candidates) if path.is_file()), None)
        if downloaded is None or not downloaded.is_relative_to(temporary.resolve()):
            raise RuntimeError("yt-dlp did not produce one audio file in the isolated download directory")
        info = audio_info(downloaded, ffprobe_binary)
        video_id = str(metadata.get("id") or "")
        title = str(metadata.get("title") or video_id or "youtube-video")
        sanitized_url = urlunparse((urlparse(url).scheme, urlparse(url).netloc, urlparse(url).path, "", "", ""))
        safe_metadata: dict[str, object] = {
            "kind": "youtube",
            "host": (urlparse(url).hostname or "").casefold(),
            "video_id": video_id,
            "title": title,
            "source_page": sanitized_url,
            "duration_seconds": round(float(info["duration_seconds"]), 6),
            "codec": info["codec"],
        }
        if metadata.get("channel_id"):
            safe_metadata["channel_id"] = str(metadata["channel_id"])
        return PreparedSource(
            path=downloaded,
            sha256=sha256_file(downloaded),
            title=title,
            extension=downloaded.suffix.casefold() or ".audio",
            metadata=safe_metadata,
            temporary_root=temporary,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _last_json_object(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("yt-dlp returned no metadata JSON")
