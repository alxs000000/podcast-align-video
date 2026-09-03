from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


SAFE_JOB_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
URL_QUERY_RE = re.compile(r'''(https?://[^\s?"'<>]+)\?[^\s"'<>]+''', re.IGNORECASE)
WINDOWS_PATH_RE = re.compile(r'''(?<!\w)[A-Za-z]:[\\/][^\s"'<>]+''')
POSIX_PATH_RE = re.compile(r'''(?<![\w/:])/(?:[^\s"'<>]+)''')
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.replace(temporary, destination)


def slugify(value: str, fallback: str = "podcast", limit: int = 72) -> str:
    value = value.casefold().encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return (value[:limit].rstrip("-") or fallback)


def ensure_job_id(value: str) -> str:
    if not SAFE_JOB_ID.fullmatch(value):
        raise ValueError("job ID must contain lowercase ASCII letters, numbers, and single hyphens")
    return value


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def run_command(
    args: Sequence[str | os.PathLike[str]],
    *,
    logger: Callable[[str], None] | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [os.fspath(item) for item in args]
    if logger:
        logger("$ " + " ".join(_display_arg(item) for item in command))
    combined_env = os.environ.copy()
    if env:
        combined_env.update(env)
    if capture:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=combined_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
            raise RuntimeError(f"command failed ({completed.returncode}): {_display_arg(command[0])}\n{detail}")
        return completed

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=combined_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        clean = line.rstrip("\r\n")
        lines.append(clean)
        if logger:
            logger(clean)
        else:
            print(clean, flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"command failed ({returncode}): {_display_arg(command[0])}\n" + "\n".join(lines[-50:]))
    return subprocess.CompletedProcess(command, returncode, "\n".join(lines), "")


def _display_arg(value: str) -> str:
    lowered = value.casefold()
    if any(marker in lowered for marker in ("token=", "authorization:", "hf_token")):
        return "<redacted>"
    if re.fullmatch(r"[A-Za-z0-9_./:=+@%-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, file=sys.stderr, flush=True)


def artifact_record(path: Path, *, name: str | None = None) -> dict[str, object]:
    return {
        "name": name or path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finite_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def safe_exception_text(error: BaseException, *private_roots: Path, limit: int = 1000) -> str:
    message = f"{type(error).__name__}: {str(error)[:limit]}"
    for root in private_roots:
        message = message.replace(str(root), "<local-path>")
    message = URL_QUERY_RE.sub(r"\1?<redacted-query>", message)
    message = HF_TOKEN_RE.sub("<redacted-token>", message)
    message = WINDOWS_PATH_RE.sub("<local-path>", message)
    message = POSIX_PATH_RE.sub("<local-path>", message)
    return message
