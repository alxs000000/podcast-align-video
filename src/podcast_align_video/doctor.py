from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ModelSpec, RunConfig
from .models import _model_directory_complete, _model_revision_matches


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(config: RunConfig, *, expensive: bool = True) -> list[Check]:
    device_literal = repr(config.runtime.device)
    cohere_model_literal = repr(str(config.cohere.resolved_path(config.data_root)))
    checks = [
        Check("platform", sys.platform.startswith("linux"), f"{platform.system()} {platform.release()} (Linux/WSL2 required)"),
        Check(
            "python",
            sys.version_info[:2] == (3, 12),
            f"{platform.python_version()} (3.12 required)",
        ),
        _command_check("ffmpeg", config.runtime.ffmpeg),
        _command_check("ffprobe", config.runtime.ffprobe),
        _command_check("nvidia-smi", "nvidia-smi"),
        _file_check("Cohere Python", config.cohere_python),
        _file_check("Qwen Python", config.qwen_python),
        _file_check("MFA", config.mfa_binary),
        _model_check("Cohere model", config.cohere, config.data_root),
        _model_check("Qwen ASR model", config.qwen_asr, config.data_root),
        _model_check("Qwen aligner model", config.qwen_aligner, config.data_root),
        _browser_check(config),
    ]
    if expensive and (_command_check("ffmpeg", config.runtime.ffmpeg).ok):
        checks.append(_ffmpeg_capability_check(config.runtime.ffmpeg))
    if expensive and config.cohere_python.is_file():
        checks.append(
            _python_check(
                "Cohere environment",
                config.cohere_python,
                "import torch, transformers, soundfile, silero_vad; from transformers import AutoProcessor; "
                "assert transformers.__version__ == '5.16.1'; "
                f"AutoProcessor.from_pretrained({cohere_model_literal}, local_files_only=True); "
                f"assert torch.cuda.is_available(); device={device_literal}; "
                "torch.empty(1, device=device); print(torch.cuda.get_device_name(torch.device(device)))",
            )
        )
    if expensive and config.qwen_python.is_file():
        checks.append(
            _python_check(
                "Qwen environment",
                config.qwen_python,
                "import torch, qwen_asr, librosa; assert torch.cuda.is_available(); "
                f"device={device_literal}; torch.empty(1, device=device); "
                "print(torch.cuda.get_device_name(torch.device(device)))",
            )
        )
    if expensive and config.mfa_binary.is_file():
        checks.append(
            _python_check(
                "MFA toolchain",
                config.mfa_python,
                "from montreal_forced_aligner.utils import check_third_party; "
                "check_third_party(); print('OpenFst, Kaldi, and SoX available')",
                env=config.mfa_subprocess_environment(),
            )
        )
        for kind, name in (
            ("acoustic", config.alignment.mfa_acoustic_model),
            ("dictionary", config.alignment.mfa_dictionary),
            ("g2p", config.alignment.mfa_g2p_model),
        ):
            checks.append(
                _mfa_model_check(
                    config.mfa_binary,
                    kind,
                    name,
                    config.mfa_subprocess_environment(),
                )
            )
    return checks


def require_ready(config: RunConfig) -> None:
    failed = [check for check in run_checks(config) if check.required and not check.ok]
    if failed:
        details = "\n".join(f"- {check.name}: {check.detail}" for check in failed)
        raise RuntimeError(
            "environment preflight failed before media processing:\n"
            f"{details}\nRun `podcast-align-video doctor` and `podcast-align-video models fetch`."
        )


def _command_check(name: str, command: str) -> Check:
    path = shutil.which(command) if "/" not in command else (command if Path(command).is_file() else None)
    return Check(name, bool(path), str(path or f"not found: {command}"))


def _file_check(name: str, path: Path) -> Check:
    return Check(name, path.is_file(), str(path if path.is_file() else "missing; run scripts/setup.sh"))


def _model_check(name: str, spec: ModelSpec, data_root: Path) -> Check:
    path = spec.resolved_path(data_root)
    if not path.is_dir() or not _model_directory_complete(path, spec.id):
        return Check(name, False, f"missing {spec.id}@{spec.revision}; run models fetch")
    if spec.path is None and not _model_revision_matches(path, spec.id, spec.revision):
        return Check(name, False, "model inventory or immutable revision marker does not match; run models fetch")
    return Check(name, True, f"{spec.id}@{spec.revision}")


def _browser_check(config: RunConfig) -> Check:
    old = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    try:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(config.browser_cache)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                return Check("Playwright Chromium", False, "browser executable missing; run scripts/setup.sh")
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content("<!doctype html><title>doctor</title>")
            browser.close()
        return Check("Playwright Chromium", True, f"launch succeeded: {executable}")
    except Exception as error:
        detail = str(error).strip().splitlines()
        summary = detail[0] if detail else "browser launch failed"
        return Check(
            "Playwright Chromium",
            False,
            f"{type(error).__name__}: {summary}; install the distro's Chromium runtime libraries",
        )
    finally:
        if old is None:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        else:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = old


def _python_check(name: str, python: Path, code: str, *, env: dict[str, str] | None = None) -> Check:
    environment = os.environ.copy()
    if env:
        environment.update(env)
    completed = subprocess.run(
        [str(python), "-c", code],
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    detail = (completed.stdout or "").strip().splitlines()
    return Check(name, completed.returncode == 0, detail[-1] if detail else f"exit {completed.returncode}")


def _ffmpeg_capability_check(binary: str) -> Check:
    filters = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    encoders = subprocess.run(
        [binary, "-hide_banner", "-encoders"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    has_ass = any(line.split()[1:2] == ["ass"] for line in filters.stdout.splitlines() if len(line.split()) >= 2)
    has_x264 = "libx264" in encoders.stdout
    ok = filters.returncode == 0 and encoders.returncode == 0 and has_ass and has_x264
    return Check("FFmpeg capabilities", ok, f"libass={'yes' if has_ass else 'no'}, libx264={'yes' if has_x264 else 'no'}")


def _mfa_model_check(binary: Path, kind: str, name: str, mfa_environment: dict[str, str]) -> Check:
    environment = os.environ.copy()
    environment.update(mfa_environment)
    completed = subprocess.run(
        [str(binary), "model", "list", kind],
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    found = completed.returncode == 0 and name in completed.stdout
    return Check(f"MFA {kind} model", found, f"{name} {'installed' if found else 'missing; run models fetch'}")


def format_checks(checks: list[Check]) -> str:
    return "\n".join(f"{'OK' if check.ok else 'FAIL':4}  {check.name:24} {check.detail}" for check in checks)
