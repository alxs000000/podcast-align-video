from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    """Small adapter contract; only the bundled Cohere adapter is guaranteed in v0.1."""

    def transcribe(self, audio: str, output_dir: str) -> dict:
        ...


@runtime_checkable
class Aligner(Protocol):
    """Small adapter contract; only the bundled Qwen + MFA adapter is guaranteed in v0.1."""

    def align(self, audio: str, transcript: str, output_dir: str) -> dict:
        ...
