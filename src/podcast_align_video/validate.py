from __future__ import annotations

import json
import math
from pathlib import Path


def validate_word_timings(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("word-timings.json must have schema_version 1")
    words = value.get("words")
    sentences = value.get("sentences")
    chunks = value.get("chunks")
    if not isinstance(words, list) or not words:
        raise ValueError("word-timings.json contains no words")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("word-timings.json contains no sentences")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("word-timings.json contains no chunks")
    duration = float(value.get("metadata", {}).get("audio_duration_seconds", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("word-timings.json has an invalid audio duration")

    previous_acoustic_end = 0.0
    previous_focus_end = 0.0
    for index, word in enumerate(words):
        if word.get("index") != index:
            raise ValueError(f"word indexes are not contiguous at {index}")
        if not isinstance(word.get("text"), str) or not word["text"].strip():
            raise ValueError(f"word {index} has no display text")
        if not isinstance(word.get("clean"), str) or not word["clean"]:
            raise ValueError(f"word {index} has no normalized text")
        if not isinstance(word.get("source"), str) or not word["source"]:
            raise ValueError(f"word {index} has no timing source")
        values = [word.get(key) for key in ("start", "end", "focus_start", "focus_end")]
        if not all(_finite(item) for item in values):
            raise ValueError(f"word {index} has non-finite timing")
        start, end, focus_start, focus_end = (float(item) for item in values)
        if not (0 <= start <= end <= duration):
            raise ValueError(f"word {index} acoustic timing is outside the audio")
        if start < previous_acoustic_end - 1e-6:
            raise ValueError(f"word {index} acoustic timing overlaps the previous word")
        if not (0 <= focus_start < focus_end <= duration):
            raise ValueError(f"word {index} focus timing is not positive and bounded")
        if focus_start < previous_focus_end - 1e-6:
            raise ValueError(f"word {index} focus timing overlaps the previous word")
        previous_acoustic_end = end
        previous_focus_end = focus_end

    covered: list[int] = []
    for index, sentence in enumerate(sentences):
        if sentence.get("index") != index:
            raise ValueError(f"sentence indexes are not contiguous at {index}")
        start = int(sentence["word_start"])
        end = int(sentence["word_end"])
        if not 0 <= start < end <= len(words):
            raise ValueError(f"sentence {index} has invalid word bounds")
        covered.extend(range(start, end))
    if covered != list(range(len(words))):
        raise ValueError("sentences must cover every word exactly once in order")
    for sentence in sentences:
        for word_index in range(int(sentence["word_start"]), int(sentence["word_end"])):
            if words[word_index].get("sentence_index") != sentence["index"]:
                raise ValueError(f"word {word_index} points to the wrong sentence")

    chunk_coverage: list[int] = []
    for position, chunk in enumerate(chunks):
        if chunk.get("index") != position:
            raise ValueError(f"chunk indexes are not contiguous at {position}")
        start = int(chunk["word_start"])
        end = int(chunk["word_end"])
        if not 0 <= start < end <= len(words):
            raise ValueError(f"chunk {position} has invalid word bounds")
        for word_index in range(start, end):
            if words[word_index].get("chunk_index") != position:
                raise ValueError(f"word {word_index} points to the wrong chunk")
        chunk_coverage.extend(range(start, end))
    if chunk_coverage != list(range(len(words))):
        raise ValueError("chunks must cover every word exactly once in order")
    return value


def _finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
