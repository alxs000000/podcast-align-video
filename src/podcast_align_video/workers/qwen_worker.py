#!/usr/bin/env python3
"""Qwen3 ASR anchors -> exact transcript alignment -> MFA corpus."""

from __future__ import annotations

import argparse
import bisect
import difflib
import gc
import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

librosa = None
np = None
sf = None
torch = None
Qwen3ASRModel = None


SR = 16_000
TAG_RE = re.compile(r"\[[^\]\n]+\]\s*")
TOKEN_RE = re.compile(r"\S+")
ALIGNMENT_ALGORITHM = "qwen-rough-forced-v2"
MAX_ALIGNMENT_WORDS = 60
MAX_CROP_SECONDS = 55.0
SEGMENTATION_SCHEMA_VERSION = 10
MAX_ROUGH_SPAN_SECONDS = 35.0
MAX_ROUGH_GAP_SECONDS = 8.0
COHERE_ANCHOR_PADDING_SECONDS = 4.0
ROUGH_ALIGNMENT_OUTLIER_SECONDS = 2.5
MIN_ACOUSTIC_WORD_DURATION_SECONDS = 0.005
MAX_QWEN_WORD_DURATION_SECONDS = 2.5
COHERE_TIMING_OUTLIER_SECONDS = 4.0


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qwen_clean(token: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", token)
        if ch == "'" or unicodedata.category(ch).startswith(("L", "N"))
    )


def norm(token: str) -> str:
    return qwen_clean(token).casefold()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def write_json(path: Path, data: object) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def safe_error(error: BaseException, limit: int = 240) -> str:
    message = str(error)
    message = re.sub(r"hf_[A-Za-z0-9]{20,}", "<redacted-token>", message)
    message = re.sub(r'''(?<!\w)(?:[A-Za-z]:[\\/]|/)[^\s"'<>]+''', "<local-path>", message)
    return f"{type(error).__name__}: {message[:limit]}"


def sec_to_vtt(value: float) -> str:
    ms = max(0, round(value * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_transcript_index(plain_path: Path, chunks_dir: Path) -> dict:
    plain_text = plain_path.read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in plain_text.splitlines() if line.strip()]
    display_words: list[dict] = []
    sentences: list[dict] = []
    for sentence_index, line in enumerate(lines):
        start = len(display_words)
        for token in tokens(line):
            cleaned = qwen_clean(token)
            if not cleaned:
                continue
            display_words.append({
                "index": len(display_words),
                "text": token,
                "clean": cleaned,
                "sentence_index": sentence_index,
            })
        sentences.append({
            "index": sentence_index,
            "text": line,
            "word_start": start,
            "word_end": len(display_words),
        })

    chunk_files = sorted(p for p in chunks_dir.glob("chunk_*.txt") if not p.name.startswith("._"))
    if not chunk_files:
        raise RuntimeError("No chunk_*.txt files found")
    chunks: list[dict] = []
    chunk_words: list[str] = []
    for chunk_index, path in enumerate(chunk_files):
        match = re.search(r"(\d+)", path.stem)
        file_index = int(match.group(1)) if match else chunk_index
        stripped = TAG_RE.sub(" ", path.read_text(encoding="utf-8-sig"))
        c_tokens = [t for t in tokens(stripped) if qwen_clean(t)]
        start = len(chunk_words)
        chunk_words.extend(c_tokens)
        chunks.append({
            "index": file_index,
            "file": path.name,
            "text": " ".join(stripped.split()),
            "word_start": start,
            "word_end": len(chunk_words),
            "word_count": len(c_tokens),
        })

    plain_tokens = [w["text"] for w in display_words]
    if [norm(x) for x in chunk_words] != [norm(x) for x in plain_tokens]:
        sm = difflib.SequenceMatcher(a=[norm(x) for x in plain_tokens], b=[norm(x) for x in chunk_words])
        raise RuntimeError(f"Chunk text does not reproduce the transcript (ratio={sm.ratio():.6f})")

    chunk_at = [None] * len(display_words)
    for c in chunks:
        for i in range(c["word_start"], c["word_end"]):
            chunk_at[i] = c["index"]
    for w, c in zip(display_words, chunk_at):
        w["chunk_index"] = c
    return {"text": plain_text, "words": display_words, "sentences": sentences, "chunks": chunks}


def _cohere_join_norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.casefold(), flags=re.UNICODE)


def load_cohere_word_timings(checkpoint_path: Path, exact_words: list[dict]) -> tuple[dict[int, tuple[float, float]], dict]:
    """Recover monotonic, coarse word anchors from the windowed Cohere STT pass."""
    report = {
        "available": False,
        "checkpoint": checkpoint_path.name,
        "window_count": 0,
        "provenance_words": 0,
        "matched_words": 0,
        "coverage": 0.0,
    }
    if not checkpoint_path.is_file():
        return {}, report
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        windows = checkpoint.get("windows", {})
        if not isinstance(windows, dict):
            return {}, report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return {}, report

    provenance: list[dict] = []
    last_provenance_end = 0.0
    for key in sorted(windows, key=lambda value: int(value)):
        entry = windows[key]
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", ""))
        right = TOKEN_RE.findall(text)
        if not right:
            continue
        left_norm = [_cohere_join_norm(item["text"]) for item in provenance]
        right_norm = [_cohere_join_norm(token) for token in right]
        overlap = 0
        upper = min(48, len(left_norm), len(right_norm))
        for size in range(upper, 1, -1):
            if left_norm[-size:] == right_norm[:size]:
                overlap = size
                break
        new_tokens = right[overlap:]
        try:
            raw_window_start = float(entry["start"])
            raw_window_end = max(raw_window_start, float(entry["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        # Speech windows overlap.  The newly appended text starts after the
        # already merged text, not necessarily at the overlapping window's
        # raw start; otherwise provenance time can jump backwards at every
        # deduplication boundary.
        window_start = max(raw_window_start, last_provenance_end)
        window_end = max(window_start + 0.01, raw_window_end)
        duration = max(0.01, window_end - window_start)
        count = max(1, len(new_tokens))
        for position, token in enumerate(new_tokens):
            start = window_start + duration * position / count
            end = window_start + duration * (position + 1) / count
            provenance.append({
                "text": token,
                "start": start,
                "end": max(start + 0.005, end),
                "window_index": int(key),
            })
        if new_tokens:
            last_provenance_end = provenance[-1]["end"]

    exact_norm = [norm(word["clean"]) for word in exact_words]
    provenance_norm = [norm(item["text"]) for item in provenance]
    matcher = difflib.SequenceMatcher(None, exact_norm, provenance_norm, autojunk=False)
    timings: dict[int, tuple[float, float]] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            exact_index = block.a + offset
            provenance_item = provenance[block.b + offset]
            timings[exact_index] = (float(provenance_item["start"]), float(provenance_item["end"]))
    report.update({
        "available": True,
        "window_count": len(windows),
        "provenance_words": len(provenance),
        "matched_words": len(timings),
        "coverage": len(timings) / max(1, len(exact_words)),
        "sequence_ratio": matcher.ratio(),
    })
    return timings, report


def build_alignment_segments(
    index: dict,
    anchor_bounds: dict[int, tuple[float, float]],
    max_words: int = MAX_ALIGNMENT_WORDS,
) -> list[dict]:
    """Split text into units that are short in both words and rough audio time.

    A word-only split is unsafe for a transcript that crosses long pauses:
    the forced aligner would receive a 50-second crop for text spread across
    several minutes and can assign every word to the crop boundary.  Rough
    ASR anchors are used only to choose safe crop boundaries; Qwen remains the
    source of the final word timings.
    """
    anchor_times = {
        exact_index: (bounds[0] + bounds[1]) / 2
        for exact_index, bounds in anchor_bounds.items()
    }
    anchor_indexes = sorted(anchor_times)

    def next_anchor_after(index: int, limit: int) -> tuple[int, float] | None:
        position = bisect.bisect_left(anchor_indexes, index)
        if position >= len(anchor_indexes):
            return None
        candidate = anchor_indexes[position]
        if candidate >= limit:
            return None
        return candidate, anchor_times[candidate]

    segments: list[dict] = []
    for chunk in index["chunks"]:
        cursor = chunk["word_start"]
        part = 0
        while cursor < chunk["word_end"]:
            start = cursor
            first_anchor = None
            last_anchor = None
            limit = cursor
            for word_index in range(cursor, chunk["word_end"]):
                if word_index - start >= max_words:
                    break
                anchor_time = anchor_times.get(word_index)
                if anchor_time is None:
                    future = next_anchor_after(word_index + 1, chunk["word_end"])
                    if (
                        last_anchor is not None
                        and future is not None
                        and word_index > start
                        and (
                            future[1] - last_anchor > MAX_ROUGH_GAP_SECONDS
                            or (
                                first_anchor is not None
                                and future[1] - first_anchor > MAX_ROUGH_SPAN_SECONDS
                            )
                        )
                    ):
                        break
                elif (
                    last_anchor is not None
                    and word_index > start
                    and (
                        anchor_time - last_anchor > MAX_ROUGH_GAP_SECONDS
                        or (
                            first_anchor is not None
                            and anchor_time - first_anchor > MAX_ROUGH_SPAN_SECONDS
                        )
                    )
                ):
                    break
                if anchor_time is not None:
                    if first_anchor is None:
                        first_anchor = anchor_time
                    last_anchor = anchor_time
                limit = word_index + 1
            if limit == start:
                limit = start + 1
            words = index["words"][cursor:limit]
            segments.append({
                "segment_id": f"c{chunk['index']:03d}-{part:03d}",
                "source_chunk": chunk["index"],
                "word_start": cursor,
                "word_end": limit,
                "word_count": len(words),
                "text": " ".join(word["text"] for word in words),
            })
            cursor = limit
            part += 1
    return segments


def save_rough(path: Path, result: object, meta: dict) -> dict:
    items = [
        {"text": it.text, "start": float(it.start_time), "end": float(it.end_time)}
        for it in result.time_stamps.items
    ]
    data = {"schema_version": 1, "meta": meta, "language": result.language, "text": result.text, "items": items}
    write_json(path, data)
    return data


def build_anchor_map(
    exact_words: list[dict], rough_items: list[dict], cohere_timings: dict[int, tuple[float, float]] | None = None
) -> tuple[dict[int, int], dict]:
    exact_norm = [norm(w["clean"]) for w in exact_words]
    candidates = []
    for rough_index, item in enumerate(rough_items):
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end - start > 0.005:
            candidates.append((rough_index, item))
    rough_norm = [norm(item["text"]) for _, item in candidates]
    sm = difflib.SequenceMatcher(None, exact_norm, rough_norm, autojunk=True)
    mapping: dict[int, int] = {}
    cohere_rejected = 0
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            exact_index = block.a + k
            rough_index, rough_item = candidates[block.b + k]
            if cohere_timings and exact_index in cohere_timings:
                rough_mid = (float(rough_item["start"]) + float(rough_item["end"])) / 2
                cohere_start, cohere_end = cohere_timings[exact_index]
                if (
                    rough_mid < cohere_start - COHERE_ANCHOR_PADDING_SECONDS
                    or rough_mid > cohere_end + COHERE_ANCHOR_PADDING_SECONDS
                ):
                    cohere_rejected += 1
                    continue
            mapping[exact_index] = rough_index
    report = {
        "exact_words": len(exact_norm),
        "rough_words": len(rough_norm),
        "rough_words_filtered": len(rough_items) - len(candidates),
        "matched_words": len(mapping),
        "coverage": len(mapping) / max(1, len(exact_norm)),
        "sequence_ratio": sm.ratio(),
        "cohere_rejected_anchors": cohere_rejected,
    }
    cohere_indexes = {
        index for index in (cohere_timings or {})
        if isinstance(index, int) and 0 <= index < len(exact_norm)
    }
    combined_indexes = set(mapping) | cohere_indexes
    report.update(
        {
            "cohere_anchor_count": len(cohere_indexes),
            "combined_anchor_count": len(combined_indexes),
            "combined_coverage": len(combined_indexes) / max(1, len(exact_norm)),
        }
    )
    if report["combined_coverage"] < 0.55:
        raise RuntimeError(f"Combined Qwen/Cohere anchor coverage too low: {report}")
    return mapping, report


def estimate_crop(
    index: dict,
    segment: dict,
    anchor_bounds: dict[int, tuple[float, float]],
    audio_duration: float,
) -> dict:
    exact_words = index["words"]
    anchor_exact = np.asarray(sorted(anchor_bounds), dtype=np.float64)
    if len(anchor_exact) < 2:
        raise RuntimeError("Not enough anchors to estimate sentence crops")
    seconds_per_word = audio_duration / max(1, len(exact_words))
    a, b = segment["word_start"], segment["word_end"]
    direct = [(i, anchor_bounds[i]) for i in range(a, b) if i in anchor_bounds]
    if direct:
        first_index, first_bounds = direct[0]
        last_index, last_bounds = direct[-1]
        first_anchor = float(first_bounds[0])
        last_anchor = float(last_bounds[1])
        first_mid = sum(first_bounds) / 2
        last_mid = sum(last_bounds) / 2
        # Extrapolate only across unmatched words inside this local segment.
        # Interpolating against the complete five-hour anchor stream is unsafe:
        # a long silence between two anchors can pull a short text crop to the
        # middle of that silence.
        first_mid -= max(0, first_index - a) * seconds_per_word
        last_mid += max(0, (b - 1) - last_index) * seconds_per_word
        est_start = min(first_mid - 0.5 * seconds_per_word, first_anchor)
        est_end = max(last_mid + 0.5 * seconds_per_word, last_anchor)
    else:
        anchor_indexes = [int(value) for value in anchor_exact]
        position = bisect.bisect_left(anchor_indexes, a)
        if position:
            left_index = anchor_indexes[position - 1]
            left_mid = sum(anchor_bounds[left_index]) / 2
            first_mid = left_mid + (a - left_index) * seconds_per_word
        elif position < len(anchor_indexes):
            right_index = anchor_indexes[position]
            right_mid = sum(anchor_bounds[right_index]) / 2
            first_mid = right_mid - (right_index - a) * seconds_per_word
        else:
            first_mid = 0.0
        last_mid = first_mid + max(0, b - a - 1) * seconds_per_word
        first_anchor = first_mid
        last_anchor = last_mid
        est_start = first_mid - 0.5 * seconds_per_word
        est_end = last_mid + 0.5 * seconds_per_word

    # When anchors are available, their observed span is a much better crop
    # estimate than a global seconds-per-word rate.  The latter can turn a
    # normal 30-second utterance into a 52-second crop with leading speech;
    # the forced aligner may then anchor the whole text at crop_start.
    estimated_span = max(0.0, last_mid - first_mid)
    expected = max(4.0, estimated_span + max(1.0, 2.0 * seconds_per_word))
    if est_end - est_start < expected:
        pad = (expected - (est_end - est_start)) / 2
        est_start -= pad
        est_end += pad
    est_start -= 1.35
    est_end += 1.35

    raw_crop_span = est_end - est_start
    crop_clipped = raw_crop_span > MAX_CROP_SECONDS
    if crop_clipped:
        center = (first_mid + last_mid) / 2
        width = min(MAX_CROP_SECONDS, max(12.0, expected + 2.7))
        est_start = center - width / 2
        est_end = center + width / 2
    crop_start = max(0.0, est_start)
    crop_end = min(audio_duration, est_end)
    if crop_end <= crop_start:
        raise RuntimeError(f"Invalid crop for segment {segment['segment_id']}")
    return {
        **segment,
        "crop_start": round(crop_start, 4),
        "crop_end": round(crop_end, 4),
        "rough_anchor_count": len(direct),
        "raw_crop_span_seconds": round(raw_crop_span, 4),
        "crop_clipped": crop_clipped,
    }


def estimate_crops(
    index: dict,
    segments: list[dict],
    anchor_bounds: dict[int, tuple[float, float]],
    audio_duration: float,
) -> list[dict]:
    return [estimate_crop(index, segment, anchor_bounds, audio_duration) for segment in segments]


def batch_crops(crops: list[dict], max_items: int, max_padded_seconds: float) -> list[list[dict]]:
    ordered = sorted(crops, key=lambda x: x["crop_end"] - x["crop_start"])
    batches: list[list[dict]] = []
    cur: list[dict] = []
    for item in ordered:
        trial = cur + [item]
        padded = max(x["crop_end"] - x["crop_start"] for x in trial) * len(trial)
        if cur and (len(trial) > max_items or padded > max_padded_seconds):
            batches.append(cur)
            cur = [item]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    return batches


def _item_text(item: object) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)


def _item_times(item: object) -> tuple[float, float] | None:
    start = getattr(item, "start_time", None)
    end = getattr(item, "end_time", None)
    if isinstance(item, dict):
        start = item.get("start_time", item.get("start"))
        end = item.get("end_time", item.get("end"))
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start_value) or not math.isfinite(end_value):
        return None
    return min(start_value, end_value), max(start_value, end_value)


def _observed_char_bounds(items: list[object]) -> tuple[str, list[tuple[float, float] | None]]:
    text_parts: list[str] = []
    bounds: list[tuple[float, float] | None] = []
    for item in items:
        value = norm(_item_text(item))
        times = _item_times(item)
        if not value or times is None:
            continue
        start, end = times
        duration = max(0.0, end - start)
        text_parts.append(value)
        for offset in range(len(value)):
            left = start + duration * offset / len(value)
            right = start + duration * (offset + 1) / len(value)
            bounds.append((left, right))
    return "".join(text_parts), bounds


def _linear_bounds(position: int, total: int, duration: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, max(0.01, duration)
    start = duration * position / total
    end = duration * (position + 1) / total
    return start, max(start + 0.005, end)


def align_result_to_words(expected: list[dict], observed: list[object], crop_duration: float) -> tuple[list[dict], dict] | None:
    """Map Qwen's timed subword stream back to the white-text word stream."""
    expected_values = [norm(word["clean"]) for word in expected]
    expected_text = "".join(expected_values)
    observed_text, observed_bounds = _observed_char_bounds(observed)
    if not expected_text or not observed_text or not observed_bounds:
        return None

    char_map: dict[int, tuple[float, float]] = {}
    exact = expected_text == observed_text
    if exact:
        for position, bounds in enumerate(observed_bounds):
            if bounds is not None:
                char_map[position] = bounds
    else:
        matcher = difflib.SequenceMatcher(None, expected_text, observed_text, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                observed_position = block.b + offset
                bounds = observed_bounds[observed_position]
                if bounds is not None:
                    char_map[block.a + offset] = bounds
        if not char_map:
            return None

    aligned: list[dict] = []
    cursor = 0
    matched_chars = 0
    for value in expected_values:
        start_char = cursor
        end_char = cursor + len(value)
        cursor = end_char
        matches = [char_map[position] for position in range(start_char, end_char) if position in char_map]
        matched_chars += len(matches)
        if matches:
            start = min(item[0] for item in matches)
            end = max(item[1] for item in matches)
        else:
            left_positions = [position for position in char_map if position < start_char]
            right_positions = [position for position in char_map if position >= end_char]
            if left_positions and right_positions:
                left = char_map[max(left_positions)][1]
                right = char_map[min(right_positions)][0]
                start = min(left, right)
                end = max(start + 0.005, right)
            else:
                start, end = _linear_bounds(start_char, max(1, len(expected_text)), crop_duration)
        aligned.append({"start": max(0.0, start), "end": min(crop_duration, max(start + 0.005, end))})
    return aligned, {
        "mode": "char_exact" if exact else "char_partial",
        "matched_chars": matched_chars,
        "expected_chars": len(expected_text),
        "observed_chars": len(observed_text),
    }


def _anchor_time_for_word(
    word_index: int,
    c: dict,
    anchor_bounds: dict[int, tuple[float, float]],
    anchor_indexes: list[int],
) -> tuple[float, float]:
    position = bisect.bisect_left(anchor_indexes, word_index)
    if word_index in anchor_bounds:
        start, end = anchor_bounds[word_index]
        return float(start), float(end)
    left_index = anchor_indexes[position - 1] if position else None
    right_index = anchor_indexes[position] if position < len(anchor_indexes) else None
    if left_index is not None and right_index is not None and right_index > left_index:
        left_start, left_end = anchor_bounds[left_index]
        right_start, right_end = anchor_bounds[right_index]
        ratio = (word_index - left_index) / (right_index - left_index)
        start = float(left_end) + (float(right_start) - float(left_end)) * ratio
        end = start + max(0.005, (float(right_start) - float(left_end)) / max(1, right_index - left_index))
        return start, end
    if left_index is not None:
        left_start, left_end = anchor_bounds[left_index]
        width = max(0.005, float(left_end) - float(left_start))
        delta = max(0, word_index - left_index) * width
        return float(left_end) + delta, float(left_end) + delta + width
    if right_index is not None:
        right_start, right_end = anchor_bounds[right_index]
        width = max(0.005, float(right_end) - float(right_start))
        delta = max(0, right_index - word_index) * width
        return max(0.0, float(right_start) - delta - width), max(0.005, float(right_start) - delta)
    duration = c["crop_end"] - c["crop_start"]
    return c["crop_start"], c["crop_start"] + max(0.005, duration)


def rough_fallback_items(
    c: dict, index: dict, anchor_bounds: dict[int, tuple[float, float]], anchor_indexes: list[int]
) -> list[dict]:
    items: list[dict] = []
    duration = max(0.005, c["crop_end"] - c["crop_start"])
    for word_index in range(c["word_start"], c["word_end"]):
        start, end = _anchor_time_for_word(word_index, c, anchor_bounds, anchor_indexes)
        start = min(duration, max(0.0, start - c["crop_start"]))
        end = min(duration, max(start + 0.005, end - c["crop_start"]))
        items.append({"start": start, "end": end})
    return items


def _is_oom_error(error: Exception) -> bool:
    message = str(error).casefold()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message or "cuda error" in message and "memory" in message


def _is_degenerate_alignment(relative_items: list[dict]) -> bool:
    """Reject the forced aligner's collapsed output instead of hiding it later."""
    if len(relative_items) <= 1:
        return False
    spans = []
    for item in relative_items:
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            return True
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            return True
        spans.append(max(0.0, end - start))
    collapsed = sum(span <= 0.006 for span in spans)
    total_span = max(float(item["end"]) for item in relative_items) - min(float(item["start"]) for item in relative_items)
    return total_span <= 0.05 or (len(spans) >= 6 and collapsed / len(spans) >= 0.9)


def run_exact_alignment(
    aligner: object,
    wav: np.ndarray,
    index: dict,
    crops: list[dict],
    checkpoint: Path,
    checkpoint_meta: dict,
    anchor_bounds: dict[int, tuple[float, float]],
    max_items: int,
    max_padded_seconds: float,
) -> tuple[dict[str, dict], dict]:
    completed: dict[str, dict] = {}
    recoverable_errors: list[dict[str, Any]] = []

    def valid_segment(segment_id: str, value: object) -> bool:
        if not isinstance(value, dict) or value.get("segment_id") != segment_id:
            return False
        items = value.get("items")
        if not isinstance(items, list) or not items:
            return False
        indexes: list[int] = []
        for item in items:
            if not isinstance(item, dict):
                return False
            try:
                word_index = int(item["word_index"])
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                return False
            indexes.append(word_index)
        return len(indexes) == len(set(indexes))

    if checkpoint.is_file():
        try:
            old = json.loads(checkpoint.read_text(encoding="utf-8"))
            if old.get("meta") == checkpoint_meta and isinstance(old.get("segments"), dict):
                completed = {
                    str(key): value
                    for key, value in old["segments"].items()
                    if valid_segment(str(key), value)
                }
                old_errors = old.get("recoverable_errors", [])
                if isinstance(old_errors, list):
                    recoverable_errors = [item for item in old_errors if isinstance(item, dict)]
                discarded = len(old["segments"]) - len(completed)
                if discarded:
                    log(f"Discarded {discarded} invalid alignment checkpoint segments")
                log(f"Resuming {len(completed)} validated alignment segments")
            else:
                log("Ignoring stale or incompatible alignment checkpoint")
        except Exception:
            log("Ignoring unreadable alignment checkpoint")
    resumed_segment_ids = set(completed)

    def crop_complete(crop: dict) -> bool:
        segment_id = str(crop["segment_id"])
        indexes: list[int] = []
        for key, value in completed.items():
            if key == segment_id or key.startswith(segment_id + "."):
                indexes.extend(int(item["word_index"]) for item in value["items"])
        expected = list(range(int(crop["word_start"]), int(crop["word_end"])))
        return sorted(indexes) == expected

    def has_descendants(crop: dict) -> bool:
        prefix = str(crop["segment_id"]) + "."
        return any(key.startswith(prefix) for key in completed)

    pending = [c for c in crops if not crop_complete(c)]
    batches = batch_crops(pending, max_items=max_items, max_padded_seconds=max_padded_seconds)
    log(f"Exact pass: {len(pending)} segments in {len(batches)} duration-bucketed batches")
    anchor_indexes = sorted(anchor_bounds)
    stats: dict[str, Any] = {
        "modes": {},
        "fallback_segments": 0,
        "recoverable_errors": recoverable_errors,
    }

    def write_checkpoint() -> None:
        write_json(
            checkpoint,
            {
                "meta": checkpoint_meta,
                "segments": completed,
                "recoverable_errors": recoverable_errors,
            },
        )

    def split_segment(c: dict) -> list[dict]:
        count = c["word_end"] - c["word_start"]
        if count <= 1:
            return []
        midpoint = c["word_start"] + count // 2
        children: list[dict] = []
        for suffix, (start, end) in enumerate(((c["word_start"], midpoint), (midpoint, c["word_end"]))):
            child = {
                "segment_id": f"{c['segment_id']}.{suffix}",
                "source_chunk": c["source_chunk"],
                "word_start": start,
                "word_end": end,
                "word_count": end - start,
                "text": " ".join(word["text"] for word in index["words"][start:end]),
            }
            children.append(estimate_crop(index, child, anchor_bounds, len(wav) / SR))
        return children

    def store(c: dict, relative_items: list[dict], mode: str, details: dict) -> None:
        output_items = []
        for word_index, timing in zip(range(c["word_start"], c["word_end"]), relative_items):
            output_items.append({
                "text": index["words"][word_index]["text"],
                "clean": index["words"][word_index]["clean"],
                "word_index": word_index,
                "start": round(c["crop_start"] + timing["start"], 3),
                "end": round(c["crop_start"] + timing["end"], 3),
            })
        segment_id = str(c["segment_id"])
        for stale in [key for key in completed if key.startswith(segment_id + ".")]:
            del completed[stale]
        completed[segment_id] = {
            "segment_id": c["segment_id"],
            "crop_start": c["crop_start"],
            "crop_end": c["crop_end"],
            "rough_anchor_count": c["rough_anchor_count"],
            "alignment_mode": mode,
            "details": details,
            "items": output_items,
        }
        stats["modes"][mode] = stats["modes"].get(mode, 0) + 1
        if mode.startswith("rough_"):
            stats["fallback_segments"] += 1
        write_checkpoint()

    def process(batch: list[dict]) -> None:
        batch = [crop for crop in batch if not crop_complete(crop)]
        if not batch:
            return
        partial = [crop for crop in batch if has_descendants(crop)]
        if partial:
            untouched = [crop for crop in batch if crop not in partial]
            process(untouched)
            for crop in partial:
                children = split_segment(crop)
                if not children:
                    raise RuntimeError(f"partial split checkpoint cannot be resumed: {crop['segment_id']}")
                for child in children:
                    process([child])
            return
        clipped = [c for c in batch if c.get("crop_clipped")]
        if clipped:
            safe = [c for c in batch if not c.get("crop_clipped")]
            if safe:
                process(safe)
            for c in clipped:
                children = split_segment(c)
                if children:
                    log(f"Splitting unsafe audio crop {c['segment_id']} ({c['raw_crop_span_seconds']:.1f}s)")
                    recoverable_errors.append({
                        "kind": "UnsafeCrop",
                        "segment_id": c["segment_id"],
                        "raw_crop_span_seconds": c["raw_crop_span_seconds"],
                    })
                    for child in children:
                        process([child])
                    continue
                log(f"Rough fallback for irreducibly unsafe crop {c['segment_id']}")
                store(c, rough_fallback_items(c, index, anchor_bounds, anchor_indexes), "rough_unsafe_crop_fallback", {
                    "raw_crop_span_seconds": c["raw_crop_span_seconds"],
                })
            return
        audios = []
        texts = []
        for c in batch:
            s = max(0, int(math.floor(c["crop_start"] * SR)))
            e = min(len(wav), int(math.ceil(c["crop_end"] * SR)))
            audios.append((np.ascontiguousarray(wav[s:e]), SR))
            texts.append(c["text"])
        try:
            results = list(aligner.align(audio=audios, text=texts, language=["English"] * len(batch)))
            if len(results) != len(batch):
                raise RuntimeError(f"aligner returned {len(results)} results for {len(batch)} inputs")
        except Exception as error:
            torch.cuda.empty_cache()
            if len(batch) > 1:
                mid = len(batch) // 2
                log(f"Alignment retry after {_is_oom_error(error) and 'OOM' or 'error'}: {len(batch)} -> {mid}+{len(batch)-mid}")
                recoverable_errors.append(
                    {"kind": type(error).__name__, "message": safe_error(error), "batch_size": len(batch)}
                )
                process(batch[:mid])
                process(batch[mid:])
                return
            children = split_segment(batch[0])
            if children:
                log(f"Alignment retry with smaller segment after {type(error).__name__}: {batch[0]['segment_id']}")
                recoverable_errors.append(
                    {
                        "kind": type(error).__name__,
                        "message": safe_error(error),
                        "segment_id": batch[0]["segment_id"],
                    }
                )
                for child in children:
                    process([child])
                return
            log(f"Alignment fallback for {batch[0]['segment_id']} after {type(error).__name__}")
            recoverable_errors.append(
                {
                    "kind": type(error).__name__,
                    "message": safe_error(error),
                    "segment_id": batch[0]["segment_id"],
                }
            )
            store(batch[0], rough_fallback_items(batch[0], index, anchor_bounds, anchor_indexes), "rough_error_fallback", {"error": type(error).__name__})
            return

        for c, result in zip(batch, results):
            expected = index["words"][c["word_start"]:c["word_end"]]
            aligned = align_result_to_words(expected, list(result.items), c["crop_end"] - c["crop_start"])
            if aligned is None:
                store(c, rough_fallback_items(c, index, anchor_bounds, anchor_indexes), "rough_token_fallback", {"reason": "no_usable_character_match"})
            else:
                relative_items, details = aligned
                if _is_degenerate_alignment(relative_items):
                    children = split_segment(c)
                    if children:
                        log(f"Splitting degenerate alignment result {c['segment_id']}")
                        recoverable_errors.append({
                            "kind": "DegenerateAlignment",
                            "segment_id": c["segment_id"],
                            "details": details,
                        })
                        for child in children:
                            process([child])
                    else:
                        store(c, rough_fallback_items(c, index, anchor_bounds, anchor_indexes), "rough_degenerate_fallback", {
                            "reason": "degenerate_forced_alignment",
                            **details,
                        })
                else:
                    store(c, relative_items, details["mode"], details)

    for batch_no, batch in enumerate(batches, 1):
        process(batch)
        if batch_no == 1 or batch_no % 20 == 0 or batch_no == len(batches):
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            log(f"Exact progress {len(completed)} segments; GPU {allocated:.2f} GiB, peak {peak:.2f} GiB")
    # Derive totals from every persisted segment, including work loaded at the
    # beginning of a resumed run.  The old implementation counted only work
    # performed in the current process.
    all_modes: dict[str, int] = {}
    fallback_segments = 0
    for item in completed.values():
        mode = str(item.get("alignment_mode", "unknown"))
        all_modes[mode] = all_modes.get(mode, 0) + 1
        if mode.startswith("rough_"):
            fallback_segments += 1
    stats.update(
        {
            "modes": all_modes,
            "fallback_segments": fallback_segments,
            "segment_count": len(completed),
            "resumed_segments": len(set(completed) & resumed_segment_ids),
            "new_segments": len(set(completed) - resumed_segment_ids),
        }
    )
    return completed, stats


def create_mfa_corpus(wav: np.ndarray, index: dict, qwen_words: list[dict], output_dir: Path) -> list[dict]:
    corpus = output_dir / "mfa_corpus" / "speaker1"
    corpus.mkdir(parents=True, exist_ok=True)
    for stale in (*corpus.glob("utt_*.wav"), *corpus.glob("utt_*.lab")):
        stale.unlink()
    manifest: list[dict] = []
    for sentence in index["sentences"]:
        sid = sentence["index"]
        items = qwen_words[sentence["word_start"]:sentence["word_end"]]
        start = max(0.0, items[0]["start"] - 0.35)
        end = min(len(wav) / SR, items[-1]["end"] + 0.35)
        utt_id = f"utt_{sid:05d}"
        wav_path = corpus / f"{utt_id}.wav"
        lab_path = corpus / f"{utt_id}.lab"
        s = max(0, int(math.floor(start * SR)))
        e = min(len(wav), int(math.ceil(end * SR)))
        temporary_wav = wav_path.with_name(f".{wav_path.stem}.{os.getpid()}.tmp.wav")
        temporary_wav.unlink(missing_ok=True)
        try:
            sf.write(temporary_wav, wav[s:e], SR, subtype="PCM_16")
            info = sf.info(str(temporary_wav))
            if int(info.samplerate) != SR or info.channels != 1 or info.frames != e - s:
                raise RuntimeError(f"invalid temporary MFA audio for {utt_id}")
            with temporary_wav.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_wav, wav_path)
        finally:
            temporary_wav.unlink(missing_ok=True)
        write_text(lab_path, sentence["text"] + "\n")
        manifest.append({
            "utterance_id": utt_id,
            "sentence_index": sid,
            "word_start": sentence["word_start"],
            "word_end": sentence["word_end"],
            "offset": round(start, 6),
            "duration": round(end - start, 6),
        })
    write_json(output_dir / "mfa_manifest.json", manifest)
    return manifest


def calibrate_segment_timings(
    aligned: dict[str, dict],
    rough_items: list[dict],
    mapping: dict[int, int],
    cohere_timings: dict[int, tuple[float, float]],
) -> tuple[dict[int, dict], set[int], dict]:
    """Apply only local, robust corrections; never shift all later chunks cumulatively."""
    timed_by_index: dict[int, dict] = {}
    rough_guarded: set[int] = set()
    segment_reports = []
    shifts = []
    for result in aligned.values():
        items = result["items"]
        cohere_deltas = []
        for item in items:
            word_index = int(item["word_index"])
            if word_index not in cohere_timings:
                continue
            cohere_start, cohere_end = cohere_timings[word_index]
            cohere_mid = (cohere_start + cohere_end) / 2
            qwen_mid = (float(item["start"]) + float(item["end"])) / 2
            cohere_deltas.append(cohere_mid - qwen_mid)
        anchor_kind = "cohere_window" if cohere_deltas else "qwen_rough"
        deltas = cohere_deltas
        if not deltas:
            for item in items:
                word_index = int(item["word_index"])
                if word_index not in mapping:
                    continue
                rough_item = rough_items[mapping[word_index]]
                rough_mid = (float(rough_item["start"]) + float(rough_item["end"])) / 2
                qwen_mid = (float(item["start"]) + float(item["end"])) / 2
                deltas.append(rough_mid - qwen_mid)
        shift = statistics.median(deltas) if deltas else 0.0
        shifts.append(float(shift))
        guarded_in_segment = 0
        for item in items:
            word_index = int(item["word_index"])
            start = float(item["start"]) + shift
            end = max(start + 0.005, float(item["end"]) + shift)
            timed_by_index[word_index] = {
                "start": round(start, 3),
                "end": round(end, 3),
            }
        segment_reports.append({
            "segment_id": result["segment_id"],
            "word_count": len(items),
            "anchor_kind": anchor_kind,
            "anchor_count": len(deltas),
            "shift_seconds": round(shift, 3),
            "rough_guarded_words": guarded_in_segment,
        })
    report = {
        "segment_count": len(segment_reports),
        "segments_with_nontrivial_shift": sum(abs(value) > 0.25 for value in shifts),
        "max_abs_shift_seconds": round(max((abs(value) for value in shifts), default=0.0), 3),
        "rough_guarded_words": len(rough_guarded),
        "segment_reports": segment_reports,
    }
    return timed_by_index, rough_guarded, report


def _isotonic_centers(values: list[float], minimum_step: float) -> tuple[list[float], int]:
    """Project word centers onto a forward-only timeline with a minimum step."""
    adjusted = [float(value) - index * minimum_step for index, value in enumerate(values)]
    blocks: list[list[float]] = []
    for index, value in enumerate(adjusted):
        blocks.append([float(index), float(index), value, 1.0])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left[2] / left[3] <= right[2] / right[3] + 1e-12:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([
                left[0],
                right[1],
                left[2] + right[2],
                left[3] + right[3],
            ])
    projected = [0.0] * len(values)
    for start, end, total, count in blocks:
        mean = total / count
        for index in range(int(start), int(end) + 1):
            projected[index] = mean
    return [projected[index] + index * minimum_step for index in range(len(values))], len(blocks)


def cleanup_acoustic_timings(
    words: list[dict],
    cohere_timings: dict[int, tuple[float, float]],
    audio_duration: float,
) -> tuple[dict, set[int]]:
    """Remove local forced-aligner collapses before MFA and player rendering.

    Qwen usually supplies the useful fine-grained boundaries, but a forced
    aligner can occasionally put a long span or an outlier several seconds
    away from its surrounding words.  Cohere's window provenance is coarse,
    yet it is a reliable fallback for those isolated failures.  The final
    center projection keeps every acoustic interval in transcript order;
    this is deliberately separate from the player focus spans.
    """
    raw_centers: list[float] = []
    durations: list[float] = []
    cohere_guarded: set[int] = set()
    long_duration_words = 0
    cohere_outlier_words = 0

    for word in words:
        word_index = int(word["index"])
        try:
            raw_start = float(word["start"])
            raw_end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            raw_start, raw_end = 0.0, MIN_ACOUSTIC_WORD_DURATION_SECONDS
        if not math.isfinite(raw_start) or not math.isfinite(raw_end):
            raw_start, raw_end = 0.0, MIN_ACOUSTIC_WORD_DURATION_SECONDS
        raw_duration = max(0.0, raw_end - raw_start)
        center = (raw_start + raw_end) / 2
        cohere_bounds = cohere_timings.get(word_index)
        reasons = []
        if raw_duration > MAX_QWEN_WORD_DURATION_SECONDS:
            long_duration_words += 1
            reasons.append("long_duration")
        if cohere_bounds:
            cohere_center = (float(cohere_bounds[0]) + float(cohere_bounds[1])) / 2
            if abs(center - cohere_center) > COHERE_TIMING_OUTLIER_SECONDS:
                cohere_outlier_words += 1
                reasons.append("cohere_outlier")
        if reasons and cohere_bounds:
            cohere_start, cohere_end = map(float, cohere_bounds)
            center = (cohere_start + cohere_end) / 2
            raw_duration = max(
                MIN_ACOUSTIC_WORD_DURATION_SECONDS,
                cohere_end - cohere_start,
            )
            cohere_guarded.add(word_index)
        raw_centers.append(center)
        durations.append(min(
            MAX_QWEN_WORD_DURATION_SECONDS,
            max(MIN_ACOUSTIC_WORD_DURATION_SECONDS, raw_duration),
        ))

    centers, block_count = _isotonic_centers(
        raw_centers, MIN_ACOUSTIC_WORD_DURATION_SECONDS
    )
    centers = [min(audio_duration, max(0.0, center)) for center in centers]
    for index in range(1, len(centers)):
        centers[index] = max(centers[index], centers[index - 1] + MIN_ACOUSTIC_WORD_DURATION_SECONDS)
    if centers and centers[-1] > audio_duration - MIN_ACOUSTIC_WORD_DURATION_SECONDS / 2:
        centers[-1] = audio_duration - MIN_ACOUSTIC_WORD_DURATION_SECONDS / 2
    for index in range(len(centers) - 2, -1, -1):
        centers[index] = min(centers[index], centers[index + 1] - MIN_ACOUSTIC_WORD_DURATION_SECONDS)
    if centers and centers[0] < MIN_ACOUSTIC_WORD_DURATION_SECONDS / 2:
        raise RuntimeError("Not enough timeline space for acoustic word timings")

    center_shifts = []
    for index, word in enumerate(words):
        center_shifts.append(abs(centers[index] - raw_centers[index]))
        left_limit = 0.0 if index == 0 else (centers[index - 1] + centers[index]) / 2
        right_limit = audio_duration if index == len(words) - 1 else (centers[index] + centers[index + 1]) / 2
        desired = durations[index]
        start = max(left_limit, centers[index] - desired / 2)
        end = min(right_limit, centers[index] + desired / 2)
        if end - start < MIN_ACOUSTIC_WORD_DURATION_SECONDS:
            start = max(left_limit, centers[index] - MIN_ACOUSTIC_WORD_DURATION_SECONDS / 2)
            end = min(right_limit, start + MIN_ACOUSTIC_WORD_DURATION_SECONDS)
            start = max(left_limit, end - MIN_ACOUSTIC_WORD_DURATION_SECONDS)
        # Keep enough precision that a 5 ms projected gap is not inverted by
        # decimal rounding at adjacent boundaries.
        word["start"] = round(start, 9)
        word["end"] = round(max(start + MIN_ACOUSTIC_WORD_DURATION_SECONDS, end), 9)

    acoustic_monotonic = all(
        words[index]["start"] <= words[index]["end"] <= words[index + 1]["start"]
        for index in range(len(words) - 1)
    ) and (not words or words[-1]["start"] <= words[-1]["end"])
    if not acoustic_monotonic:
        raise RuntimeError("Acoustic timing cleanup did not produce a monotonic timeline")
    return {
        "algorithm": "cohere_guarded_isotonic_centers_v1",
        "cohere_guarded_words": len(cohere_guarded),
        "long_duration_words": long_duration_words,
        "cohere_outlier_words": cohere_outlier_words,
        "isotonic_block_count": block_count,
        "center_adjusted_words": sum(value > 0.001 for value in center_shifts),
        "max_center_shift_seconds": round(max(center_shifts, default=0.0), 3),
        "p95_center_shift_seconds": round(
            sorted(center_shifts)[int(0.95 * (len(center_shifts) - 1))]
            if center_shifts else 0.0,
            3,
        ),
        "minimum_acoustic_word_duration_seconds": MIN_ACOUSTIC_WORD_DURATION_SECONDS,
        "maximum_acoustic_word_duration_seconds": MAX_QWEN_WORD_DURATION_SECONDS,
        "acoustic_monotonic": True,
    }, cohere_guarded


def export_qwen_outputs(
    index: dict,
    aligned: dict[str, dict],
    output_dir: Path,
    metadata: dict,
    mapping: dict[int, int],
    rough_items: list[dict],
    cohere_timings: dict[int, tuple[float, float]],
) -> list[dict]:
    timed_by_index, rough_guarded, calibration = calibrate_segment_timings(
        aligned, rough_items, mapping, cohere_timings
    )
    if len(timed_by_index) != len(index["words"]):
        raise RuntimeError(f"Exact alignment coverage mismatch: {len(timed_by_index)}/{len(index['words'])}")
    words = []
    for base in index["words"]:
        item = timed_by_index[base["index"]]
        source = "qwen3_forced_aligner_rough_guard" if base["index"] in rough_guarded else "qwen3_forced_aligner"
        words.append({**base, "start": item["start"], "end": item["end"], "source": source})
    cleanup_report, cohere_guarded = cleanup_acoustic_timings(
        words,
        cohere_timings,
        float(metadata["audio_duration_seconds"]),
    )
    for word in words:
        if int(word["index"]) in cohere_guarded:
            word["source"] = "cohere_window_timing_guard"
    metadata["qwen_segment_timing_calibration"] = calibration
    metadata["qwen_acoustic_timing_cleanup"] = cleanup_report
    sentences = []
    for sentence in index["sentences"]:
        a, b = sentence["word_start"], sentence["word_end"]
        sentences.append({
            **sentence,
            "start": words[a]["start"],
            "end": words[b - 1]["end"],
        })
    write_json(
        output_dir / "qwen_word_timings.json",
        {
            "schema_version": 1,
            "metadata": metadata,
            "words": words,
            "sentences": sentences,
            "chunks": index["chunks"],
        },
    )
    cues = ["WEBVTT", ""]
    for i, s in enumerate(sentences, 1):
        cues.extend([str(i), f"{sec_to_vtt(s['start'])} --> {sec_to_vtt(s['end'])}", s["text"], ""])
    write_text(output_dir / "qwen_sentences.vtt", "\n".join(cues))
    return words


def main() -> None:
    global librosa, np, sf, torch, Qwen3ASRModel
    import librosa as _librosa
    import numpy as _np
    import soundfile as _sf
    import torch as _torch
    from transformers.utils import logging as transformers_logging
    from qwen_asr import Qwen3ASRModel as _Qwen3ASRModel

    transformers_logging.disable_progress_bar()

    librosa, np, sf, torch, Qwen3ASRModel = _librosa, _np, _sf, _torch, _Qwen3ASRModel
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--cohere-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--asr-revision", required=True)
    parser.add_argument("--aligner-model", type=Path, required=True)
    parser.add_argument("--aligner-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batch-items", type=int, default=8)
    parser.add_argument("--max-padded-seconds", type=float, default=100.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, path in (
        ("audio", args.audio),
        ("transcript", args.transcript),
        ("Cohere checkpoint", args.cohere_checkpoint),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")
    if not args.chunks_dir.is_dir():
        raise RuntimeError(f"chunks directory is missing: {args.chunks_dir}")
    index = build_transcript_index(args.transcript, args.chunks_dir)
    log(
        f"Verified transcript: {len(index['words'])} words, "
        f"{len(index['sentences'])} sentences, {len(index['chunks'])} chunks"
    )

    started = time.time()
    rough_path = args.output_dir / "rough_asr_anchors_full.json"
    rough_meta = {
        "audio_sha256": sha256(args.audio),
        "asr_model_revision": args.asr_revision,
        "aligner_model_revision": args.aligner_revision,
        "device": args.device,
    }
    torch.cuda.reset_peak_memory_stats()
    rough = None
    if rough_path.is_file():
        try:
            candidate = json.loads(rough_path.read_text(encoding="utf-8"))
            if candidate.get("schema_version") == 1 and candidate.get("meta") == rough_meta and candidate.get("items"):
                rough = candidate
        except Exception:
            pass
    if rough is None:
        log("Loading Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B")
        asr = Qwen3ASRModel.from_pretrained(
            str(args.asr_model),
            forced_aligner=str(args.aligner_model),
            forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": args.device, "low_cpu_mem_usage": True},
            max_inference_batch_size=2,
            max_new_tokens=1024,
            dtype=torch.bfloat16,
            device_map=args.device,
            low_cpu_mem_usage=True,
        )
        try:
            result = asr.transcribe(audio=str(args.audio), language="English", return_time_stamps=True)[0]
        except torch.cuda.OutOfMemoryError:
            log("Rough-pass VRAM guard: retrying with batch size 1")
            torch.cuda.empty_cache()
            asr.max_inference_batch_size = 1
            result = asr.transcribe(audio=str(args.audio), language="English", return_time_stamps=True)[0]
        rough = save_rough(rough_path, result, rough_meta)
        aligner = asr.forced_aligner
        asr.forced_aligner = None
        del asr, result
        gc.collect()
        torch.cuda.empty_cache()
    else:
        log("Using validated rough ASR checkpoint")
        from qwen_asr import Qwen3ForcedAligner
        aligner = Qwen3ForcedAligner.from_pretrained(
            str(args.aligner_model), dtype=torch.bfloat16, device_map=args.device, low_cpu_mem_usage=True
        )

    cohere_timings, cohere_report = load_cohere_word_timings(args.cohere_checkpoint, index["words"])
    if cohere_report["available"]:
        log(
            f"Cohere provenance anchors {cohere_report['coverage']:.2%} "
            f"({cohere_report['matched_words']}/{len(index['words'])})"
        )
    else:
        log("Cohere provenance checkpoint unavailable; using Qwen rough anchors only")
    mapping, anchor_report = build_anchor_map(index["words"], rough["items"], cohere_timings)
    log(
        f"Rough anchor coverage {anchor_report['coverage']:.2%}; "
        f"combined {anchor_report['combined_coverage']:.2%} "
        f"({anchor_report['combined_anchor_count']}/{anchor_report['exact_words']})"
    )
    anchor_bounds = dict(cohere_timings)
    for exact_index, rough_index in mapping.items():
        rough_item = rough["items"][rough_index]
        anchor_bounds.setdefault(exact_index, (float(rough_item["start"]), float(rough_item["end"])))
    if len(anchor_bounds) < 2:
        raise RuntimeError("Not enough reliable timing anchors for exact alignment")
    log("Loading full audio once for the exact transcript pass")
    wav, _ = librosa.load(args.audio, sr=SR, mono=True, dtype=np.float32)
    duration = len(wav) / SR
    segments = build_alignment_segments(index, anchor_bounds)
    crops = estimate_crops(index, segments, anchor_bounds, duration)
    checkpoint_meta = {
        "schema_version": SEGMENTATION_SCHEMA_VERSION,
        "alignment_algorithm": ALIGNMENT_ALGORITHM,
        "audio_sha256": sha256(args.audio),
        "transcript_sha256": sha256(args.transcript),
        "word_count": len(index["words"]),
        "segment_count": len(segments),
        "max_rough_span_seconds": MAX_ROUGH_SPAN_SECONDS,
        "max_rough_gap_seconds": MAX_ROUGH_GAP_SECONDS,
        "cohere_anchor_count": len(cohere_timings),
        "cohere_checkpoint_sha256": sha256(args.cohere_checkpoint),
        "rough_anchors_sha256": sha256(rough_path),
        "reliable_rough_anchor_count": len(mapping),
        "asr_model_revision": args.asr_revision,
        "aligner_model_revision": args.aligner_revision,
        "device": args.device,
        "max_batch_items": args.max_batch_items,
        "max_padded_seconds": args.max_padded_seconds,
    }
    aligned, alignment_stats = run_exact_alignment(
        aligner, wav, index, crops, args.output_dir / "qwen_alignment_checkpoint_v2.json",
        checkpoint_meta, anchor_bounds,
        max_items=args.max_batch_items, max_padded_seconds=args.max_padded_seconds,
    )
    metadata = {
        "alignment_algorithm": ALIGNMENT_ALGORITHM,
        "audio_file": args.audio.name,
        "audio_sha256": sha256(args.audio),
        "transcript_file": args.transcript.name,
        "transcript_sha256": sha256(args.transcript),
        "audio_duration_seconds": round(duration, 6),
        "word_count": len(index["words"]),
        "sentence_count": len(index["sentences"]),
        "tts_chunk_count": len(index["chunks"]),
        "alignment_segment_count": len(segments),
        "segmentation_schema_version": SEGMENTATION_SCHEMA_VERSION,
        "max_rough_span_seconds": MAX_ROUGH_SPAN_SECONDS,
        "max_rough_gap_seconds": MAX_ROUGH_GAP_SECONDS,
        "cohere_anchor_report": cohere_report,
        "rough_anchor_report": anchor_report,
        "alignment_stats": alignment_stats,
        "qwen_models": {
            "asr": {"id": "Qwen/Qwen3-ASR-0.6B", "revision": args.asr_revision},
            "forced_aligner": {
                "id": "Qwen/Qwen3-ForcedAligner-0.6B",
                "revision": args.aligner_revision,
            },
        },
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    qwen_words = export_qwen_outputs(
        index, aligned, args.output_dir, metadata, mapping, rough["items"], cohere_timings
    )
    create_mfa_corpus(wav, index, qwen_words, args.output_dir)
    write_json(args.output_dir / "qwen_report.json", metadata)
    log(f"Qwen stage complete in {(time.time()-started)/60:.1f} min; MFA corpus ready")


if __name__ == "__main__":
    main()
