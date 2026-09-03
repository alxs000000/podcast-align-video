#!/usr/bin/env python3
"""Merge MFA boundaries with Qwen timings.

MFA is a refinement layer here, not the sole source of timings.  When MFA
cannot produce a TextGrid for an utterance, the Qwen ForcedAligner timings
are retained and the output metadata records the fallback explicitly.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import statistics
import unicodedata
from pathlib import Path

NON_TERMINAL_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
    "no",
    "inc",
    "corp",
    "ltd",
    "co",
    "fig",
    "approx",
}


def norm(token: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", token).casefold()
        if ch == "'" or unicodedata.category(ch).startswith(("L", "N"))
    )


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


def safe_error(error: BaseException, path: Path, limit: int = 240) -> str:
    message = str(error).replace(str(path), path.name)
    message = re.sub(r"hf_[A-Za-z0-9]{20,}", "<redacted-token>", message)
    message = re.sub(r'''(?<!\w)(?:[A-Za-z]:[\\/]|/)[^\s"'<>]+''', "<local-path>", message)
    return f"{type(error).__name__}: {message[:limit]}"


def sec_to_vtt(value: float) -> str:
    ms = max(0, round(value * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def entry_values(entry: object) -> tuple[float, float, str]:
    if hasattr(entry, "start"):
        return float(entry.start), float(entry.end), str(entry.label)
    return float(entry[0]), float(entry[1]), str(entry[2])


def read_words(path: Path) -> list[tuple[float, float, str]]:
    from praatio import textgrid

    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    names = list(tg.tierNames)
    selected = next((n for n in names if n.casefold() == "words"), None)
    if selected is None:
        selected = next((n for n in names if "word" in n.casefold()), None)
    if selected is None:
        raise RuntimeError(f"No word tier in {path.name}: {names}")
    tier = tg.getTier(selected)
    return [entry_values(x) for x in tier.entries if entry_values(x)[2].strip()]


def resolve_conflicts(words: list[dict], qwen_words: list[dict]) -> dict:
    """Trim feasible adjacent overlaps; use Qwen only for irreconcilable spans."""
    fallback_indexes = set()
    boundary_adjustments = 0
    for _ in range(len(words)):
        bad = set()
        for i, w in enumerate(words):
            if w["end"] < w["start"]:
                bad.add(i)
        for i in bad:
            words[i]["start"] = qwen_words[i]["start"]
            words[i]["end"] = qwen_words[i]["end"]
            words[i]["source"] = "qwen3_forced_aligner_conflict_fallback"
        fallback_indexes.update(bad)

        impossible = set()
        adjusted_this_pass = 0
        for i in range(len(words) - 1):
            if words[i]["end"] > words[i + 1]["start"]:
                left, right = words[i], words[i + 1]
                if left["start"] <= right["end"]:
                    boundary = (left["end"] + right["start"]) / 2
                    boundary = round(min(max(boundary, left["start"]), right["end"]), 3)
                    left["end"] = boundary
                    right["start"] = boundary
                    if left["source"].startswith("mfa_"):
                        left["source"] = "mfa_3.4.1_fine_tuned_boundary_adjusted"
                    if right["source"].startswith("mfa_"):
                        right["source"] = "mfa_3.4.1_fine_tuned_boundary_adjusted"
                    adjusted_this_pass += 1
                else:
                    impossible.update((i, i + 1))
        for i in impossible:
            words[i]["start"] = qwen_words[i]["start"]
            words[i]["end"] = qwen_words[i]["end"]
            words[i]["source"] = "qwen3_forced_aligner_conflict_fallback"
        fallback_indexes.update(impossible)
        boundary_adjustments += adjusted_this_pass
        if not bad and not impossible and adjusted_this_pass == 0:
            return {"fallback_words": len(fallback_indexes), "boundary_adjustments": boundary_adjustments}
    # Expand the fallback only around boundaries that remain inconsistent.
    # Because the Qwen checkpoint is monotonic, this converges even when a
    # run of neighboring MFA words is unusable.
    for _ in range(len(words) + 1):
        offenders = set()
        for index, word in enumerate(words):
            if word["end"] < word["start"]:
                offenders.add(index)
        for index in range(len(words) - 1):
            if words[index]["end"] > words[index + 1]["start"]:
                offenders.update((index, index + 1))
        if not offenders:
            return {"fallback_words": len(fallback_indexes), "boundary_adjustments": boundary_adjustments}
        for index in offenders:
            words[index]["start"] = qwen_words[index]["start"]
            words[index]["end"] = qwen_words[index]["end"]
            words[index]["source"] = "qwen3_forced_aligner_conflict_fallback"
        fallback_indexes.update(offenders)
    raise RuntimeError("Qwen checkpoint is not monotonic enough to resolve boundary conflicts")


def is_acoustic_monotonic(words: list[dict]) -> bool:
    return all(
        words[index]["start"] <= words[index]["end"] <= words[index + 1]["start"]
        for index in range(len(words) - 1)
    ) and (not words or words[-1]["start"] <= words[-1]["end"])


def ensure_focusable_timings(words: list[dict], audio_duration: float, minimum: float = 0.02) -> dict:
    """Preserve acoustic spans and add positive, non-overlapping UI focus spans."""
    clamped = 0
    for w in words:
        old = (w["start"], w["end"])
        w["start"] = min(audio_duration, max(0.0, w["start"]))
        w["end"] = min(audio_duration, max(w["start"], w["end"]))
        if (w["start"], w["end"]) != old:
            clamped += 1
    raw_centers = [min(audio_duration, max(0.0, (w["start"] + w["end"]) / 2)) for w in words]
    centers = list(raw_centers)
    for i in range(1, len(centers)):
        centers[i] = max(centers[i], centers[i - 1] + minimum)
    centers[-1] = min(centers[-1], audio_duration - minimum / 2)
    for i in range(len(centers) - 2, -1, -1):
        centers[i] = min(centers[i], centers[i + 1] - minimum)
    if centers[0] < minimum / 2:
        raise RuntimeError("Not enough timeline space for positive word durations")

    changed = 0
    zero_before = 0
    max_center_shift = 0.0
    for i, w in enumerate(words):
        old_start, old_end = w["start"], w["end"]
        old_duration = max(0.0, old_end - old_start)
        if old_duration <= 0:
            zero_before += 1
        left_limit = 0.0 if i == 0 else (centers[i - 1] + centers[i]) / 2
        right_limit = audio_duration if i == len(words) - 1 else (centers[i] + centers[i + 1]) / 2
        desired = max(minimum, old_duration)
        start = max(left_limit, centers[i] - desired / 2)
        end = min(right_limit, centers[i] + desired / 2)
        if end - start < minimum - 1e-9:
            start = max(left_limit, centers[i] - minimum / 2)
            end = min(right_limit, start + minimum)
            start = max(left_limit, end - minimum)
        w["focus_start"] = round(start, 3)
        w["focus_end"] = min(math.floor(audio_duration * 1000) / 1000, round(end, 3))
        if abs(w["focus_start"] - old_start) > 0.001 or abs(w["focus_end"] - old_end) > 0.001:
            changed += 1
        max_center_shift = max(max_center_shift, abs(centers[i] - raw_centers[i]))
    return {
        "minimum_word_duration_seconds": minimum,
        "acoustic_zero_duration_words": zero_before,
        "acoustic_clamped_words": clamped,
        "focus_adjusted_words": changed,
        "max_center_shift_seconds": round(max_center_shift, 3),
    }


def is_sentence_boundary(word: str, next_word: str | None = None) -> bool:
    """Mirror the player-side display sentence boundary rules."""
    token = re.sub(r'''["”’')\]}]+$''', "", word)
    if re.search(r"[!?]+$", token):
        return True
    if not token.endswith("."):
        return False
    stem = token[:-1]
    if stem.casefold() in NON_TERMINAL_ABBREVIATIONS:
        return False
    is_initialism = bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", stem))
    if is_initialism and next_word and re.match(r"^[a-z]", next_word):
        return False
    return True


def split_sentence_ranges(sentence: dict, words: list[dict]) -> list[tuple[int, int]]:
    """Split an STT sentence exactly where the web player splits its display."""
    start = int(sentence["word_start"])
    end = int(sentence["word_end"])
    ranges = []
    segment_start = start
    for index in range(start, end):
        is_last = index == end - 1
        if not is_last and not is_sentence_boundary(words[index]["text"], words[index + 1]["text"]):
            continue
        ranges.append((segment_start, index + 1))
        segment_start = index + 1
    return ranges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--qwen-dir", type=Path, required=True)
    p.add_argument("--textgrids-dir", type=Path, required=True)
    p.add_argument(
        "--allow-qwen-fallback",
        action="store_true",
        help="keep Qwen timings for utterances without usable MFA output",
    )
    p.add_argument("--mfa-version", default="3.4.1")
    p.add_argument("--mfa-acoustic-model", default="english_mfa")
    p.add_argument("--mfa-dictionary", default="english_mfa")
    p.add_argument("--mfa-g2p-model", default="english_us_mfa")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    qwen = json.loads((args.qwen_dir / "qwen_word_timings.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.qwen_dir / "mfa_manifest.json").read_text(encoding="utf-8"))
    base_qwen_words = [dict(w) for w in qwen["words"]]
    final_words = [dict(w) for w in base_qwen_words]
    by_id = {m["utterance_id"]: m for m in manifest}
    matched = 0
    total_mfa = 0
    deltas = []
    missing_textgrids = []
    invalid_textgrids = []

    grids = {path.stem: path for path in args.textgrids_dir.rglob("*.TextGrid")}
    for utt_id, m in by_id.items():
        path = grids.get(utt_id)
        expected_indexes = list(range(m["word_start"], m["word_end"]))
        if path is None:
            missing_textgrids.append(utt_id)
            for wi in expected_indexes:
                final_words[wi]["source"] = "qwen3_forced_aligner_mfa_missing_fallback"
            continue
        try:
            intervals = read_words(path)
        except Exception as error:
            invalid_textgrids.append({"utterance_id": utt_id, "error": safe_error(error, path)})
            for wi in expected_indexes:
                final_words[wi]["source"] = "qwen3_forced_aligner_mfa_invalid_fallback"
            continue
        total_mfa += len(intervals)
        expected = [norm(final_words[i]["text"]) for i in expected_indexes]
        observed = [norm(label) for _, _, label in intervals]
        sm = difflib.SequenceMatcher(None, expected, observed, autojunk=False)
        pairs = []
        for block in sm.get_matching_blocks():
            for k in range(block.size):
                wi = expected_indexes[block.a + k]
                local_start, local_end, _ = intervals[block.b + k]
                pairs.append((wi, local_start, local_end))
        paired_indexes = {wi for wi, _, _ in pairs}
        for wi in expected_indexes:
            if wi not in paired_indexes:
                final_words[wi]["source"] = "qwen3_forced_aligner_mfa_unmatched_fallback"
        offsets = []
        for wi, local_start, local_end in pairs:
            mfa_mid = (local_start + local_end) / 2 + m["offset"]
            qwen_mid = (base_qwen_words[wi]["start"] + base_qwen_words[wi]["end"]) / 2
            offsets.append(qwen_mid - mfa_mid)
        utterance_shift = statistics.median(offsets) if offsets else 0.0
        for wi, local_start, local_end in pairs:
            start = round(local_start + m["offset"] + utterance_shift, 3)
            end = round(local_end + m["offset"] + utterance_shift, 3)
            deltas.append(abs(start - final_words[wi]["start"]))
            deltas.append(abs(end - final_words[wi]["end"]))
            final_words[wi]["start"] = start
            final_words[wi]["end"] = end
            final_words[wi]["source"] = "mfa_3.4.1_fine_tuned"
            matched += 1

    mfa_merge_fallback = None
    conflict_resolution = resolve_conflicts(final_words, base_qwen_words)
    if not is_acoustic_monotonic(final_words):
        if not is_acoustic_monotonic(base_qwen_words):
            raise RuntimeError("Neither MFA merge nor the Qwen checkpoint has monotonic acoustic timings")
        raise RuntimeError("MFA merge remained non-monotonic after localized Qwen fallback")
    focus_timing_normalization = ensure_focusable_timings(
        final_words, float(qwen["metadata"]["audio_duration_seconds"])
    )
    sentences = []
    for s in qwen["sentences"]:
        ranges = split_sentence_ranges(s, final_words)
        for a, b in ranges:
            sentence_text = s["text"] if (a, b) == (s["word_start"], s["word_end"]) else " ".join(
                word["text"] for word in final_words[a:b]
            )
            final_sentence_index = len(sentences)
            for word_index in range(a, b):
                final_words[word_index]["sentence_index"] = final_sentence_index
            sentences.append({
                **s,
                "index": final_sentence_index,
                "text": sentence_text,
                "word_start": a,
                "word_end": b,
                "start": final_words[a]["focus_start"],
                "end": final_words[b - 1]["focus_end"],
            })
    chunks = []
    for c in qwen["chunks"]:
        a, b = c["word_start"], c["word_end"]
        chunks.append({
            **c,
            "start": final_words[a]["focus_start"],
            "end": final_words[b - 1]["focus_end"],
        })

    source_counts = {}
    for w in final_words:
        source_counts[w["source"]] = source_counts.get(w["source"], 0) + 1
    acoustic_monotonic = is_acoustic_monotonic(final_words)
    focus_monotonic = all(
        final_words[i]["focus_start"] <= final_words[i]["focus_end"] <= final_words[i + 1]["focus_start"]
        for i in range(len(final_words) - 1)
    ) and final_words[-1]["focus_start"] <= final_words[-1]["focus_end"]
    if not focus_monotonic:
        raise RuntimeError("Final focus timings are not monotonic")
    effective_mfa_words = sum(v for k, v in source_counts.items() if k.startswith("mfa_"))
    coverage = effective_mfa_words / max(1, len(final_words))
    metadata = dict(qwen["metadata"])
    metadata.update({
        "player_sentence_boundary_version": 1,
        "sentence_count": len(sentences),
        "final_model": (
            f"Montreal Forced Aligner {args.mfa_version} / "
            f"{args.mfa_acoustic_model} / fine_tune with localized Qwen fallback"
        ),
        "mfa_version": args.mfa_version,
        "mfa_acoustic_model": args.mfa_acoustic_model,
        "mfa_dictionary": args.mfa_dictionary,
        "mfa_g2p_model": args.mfa_g2p_model,
        "mfa_word_coverage": coverage,
        "mfa_matched_words": matched,
        "mfa_effective_words": effective_mfa_words,
        "mfa_output_words": total_mfa,
        "source_counts": source_counts,
        "median_qwen_mfa_boundary_delta_seconds": round(statistics.median(deltas), 4) if deltas else None,
        "p95_qwen_mfa_boundary_delta_seconds": round(sorted(deltas)[int(0.95 * (len(deltas) - 1))], 4) if deltas else None,
        "conflict_resolution": conflict_resolution,
        "focus_timing_normalization": focus_timing_normalization,
        "missing_textgrids": missing_textgrids,
        "invalid_textgrids": invalid_textgrids,
        "mfa_merge_fallback": mfa_merge_fallback,
        "monotonic": focus_monotonic,
        "acoustic_monotonic": acoustic_monotonic,
    })
    if coverage < 0.90:
        metadata["mfa_coverage_warning"] = (
            "MFA coverage below 90%; Qwen ForcedAligner timings were retained "
            "for unmatched words or missing utterance TextGrids."
        )
    final = {
        "schema_version": 1,
        "metadata": metadata,
        "words": final_words,
        "sentences": sentences,
        "chunks": chunks,
    }
    write_json(out / "word_timings.json", final)
    write_json(out / "alignment_report.json", metadata)

    cues = ["WEBVTT", ""]
    for i, s in enumerate(sentences, 1):
        cues.extend([str(i), f"{sec_to_vtt(s['start'])} --> {sec_to_vtt(s['end'])}", s["text"], ""])
    write_text(out / "sentences.vtt", "\n".join(cues))
    print(
        json.dumps(
            {
                "mfa_word_coverage": metadata["mfa_word_coverage"],
                "mfa_effective_words": metadata["mfa_effective_words"],
                "missing_textgrid_count": len(missing_textgrids),
                "invalid_textgrid_count": len(invalid_textgrids),
                "source_counts": source_counts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
