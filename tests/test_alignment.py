from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from podcast_align_video.validate import validate_word_timings
from podcast_align_video.workers.mfa_merge import ensure_focusable_timings, resolve_conflicts
from podcast_align_video.workers import qwen_worker
from podcast_align_video.workers.qwen_worker import align_result_to_words


def test_qwen_character_alignment_maps_words() -> None:
    expected = [{"clean": "Hello"}, {"clean": "world"}]
    observed = [
        {"text": "Hello", "start": 0.1, "end": 0.5},
        {"text": "world", "start": 0.6, "end": 1.0},
    ]
    result = align_result_to_words(expected, observed, 1.2)
    assert result is not None
    words, details = result
    assert details["mode"] == "char_exact"
    assert words[0]["start"] == pytest.approx(0.1)
    assert words[1]["end"] == pytest.approx(1.0)


def test_low_qwen_rough_coverage_uses_complete_cohere_anchors() -> None:
    exact = [{"clean": token} for token in ("one", "two", "three", "four")]
    rough = [{"text": "unrelated", "start": 0.0, "end": 0.5}]
    cohere = {index: (float(index), float(index + 1)) for index in range(4)}
    mapping, report = qwen_worker.build_anchor_map(exact, rough, cohere)
    assert mapping == {}
    assert report["coverage"] == 0.0
    assert report["combined_coverage"] == 1.0


def test_low_combined_anchor_coverage_is_rejected() -> None:
    exact = [{"clean": token} for token in ("one", "two", "three", "four")]
    rough = [{"text": "one", "start": 0.0, "end": 0.5}]
    with pytest.raises(RuntimeError, match="Combined Qwen/Cohere anchor coverage too low"):
        qwen_worker.build_anchor_map(exact, rough, {})


def test_qwen_oom_batch_is_split_and_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def memory_allocated() -> int:
            return 0

        @staticmethod
        def max_memory_allocated() -> int:
            return 0

    class FakeNumpy:
        float64 = float

        @staticmethod
        def asarray(value, dtype=None):
            return list(value)

        @staticmethod
        def ascontiguousarray(value):
            return value

    class SplittingAligner:
        def align(self, *, audio, text, language):
            if len(audio) > 1:
                raise RuntimeError("CUDA out of memory")
            return [
                SimpleNamespace(items=[SimpleNamespace(text=text[0], start_time=0.1, end_time=0.4)])
            ]

    monkeypatch.setattr(qwen_worker, "torch", SimpleNamespace(cuda=FakeCuda))
    monkeypatch.setattr(qwen_worker, "np", FakeNumpy)
    index = {
        "words": [
            {"text": "one", "clean": "one"},
            {"text": "two", "clean": "two"},
        ]
    }
    crops = [
        {"segment_id": "a", "source_chunk": 0, "word_start": 0, "word_end": 1, "text": "one", "crop_start": 0.0, "crop_end": 1.0, "rough_anchor_count": 1},
        {"segment_id": "b", "source_chunk": 0, "word_start": 1, "word_end": 2, "text": "two", "crop_start": 1.0, "crop_end": 2.0, "rough_anchor_count": 1},
    ]
    completed, stats = qwen_worker.run_exact_alignment(
        SplittingAligner(),
        [0.0] * (qwen_worker.SR * 2),
        index,
        crops,
        tmp_path / "segments.json",
        {"schema_version": 9},
        {0: (0.1, 0.4), 1: (1.1, 1.4)},
        max_items=2,
        max_padded_seconds=10,
    )
    assert set(completed) == {"a", "b"}
    assert stats["modes"] == {"char_exact": 2}
    assert stats["recoverable_errors"][0]["batch_size"] == 2


def test_mfa_conflict_falls_back_locally() -> None:
    qwen = [
        {"start": 0.0, "end": 0.4, "source": "qwen"},
        {"start": 0.4, "end": 0.8, "source": "qwen"},
        {"start": 0.8, "end": 1.2, "source": "qwen"},
    ]
    words = [dict(item) for item in qwen]
    words[1].update({"start": 0.7, "end": 0.2, "source": "mfa_3.4.1_fine_tuned"})
    report = resolve_conflicts(words, qwen)
    assert report["fallback_words"] == 1
    assert words[1]["source"].endswith("conflict_fallback")


def test_focus_normalization_is_positive_and_nonoverlapping() -> None:
    words = [
        {"start": 0.1, "end": 0.1},
        {"start": 0.1, "end": 0.1},
        {"start": 0.1, "end": 0.2},
    ]
    ensure_focusable_timings(words, 1.0)
    assert all(word["focus_end"] > word["focus_start"] for word in words)
    assert all(words[index]["focus_end"] <= words[index + 1]["focus_start"] for index in range(2))


def test_word_timing_schema_requires_exact_sentence_coverage(tmp_path: Path) -> None:
    value = {
        "schema_version": 1,
        "metadata": {"audio_duration_seconds": 2.0},
        "words": [
            {"index": 0, "text": "one", "clean": "one", "source": "qwen", "sentence_index": 0, "chunk_index": 0, "start": 0.0, "end": 0.4, "focus_start": 0.0, "focus_end": 0.4},
            {"index": 1, "text": "two", "clean": "two", "source": "qwen", "sentence_index": 0, "chunk_index": 0, "start": 0.5, "end": 0.9, "focus_start": 0.5, "focus_end": 0.9},
        ],
        "sentences": [{"index": 0, "word_start": 0, "word_end": 1}],
        "chunks": [{"index": 0, "word_start": 0, "word_end": 2}],
    }
    path = tmp_path / "word-timings.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="cover every word"):
        validate_word_timings(path)


def test_resumed_alignment_stats_are_recomputed_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "segments.json"
    meta = {"schema_version": 8}
    checkpoint.write_text(
        json.dumps(
            {
                "meta": meta,
                "segments": {
                    "0": {
                        "segment_id": "0",
                        "alignment_mode": "char_exact",
                        "items": [{"word_index": 0, "start": 0.1, "end": 0.2}],
                    },
                    "1": {
                        "segment_id": "1",
                        "alignment_mode": "rough_error_fallback",
                        "items": [{"word_index": 1, "start": 0.3, "end": 0.4}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    crops = [
        {"segment_id": "0", "word_start": 0, "word_end": 1},
        {"segment_id": "1", "word_start": 1, "word_end": 2},
    ]
    completed, stats = qwen_worker.run_exact_alignment(
        object(),
        [],
        {"words": []},
        crops,
        checkpoint,
        meta,
        {},
        max_items=4,
        max_padded_seconds=80,
    )
    assert len(completed) == 2
    assert stats["modes"] == {"char_exact": 1, "rough_error_fallback": 1}
    assert stats["fallback_segments"] == 1
    assert stats["resumed_segments"] == 2
    assert stats["new_segments"] == 0


def test_partial_split_alignment_checkpoint_resumes_only_missing_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCuda:
        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def memory_allocated() -> int:
            return 0

        @staticmethod
        def max_memory_allocated() -> int:
            return 0

    class FakeNumpy:
        float64 = float

        @staticmethod
        def asarray(value, dtype=None):
            return list(value)

        @staticmethod
        def ascontiguousarray(value):
            return value

    class CountingAligner:
        calls = 0

        def align(self, *, audio, text, language):
            self.calls += 1
            return [SimpleNamespace(items=[SimpleNamespace(text=text[0], start_time=0.1, end_time=0.4)])]

    monkeypatch.setattr(qwen_worker, "torch", SimpleNamespace(cuda=FakeCuda))
    monkeypatch.setattr(qwen_worker, "np", FakeNumpy)
    checkpoint = tmp_path / "segments.json"
    meta = {"schema_version": 10}
    checkpoint.write_text(
        json.dumps(
            {
                "meta": meta,
                "segments": {
                    "root.0": {
                        "segment_id": "root.0",
                        "alignment_mode": "char_exact",
                        "items": [{"word_index": 0, "start": 0.1, "end": 0.4}],
                    }
                },
                "recoverable_errors": [{"kind": "OutOfMemoryError"}],
            }
        ),
        encoding="utf-8",
    )
    index = {"words": [{"text": "one", "clean": "one"}, {"text": "two", "clean": "two"}]}
    crops = [
        {
            "segment_id": "root",
            "source_chunk": 0,
            "word_start": 0,
            "word_end": 2,
            "text": "one two",
            "crop_start": 0.0,
            "crop_end": 2.0,
            "rough_anchor_count": 2,
        }
    ]
    aligner = CountingAligner()
    completed, stats = qwen_worker.run_exact_alignment(
        aligner,
        [0.0] * (qwen_worker.SR * 2),
        index,
        crops,
        checkpoint,
        meta,
        {0: (0.1, 0.4), 1: (1.1, 1.4)},
        max_items=2,
        max_padded_seconds=10,
    )
    assert aligner.calls == 1
    assert set(completed) == {"root.0", "root.1"}
    assert stats["resumed_segments"] == 1
    assert stats["new_segments"] == 1
    assert stats["recoverable_errors"] == [{"kind": "OutOfMemoryError"}]


def test_missing_mfa_textgrid_falls_back_only_for_that_utterance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_align_video.workers import mfa_merge

    qwen_dir = tmp_path / "qwen"
    output_dir = tmp_path / "mfa"
    textgrids = output_dir / "textgrids"
    qwen_dir.mkdir()
    textgrids.mkdir(parents=True)
    qwen = {
        "schema_version": 1,
        "metadata": {"audio_duration_seconds": 2.0},
        "words": [
            {"index": 0, "text": "one", "clean": "one", "sentence_index": 0, "chunk_index": 0, "start": 0.1, "end": 0.5, "source": "qwen3_forced_aligner"},
            {"index": 1, "text": "two.", "clean": "two", "sentence_index": 0, "chunk_index": 0, "start": 0.6, "end": 1.0, "source": "qwen3_forced_aligner"},
        ],
        "sentences": [{"index": 0, "text": "one two.", "word_start": 0, "word_end": 2}],
        "chunks": [{"index": 0, "word_start": 0, "word_end": 2}],
    }
    (qwen_dir / "qwen_word_timings.json").write_text(json.dumps(qwen), encoding="utf-8")
    (qwen_dir / "mfa_manifest.json").write_text(
        json.dumps([{"utterance_id": "utt_00000", "word_start": 0, "word_end": 2, "offset": 0.0}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mfa_merge.py",
            "--output-dir", str(output_dir),
            "--qwen-dir", str(qwen_dir),
            "--textgrids-dir", str(textgrids),
            "--allow-qwen-fallback",
        ],
    )
    mfa_merge.main()
    result = json.loads((output_dir / "word_timings.json").read_text(encoding="utf-8"))
    assert result["metadata"]["missing_textgrids"] == ["utt_00000"]
    assert result["metadata"]["mfa_word_coverage"] == 0.0
    assert all("mfa_missing_fallback" in word["source"] for word in result["words"])
