from podcast_align_video.workers.cohere_worker import (
    full_audio_schedule,
    merge_overlapping_transcripts,
    merge_speech_regions,
    sentence_lines,
    speech_schedule,
    transcript_quality_report,
)


def test_window_schedule_uses_overlap() -> None:
    assert full_audio_schedule(60.0, 28.0, 4.0) == [
        (0, 0.0, 28.0),
        (1, 24.0, 52.0),
        (2, 48.0, 60.0),
    ]


def test_vad_merge_and_schedule() -> None:
    regions = merge_speech_regions([(2, 4), (4.5, 8), (20, 22)], 30, 0.35, 0.9)
    assert regions == [(1.65, 8.35), (19.65, 22.35)]
    schedule = speech_schedule(regions, 4.0, 1.0)
    assert schedule[0] == (0, 1.65, 5.65)
    assert schedule[-1][2] == 22.35


def test_overlap_merge_does_not_drop_nonoverlap_words() -> None:
    result = merge_overlapping_transcripts(
        "The quick brown fox jumps over the lazy dog",
        "over the lazy dog and keeps running",
    )
    assert result == "The quick brown fox jumps over the lazy dog and keeps running"


def test_repetition_detection() -> None:
    text = " ".join(["loop one two three"] * 30)
    report = transcript_quality_report(text, 30)
    assert not report["ok"]
    assert "consecutive_ngram_loop" in report["reasons"]


def test_sentence_split_preserves_every_token() -> None:
    text = "Dr. Smith arrived at 3.5 p.m. He spoke clearly! Then everyone left"
    lines = sentence_lines(text, max_words=8)
    assert " ".join(lines).split() == text.split()
    assert len(lines) >= 2
