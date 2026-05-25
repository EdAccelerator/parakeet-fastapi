"""Unit tests for the timestamp/token grouping helpers in app.asr."""

from __future__ import annotations

from app.asr import _segments_from_words, _words_from_tokens


def test_words_from_tokens_groups_bpe_tokens_into_words():
    # Matches the actual onnx-asr output shape for "I don't see"
    tokens = [" I", " don", "'", "t", " see"]
    timestamps = [0.0, 0.2, 0.3, 0.32, 0.5]
    words = _words_from_tokens(tokens, timestamps)
    assert [w.word for w in words] == ["I", "don't", "see"]
    # Start of "don't" is at 0.2 (the " don" token).
    assert words[1].word == "don't"
    assert words[1].start == 0.2
    # End of "don't" should be at the start of "see".
    assert words[1].end == 0.5
    # All starts non-decreasing.
    starts = [w.start for w in words]
    assert starts == sorted(starts)


def test_words_from_tokens_handles_empty_input():
    assert _words_from_tokens([], []) == []
    assert _words_from_tokens(["foo"], []) == []
    assert _words_from_tokens(["foo"], [0.0, 1.0]) == []  # mismatched lengths


def test_words_from_tokens_single_word():
    tokens = [" hello"]
    timestamps = [0.5]
    words = _words_from_tokens(tokens, timestamps)
    assert len(words) == 1
    assert words[0].word == "hello"
    assert words[0].start == 0.5
    # End falls back to a small fixed offset when we can't infer a step.
    assert words[0].end > words[0].start


def test_words_from_tokens_starts_word_without_leading_space():
    # First token sometimes lacks a leading space.
    tokens = ["Well", ",", " I", " don"]
    timestamps = [0.0, 0.1, 0.3, 0.5]
    words = _words_from_tokens(tokens, timestamps)
    assert [w.word for w in words] == ["Well,", "I", "don"]
    assert words[0].start == 0.0


def test_segments_from_words_returns_single_span():
    from app.asr import Word

    words = [
        Word(word="hello", start=0.0, end=0.5),
        Word(word="world", start=0.5, end=1.0),
    ]
    segs = _segments_from_words(words, "hello world")
    assert len(segs) == 1
    assert segs[0].start == 0.0
    assert segs[0].end == 1.0
    assert segs[0].text == "hello world"


def test_segments_from_words_empty_when_no_words():
    assert _segments_from_words([], "anything") == []
