"""SRT/VTT formatting helpers."""

from __future__ import annotations

from app.asr import Segment
from app.routers.native import _format_srt, _format_timestamp, _format_vtt


def test_format_timestamp_srt():
    # 1h 2m 3.456s → 01:02:03,456
    assert _format_timestamp(3723.456, vtt=False) == "01:02:03,456"


def test_format_timestamp_vtt_uses_dot():
    assert _format_timestamp(3.5, vtt=True) == "00:00:03,500".replace(",", ".")


def test_format_timestamp_clamps_negative_to_zero():
    assert _format_timestamp(-1.0, vtt=False) == "00:00:00,000"


def test_format_srt_with_segments():
    segs = [
        Segment(start=0.0, end=1.5, text="hello"),
        Segment(start=1.5, end=2.0, text="world"),
    ]
    out = _format_srt(segs, fallback_text="", duration=2.0)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert lines[2] == "hello"
    assert "2" in lines
    assert "world" in out


def test_format_srt_falls_back_to_single_segment():
    out = _format_srt([], fallback_text="hi there", duration=3.0)
    assert "00:00:00,000 --> 00:00:03,000" in out
    assert "hi there" in out


def test_format_vtt_starts_with_header():
    segs = [Segment(start=0.0, end=1.0, text="hello")]
    out = _format_vtt(segs, fallback_text="", duration=1.0)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in out
    assert "hello" in out


def test_format_srt_empty_when_no_text():
    out = _format_srt([], fallback_text="", duration=1.0)
    assert out == ""
