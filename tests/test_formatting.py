"""Testes dos utilitarios de formatacao de tempo."""

from __future__ import annotations

import datetime as dt

import pytest

from judo_analysis.formatting import (
    format_duration,
    format_duration_precise,
    parse_time,
    seconds_to_time,
    slider_time_format,
    time_to_seconds,
)


def test_format_duration_minutes_seconds() -> None:
    assert format_duration(0) == "00:00"
    assert format_duration(75) == "01:15"
    assert format_duration(599) == "09:59"


def test_format_duration_with_hours() -> None:
    assert format_duration(3661) == "1:01:01"


def test_format_duration_precise_keeps_integers_clean() -> None:
    assert format_duration_precise(13) == "00:13"
    assert format_duration_precise(75.0) == "01:15"


def test_format_duration_precise_shows_centiseconds() -> None:
    assert format_duration_precise(1.55) == "00:01.55"
    assert format_duration_precise(60.4583) == "01:00.46"
    assert format_duration_precise(61.375) == "01:01.38"
    assert format_duration_precise(3661.25) == "1:01:01.25"


def test_format_duration_precise_accepts_other_precisions() -> None:
    assert format_duration_precise(60.4583, decimals=1) == "01:00.5"
    assert format_duration_precise(60.4583, decimals=3) == "01:00.458"


def test_format_duration_precise_does_not_overflow_the_seconds_field() -> None:
    # 59.999 esta a menos de meio centesimo de 60 s: sai como 01:00, nao 00:60.
    assert format_duration_precise(59.999) == "01:00"
    assert format_duration_precise(59.994) == "00:59.99"


def test_format_duration_precise_roundtrips() -> None:
    for value in (1.55, 60.4583, 61.375, 12.5):
        assert parse_time(format_duration_precise(value)) == pytest.approx(value, abs=0.01)


def test_slider_time_format() -> None:
    assert slider_time_format(120) == "mm:ss"
    assert slider_time_format(3600) == "h:mm:ss"


def test_slider_time_format_with_centiseconds() -> None:
    # O slider que escolhe o frame-chave mostra os centesimos.
    assert slider_time_format(120, precise=True) == "mm:ss.SS"
    assert slider_time_format(3600, precise=True) == "h:mm:ss.SS"


def test_seconds_and_time_roundtrip() -> None:
    value = seconds_to_time(195)  # 03:15
    assert value == dt.time(0, 3, 15)
    assert time_to_seconds(value) == 195.0


def test_seconds_to_time_preserves_centiseconds() -> None:
    value = seconds_to_time(1.55)
    assert value == dt.time(0, 0, 1, 550000)
    assert time_to_seconds(value) == pytest.approx(1.55)


def test_seconds_to_time_rounds_to_the_nearest_microsecond() -> None:
    assert seconds_to_time(0.1234567) == dt.time(0, 0, 0, 123457)


def test_seconds_to_time_clamps_above_max() -> None:
    assert seconds_to_time(10 ** 9).hour <= 23


def test_parse_time_minutes_seconds() -> None:
    assert parse_time("00:05") == 5.0
    assert parse_time("01:15") == 75.0
    assert parse_time(" 2:03 ") == 123.0


def test_parse_time_with_centiseconds() -> None:
    assert parse_time("00:01.55") == pytest.approx(1.55)
    assert parse_time("01:02.34") == pytest.approx(62.34)
    assert parse_time("1:00:00.5") == pytest.approx(3600.5)


def test_parse_time_hours_and_plain_seconds() -> None:
    assert parse_time("1:00:00") == 3600.0
    assert parse_time("12.5") == 12.5


def test_parse_time_invalid_returns_none() -> None:
    assert parse_time("") is None
    assert parse_time("abc") is None
    assert parse_time("1:2:3:4") is None
    assert parse_time("-1:00") is None
