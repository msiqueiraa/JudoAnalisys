"""Testes dos utilitarios estatisticos e da escolha do frame-chave."""

from __future__ import annotations

import pytest

from judo_analysis.domain.entities.analysis import AnalysisRun, FrameMetrics
from judo_analysis.domain.entities.video import TimeRange, Video
from judo_analysis.domain.services.statistics import (
    aggregate_run,
    frame_by_index,
    nearest_frame,
    peak_speed_frame,
)
from judo_analysis.domain.value_objects.joint_angles import JointAngles
from judo_analysis.domain.value_objects.speed import Speed

_FPS = 10.0


def _speed(pixels: float) -> Speed:
    return Speed(pixels_per_second=pixels, body_heights_per_second=pixels / 300.0)


def _run(speeds: list[float | None]) -> AnalysisRun:
    frames = [
        FrameMetrics(
            frame_index=index,
            timestamp=index / _FPS,
            angles=JointAngles(),
            hand_speed=None if speed is None else _speed(speed),
        )
        for index, speed in enumerate(speeds)
    ]
    return AnalysisRun(
        label="Referencia",
        video=Video(
            path="fake.mp4", fps=_FPS, frame_count=len(frames), width=100, height=100
        ),
        time_range=TimeRange(0.0, max(len(frames) / _FPS, 1.0)),
        person_id=0,
        frames=frames,
    )


def test_peak_speed_frame_is_the_strike() -> None:
    run = _run([100.0, 250.0, 900.0, 300.0, 120.0])

    assert peak_speed_frame(run).frame_index == 2


def test_peak_speed_frame_without_speed_uses_the_first() -> None:
    run = _run([None, None, None])

    assert peak_speed_frame(run).frame_index == 0


def test_peak_speed_frame_of_an_empty_run_is_none() -> None:
    assert peak_speed_frame(_run([])) is None


def test_aggregate_run_marks_the_peak_as_the_key_frame() -> None:
    run = _run([100.0, 250.0, 900.0, 300.0])

    aggregate_run(run)

    assert run.key_frame_index == 2
    assert run.stats["hand_speed"].maximum == pytest.approx(900.0)


def test_nearest_frame_snaps_to_the_closest_valid_frame() -> None:
    run = _run([100.0, 250.0, 900.0, 300.0])

    assert nearest_frame(run, 0.31).frame_index == 3
    assert nearest_frame(run, 0.19).frame_index == 2
    assert nearest_frame(run, -5.0).frame_index == 0
    assert nearest_frame(run, 99.0).frame_index == 3


def test_nearest_frame_of_an_empty_run_is_none() -> None:
    assert nearest_frame(_run([]), 1.0) is None


def test_frame_by_index_falls_back_to_the_first_frame() -> None:
    run = _run([100.0, 250.0])

    assert frame_by_index(run, 1).frame_index == 1
    # Um indice que nao existe mais (frame descartado) cai no primeiro.
    assert frame_by_index(run, 999).frame_index == 0
