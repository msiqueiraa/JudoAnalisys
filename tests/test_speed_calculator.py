"""Testes do calculo de velocidade do golpe."""

from __future__ import annotations

import pytest

from judo_analysis.domain.services.speed_calculator import SpeedCalculator
from tests.conftest import make_pose


def _pose_with_wrist(x: float) -> object:
    return make_pose(
        LEFT_WRIST=(x, 0.4),
        LEFT_SHOULDER=(0.5, 0.3),
        RIGHT_SHOULDER=(0.5, 0.3),
        LEFT_ANKLE=(0.5, 0.9),
        RIGHT_ANKLE=(0.5, 0.9),
    )


def test_first_frame_has_zero_speed() -> None:
    calculator = SpeedCalculator(width=100, height=100, fps=10, smoothing_window=1)
    speed = calculator.update(_pose_with_wrist(0.1))
    assert speed is not None
    assert speed.pixels_per_second == pytest.approx(0.0)


def test_speed_matches_displacement_times_fps() -> None:
    calculator = SpeedCalculator(width=100, height=100, fps=10, smoothing_window=1)
    calculator.update(_pose_with_wrist(0.1))
    speed = calculator.update(_pose_with_wrist(0.2))
    assert speed is not None
    # 0.1 de 100px = 10px; a 10 fps -> 100 px/s.
    assert speed.pixels_per_second == pytest.approx(100.0)


def test_speed_is_normalized_by_body_height() -> None:
    calculator = SpeedCalculator(width=100, height=100, fps=10, smoothing_window=1)
    calculator.update(_pose_with_wrist(0.1))
    speed = calculator.update(_pose_with_wrist(0.2))
    assert speed is not None
    # Altura do corpo = ombro(0.3) -> tornozelo(0.9) = 60px.
    assert calculator.body_height_px == pytest.approx(60.0)
    assert speed.body_heights_per_second == pytest.approx(100.0 / 60.0)


def test_missing_pose_resets_tracking() -> None:
    calculator = SpeedCalculator(width=100, height=100, fps=10, smoothing_window=1)
    calculator.update(_pose_with_wrist(0.1))
    assert calculator.update(None) is None
    # Sem frame anterior, a velocidade volta a ser zero.
    speed = calculator.update(_pose_with_wrist(0.9))
    assert speed is not None
    assert speed.pixels_per_second == pytest.approx(0.0)
