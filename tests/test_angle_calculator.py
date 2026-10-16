"""Testes do calculo de angulos articulares."""

from __future__ import annotations

import pytest

from judo_analysis.domain.services.angle_calculator import AngleCalculator
from tests.conftest import make_pose


@pytest.fixture
def calculator() -> AngleCalculator:
    return AngleCalculator()


def test_knee_angle_straight_leg(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_HIP=(0.5, 0.3),
        LEFT_KNEE=(0.5, 0.5),
        LEFT_ANKLE=(0.5, 0.7),
    )
    angles = calculator.compute(pose)
    assert angles.knee_left == pytest.approx(180.0, abs=0.5)


def test_knee_angle_bent_90_degrees(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_HIP=(0.5, 0.3),
        LEFT_KNEE=(0.5, 0.5),
        LEFT_ANKLE=(0.7, 0.5),
    )
    angles = calculator.compute(pose)
    assert angles.knee_left == pytest.approx(90.0, abs=0.5)


def test_missing_landmark_returns_none(calculator: AngleCalculator) -> None:
    pose = make_pose(LEFT_HIP=(0.5, 0.3), LEFT_KNEE=(0.5, 0.5))
    assert calculator.compute(pose).knee_left is None


def test_spine_upright_is_zero(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_SHOULDER=(0.45, 0.2),
        RIGHT_SHOULDER=(0.55, 0.2),
        LEFT_HIP=(0.45, 0.6),
        RIGHT_HIP=(0.55, 0.6),
    )
    assert calculator.compute(pose).spine == pytest.approx(0.0, abs=0.5)


def test_spine_tilted(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_SHOULDER=(0.6, 0.2),
        RIGHT_SHOULDER=(0.7, 0.2),
        LEFT_HIP=(0.45, 0.6),
        RIGHT_HIP=(0.55, 0.6),
    )
    # Tronco inclinado ~20.6 graus em relacao a vertical.
    assert calculator.compute(pose).spine == pytest.approx(20.6, abs=1.0)


def test_neck_upright_is_zero(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_SHOULDER=(0.45, 0.4),
        RIGHT_SHOULDER=(0.55, 0.4),
        LEFT_EAR=(0.45, 0.2),
        RIGHT_EAR=(0.55, 0.2),
    )
    assert calculator.compute(pose).neck == pytest.approx(0.0, abs=0.5)


def test_leg_opening_parallel_is_near_zero(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_HIP=(0.45, 0.3),
        RIGHT_HIP=(0.55, 0.3),
        LEFT_KNEE=(0.45, 0.6),
        RIGHT_KNEE=(0.55, 0.6),
    )
    assert calculator.compute(pose).leg_opening == pytest.approx(0.0, abs=0.5)


def test_leg_opening_spread(calculator: AngleCalculator) -> None:
    pose = make_pose(
        LEFT_HIP=(0.45, 0.3),
        RIGHT_HIP=(0.55, 0.3),
        LEFT_KNEE=(0.2, 0.6),
        RIGHT_KNEE=(0.8, 0.6),
    )
    opening = calculator.compute(pose).leg_opening
    assert opening is not None
    assert opening > 30.0


def test_degenerate_points_do_not_raise(calculator: AngleCalculator) -> None:
    # Todos os pontos coincidentes: nao deve levantar excecao nem gerar NaN.
    pose = make_pose(
        LEFT_HIP=(0.5, 0.5),
        LEFT_KNEE=(0.5, 0.5),
        LEFT_ANKLE=(0.5, 0.5),
    )
    assert calculator.compute(pose).knee_left is None
