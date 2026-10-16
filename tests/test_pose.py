"""Testes das propriedades geometricas da entidade Pose."""

from __future__ import annotations

import pytest

from tests.conftest import make_pose


def test_pelvis_is_hip_midpoint() -> None:
    pose = make_pose(LEFT_HIP=(0.4, 0.5), RIGHT_HIP=(0.6, 0.5))
    assert pose.pelvis == pytest.approx((0.5, 0.5))


def test_pelvis_falls_back_to_shoulders() -> None:
    pose = make_pose(LEFT_SHOULDER=(0.4, 0.2), RIGHT_SHOULDER=(0.6, 0.2))
    assert pose.pelvis == pytest.approx((0.5, 0.2))


def test_pelvis_none_without_reference_landmarks() -> None:
    assert make_pose().pelvis is None


def test_bbox_encloses_visible_landmarks() -> None:
    pose = make_pose(LEFT_HIP=(0.3, 0.6), RIGHT_KNEE=(0.7, 0.8))
    assert pose.bbox == pytest.approx((0.3, 0.6, 0.7, 0.8))


def test_body_scale_is_shoulder_to_ankle_distance() -> None:
    pose = make_pose(
        LEFT_SHOULDER=(0.45, 0.2),
        RIGHT_SHOULDER=(0.55, 0.2),
        LEFT_ANKLE=(0.45, 0.8),
        RIGHT_ANKLE=(0.55, 0.8),
    )
    assert pose.body_scale == pytest.approx(0.6)


def test_body_scale_none_when_landmark_missing() -> None:
    pose = make_pose(LEFT_SHOULDER=(0.45, 0.2), RIGHT_SHOULDER=(0.55, 0.2))
    assert pose.body_scale is None
