"""Testes do detector de pose (deduplicacao, segunda tentativa e cor)."""

from __future__ import annotations

import numpy as np

from judo_analysis.application.pose_detection import PoseDetector, detect_in_halves
from judo_analysis.domain.value_objects.gi_color import GiColor
from tests.conftest import make_pose

FRAME = np.zeros((100, 200, 3), dtype=np.uint8)


def _pose(cx: float, cy: float = 0.5, scale: float = 0.2) -> object:
    return make_pose(
        LEFT_HIP=(cx - scale * 0.15, cy),
        RIGHT_HIP=(cx + scale * 0.15, cy),
        LEFT_SHOULDER=(cx - scale * 0.25, cy - scale),
        RIGHT_SHOULDER=(cx + scale * 0.25, cy - scale),
        LEFT_ANKLE=(cx - scale * 0.1, cy + scale),
        RIGHT_ANKLE=(cx + scale * 0.1, cy + scale),
    )


class _Estimator:
    """Estimador de teste: devolve poses conforme a largura recebida."""

    def __init__(self, full, halves=None) -> None:
        self.full = full
        self.halves = halves
        self.calls: list[int] = []

    def detect(self, frame, timestamp_ms=None):
        self.calls.append(frame.shape[1])
        if frame.shape[1] == FRAME.shape[1]:
            return list(self.full)
        if self.halves is not None:
            return list(self.halves)
        # Uma pose no centro de cada metade (vira 0.25 e 0.75 no quadro todo).
        return [_pose(0.5)]

    def close(self) -> None:
        pass


class _Sampler:
    def __init__(self, color: GiColor) -> None:
        self.color = color
        self.count = 0

    def sample(self, frame, pose):
        self.count += 1
        return self.color


def test_deduplicates_what_the_estimator_returns() -> None:
    detector = PoseDetector(_Estimator([_pose(0.5), _pose(0.5)]), retry=False)
    assert len(detector.detect(FRAME)) == 1


def test_retries_in_halves_whenever_fewer_than_two_people() -> None:
    """Regressao: a segunda tentativa nao pode depender de ja ter visto dois.

    Era esse o bug que fazia o rastreamento alternar: quando o detector fundia os
    dois atletas, ele nunca via dois e a segunda tentativa nunca acontecia.
    """

    estimator = _Estimator([_pose(0.5)])
    detector = PoseDetector(estimator)
    poses = detector.detect(FRAME)
    assert len(poses) == 3
    assert 200 in estimator.calls


def test_does_not_retry_when_two_people_are_found() -> None:
    estimator = _Estimator([_pose(0.2), _pose(0.8)], halves=[_pose(0.5)])
    detector = PoseDetector(estimator)
    poses = detector.detect(FRAME)
    assert len(poses) == 2
    assert estimator.calls == [200]


def test_retry_can_be_disabled() -> None:
    estimator = _Estimator([_pose(0.5)], halves=[_pose(0.2)])
    detector = PoseDetector(estimator, retry=False)
    assert len(detector.detect(FRAME)) == 1


def test_keeps_the_largest_number_of_people_seen() -> None:
    estimator = _Estimator([_pose(0.5)])
    detector = PoseDetector(estimator)
    detector.detect(FRAME)
    assert detector.max_people_seen == 3


def test_samples_the_gi_color_of_every_pose() -> None:
    sampler = _Sampler(GiColor(l=70.0, a=1.0, b=2.0))
    detector = PoseDetector(_Estimator([_pose(0.2), _pose(0.8)]), color_sampler=sampler)
    poses = detector.detect(FRAME)
    assert sampler.count == 2
    assert all(pose.gi_color is not None for pose in poses)


def test_detect_in_halves_shifts_poses_to_the_full_frame() -> None:
    class _HalfEstimator:
        def detect(self, frame, timestamp_ms=None):
            # Pelve no centro de qualquer recorte recebido.
            return [_pose(0.5)]

        def close(self) -> None:
            pass

    poses = detect_in_halves(_HalfEstimator(), FRAME)
    assert [round(pose.pelvis[0], 2) for pose in poses] == [0.25, 0.75]


def test_detect_in_halves_skips_tiny_frames() -> None:
    narrow = np.zeros((50, 20, 3), dtype=np.uint8)
    assert detect_in_halves(_Estimator([]), narrow) == []
