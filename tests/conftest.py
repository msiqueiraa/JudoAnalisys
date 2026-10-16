"""Fixtures compartilhadas pelos testes."""

from __future__ import annotations

import pytest

from judo_analysis.domain.entities.landmark import Landmark
from judo_analysis.domain.entities.pose import Pose, PoseLandmark

_LANDMARK_COUNT = len(PoseLandmark)


def make_pose(**points: tuple[float, float]) -> Pose:
    """Cria uma pose com os landmarks nomeados visiveis e o resto invisivel.

    Exemplo::

        make_pose(LEFT_HIP=(0.5, 0.4), LEFT_KNEE=(0.5, 0.6), LEFT_ANKLE=(0.5, 0.8))

    """

    landmarks = [Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(_LANDMARK_COUNT)]
    for name, (x, y) in points.items():
        landmarks[int(PoseLandmark[name])] = Landmark(x=x, y=y, z=0.0, visibility=1.0)
    return Pose(person_id=0, landmarks=landmarks)


@pytest.fixture
def make_pose_fixture():
    return make_pose
