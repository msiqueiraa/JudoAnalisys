"""Entidade que representa a pose estimada de uma pessoa em um frame."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from ...config import MIN_VISIBILITY
from ..value_objects.gi_color import GiColor
from .landmark import Landmark


class PoseLandmark(IntEnum):
    """Indices dos 33 landmarks do MediaPipe BlazePose."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass(slots=True)
class Pose:
    """Pose de uma pessoa: conjunto ordenado de landmarks.

    Attributes:
        person_id: Identificador da pessoa dentro do frame (0..n-1).
        landmarks: Lista de 33 landmarks indexados por :class:`PoseLandmark`.
        score: Confianca global da deteccao (0..1).
        gi_color: Cor do kimono medida no frame (pista opcional de identidade).
    """

    person_id: int
    landmarks: list[Landmark] = field(default_factory=list)
    score: float = 1.0
    gi_color: GiColor | None = None

    def get(self, landmark: PoseLandmark) -> Landmark | None:
        """Retorna o landmark se ele existir e estiver visivel o suficiente."""

        index = int(landmark)
        if index < 0 or index >= len(self.landmarks):
            return None
        point = self.landmarks[index]
        if point.visibility < MIN_VISIBILITY:
            return None
        return point

    @property
    def centroid(self) -> tuple[float, float] | None:
        """Centro medio dos landmarks visiveis (usado no rastreamento)."""

        visible = [lm for lm in self.landmarks if lm.visibility >= MIN_VISIBILITY]
        if not visible:
            return None
        return (
            sum(lm.x for lm in visible) / len(visible),
            sum(lm.y for lm in visible) / len(visible),
        )

    @property
    def pelvis(self) -> tuple[float, float] | None:
        """Centro do quadril (pelve), mais estavel que o centroide geral.

        Cai para o centro dos ombros e, por fim, para o centroide, caso os
        quadris nao estejam visiveis.
        """

        left_hip = self.get(PoseLandmark.LEFT_HIP)
        right_hip = self.get(PoseLandmark.RIGHT_HIP)
        if left_hip is not None and right_hip is not None:
            return _midpoint(left_hip, right_hip)

        left_shoulder = self.get(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get(PoseLandmark.RIGHT_SHOULDER)
        if left_shoulder is not None and right_shoulder is not None:
            return _midpoint(left_shoulder, right_shoulder)

        return self.centroid

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """Caixa envolvente ``(x_min, y_min, x_max, y_max)`` dos landmarks visiveis."""

        visible = [lm for lm in self.landmarks if lm.visibility >= MIN_VISIBILITY]
        if not visible:
            return None
        xs = [lm.x for lm in visible]
        ys = [lm.y for lm in visible]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def body_scale(self) -> float | None:
        """Altura do corpo normalizada (ombro medio -> tornozelo medio)."""

        left_shoulder = self.get(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get(PoseLandmark.RIGHT_SHOULDER)
        left_ankle = self.get(PoseLandmark.LEFT_ANKLE)
        right_ankle = self.get(PoseLandmark.RIGHT_ANKLE)
        if None in (left_shoulder, right_shoulder, left_ankle, right_ankle):
            return None

        mid_shoulder = _midpoint(left_shoulder, right_shoulder)
        mid_ankle = _midpoint(left_ankle, right_ankle)
        return math.hypot(mid_shoulder[0] - mid_ankle[0], mid_shoulder[1] - mid_ankle[1])

    @property
    def visible_count(self) -> int:
        """Quantidade de landmarks com visibilidade suficiente."""

        return sum(1 for lm in self.landmarks if lm.visibility >= MIN_VISIBILITY)


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
