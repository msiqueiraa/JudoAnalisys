"""Calculo da velocidade do golpe.

A velocidade e medida a partir do deslocamento quadro a quadro dos punhos
(maos) e do centro do quadril. E reportada em pixels/s e normalizada pela
altura do corpo (alturas de corpo/s), o que permite comparar videos com
enquadramentos e resolucoes diferentes.
"""

from __future__ import annotations

import math
from collections import deque

from ..entities.landmark import Landmark
from ..entities.pose import Pose, PoseLandmark
from ..value_objects.speed import Speed

_Point = tuple[float, float]


def _midpoint(a: Landmark, b: Landmark) -> _Point:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


class SpeedCalculator:
    """Acumula posicoes ao longo dos frames e devolve a velocidade do golpe."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        smoothing_window: int = 3,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps if fps > 0 else 30.0
        self._prev_points: dict[str, _Point] | None = None
        self._body_height_px: float | None = None
        self._history: deque[float] = deque(maxlen=max(1, smoothing_window))

    @property
    def body_height_px(self) -> float | None:
        return self._body_height_px

    def update(self, pose: Pose | None) -> Speed | None:
        """Atualiza o estado com a pose do frame atual."""

        if pose is None:
            self._prev_points = None
            return None

        points = self._tracked_points(pose)
        if not points:
            self._prev_points = None
            return None

        self._update_body_height(pose)

        raw_speed = 0.0
        if self._prev_points is not None:
            speeds = []
            for key, point in points.items():
                previous = self._prev_points.get(key)
                if previous is None:
                    continue
                distance = math.dist(point, previous)
                speeds.append(distance * self._fps)
            if speeds:
                raw_speed = max(speeds)

        self._prev_points = points
        self._history.append(raw_speed)
        smoothed = sum(self._history) / len(self._history)

        normalized = 0.0
        if self._body_height_px:
            normalized = smoothed / self._body_height_px

        return Speed(pixels_per_second=smoothed, body_heights_per_second=normalized)

    def _tracked_points(self, pose: Pose) -> dict[str, _Point]:
        """Pontos rastreados para o calculo de velocidade, em pixels."""

        points: dict[str, _Point] = {}
        for key, landmark in (
            ("left_wrist", PoseLandmark.LEFT_WRIST),
            ("right_wrist", PoseLandmark.RIGHT_WRIST),
        ):
            point = pose.get(landmark)
            if point is not None:
                points[key] = point.as_pixel(self._width, self._height)

        left_hip = pose.get(PoseLandmark.LEFT_HIP)
        right_hip = pose.get(PoseLandmark.RIGHT_HIP)
        if left_hip is not None and right_hip is not None:
            mid = _midpoint(left_hip, right_hip)
            points["hip_center"] = (mid[0] * self._width, mid[1] * self._height)

        return points

    def _update_body_height(self, pose: Pose) -> None:
        left_shoulder = pose.get(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = pose.get(PoseLandmark.RIGHT_SHOULDER)
        left_ankle = pose.get(PoseLandmark.LEFT_ANKLE)
        right_ankle = pose.get(PoseLandmark.RIGHT_ANKLE)
        if None in (left_shoulder, right_shoulder, left_ankle, right_ankle):
            return

        mid_shoulder = _midpoint(left_shoulder, right_shoulder)
        mid_ankle = _midpoint(left_ankle, right_ankle)
        shoulder_px = (mid_shoulder[0] * self._width, mid_shoulder[1] * self._height)
        ankle_px = (mid_ankle[0] * self._width, mid_ankle[1] * self._height)
        height = math.dist(shoulder_px, ankle_px)
        if height <= 1.0:
            return

        if self._body_height_px is None:
            self._body_height_px = height
        else:
            # Media exponencial suave para estabilizar a referencia de escala.
            self._body_height_px = 0.9 * self._body_height_px + 0.1 * height
