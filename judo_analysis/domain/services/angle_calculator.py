"""Calculo dos angulos articulares relevantes para o judo.

Toda a matematica fica isolada aqui, sem dependencia de MediaPipe/OpenCV,
o que torna o comportamento testavel e previsivel.
"""

from __future__ import annotations

import math

from ..entities.landmark import Landmark
from ..entities.pose import Pose, PoseLandmark
from ..value_objects.joint_angles import JointAngles

_EPSILON = 1e-9

# Direcao "para cima" em coordenadas de imagem (o eixo Y cresce para baixo).
_VERTICAL_UP = (0.0, -1.0)


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float | None:
    """Angulo (graus) entre dois vetores, com protecao contra ``NaN``."""

    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < _EPSILON or n2 < _EPSILON:
        return None
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    # Clip evita `ValueError`/`NaN` por erro de ponto flutuante.
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


class AngleCalculator:
    """Calcula os angulos de joelho, coluna, pescoco e abertura das pernas."""

    def compute(self, pose: Pose) -> JointAngles:
        return JointAngles(
            knee_left=self._joint_angle(
                pose, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE
            ),
            knee_right=self._joint_angle(
                pose, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE
            ),
            spine=self._spine_angle(pose),
            neck=self._neck_angle(pose),
            leg_opening=self._leg_opening(pose),
        )

    @staticmethod
    def _joint_angle(pose: Pose, a: PoseLandmark, b: PoseLandmark, c: PoseLandmark) -> float | None:
        pa, pb, pc = pose.get(a), pose.get(b), pose.get(c)
        if pa is None or pb is None or pc is None:
            return None
        return _angle_between(_sub(pa.as_tuple(), pb.as_tuple()), _sub(pc.as_tuple(), pb.as_tuple()))

    @staticmethod
    def _spine_angle(pose: Pose) -> float | None:
        """Inclinacao do tronco em relacao a vertical."""

        left_shoulder = pose.get(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = pose.get(PoseLandmark.RIGHT_SHOULDER)
        left_hip = pose.get(PoseLandmark.LEFT_HIP)
        right_hip = pose.get(PoseLandmark.RIGHT_HIP)
        if None in (left_shoulder, right_shoulder, left_hip, right_hip):
            return None

        mid_shoulder = _midpoint(left_shoulder, right_shoulder)
        mid_hip = _midpoint(left_hip, right_hip)
        trunk = _sub(mid_shoulder, mid_hip)
        return _angle_between(trunk, _VERTICAL_UP)

    @staticmethod
    def _neck_angle(pose: Pose) -> float | None:
        """Inclinacao da cabeca/pescoco em relacao a vertical."""

        left_shoulder = pose.get(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = pose.get(PoseLandmark.RIGHT_SHOULDER)
        if left_shoulder is None or right_shoulder is None:
            return None

        head = AngleCalculator._head_point(pose)
        if head is None:
            return None

        mid_shoulder = _midpoint(left_shoulder, right_shoulder)
        neck = _sub(head, mid_shoulder)
        return _angle_between(neck, _VERTICAL_UP)

    @staticmethod
    def _head_point(pose: Pose) -> tuple[float, float] | None:
        left_ear = pose.get(PoseLandmark.LEFT_EAR)
        right_ear = pose.get(PoseLandmark.RIGHT_EAR)
        if left_ear is not None and right_ear is not None:
            return _midpoint(left_ear, right_ear)
        nose = pose.get(PoseLandmark.NOSE)
        if nose is not None:
            return nose.as_tuple()
        return left_ear.as_tuple() if left_ear else (right_ear.as_tuple() if right_ear else None)

    @staticmethod
    def _leg_opening(pose: Pose) -> float | None:
        """Angulo entre as coxas, medido a partir dos quadris.

        Quando as pernas estao paralelas o angulo tende a 0; quanto mais
        abertas, maior o valor.
        """

        left_hip = pose.get(PoseLandmark.LEFT_HIP)
        right_hip = pose.get(PoseLandmark.RIGHT_HIP)
        left_knee = pose.get(PoseLandmark.LEFT_KNEE)
        right_knee = pose.get(PoseLandmark.RIGHT_KNEE)
        if None in (left_hip, right_hip, left_knee, right_knee):
            return None

        left_thigh = _sub(left_knee.as_tuple(), left_hip.as_tuple())
        right_thigh = _sub(right_knee.as_tuple(), right_hip.as_tuple())
        return _angle_between(left_thigh, right_thigh)
