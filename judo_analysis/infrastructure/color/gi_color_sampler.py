"""Medicao da cor do kimono a partir dos pixels do frame (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from ...application.ports.gi_color_sampler import IGiColorSampler
from ...config import (
    GI_COLOR_MIN_PATCHES,
    GI_COLOR_PATCH_MAX_MAD,
    GI_COLOR_PATCH_RADIUS,
)
from ...domain.entities.pose import Pose, PoseLandmark
from ...domain.value_objects.gi_color import GiColor

# Menor patch aproveitavel (px no total).
_MIN_PATCH_PIXELS = 9
# Raio minimo (px) para medir alguma coisa em deteccoes pequenas.
_MIN_RADIUS = 3
# Altura de corpo assumida quando os tornozelos nao estao visiveis.
_FALLBACK_SCALE = 0.3
# Quanto o ponto do peito desce em direcao ao quadril (evita rosto/pescoco).
_CHEST_TRAVEL = 0.25


class OpenCVGiColorSampler(IGiColorSampler):
    """Cor media do kimono medida em patches do peito, quadril e joelhos.

    A cor de cada patch e a **mediana** dos pixels (robusta a pele das maos e a
    pequenas oclusoes) e a cor final e a mediana entre os patches validos.
    Patches muito heterogeneos - que misturam kimono com o fundo ou com o
    adversario - sao descartados pelo desvio absoluto mediano do brilho.
    """

    def __init__(
        self,
        patch_radius: float = GI_COLOR_PATCH_RADIUS,
        max_mad: float = GI_COLOR_PATCH_MAX_MAD,
        min_patches: int = GI_COLOR_MIN_PATCHES,
    ) -> None:
        self._patch_radius = patch_radius
        self._max_mad = max_mad
        self._min_patches = max(1, min_patches)

    def sample(self, frame: np.ndarray, pose: Pose) -> GiColor | None:
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None

        radius = self._radius(pose, height)
        colors = [
            color
            for center in self._centers(pose)
            if (color := self._patch_color(frame, center, radius)) is not None
        ]
        if len(colors) < self._min_patches:
            return None

        return GiColor(
            l=float(np.median([color.l for color in colors])),
            a=float(np.median([color.a for color in colors])),
            b=float(np.median([color.b for color in colors])),
        )

    def _radius(self, pose: Pose, height: int) -> int:
        scale = pose.body_scale or _FALLBACK_SCALE
        return max(_MIN_RADIUS, int(round(self._patch_radius * scale * height)))

    @staticmethod
    def _centers(pose: Pose) -> list[tuple[float, float]]:
        """Pontos (normalizados) onde o kimono costuma estar visivel."""

        chest = _midpoint(
            pose.get(PoseLandmark.LEFT_SHOULDER), pose.get(PoseLandmark.RIGHT_SHOULDER)
        )
        hips = _midpoint(pose.get(PoseLandmark.LEFT_HIP), pose.get(PoseLandmark.RIGHT_HIP))
        if chest is not None and hips is not None:
            chest = _between(chest, hips, _CHEST_TRAVEL)
        knees = _midpoint(
            pose.get(PoseLandmark.LEFT_KNEE), pose.get(PoseLandmark.RIGHT_KNEE)
        )
        return [point for point in (chest, hips, knees) if point is not None]

    def _patch_color(
        self, frame: np.ndarray, center: tuple[float, float], radius: int
    ) -> GiColor | None:
        height, width = frame.shape[:2]
        cx = int(round(center[0] * width))
        cy = int(round(center[1] * height))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        if (x1 - x0) * (y1 - y0) < _MIN_PATCH_PIXELS:
            return None

        patch = np.ascontiguousarray(frame[y0:y1, x0:x1, :3])
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        median = np.median(lab, axis=0)
        # Desvio absoluto mediano do brilho: alto = patch com regioes diferentes.
        if float(np.median(np.abs(lab[:, 0] - median[0]))) > self._max_mad:
            return None
        return _to_gi_color(median)


def _midpoint(first, second) -> tuple[float, float] | None:
    """Ponto medio de dois landmarks (exige que os dois estejam visiveis)."""

    if first is None or second is None:
        return None
    return ((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)


def _between(
    start: tuple[float, float], end: tuple[float, float], factor: float
) -> tuple[float, float]:
    """Ponto a ``factor`` do caminho de ``start`` ate ``end``."""

    return (
        start[0] + factor * (end[0] - start[0]),
        start[1] + factor * (end[1] - start[1]),
    )


def _to_gi_color(values: np.ndarray) -> GiColor:
    """Converte o LAB do OpenCV (0..255) para o LAB padrao (L 0..100, a/b +-128)."""

    return GiColor(
        l=float(values[0]) * 100.0 / 255.0,
        a=float(values[1]) - 128.0,
        b=float(values[2]) - 128.0,
    )
