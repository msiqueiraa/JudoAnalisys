"""Geometria basica compartilhada pelos servicos de dominio."""

from __future__ import annotations

# Caixa envolvente ``(x_min, y_min, x_max, y_max)`` em coordenadas normalizadas.
Box = tuple[float, float, float, float]
Point = tuple[float, float]

_EPSILON = 1e-9


def iou(first: Box | None, second: Box | None) -> float:
    """Intersecao sobre uniao de duas caixas (0 quando nao se sobrepoem)."""

    if first is None or second is None:
        return 0.0

    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_x * inter_y

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > _EPSILON else 0.0


def ratio(first: float | None, second: float | None) -> float:
    """Razao entre dois valores positivos (1.0 quando nao ha como comparar)."""

    if not first or not second or first <= 0 or second <= 0:
        return 1.0
    return max(first / second, second / first)


def midpoint(a: Point, b: Point) -> Point:
    """Ponto medio entre dois pontos."""

    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
