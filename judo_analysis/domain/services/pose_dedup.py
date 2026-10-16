"""Deduplicacao de poses que descrevem o MESMO corpo.

O detector as vezes devolve duas poses sobre o mesmo atleta - tipico quando um
esta por cima do outro - e nenhuma para o adversario. Tratar isso como "duas
pessoas" engana todo o restante do pipeline: o rastreador acredita que a cena
tem dois atletas, o gate de identidade nunca entra em vigor e o atleta errado
pode ser adotado.

Este modulo unifica as duplicatas **antes** de qualquer contagem, mantendo em
cada grupo a deteccao mais completa (a que tem mais landmarks visiveis).
"""

from __future__ import annotations

import math

from ...config import (
    DEDUP_ANCHOR_DISTANCE,
    DEDUP_IOU,
    DEDUP_IOU_LANDMARK_DISTANCE,
    DEDUP_LANDMARK_DISTANCE,
    MIN_VISIBILITY,
)
from ..entities.pose import Pose
from .geometry import iou

_EPSILON = 1e-9


def deduplicate(
    poses: list[Pose],
    iou_threshold: float = DEDUP_IOU,
    distance_threshold: float = DEDUP_LANDMARK_DISTANCE,
    iou_distance_threshold: float = DEDUP_IOU_LANDMARK_DISTANCE,
    anchor_threshold: float = DEDUP_ANCHOR_DISTANCE,
) -> list[Pose]:
    """Remove poses duplicadas e reindexa da esquerda para a direita.

    Devolve uma lista nova (as poses originais sao reutilizadas), ordenada pela
    posicao horizontal e com ``person_id`` de 0 a ``len(resultado) - 1``.
    """

    ordered = sorted(poses, key=lambda pose: pose.visible_count, reverse=True)
    kept: list[Pose] = []
    for pose in ordered:
        if any(
            is_same_body(
                pose,
                other,
                iou_threshold=iou_threshold,
                distance_threshold=distance_threshold,
                iou_distance_threshold=iou_distance_threshold,
                anchor_threshold=anchor_threshold,
            )
            for other in kept
        ):
            continue
        kept.append(pose)

    kept.sort(key=_horizontal_position)
    for index, pose in enumerate(kept):
        pose.person_id = index
    return kept


def is_same_body(
    first: Pose,
    second: Pose,
    iou_threshold: float = DEDUP_IOU,
    distance_threshold: float = DEDUP_LANDMARK_DISTANCE,
    iou_distance_threshold: float = DEDUP_IOU_LANDMARK_DISTANCE,
    anchor_threshold: float = DEDUP_ANCHOR_DISTANCE,
) -> bool:
    """Indica se duas deteccoes descrevem o mesmo corpo."""

    # Duas deteccoes praticamente no mesmo ponto do quadro sao o mesmo corpo,
    # mesmo que os landmarks divirjam (o detector as vezes "ve" o atleta duas
    # vezes, com uma pose um pouco diferente).
    if _anchor_distance(first, second) <= anchor_threshold:
        return True

    distance = relative_landmark_distance(first, second)
    if distance is None:
        return iou(first.bbox, second.bbox) >= iou_threshold
    if distance <= distance_threshold:
        return True
    return (
        distance <= iou_distance_threshold
        and iou(first.bbox, second.bbox) >= iou_threshold
    )


def _anchor_distance(first: Pose, second: Pose) -> float:
    """Distancia entre as pelves (ou centroides) de duas poses."""

    first_anchor = first.pelvis or first.centroid
    second_anchor = second.pelvis or second.centroid
    if first_anchor is None or second_anchor is None:
        return math.inf
    return math.dist(first_anchor, second_anchor)


def relative_landmark_distance(first: Pose, second: Pose) -> float | None:
    """Distancia media entre landmarks visiveis nos dois, relativa a altura.

    ``None`` quando nao ha landmarks em comum ou quando nenhuma das poses tem
    altura de corpo para normalizar a distancia.
    """

    pairs = [
        (point_a, point_b)
        for point_a, point_b in zip(first.landmarks, second.landmarks)
        if point_a.visibility >= MIN_VISIBILITY and point_b.visibility >= MIN_VISIBILITY
    ]
    if not pairs:
        return None

    scales = [scale for scale in (first.body_scale, second.body_scale) if scale]
    if not scales:
        return None

    mean = sum(math.dist(a.as_tuple(), b.as_tuple()) for a, b in pairs) / len(pairs)
    return mean / max(sum(scales) / len(scales), _EPSILON)


def _horizontal_position(pose: Pose) -> float:
    anchor = pose.pelvis or pose.centroid
    return anchor[0] if anchor else 0.0
