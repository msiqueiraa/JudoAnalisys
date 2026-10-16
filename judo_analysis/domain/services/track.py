"""Estado de uma pessoa rastreada e comparacao de cor do kimono.

Cada pessoa acompanhada tem um :class:`Track`: sua ultima posicao (ancora),
velocidade, escala do corpo, caixa e cor do kimono. Os custos de associacao
combinam posicao, sobreposicao de caixa (IoU), escala e cor - todos
normalizados para que nenhuma pista domine as outras sozinha.
"""

from __future__ import annotations

import math

from ...config import (
    GI_COLOR_DIFFERENT_DELTA_E,
    GI_COLOR_MAX_REJECTIONS,
    GI_COLOR_REJECT_MARGIN,
    TRACKER_TRUST_COST,
    TRACKER_WEIGHT_COLOR,
    TRACKER_WEIGHT_IOU,
    TRACKER_WEIGHT_POSITION,
    TRACKER_WEIGHT_SCALE,
)
from ..entities.pose import Pose
from ..value_objects.gi_color import GiColor
from .geometry import iou, ratio

Point = tuple[float, float]

_EPSILON = 1e-9


def is_trusted(cost: float) -> bool:
    """Indica se um casamento e confiavel o bastante para atualizar a velocidade."""

    return cost <= TRACKER_TRUST_COST


class ColorGate:
    """Compara a cor do kimono dos candidatos com a do atleta rastreado.

    A cor so rejeita um candidato quando as duas referencias (a do atleta
    rastreado e a do adversario) sao **conhecidas** e claramente diferentes
    entre si, e o candidato parece mais com o adversario. Fora disso a cor
    entra apenas como um peso suave no custo.
    """

    def __init__(self, reference: GiColor | None, other: GiColor | None = None) -> None:
        self._reference = reference
        self._other = other
        self.enabled = reference is not None
        self.hard = (
            self.enabled
            and other is not None
            and reference.is_clearly_different_from(other)
        )
        self.rejections = 0

    @property
    def reference(self) -> GiColor | None:
        return self._reference

    def allows(self, pose: Pose) -> bool:
        """``False`` quando a cor prova que a pose e do outro atleta."""

        if not self.hard or self._reference is None or self._other is None:
            return True
        color = pose.gi_color
        if color is None:
            return True

        of_other = color.distance(self._other)
        of_reference = color.distance(self._reference)
        if of_other + GI_COLOR_REJECT_MARGIN < of_reference:
            self.rejections += 1
            # Depois de muitas rejeicoes seguidas o gate rigido se desliga: pode
            # ser que a referencia nao descreva bem o kimono deste trecho.
            if self.rejections >= GI_COLOR_MAX_REJECTIONS:
                self.hard = False
            return False
        self.rejections = 0
        return True

    def penalty(self, pose: Pose) -> float:
        """Custo (0..1) de a cor do candidato parecer mais com o OUTRO atleta.

        A cor so entra como diferenca **relativa** entre as duas referencias: um
        kimono claro amostrado na sombra continua sendo mais parecido com a sua
        referencia do que com a do adversario, e entao nao e penalizado. Isso
        evita inflar o custo ate de casamentos corretos (o que fazia o
        rastreador preferir descartar frames bons).
        """

        if not self.enabled or self._reference is None or self._other is None:
            return 0.0
        color = pose.gi_color
        if color is None:
            return 0.0
        advantage = color.distance(self._reference) - color.distance(self._other)
        return min(1.0, max(0.0, advantage) / GI_COLOR_DIFFERENT_DELTA_E)


class Track:
    """Posicao, escala, caixa e cor de uma pessoa ao longo dos frames."""

    def __init__(
        self,
        seed: Pose,
        max_velocity: float = 0.25,
        velocity_smoothing: float = 0.5,
        coast_decay: float = 0.85,
        color_gate: ColorGate | None = None,
    ) -> None:
        self.anchor: Point | None = seed.pelvis or seed.centroid
        self.velocity: Point = (0.0, 0.0)
        self.scale: float | None = seed.body_scale
        self.bbox = seed.bbox
        self.color = color_gate if color_gate is not None else ColorGate(None)
        self._max_velocity = max_velocity
        self._velocity_smoothing = velocity_smoothing
        self._coast_decay = coast_decay

    @property
    def active(self) -> bool:
        return self.anchor is not None

    def predicted(self) -> Point:
        assert self.anchor is not None
        return (self.anchor[0] + self.velocity[0], self.anchor[1] + self.velocity[1])

    def accepts(self, pose: Pose, limit: float, max_scale_ratio: float) -> bool:
        """Aplica os gates de distancia, escala e cor."""

        position = pose.pelvis
        if position is None or self.anchor is None:
            return False
        if math.dist(position, self.predicted()) > limit:
            return False
        if ratio(self.scale, pose.body_scale) > max_scale_ratio:
            return False
        return self.color.allows(pose)

    def cost(self, pose: Pose, limit: float) -> float:
        """Custo de associacao (menor = mais parecido com a pessoa rastreada)."""

        position = pose.pelvis
        assert position is not None
        distance = math.dist(position, self.predicted())
        scale_ratio = ratio(self.scale, pose.body_scale)
        normalized_distance = min(1.0, distance / limit) if limit > _EPSILON else 1.0
        return (
            TRACKER_WEIGHT_POSITION * normalized_distance
            + TRACKER_WEIGHT_IOU * (1.0 - iou(self.bbox, pose.bbox))
            + TRACKER_WEIGHT_SCALE * min(1.0, abs(scale_ratio - 1.0))
            + TRACKER_WEIGHT_COLOR * self.color.penalty(pose)
        )

    def update(self, pose: Pose, trusted: bool = True) -> None:
        position = pose.pelvis or pose.centroid
        if position is None or self.anchor is None:
            return

        delta = (position[0] - self.anchor[0], position[1] - self.anchor[1])
        if trusted:
            factor = self._velocity_smoothing
            velocity = (
                (1 - factor) * self.velocity[0] + factor * delta[0],
                (1 - factor) * self.velocity[1] + factor * delta[1],
            )
            self.velocity = _clamp_vector(velocity, self._max_velocity)
        else:
            # Casamento incerto: nao deixa a velocidade "envenenar" os proximos
            # frames - era o que fazia a identidade alternar entre os atletas.
            self.velocity = self._decayed_velocity()
        self.anchor = position

        scale = pose.body_scale
        if scale:
            self.scale = 0.8 * self.scale + 0.2 * scale if self.scale else scale

        bbox = pose.bbox
        if bbox is not None:
            self.bbox = bbox

    def coast(self) -> None:
        """Sem correspondencia: segue a previsao e desacelera aos poucos."""

        if self.anchor is None:
            return
        self.anchor = self.predicted()
        self.velocity = self._decayed_velocity()

    def _decayed_velocity(self) -> Point:
        return (
            self.velocity[0] * self._coast_decay,
            self.velocity[1] * self._coast_decay,
        )


def _clamp_vector(vector: Point, maximum: float) -> Point:
    magnitude = math.hypot(vector[0], vector[1])
    if magnitude <= maximum or magnitude < _EPSILON:
        return vector
    scale = maximum / magnitude
    return (vector[0] * scale, vector[1] * scale)
