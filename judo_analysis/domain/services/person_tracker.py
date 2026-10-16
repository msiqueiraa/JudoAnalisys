"""Rastreamento da pessoa escolhida ao longo do video.

O rastreamento por centroide puro se perde em esportes de contato: quando os
dois atletas se aproximam, o centroide do atleta rastreado fica mais perto da
outra pessoa e a identidade troca. Este rastreador reduz isso combinando:

- **ancora na pelve** (centro do quadril), estavel mesmo com oclusoes;
- **predicao de movimento** (velocidade constante);
- **sobreposicao de caixa (IoU)** entre o frame atual e o anterior;
- **escala do corpo** (altura ombro->tornozelo) para descartar candidatos de
  tamanho muito diferente;
- **cor do kimono** (opcional) para separar atletas de cores diferentes;
- **histerese** de identidade: so troca de pessoa se o ganho for claro;
- **gate** de distancia/escala e continuidade quando os atletas se sobrepoem.

Quando a cena tem dois atletas, prefira :class:`~...pair_tracker.PairTracker`,
que rastreia os dois de uma vez e reduz ainda mais as trocas de identidade.
"""

from __future__ import annotations

import math

from ...config import TRACKER_SWITCH_MARGIN
from ..entities.pose import Pose
from ..value_objects.gi_color import GiColor
from .track import ColorGate, Track, is_trusted


class PersonTracker:
    """Mantem a identidade de uma pessoa entre frames."""

    def __init__(
        self,
        seed: Pose,
        max_distance: float = 0.45,
        max_scale_ratio: float = 3.0,
        max_velocity: float = 0.25,
        velocity_smoothing: float = 0.5,
        coast_decay: float = 0.85,
        keep_identity: bool = True,
        strict_factor: float = 0.5,
        reference_color: GiColor | None = None,
        other_color: GiColor | None = None,
        switch_margin: float = TRACKER_SWITCH_MARGIN,
    ) -> None:
        self._track = Track(
            seed,
            max_velocity=max_velocity,
            velocity_smoothing=velocity_smoothing,
            coast_decay=coast_decay,
            color_gate=ColorGate(reference_color, other_color),
        )
        self._max_distance = max_distance
        self._max_scale_ratio = max_scale_ratio
        self._keep_identity = keep_identity
        self._strict_factor = strict_factor
        self._switch_margin = switch_margin
        # Vira ``True`` quando a cena ja teve mais de uma pessoa detectada.
        self._multi_seen = False

    @property
    def anchor(self) -> tuple[float, float] | None:
        return self._track.anchor

    def select(self, poses: list[Pose]) -> Pose | None:
        """Escolhe a pose correspondente a pessoa rastreada (ou ``None``)."""

        if not self._track.active:
            return None

        if len(poses) >= 2:
            self._multi_seen = True

        # Com uma unica deteccao apos ja ter havido duas pessoas, exige-se uma
        # correspondencia mais proxima: evita "pular" para o corpo errado quando
        # os atletas se sobrepoem (o uke por cima do tori).
        limit = self._max_distance
        if self._keep_identity and self._multi_seen and len(poses) <= 1:
            limit *= self._strict_factor

        candidates = [
            (pose, self._track.cost(pose, limit))
            for pose in poses
            if self._track.accepts(pose, limit, self._max_scale_ratio)
        ]
        if not candidates:
            self._track.coast()
            return None

        best = min(candidates, key=lambda item: item[1])
        # Histerese: manter a identidade atual (o candidato mais perto da ultima
        # posicao) a menos que outro seja claramente melhor. E isso que impede a
        # identidade de "alternar" entre os atletas.
        staying = self._closest_to_anchor(candidates)
        if staying is not None and staying[0] is not best[0]:
            if staying[1] <= best[1] + self._switch_margin:
                best = staying

        self._track.update(best[0], trusted=is_trusted(best[1]))
        return best[0]

    def _closest_to_anchor(
        self, candidates: list[tuple[Pose, float]]
    ) -> tuple[Pose, float] | None:
        anchor = self._track.anchor
        if anchor is None:
            return None
        usable = [item for item in candidates if item[0].pelvis is not None]
        if not usable:
            return None
        return min(usable, key=lambda item: math.dist(item[0].pelvis, anchor))
