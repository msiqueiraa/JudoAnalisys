"""Rastreamento simultaneo dos DOIS atletas.

Rastrear apenas uma pessoa e escolher o candidato de menor custo permite que,
em um frame ambiguo, o rastreador "roube" a deteccao do adversario - e o erro
contamina a previsao de movimento, fazendo a identidade alternar nos frames
seguintes.

Aqui os dois atletas sao rastreados juntos: em cada frame as deteccoes sao
distribuidas entre as duas pessoas de uma vez (atribuicao global). Assim, a
troca de identidade deixa de ser uma decisao isolada: ela so acontece se os
DOIS casamentos piorarem ao mesmo tempo, o que e muito mais raro.

O rastreador devolve sempre a pose da pessoa principal (a escolhida pelo
usuario); o adversario e acompanhado apenas como "distrator", para nao ser
confundido com ela.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from ...config import TRACKER_COAST_COST
from ..entities.pose import Pose
from ..value_objects.gi_color import GiColor
from .track import ColorGate, Track, is_trusted

# Pequena vantagem para a pessoa principal em empates tecnicos.
_PRIMARY_BIAS = 0.05


class PairTracker:
    """Mantem a identidade de dois atletas entre frames."""

    def __init__(
        self,
        primary: Pose,
        secondary: Pose | None = None,
        max_distance: float = 0.45,
        max_scale_ratio: float = 3.0,
        max_velocity: float = 0.25,
        velocity_smoothing: float = 0.5,
        coast_decay: float = 0.85,
        reference_color: GiColor | None = None,
        other_color: GiColor | None = None,
        coast_cost: float = TRACKER_COAST_COST,
    ) -> None:
        self._tracks = [
            Track(
                primary,
                max_velocity=max_velocity,
                velocity_smoothing=velocity_smoothing,
                coast_decay=coast_decay,
                color_gate=ColorGate(reference_color, other_color),
            )
        ]
        if secondary is not None:
            self._tracks.append(
                Track(
                    secondary,
                    max_velocity=max_velocity,
                    velocity_smoothing=velocity_smoothing,
                    coast_decay=coast_decay,
                    color_gate=ColorGate(other_color, reference_color),
                )
            )
        self._max_distance = max_distance
        self._max_scale_ratio = max_scale_ratio
        self._coast_cost = coast_cost

    @property
    def anchor(self) -> tuple[float, float] | None:
        return self._tracks[0].anchor

    @property
    def secondary_anchor(self) -> tuple[float, float] | None:
        return self._tracks[1].anchor if len(self._tracks) > 1 else None

    def select(self, poses: list[Pose]) -> Pose | None:
        """Escolhe a pose da pessoa principal (ou ``None``) e atualiza o par."""

        if not self._tracks[0].active:
            return None

        assignment = self._assign(poses)
        for track, pose in zip(self._tracks, assignment):
            if pose is None:
                track.coast()
            else:
                track.update(pose, trusted=is_trusted(track.cost(pose, self._max_distance)))
        return assignment[0]

    def _assign(self, poses: list[Pose]) -> tuple[Pose | None, ...]:
        """Distribui as poses entre as pessoas rastreadas com custo total minimo."""

        best_total = math.inf
        best: tuple[Pose | None, ...] = tuple(None for _ in self._tracks)

        for option in _assignments(len(poses), len(self._tracks)):
            total = 0.0
            chosen: list[Pose | None] = []
            for position, index in enumerate(option):
                track = self._tracks[position]
                if index is None:
                    total += self._coast_cost
                    chosen.append(None)
                    continue
                pose = poses[index]
                if not track.accepts(pose, self._max_distance, self._max_scale_ratio):
                    total = math.inf
                    break
                cost = track.cost(pose, self._max_distance)
                total += cost - (_PRIMARY_BIAS if position == 0 else 0.0)
                chosen.append(pose)
            if total < best_total:
                best_total = total
                best = tuple(chosen)

        return best


def _assignments(count: int, tracks: int) -> Iterator[tuple[int | None, ...]]:
    """Todas as formas de distribuir ``count`` poses entre ``tracks`` pessoas."""

    def walk(slot: int, used: list[int | None]) -> Iterator[tuple[int | None, ...]]:
        if slot == tracks:
            yield tuple(used)
            return
        used.append(None)
        yield from walk(slot + 1, used)
        used.pop()
        for index in range(count):
            if index in used:
                continue
            used.append(index)
            yield from walk(slot + 1, used)
            used.pop()

    yield from walk(0, [])
