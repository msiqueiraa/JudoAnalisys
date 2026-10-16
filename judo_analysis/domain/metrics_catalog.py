"""Catalogo central das metricas analisadas.

Define rotulos, unidades e escalas usadas tanto na UI quanto no calculo de
similaridade. Manter tudo aqui evita "numeros magicos" espalhados pelo codigo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricMeta:
    """Metadados de uma metrica."""

    key: str
    label: str
    unit: str
    scale: float
    description: str


METRICS: tuple[MetricMeta, ...] = (
    MetricMeta(
        key="knee_left",
        label="Joelho esquerdo",
        unit="deg",
        scale=45.0,
        description="Angulo quadril-joelho-tornozelo (esquerdo).",
    ),
    MetricMeta(
        key="knee_right",
        label="Joelho direito",
        unit="deg",
        scale=45.0,
        description="Angulo quadril-joelho-tornozelo (direito).",
    ),
    MetricMeta(
        key="spine",
        label="Coluna",
        unit="deg",
        scale=30.0,
        description="Inclinacao do tronco em relacao a vertical.",
    ),
    MetricMeta(
        key="neck",
        label="Pescoco",
        unit="deg",
        scale=30.0,
        description="Inclinacao da cabeca/pescoco em relacao a vertical.",
    ),
    MetricMeta(
        key="leg_opening",
        label="Abertura das pernas",
        unit="deg",
        scale=45.0,
        description="Angulo entre as duas coxas medido no quadril.",
    ),
    MetricMeta(
        key="hand_speed",
        label="Velocidade do golpe",
        unit="px/s",
        scale=400.0,
        description="Velocidade da mao (punho) no pico do golpe.",
    ),
)

METRICS_BY_KEY: dict[str, MetricMeta] = {metric.key: metric for metric in METRICS}

# Metricas exibidas em graficos de serie temporal.
TIME_SERIES_METRICS: tuple[str, ...] = (
    "knee_left",
    "knee_right",
    "spine",
    "neck",
    "leg_opening",
    "hand_speed",
)

# Metricas dependentes de velocidade (invalidas se os videos tiverem ritmos
# diferentes, ex.: um em slow-motion e outro na velocidade real).
SPEED_METRICS: frozenset[str] = frozenset({"hand_speed"})
