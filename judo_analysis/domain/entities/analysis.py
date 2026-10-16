"""Entidades que descrevem o resultado de uma analise."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..value_objects.joint_angles import JointAngles
from ..value_objects.speed import Speed
from .pose import Pose
from .video import TimeRange, Video


@dataclass(frozen=True, slots=True)
class MetricStats:
    """Estatisticas descritivas de uma serie de metricas."""

    mean: float
    maximum: float
    minimum: float
    std: float
    count: int = 0


@dataclass(slots=True)
class FrameMetrics:
    """Metricas medidas em um unico frame."""

    frame_index: int
    timestamp: float
    angles: JointAngles
    hand_speed: Speed | None = None
    pose: Pose | None = None


@dataclass(slots=True)
class AnalysisRun:
    """Resultado completo da analise de uma pessoa em um video.

    Attributes:
        label: Rotulo curto (ex.: "referencia" ou "analisado").
        video: Metadados do video analisado.
        time_range: Intervalo de tempo analisado.
        person_id: Identificador da pessoa rastreada.
        frames: Metricas por frame, em ordem temporal.
        stats: Estatisticas agregadas por metrica.
        key_frame_index: Frame de maior velocidade (o "golpe").
    """

    label: str
    video: Video
    time_range: TimeRange
    person_id: int
    frames: list[FrameMetrics] = field(default_factory=list)
    stats: dict[str, MetricStats] = field(default_factory=dict)
    key_frame_index: int = 0

    def series(self, metric: str) -> list[float]:
        """Retorna a serie temporal (valores nao nulos) de uma metrica."""

        values: list[float] = []
        for frame in self.frames:
            value = _metric_value(frame, metric)
            if value is not None:
                values.append(value)
        return values

    def series_with_time(self, metric: str) -> list[tuple[float, float]]:
        """Retorna pares ``(timestamp, valor)`` nao nulos de uma metrica."""

        points: list[tuple[float, float]] = []
        for frame in self.frames:
            value = _metric_value(frame, metric)
            if value is not None:
                points.append((frame.timestamp, value))
        return points


def _metric_value(frame: FrameMetrics, metric: str) -> float | None:
    if metric == "hand_speed":
        return frame.hand_speed.pixels_per_second if frame.hand_speed else None
    if metric == "hand_speed_normalized":
        return frame.hand_speed.body_heights_per_second if frame.hand_speed else None
    return frame.angles.as_dict().get(metric)
