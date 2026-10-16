"""Utilitarios estatisticos usados na agregacao das metricas."""

from __future__ import annotations

import statistics

from ..entities.analysis import AnalysisRun, FrameMetrics, MetricStats
from ..metrics_catalog import METRICS


def compute_stats(values: list[float]) -> MetricStats | None:
    """Estatisticas descritivas de uma serie. ``None`` se a serie for vazia."""

    if not values:
        return None
    return MetricStats(
        mean=statistics.fmean(values),
        maximum=max(values),
        minimum=min(values),
        std=statistics.pstdev(values) if len(values) > 1 else 0.0,
        count=len(values),
    )


def aggregate_run(run: AnalysisRun) -> AnalysisRun:
    """Preenche ``run.stats`` a partir dos frames e define o frame-chave."""

    stats: dict[str, MetricStats] = {}
    for metric in METRICS:
        result = compute_stats(run.series(metric.key))
        if result is not None:
            stats[metric.key] = result
    run.stats = stats
    run.key_frame_index = peak_speed_frame(run).frame_index if run.frames else 0
    return run


def peak_speed_frame(run: AnalysisRun) -> FrameMetrics | None:
    """Frame com maior velocidade da mao (o momento do golpe)."""

    if not run.frames:
        return None
    return max(
        run.frames,
        key=lambda frame: frame.hand_speed.pixels_per_second if frame.hand_speed else 0.0,
    )


def nearest_frame(run: AnalysisRun, timestamp: float) -> FrameMetrics | None:
    """Frame valido mais proximo de um instante (``None`` se nao houver frames)."""

    if not run.frames:
        return None
    return min(run.frames, key=lambda frame: abs(frame.timestamp - timestamp))


def frame_by_index(run: AnalysisRun, frame_index: int) -> FrameMetrics | None:
    """Frame com o indice de video pedido (ou o primeiro, se nao existir)."""

    for frame in run.frames:
        if frame.frame_index == frame_index:
            return frame
    return run.frames[0] if run.frames else None
