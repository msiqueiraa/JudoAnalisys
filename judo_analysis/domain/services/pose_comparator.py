"""Comparacao entre a analise de referencia e a analise do atleta.

A similaridade de cada metrica e calculada a partir da diferenca entre a
"caracteristica" dos dois videos:

- para angulos, usamos a media (postura tipica);
- para velocidade, usamos o pico (a forca do golpe).

A escala de normalizacao de cada metrica vem do catalogo de metricas.
"""

from __future__ import annotations

import math

from ...config import MIN_METRIC_COVERAGE, MIN_METRIC_SAMPLES
from ..entities.analysis import AnalysisRun, MetricStats
from ..metrics_catalog import METRICS
from ..value_objects.comparison import ComparisonResult, MetricComparison

# Metricas cuja caracteristica relevante e o pico, nao a media.
_PEAK_METRICS = {"hand_speed"}


class PoseComparator:
    """Compara duas execucoes e devolve um score de similaridade por metrica."""

    def compare(
        self,
        reference: AnalysisRun,
        analyzed: AnalysisRun,
        ignore: set[str] | None = None,
    ) -> ComparisonResult:
        ignored = ignore or set()
        comparisons: list[MetricComparison] = []
        available_similarities: list[float] = []
        reference_total = len(reference.frames)
        analyzed_total = len(analyzed.frames)

        for metric in METRICS:
            reference_stats = reference.stats.get(metric.key)
            analyzed_stats = analyzed.stats.get(metric.key)
            is_ignored = metric.key in ignored
            similarity = (
                None
                if is_ignored
                else self._similarity(
                    metric.key,
                    metric.scale,
                    reference_stats,
                    analyzed_stats,
                    reference_total,
                    analyzed_total,
                )
            )

            comparisons.append(
                MetricComparison(
                    key=metric.key,
                    label=metric.label,
                    unit=metric.unit,
                    reference_mean=reference_stats.mean if reference_stats else None,
                    analyzed_mean=analyzed_stats.mean if analyzed_stats else None,
                    reference_peak=reference_stats.maximum if reference_stats else None,
                    analyzed_peak=analyzed_stats.maximum if analyzed_stats else None,
                    similarity=similarity,
                    ignored=is_ignored,
                )
            )
            if similarity is not None:
                available_similarities.append(similarity)

        overall = (
            sum(available_similarities) / len(available_similarities)
            if available_similarities
            else None
        )
        return ComparisonResult(metrics=comparisons, overall_similarity=overall)

    @staticmethod
    def _similarity(
        key: str,
        scale: float,
        reference: MetricStats | None,
        analyzed: MetricStats | None,
        reference_total: int,
        analyzed_total: int,
    ) -> float | None:
        if reference is None or analyzed is None:
            return None
        # Metrica medida em poucos frames (dados espurios) nao entra na analise.
        if not _enough_data(reference, reference_total) or not _enough_data(
            analyzed, analyzed_total
        ):
            return None

        if key in _PEAK_METRICS:
            reference_value = reference.maximum
            analyzed_value = analyzed.maximum
        else:
            reference_value = reference.mean
            analyzed_value = analyzed.mean

        difference = abs(reference_value - analyzed_value)
        similarity = max(0.0, 1.0 - difference / scale) * 100.0
        return round(similarity, 1)


def _enough_data(stats: MetricStats, total_frames: int) -> bool:
    """Indica se a metrica tem dados suficientes para ser comparada."""

    required = max(MIN_METRIC_SAMPLES, math.ceil(MIN_METRIC_COVERAGE * total_frames))
    return stats.count >= required
