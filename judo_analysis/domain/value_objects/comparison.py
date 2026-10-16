"""Value objects do resultado de comparacao entre referencia e analisado."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Comparacao de uma metrica entre os dois videos.

    Attributes:
        key: Chave da metrica (ex.: ``knee_left``).
        label: Nome legivel (ex.: "Joelho esquerdo").
        unit: Unidade de medida.
        reference_mean: Media da metrica no video de referencia.
        analyzed_mean: Media da metrica no video analisado.
        reference_peak: Pico da metrica no video de referencia.
        analyzed_peak: Pico da metrica no video analisado.
        similarity: Similaridade de 0 a 100 (100 = identico).
    """

    key: str
    label: str
    unit: str
    reference_mean: float | None
    analyzed_mean: float | None
    reference_peak: float | None
    analyzed_peak: float | None
    similarity: float | None
    ignored: bool = False

    @property
    def mean_difference(self) -> float | None:
        if self.reference_mean is None or self.analyzed_mean is None:
            return None
        return self.analyzed_mean - self.reference_mean

    @property
    def peak_difference(self) -> float | None:
        if self.reference_peak is None or self.analyzed_peak is None:
            return None
        return self.analyzed_peak - self.reference_peak


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Resultado consolidado da comparacao entre referencia e analisado."""

    metrics: list[MetricComparison] = field(default_factory=list)
    overall_similarity: float | None = None

    def by_key(self, key: str) -> MetricComparison | None:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        return None

    def analyzed_metrics(self) -> list[MetricComparison]:
        """Metricas que puderam ser comparadas (valor presente nos dois lados)."""

        return [metric for metric in self.metrics if metric.similarity is not None]

    def excluded_metrics(self) -> list[MetricComparison]:
        """Metricas descartadas por faltar valor em um dos lados."""

        return [
            metric
            for metric in self.metrics
            if metric.similarity is None and not metric.ignored
        ]

    def ignored_metrics(self) -> list[MetricComparison]:
        """Metricas descartadas por escolha do usuario (ex.: velocidade)."""

        return [metric for metric in self.metrics if metric.ignored]

    def analyzed_keys(self) -> set[str]:
        return {metric.key for metric in self.analyzed_metrics()}
