"""Caso de uso: comparar a analise de referencia com a analise do atleta."""

from __future__ import annotations

from ...domain.entities.analysis import AnalysisRun
from ...domain.services.pose_comparator import PoseComparator
from ...domain.value_objects.comparison import ComparisonResult


class CompareAnalysesUseCase:
    """Produz o score de similaridade entre referencia e analisado."""

    def __init__(self, comparator: PoseComparator | None = None) -> None:
        self._comparator = comparator or PoseComparator()

    def execute(
        self,
        reference: AnalysisRun,
        analyzed: AnalysisRun,
        ignore: set[str] | None = None,
    ) -> ComparisonResult:
        return self._comparator.compare(reference, analyzed, ignore)
