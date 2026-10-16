"""Porta para geracao de relatorios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...domain.entities.analysis import AnalysisRun
from ...domain.value_objects.comparison import ComparisonResult
from ..dto import AnalysisResult


class IReportWriter(ABC):
    """Contrato de um gerador de relatorio de analise."""

    @abstractmethod
    def write(
        self,
        reference: tuple[AnalysisRun, AnalysisResult],
        analyzed: tuple[AnalysisRun, AnalysisResult],
        comparison: ComparisonResult,
        output_path: Path,
    ) -> Path:
        """Gera o relatorio e devolve o caminho do arquivo criado."""
