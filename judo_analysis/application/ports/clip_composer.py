"""Porta para composicao de um recorte lado a lado dos dois videos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...domain.entities.analysis import AnalysisRun


class IClipComposer(ABC):
    """Contrato que gera um video com dois recortes lado a lado."""

    @abstractmethod
    def compose(
        self,
        reference: AnalysisRun,
        analyzed: AnalysisRun,
        output_path: Path,
        similarity: float | None = None,
    ) -> Path:
        """Gera o recorte anotado e devolve o caminho do arquivo criado.

        As marcacoes do esqueleto e as metricas do instante
        (``FrameMetrics.pose``, ``angles`` e ``hand_speed``) sao desenhadas sobre
        cada lado; ``similarity`` (0 a 100) aparece no cartao final.
        """
