"""Caso de uso: gerar o recorte lado a lado para o player comparativo."""

from __future__ import annotations

from pathlib import Path

from ...domain.entities.analysis import AnalysisRun
from ..ports.clip_composer import IClipComposer


class BuildComparisonClipUseCase:
    """Monta um video com os dois recortes lado a lado."""

    def __init__(self, composer: IClipComposer) -> None:
        self._composer = composer

    def execute(
        self,
        reference: AnalysisRun,
        analyzed: AnalysisRun,
        output_path: Path,
        similarity: float | None = None,
    ) -> Path:
        return self._composer.compose(reference, analyzed, output_path, similarity)
