"""Componente do player comparativo (recorte dos dois videos lado a lado)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ...application.dto import AnalysisResult
from ...application.use_cases import BuildComparisonClipUseCase
from ...domain.value_objects.comparison import ComparisonResult
from ...infrastructure.video import FfmpegClipComposer


def render_comparison_clip(
    reference: AnalysisResult,
    analyzed: AnalysisResult,
    comparison: ComparisonResult,
    key: str = "clip",
) -> None:
    """Gera e exibe o recorte dos dois videos lado a lado em um unico player."""

    st.markdown("**Player comparativo**")
    st.caption(
        "Referencia a esquerda, analisado a direita, tocando ao mesmo tempo, "
        "com o esqueleto e as metricas do instante sobre cada video. No fim "
        "aparece a similaridade geral."
    )

    similarity = comparison.overall_similarity
    signature = None if similarity is None else round(similarity, 1)
    if st.session_state.get(f"{key}_signature") != signature:
        st.session_state.pop(f"{key}_path", None)
        st.session_state[f"{key}_signature"] = signature

    if st.button("Gerar recorte dos dois videos", key=f"{key}_generate"):
        try:
            with st.spinner("Montando o recorte comparado..."):
                output = Path(tempfile.mkdtemp()) / "recorte_comparado.mp4"
                BuildComparisonClipUseCase(FfmpegClipComposer()).execute(
                    reference.run,
                    analyzed.run,
                    output,
                    similarity,
                )
                st.session_state[f"{key}_path"] = str(output)
        except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
            st.error(f"Falha ao gerar o recorte: {error}")

    path = st.session_state.get(f"{key}_path")
    if path and Path(path).exists():
        try:
            st.video(path)
        except Exception:  # noqa: BLE001
            st.error("Nao foi possivel exibir o recorte gerado.")
