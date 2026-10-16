"""Componente de geracao e download do relatorio PDF."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ...application.dto import AnalysisResult
from ...domain.value_objects.comparison import ComparisonResult
from ...infrastructure.report import PdfReportWriter


def render_report_download(
    reference: AnalysisResult,
    analyzed: AnalysisResult,
    comparison: ComparisonResult,
    key: str = "report",
) -> None:
    """Botao para gerar e baixar o relatorio em PDF."""

    signature = tuple(sorted(metric.key for metric in comparison.ignored_metrics()))
    if st.session_state.get(f"{key}_signature") != signature:
        st.session_state.pop(f"{key}_bytes", None)
        st.session_state[f"{key}_signature"] = signature

    if st.button("Gerar relatorio PDF", key=f"{key}_generate"):
        try:
            with st.spinner("Gerando relatorio..."):
                output = Path(tempfile.mkdtemp()) / "relatorio_judo.pdf"
                PdfReportWriter().write(
                    (reference.run, reference),
                    (analyzed.run, analyzed),
                    comparison,
                    output,
                )
                st.session_state[f"{key}_bytes"] = output.read_bytes()
        except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
            st.error(f"Falha ao gerar o PDF: {error}")

    pdf_bytes = st.session_state.get(f"{key}_bytes")
    if pdf_bytes:
        st.download_button(
            "Baixar relatorio (PDF)",
            data=pdf_bytes,
            file_name="relatorio_judo.pdf",
            mime="application/pdf",
            key=f"{key}_download",
        )
