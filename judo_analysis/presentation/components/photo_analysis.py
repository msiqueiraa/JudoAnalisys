"""Aba de analise de foto: angulos de uma foto (ou de duas, comparadas).

Uma foto tem um unico quadro: os angulos sao medidos nela e a comparacao usa a
mesma tabela/score do video, sem a velocidade (que nao existe em imagem parada).
"""

from __future__ import annotations

import streamlit as st

from ...application.dto import AnalysisResult, PersonPreview, PersonSelection
from ...application.use_cases import (
    AnalyzePhotoUseCase,
    CompareAnalysesUseCase,
    DetectPeopleUseCase,
)
from ...infrastructure.image import SingleImageReader, read_image
from ..analysis import forget_outputs
from ..resources import make_pose_detector
from .photo_input import photo_signature, render_photo_input
from .pose_selection import pose_label, render_pose_selector
from .results import render_analysis_panel
from .results_view import SideSpec, render_results_view

# Chaves de estado da aba de foto (prefixos dos widgets de cada lado).
REFERENCE_KEY = "fref"
ANALYZED_KEY = "fana"
REFERENCE_STATE = "photo_reference_result"
ANALYZED_STATE = "photo_analyzed_result"
REPORT_KEY = "photo_report"


def render_photo_mode(num_poses: int, active: bool = True) -> None:
    """Desenha a aba de foto: upload, escolha do atleta em cada foto e resultado."""

    left, right = st.columns(2)
    with left:
        reference_path = render_photo_input("Foto de referencia (gabarito)", REFERENCE_KEY)
    with right:
        analyzed_path = render_photo_input("Foto analisada", ANALYZED_KEY)

    if not reference_path and not analyzed_path:
        st.info(
            "Carregue a foto analisada para ver os angulos do atleta. Com a foto "
            "de referencia tambem carregada, as duas sao comparadas."
        )
        return

    with left:
        reference_selection = render_photo_selection(
            "Referencia", REFERENCE_KEY, reference_path, num_poses
        )
    with right:
        analyzed_selection = render_photo_selection(
            "Analisada", ANALYZED_KEY, analyzed_path, num_poses
        )

    ready = bool(analyzed_path and analyzed_selection)
    if not analyzed_path:
        st.info("Carregue a foto analisada para medir os angulos do atleta.")
    elif not analyzed_selection:
        st.warning("Escolha o atleta na foto analisada para poder analisar.")

    if st.button("Analisar foto", type="primary", disabled=not ready):
        _run_photo_analysis(
            reference_path, reference_selection, analyzed_path, analyzed_selection
        )

    if active:
        _render_photo_results()


def render_photo_selection(
    label: str, key: str, path: str | None, num_poses: int
) -> PersonSelection | None:
    """Detecta as pessoas da foto e deixa o usuario escolher o atleta.

    A deteccao roda uma unica vez por foto (fica em cache na sessao): refazer a
    cada rerun deixaria a interface lenta sem ganho nenhum.
    """

    if not path:
        return None

    preview = _photo_preview(key, path, num_poses)
    if preview is None:
        return None
    if not preview.poses:
        st.warning(f"{label}: nenhuma pessoa foi detectada nesta foto.")
        return None

    render_pose_selector(
        preview.frame,
        preview.poses,
        key,
        caption="Clique sobre o atleta desejado (ou escolha na lista).",
    )
    selected = st.selectbox(
        "Pessoa",
        options=list(range(len(preview.poses))),
        format_func=lambda index: pose_label(index, preview.poses[index]),
        key=f"{key}_person",
    )
    chosen = preview.poses[selected]
    return PersonSelection(person_id=chosen.person_id, pose=chosen)


def analyze_photo(label: str, path: str, selection: PersonSelection) -> AnalysisResult:
    """Mede os angulos da pessoa escolhida em uma foto."""

    return AnalyzePhotoUseCase().execute(label, read_image(path), selection, path)


def _photo_preview(key: str, path: str, num_poses: int) -> PersonPreview | None:
    """Pessoas detectadas na foto (uma deteccao por foto/limite de pessoas)."""

    preview_key = f"{key}_photo_preview"
    signature = f"{photo_signature(path)}:{num_poses}"
    if st.session_state.get(f"{key}_photo_preview_signature") != signature:
        st.session_state[preview_key] = None
        st.session_state[f"{key}_photo_preview_signature"] = signature

    cached = st.session_state.get(preview_key)
    if cached is not None:
        return cached

    try:
        with st.spinner("Detectando pessoas na foto..."):
            detector = make_pose_detector(num_poses)
            with SingleImageReader(path) as reader:
                preview = DetectPeopleUseCase(reader, detector).execute(0.0)
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Nao foi possivel analisar a foto: {error}")
        return None

    st.session_state[preview_key] = preview
    return preview


def _run_photo_analysis(
    reference_path: str | None,
    reference_selection: PersonSelection | None,
    analyzed_path: str | None,
    analyzed_selection: PersonSelection | None,
) -> None:
    assert analyzed_path and analyzed_selection

    try:
        with st.spinner("Analisando a foto..."):
            analyzed_result = analyze_photo("Analisada", analyzed_path, analyzed_selection)
        reference_result = None
        if reference_path and reference_selection:
            with st.spinner("Analisando a foto de referencia..."):
                reference_result = analyze_photo(
                    "Referencia", reference_path, reference_selection
                )
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Falha na analise: {error}")
        return

    st.session_state[REFERENCE_STATE] = reference_result
    st.session_state[ANALYZED_STATE] = analyzed_result
    forget_outputs(REPORT_KEY)
    st.rerun()


def _render_photo_results() -> None:
    analyzed: AnalysisResult | None = st.session_state.get(ANALYZED_STATE)
    reference: AnalysisResult | None = st.session_state.get(REFERENCE_STATE)
    if analyzed is None:
        return

    if reference is None:
        st.divider()
        st.header("Resultado da analise")
        render_analysis_panel(analyzed, ANALYZED_KEY, ignore_speed=True)
        st.info("Carregue a foto de referencia para comparar as duas.")
        return

    comparison = CompareAnalysesUseCase().execute(reference.run, analyzed.run)
    render_results_view(
        SideSpec(reference, REFERENCE_KEY, REFERENCE_STATE, ignore_speed=True),
        SideSpec(analyzed, ANALYZED_KEY, ANALYZED_STATE, ignore_speed=True),
        comparison,
        report_key=REPORT_KEY,
        outputs=(REPORT_KEY,),
    )
