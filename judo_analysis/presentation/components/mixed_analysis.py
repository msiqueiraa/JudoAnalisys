"""Aba Foto x video: compara uma foto com um video (gabarito ou golpe).

Serve para conferir um instante do seu golpe (uma foto) contra o video de
referencia - ou o contrario. A velocidade do golpe sai da comparacao sozinha:
uma foto nao tem tempo, logo nao tem velocidade.
"""

from __future__ import annotations

import streamlit as st

from ...application.dto import AnalysisResult, PersonSelection
from ...application.use_cases import CompareAnalysesUseCase
from ...domain.entities.video import TimeRange
from ...domain.metrics_catalog import SPEED_METRICS
from ..analysis import (
    analyze_video,
    forget_outputs,
    remember_kimono_hint,
    reset_key_frame_sliders,
)
from ..resources import read_metadata
from .person_selection import render_person_selection
from .photo_analysis import analyze_photo, render_photo_selection
from .photo_input import render_photo_input
from .results_view import SideSpec, render_results_view
from .video_input import render_video_input

VIDEO_KEY = "mvideo"
PHOTO_KEY = "mphoto"
REFERENCE_STATE = "mixed_reference_result"
ANALYZED_STATE = "mixed_analyzed_result"
REPORT_KEY = "mixed_report"
# Qual papel a foto tem na comparacao (guardado junto com o resultado).
PHOTO_SIDE_STATE = "mixed_photo_side"

_PHOTO_SIDE_WIDGET = "mixed_photo_side_widget"
_PHOTO_AS_ANALYZED = "O golpe analisado"
_PHOTO_AS_REFERENCE = "O gabarito"


def render_mixed_mode(
    num_poses: int,
    keep_identity: bool,
    ignore_speed: bool,
    active: bool = True,
) -> None:
    """Desenha a aba foto x video: um lado em video e o outro em foto."""

    st.caption(
        "Compare uma foto do instante-chave com o video (gabarito ou golpe). "
        "A velocidade do golpe nao entra: uma foto nao tem tempo."
    )
    photo_is_analyzed = (
        st.radio(
            "Na comparacao, a foto e...",
            [_PHOTO_AS_ANALYZED, _PHOTO_AS_REFERENCE],
            horizontal=True,
            key=_PHOTO_SIDE_WIDGET,
        )
        == _PHOTO_AS_ANALYZED
    )
    _drop_results_for_the_other_side(photo_is_analyzed)

    video_column, photo_column = st.columns(2)
    with video_column:
        video_label = (
            "Video analisado" if photo_is_analyzed else "Video de referencia (gabarito)"
        )
        video_path = render_video_input(video_label, VIDEO_KEY)
    with photo_column:
        photo_label = (
            "Foto de referencia (gabarito)" if photo_is_analyzed else "Foto analisada"
        )
        photo_path = render_photo_input(photo_label, PHOTO_KEY)

    if not video_path or not photo_path:
        st.info("Carregue o video e a foto para comparar.")
        return

    try:
        video_meta = read_metadata(video_path)
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Nao foi possivel ler o video: {error}")
        return

    with video_column:
        video_selection, video_range, video_diff = render_person_selection(
            "Video", VIDEO_KEY, video_path, video_meta.duration, num_poses
        )
    with photo_column:
        photo_selection = render_photo_selection("Foto", PHOTO_KEY, photo_path, num_poses)

    ready = all([video_selection, video_range, photo_selection])
    if st.button("Analisar", type="primary", disabled=not ready, key=f"{VIDEO_KEY}_analyze"):
        _run_mixed_analysis(
            video_path,
            video_selection,
            video_range,
            video_diff,
            photo_path,
            photo_selection,
            num_poses,
            keep_identity,
            photo_is_analyzed,
        )

    if active:
        _render_mixed_results(num_poses, keep_identity, ignore_speed)


def _drop_results_for_the_other_side(photo_is_analyzed: bool) -> None:
    """Esquece o resultado quando o usuario troca o papel da foto.

    Os resultados guardam os rotulos "Referencia"/"Analisado" da analise; manter
    o painel antigo depois da troca mostraria o mesmo lado com dois papeis.
    """

    stored = st.session_state.get(PHOTO_SIDE_STATE)
    if stored is None or stored == photo_is_analyzed:
        return
    st.session_state[REFERENCE_STATE] = None
    st.session_state[ANALYZED_STATE] = None
    st.session_state[PHOTO_SIDE_STATE] = photo_is_analyzed


def _run_mixed_analysis(
    video_path: str,
    video_selection: PersonSelection | None,
    video_range: TimeRange | None,
    video_diff: bool,
    photo_path: str,
    photo_selection: PersonSelection | None,
    num_poses: int,
    keep_identity: bool,
    photo_is_analyzed: bool,
) -> None:
    assert video_selection and video_range and photo_selection

    # Guardado para as reanalises do lado em video manterem a dica de cor.
    remember_kimono_hint(VIDEO_KEY, video_diff)

    video_label = "Analisado" if photo_is_analyzed else "Referencia"
    photo_label = "Referencia" if photo_is_analyzed else "Analisada"
    try:
        with st.spinner("Analisando o video..."):
            video_result = analyze_video(
                video_label,
                video_path,
                video_selection,
                video_range,
                num_poses,
                keep_identity,
                different_kimono=video_diff,
            )
        with st.spinner("Analisando a foto..."):
            photo_result = analyze_photo(photo_label, photo_path, photo_selection)
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Falha na analise: {error}")
        return

    if photo_is_analyzed:
        st.session_state[REFERENCE_STATE] = video_result
        st.session_state[ANALYZED_STATE] = photo_result
    else:
        st.session_state[REFERENCE_STATE] = photo_result
        st.session_state[ANALYZED_STATE] = video_result
    st.session_state[PHOTO_SIDE_STATE] = photo_is_analyzed

    forget_outputs(REPORT_KEY)
    reset_key_frame_sliders(VIDEO_KEY)
    st.rerun()


def _render_mixed_results(num_poses: int, keep_identity: bool, ignore_speed: bool) -> None:
    reference: AnalysisResult | None = st.session_state.get(REFERENCE_STATE)
    analyzed: AnalysisResult | None = st.session_state.get(ANALYZED_STATE)
    if reference is None or analyzed is None:
        return

    photo_is_analyzed = bool(st.session_state.get(PHOTO_SIDE_STATE, True))
    video_side = analyzed if photo_is_analyzed else reference
    photo_side = reference if photo_is_analyzed else analyzed
    video_state = ANALYZED_STATE if photo_is_analyzed else REFERENCE_STATE
    photo_state = REFERENCE_STATE if photo_is_analyzed else ANALYZED_STATE

    video_spec = SideSpec(
        video_side,
        VIDEO_KEY,
        video_state,
        review=True,
        hint_slot=VIDEO_KEY,
        ignore_speed=ignore_speed,
    )
    photo_spec = SideSpec(photo_side, PHOTO_KEY, photo_state, ignore_speed=True)

    ignore = set(SPEED_METRICS) if ignore_speed else set()
    comparison = CompareAnalysesUseCase().execute(reference.run, analyzed.run, ignore)
    reference_spec, analyzed_spec = (
        (video_spec, photo_spec) if photo_is_analyzed else (photo_spec, video_spec)
    )
    render_results_view(
        reference_spec,
        analyzed_spec,
        comparison,
        report_key=REPORT_KEY,
        outputs=(REPORT_KEY,),
        num_poses=num_poses,
        keep_identity=keep_identity,
    )
