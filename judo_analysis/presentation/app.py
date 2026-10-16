"""Aplicacao Streamlit do Judo Analysis."""

from __future__ import annotations

import streamlit as st

from ..application.dto import PersonSelection
from ..application.use_cases import CompareAnalysesUseCase
from ..config import DEFAULT_KEEP_IDENTITY, DEFAULT_NUM_POSES
from ..domain.entities.video import TimeRange
from ..domain.metrics_catalog import SPEED_METRICS
from .analysis import (
    analyze_video,
    forget_outputs,
    remember_kimono_hint,
    reset_key_frame_sliders,
)
from .components import (
    SideSpec,
    render_example_loader,
    render_mixed_mode,
    render_person_selection,
    render_photo_mode,
    render_results_view,
    render_video_input,
)
from .resources import read_metadata

# Chaves de estado da aba de video.
REFERENCE_STATE = "reference_result"
ANALYZED_STATE = "analyzed_result"
REPORT_KEY = "report"
_CLIP_KEY = "clip"


def main() -> None:
    st.set_page_config(page_title="Judo Analysis", page_icon="🥋", layout="wide")
    _render_header()

    num_poses = st.sidebar.slider(
        "Maximo de pessoas por frame",
        min_value=1,
        max_value=6,
        value=DEFAULT_NUM_POSES,
        help="Quantas pessoas o detector deve procurar em cada frame.",
    )
    keep_identity = st.sidebar.checkbox(
        "Manter identidade na sobreposicao",
        value=DEFAULT_KEEP_IDENTITY,
        help=(
            "Quando os dois atletas se sobrepoem e so uma pessoa e detectada, "
            "mantem a identidade anterior em vez de trocar de atleta."
        ),
    )
    ignore_speed = st.sidebar.checkbox(
        "Ignorar velocidade do golpe",
        value=False,
        help=(
            "Use quando os videos tiverem ritmos diferentes (ex.: slow-motion x "
            "normal). A velocidade sai da tabela, dos graficos e da nota."
        ),
    )
    render_example_loader()

    video_tab, photo_tab, mixed_tab = st.tabs(
        ["Video", "Foto", "Foto x video"], on_change="rerun", key="analysis_mode"
    )
    with video_tab:
        _render_video_mode(num_poses, keep_identity, ignore_speed, _is_open(video_tab))
    with photo_tab:
        render_photo_mode(num_poses, _is_open(photo_tab))
    with mixed_tab:
        render_mixed_mode(num_poses, keep_identity, ignore_speed, _is_open(mixed_tab))


def _is_open(tab) -> bool:
    """Indica se a aba esta aberta neste run.

    A aba aberta e a unica que desenha o resultado (tabelas, graficos e PDF): as
    outras rodam apenas os controles, que precisam continuar vivos para o
    Streamlit nao descartar o que o usuario ja escolheu.
    """

    return tab.open is not False


def _render_video_mode(
    num_poses: int, keep_identity: bool, ignore_speed: bool, active: bool = True
) -> None:
    """Aba de video: dois videos (referencia e analisado) lado a lado."""

    reference_column, analyzed_column = st.columns(2)
    with reference_column:
        reference_path = render_video_input("Video de referencia (gabarito)", "ref")
    with analyzed_column:
        analyzed_path = render_video_input("Video analisado", "ana")

    if not reference_path or not analyzed_path:
        st.info("Carregue os dois videos para comecar: um de referencia e um para analisar.")
        return

    try:
        reference_meta = read_metadata(reference_path)
        analyzed_meta = read_metadata(analyzed_path)
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Nao foi possivel ler os videos: {error}")
        return

    with reference_column:
        reference_selection, reference_range, reference_diff = render_person_selection(
            "Referencia", "ref", reference_path, reference_meta.duration, num_poses
        )
    with analyzed_column:
        analyzed_selection, analyzed_range, analyzed_diff = render_person_selection(
            "Analisado", "ana", analyzed_path, analyzed_meta.duration, num_poses
        )

    ready = all([reference_selection, reference_range, analyzed_selection, analyzed_range])
    if st.button("Analisar", type="primary", disabled=not ready):
        _run_video_analysis(
            reference_path,
            reference_selection,
            reference_range,
            analyzed_path,
            analyzed_selection,
            analyzed_range,
            num_poses,
            keep_identity,
            reference_diff,
            analyzed_diff,
        )

    if active:
        _render_video_results(num_poses, ignore_speed, keep_identity)


def _run_video_analysis(
    reference_path: str,
    reference_selection: PersonSelection | None,
    reference_range: TimeRange | None,
    analyzed_path: str,
    analyzed_selection: PersonSelection | None,
    analyzed_range: TimeRange | None,
    num_poses: int,
    keep_identity: bool,
    reference_diff: bool = False,
    analyzed_diff: bool = False,
) -> None:
    assert reference_selection and reference_range
    assert analyzed_selection and analyzed_range

    # Guardado para as reanalises (corrigir pessoa / excluir frames) manterem a
    # mesma dica de cor do kimono.
    remember_kimono_hint("ref", reference_diff)
    remember_kimono_hint("ana", analyzed_diff)

    try:
        with st.spinner("Analisando referencia..."):
            reference_result = analyze_video(
                "Referencia",
                reference_path,
                reference_selection,
                reference_range,
                num_poses,
                keep_identity,
                different_kimono=reference_diff,
            )
        with st.spinner("Analisando video analisado..."):
            analyzed_result = analyze_video(
                "Analisado",
                analyzed_path,
                analyzed_selection,
                analyzed_range,
                num_poses,
                keep_identity,
                different_kimono=analyzed_diff,
            )
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Falha na analise: {error}")
        return

    st.session_state[REFERENCE_STATE] = reference_result
    st.session_state[ANALYZED_STATE] = analyzed_result
    forget_outputs(REPORT_KEY, _CLIP_KEY)
    # O slider do frame-chave comeca de novo no pico de velocidade.
    reset_key_frame_sliders()
    st.rerun()


def _render_video_results(num_poses: int, ignore_speed: bool, keep_identity: bool) -> None:
    reference_result = st.session_state.get(REFERENCE_STATE)
    analyzed_result = st.session_state.get(ANALYZED_STATE)
    if not (reference_result and analyzed_result):
        return

    ignore = set(SPEED_METRICS) if ignore_speed else set()
    comparison = CompareAnalysesUseCase().execute(
        reference_result.run, analyzed_result.run, ignore
    )

    render_results_view(
        SideSpec(
            reference_result,
            "ref",
            REFERENCE_STATE,
            review=True,
            hint_slot="ref",
            ignore_speed=ignore_speed,
        ),
        SideSpec(
            analyzed_result,
            "ana",
            ANALYZED_STATE,
            review=True,
            hint_slot="ana",
            ignore_speed=ignore_speed,
        ),
        comparison,
        clip=True,
        report_key=REPORT_KEY,
        outputs=(REPORT_KEY, _CLIP_KEY),
        num_poses=num_poses,
        keep_identity=keep_identity,
    )


def _render_header() -> None:
    st.title("🥋 Judo Analysis")
    st.caption(
        "Compare a biomecanica de um golpe de judo entre videos (referencia e "
        "analisado) ou em fotos do instante-chave: joelho, coluna, pescoco, "
        "abertura das pernas e velocidade do golpe."
    )


if __name__ == "__main__":
    main()
