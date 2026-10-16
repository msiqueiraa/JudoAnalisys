"""Aplicacao Streamlit do Judo Analysis."""

from __future__ import annotations

import streamlit as st

from ..application.dto import AnalysisResult, PersonSelection
from ..application.use_cases import AnalyzePersonUseCase, CompareAnalysesUseCase
from ..config import DEFAULT_KEEP_IDENTITY, DEFAULT_NUM_POSES
from ..domain.entities.video import TimeRange
from ..domain.metrics_catalog import SPEED_METRICS
from ..infrastructure.video import OpenCVVideoReader
from .components import (
    render_analysis_panel,
    render_comparison,
    render_comparison_clip,
    render_example_loader,
    render_person_selection,
    render_report_download,
    render_tracking_review,
    render_video_input,
    reset_key_frame_sliders,
)
from .resources import make_pose_detector, read_metadata


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
        _run_analysis(
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

    _render_results(num_poses, ignore_speed, keep_identity)


def _run_analysis(
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
    _remember_kimono_hints(reference_diff, analyzed_diff)

    try:
        with st.spinner("Analisando referencia..."):
            reference_result = _analyze(
                "Referencia",
                reference_path,
                reference_selection,
                reference_range,
                num_poses,
                keep_identity,
                different_kimono=reference_diff,
            )
        with st.spinner("Analisando video analisado..."):
            analyzed_result = _analyze(
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

    st.session_state["reference_result"] = reference_result
    st.session_state["analyzed_result"] = analyzed_result
    st.session_state.pop("report_bytes", None)
    st.session_state.pop("clip_path", None)
    # O slider do frame-chave comeca de novo no pico de velocidade.
    reset_key_frame_sliders()
    st.rerun()


def _remember_kimono_hints(reference_diff: bool, analyzed_diff: bool) -> None:
    """Guarda a dica de cor usada na analise, para as reanalises reusarem.

    Fica em chaves proprias (``ref_kimono_hint`` / ``ana_kimono_hint``) e nao nas
    chaves dos checkboxes: o Streamlit proibe escrever na chave de um widget ja
    instanciado neste mesmo run (o botao "Analisar" roda depois dos checkboxes).
    """

    st.session_state["ref_kimono_hint"] = reference_diff
    st.session_state["ana_kimono_hint"] = analyzed_diff


def _kimono_hints() -> tuple[bool, bool]:
    """Dica de cor (referencia, analisado) usada na analise de cada video."""

    return (
        bool(st.session_state.get("ref_kimono_hint", False)),
        bool(st.session_state.get("ana_kimono_hint", False)),
    )


def _analyze(
    label: str,
    path: str,
    selection: PersonSelection,
    time_range: TimeRange,
    num_poses: int,
    keep_identity: bool,
    anchor_time: float | None = None,
    exclude: list[TimeRange] | None = None,
    different_kimono: bool = False,
    key_frame_time: float | None = None,
) -> AnalysisResult:
    detector = make_pose_detector(num_poses)
    with OpenCVVideoReader(path) as reader:
        use_case = AnalyzePersonUseCase(
            reader,
            detector,
            keep_identity=keep_identity,
            different_kimono=different_kimono,
        )
        return use_case.execute(
            label, time_range, selection, anchor_time, exclude, key_frame_time
        )


def _render_results(num_poses: int, ignore_speed: bool, keep_identity: bool) -> None:
    reference_result = st.session_state.get("reference_result")
    analyzed_result = st.session_state.get("analyzed_result")
    if not (reference_result and analyzed_result):
        return

    reference_diff, analyzed_diff = _kimono_hints()

    ignore = set(SPEED_METRICS) if ignore_speed else set()
    comparison = CompareAnalysesUseCase().execute(
        reference_result.run, analyzed_result.run, ignore
    )

    st.divider()
    st.header("Resultado da analise")

    render_comparison_clip(reference_result, analyzed_result, comparison)

    st.divider()
    left, right = st.columns(2)
    with left:
        render_analysis_panel(reference_result, "ref", ignore_speed)
        render_tracking_review(
            reference_result,
            num_poses,
            on_correct=_correction_callback(
                "reference_result", reference_result, num_poses, keep_identity, reference_diff
            ),
            on_exclude=_exclusion_callback(
                "reference_result", reference_result, num_poses, keep_identity, reference_diff
            ),
            key="ref_review",
        )
    with right:
        render_analysis_panel(analyzed_result, "ana", ignore_speed)
        render_tracking_review(
            analyzed_result,
            num_poses,
            on_correct=_correction_callback(
                "analyzed_result", analyzed_result, num_poses, keep_identity, analyzed_diff
            ),
            on_exclude=_exclusion_callback(
                "analyzed_result", analyzed_result, num_poses, keep_identity, analyzed_diff
            ),
            key="ana_review",
        )

    st.divider()
    st.subheader("Comparacao")
    render_comparison(reference_result.run, analyzed_result.run, comparison)

    st.divider()
    render_report_download(reference_result, analyzed_result, comparison)


def _correction_callback(
    state_key: str, result: AnalysisResult, num_poses: int, keep_identity: bool, different_kimono: bool
):
    def correct(selection: PersonSelection, anchor_time: float) -> None:
        _reanalyze(
            state_key,
            result,
            num_poses,
            keep_identity,
            selection=selection,
            anchor_time=anchor_time,
            different_kimono=different_kimono,
        )

    return correct


def _exclusion_callback(
    state_key: str, result: AnalysisResult, num_poses: int, keep_identity: bool, different_kimono: bool
):
    def exclude(ranges: list[TimeRange]) -> None:
        _reanalyze(
            state_key,
            result,
            num_poses,
            keep_identity,
            exclude=ranges,
            different_kimono=different_kimono,
        )

    return exclude


def _reanalyze(
    state_key: str,
    result: AnalysisResult,
    num_poses: int,
    keep_identity: bool,
    *,
    selection: PersonSelection | None = None,
    anchor_time: float | None = None,
    exclude: list[TimeRange] | None = None,
    different_kimono: bool = False,
) -> None:
    chosen = selection or result.selection
    if chosen is None:
        st.error("Nao foi possivel reanalisar: a pessoa selecionada nao esta disponivel.")
        return

    effective_exclude = result.excluded_ranges if exclude is None else exclude
    try:
        with st.spinner("Reanalisando..."):
            new_result = _analyze(
                result.run.label,
                result.run.video.path,
                chosen,
                result.run.time_range,
                num_poses,
                keep_identity,
                anchor_time,
                effective_exclude,
                different_kimono,
                # O frame-chave escolhido pelo usuario sobrevive a reanalise.
                result.key_frame_time,
            )
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Falha ao reanalisar: {error}")
        return

    st.session_state[state_key] = new_result
    st.session_state.pop("report_bytes", None)
    st.session_state.pop("clip_path", None)
    # O slider e recriado a partir do resultado novo (que mantem a escolha).
    reset_key_frame_sliders()
    st.rerun()


def _render_header() -> None:
    st.title("🥋 Judo Analysis")
    st.caption(
        "Compare a biomecanica de um golpe de judo entre um video de referencia e "
        "um video analisado: joelho, coluna, pescoco, abertura das pernas e "
        "velocidade do golpe."
    )


if __name__ == "__main__":
    main()
