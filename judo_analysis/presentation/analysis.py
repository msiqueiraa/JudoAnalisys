"""Execucao das analises de video: analisar, reanalisar e guardar dicas de cor.

As abas do app (video, foto e foto x video) compartilham estas rotinas. Nada
aqui desenha interface: apenas roda casos de uso e guarda estado na sessao.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st

from ..application.dto import AnalysisResult, PersonSelection
from ..application.use_cases import AnalyzePersonUseCase
from ..domain.entities.video import TimeRange
from ..infrastructure.video import OpenCVVideoReader
from .resources import make_pose_detector

# Paineis e saidas (PDF/recorte) da aba de video, usados por padrao.
VIDEO_PANELS = ("ref", "ana")
VIDEO_OUTPUTS = ("report", "clip")


def analyze_video(
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
    """Roda a analise de um video para a pessoa e o trecho escolhidos."""

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


def reanalyze_video(
    state_key: str,
    result: AnalysisResult,
    num_poses: int,
    keep_identity: bool,
    *,
    selection: PersonSelection | None = None,
    anchor_time: float | None = None,
    exclude: list[TimeRange] | None = None,
    different_kimono: bool = False,
    panels: Sequence[str] = VIDEO_PANELS,
    outputs: Sequence[str] = VIDEO_OUTPUTS,
) -> None:
    """Reanalisa um video (corrigir pessoa / excluir frames).

    O frame-chave escolhido pelo usuario sobrevive a reanalise.
    """

    chosen = selection or result.selection
    if chosen is None:
        st.error("Nao foi possivel reanalisar: a pessoa selecionada nao esta disponivel.")
        return

    effective_exclude = result.excluded_ranges if exclude is None else exclude
    try:
        with st.spinner("Reanalisando..."):
            new_result = analyze_video(
                result.run.label,
                result.run.video.path,
                chosen,
                result.run.time_range,
                num_poses,
                keep_identity,
                anchor_time,
                effective_exclude,
                different_kimono,
                result.key_frame_time,
            )
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.error(f"Falha ao reanalisar: {error}")
        return

    st.session_state[state_key] = new_result
    forget_outputs(*outputs)
    # Os sliders do frame-chave sao recriados a partir do resultado novo.
    reset_key_frame_sliders(*panels)
    st.rerun()


def correction_callback(
    state_key: str,
    result: AnalysisResult,
    num_poses: int,
    keep_identity: bool,
    different_kimono: bool,
    *,
    panels: Sequence[str] = VIDEO_PANELS,
    outputs: Sequence[str] = VIDEO_OUTPUTS,
) -> Callable[[PersonSelection, float], None]:
    """Callback de "Corrigir pessoa" para a faixa de verificacao do rastreamento."""

    def correct(selection: PersonSelection, anchor_time: float) -> None:
        reanalyze_video(
            state_key,
            result,
            num_poses,
            keep_identity,
            selection=selection,
            anchor_time=anchor_time,
            different_kimono=different_kimono,
            panels=panels,
            outputs=outputs,
        )

    return correct


def exclusion_callback(
    state_key: str,
    result: AnalysisResult,
    num_poses: int,
    keep_identity: bool,
    different_kimono: bool,
    *,
    panels: Sequence[str] = VIDEO_PANELS,
    outputs: Sequence[str] = VIDEO_OUTPUTS,
) -> Callable[[list[TimeRange]], None]:
    """Callback de exclusao de frames (a analise roda de novo sem eles)."""

    def exclude(ranges: list[TimeRange]) -> None:
        reanalyze_video(
            state_key,
            result,
            num_poses,
            keep_identity,
            exclude=ranges,
            different_kimono=different_kimono,
            panels=panels,
            outputs=outputs,
        )

    return exclude


def remember_kimono_hint(slot: str, different_kimono: bool) -> None:
    """Guarda a dica de cor usada na analise, para as reanalises reusarem.

    Fica em chaves proprias (``<slot>_kimono_hint``) e nao nas chaves dos
    checkboxes: o Streamlit proibe escrever na chave de um widget ja instanciado
    neste mesmo run (o botao "Analisar" roda depois dos checkboxes).
    """

    st.session_state[f"{slot}_kimono_hint"] = different_kimono


def kimono_hint(slot: str) -> bool:
    """Dica de cor do kimono guardada para um lado da analise."""

    return bool(st.session_state.get(f"{slot}_kimono_hint", False))


def forget_outputs(*prefixes: str) -> None:
    """Esquece PDFs e recortes ja gerados: sao refeitos com a analise nova."""

    for prefix in prefixes:
        st.session_state.pop(f"{prefix}_bytes", None)
        st.session_state.pop(f"{prefix}_path", None)
        st.session_state.pop(f"{prefix}_signature", None)


def reset_key_frame_sliders(*prefixes: str) -> None:
    """Esquece o instante escolhido nos sliders do frame-chave.

    Chamado quando uma analise e (re)feita: os sliders sao recriados a partir do
    resultado novo, que ja carrega a escolha do usuario quando houver.
    """

    for prefix in prefixes or VIDEO_PANELS:
        st.session_state.pop(f"{prefix}_keyframe_slider", None)
