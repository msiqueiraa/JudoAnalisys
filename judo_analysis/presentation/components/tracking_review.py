"""Verificacao visual do rastreamento e correcao/exclusao de frames."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import cv2
import streamlit as st

from ...application.dto import AnalysisResult, PersonSelection
from ...application.use_cases import DetectPeopleUseCase
from ...domain.entities.video import TimeRange
from ...formatting import format_duration_precise
from ...infrastructure.render import FrameAnnotator
from ...infrastructure.video import OpenCVVideoReader
from ..resources import make_pose_detector
from .exclusion import merge_ranges, render_frame_exclusion, sample_cell
from .pose_selection import (
    click_enabled,
    is_new_click,
    pose_label,
    render_clickable_image,
    render_pose_selector,
)

_THUMBNAIL_WIDTH = 360
# Passo do slider da correcao: tambem alinha o instante vindo da miniatura.
_OFFSET_STEP = 0.1


def render_tracking_review(
    result: AnalysisResult,
    num_poses: int,
    on_correct: Callable[[PersonSelection, float], None],
    on_exclude: Callable[[list[TimeRange]], None],
    key: str,
) -> None:
    """Mostra o rastreamento por amostras e permite corrigir/descartar frames."""

    st.markdown(f"**Verificacao do rastreamento ({result.run.label})**")
    _render_strip(result, on_exclude, key)
    _render_correction(result, num_poses, on_correct, key)
    render_frame_exclusion(result, on_exclude, key)


def _render_strip(
    result: AnalysisResult,
    on_exclude: Callable[[list[TimeRange]], None],
    key: str,
) -> None:
    samples = result.tracking_samples
    if not samples:
        st.caption("Sem amostras de rastreamento.")
        return

    st.caption(_strip_hint())
    annotator = FrameAnnotator()
    timestamps = [sample.timestamp for sample in samples]
    pending: list[int] = []

    with contextlib.ExitStack() as stack:
        try:
            reader = stack.enter_context(OpenCVVideoReader(result.run.video.path))
        except Exception:  # noqa: BLE001 - a imagem e opcional; os botoes continuam
            reader = None

        columns = st.columns(len(samples))
        for index, (column, sample) in enumerate(zip(columns, samples)):
            with column:
                image = _annotated_image(reader, annotator, sample)
                _render_thumbnail(image, index, result.run.time_range, sample, key)
                st.caption(_sample_caption(sample, _is_marked(key, sample.timestamp)))

                if st.button(
                    "X",
                    key=f"{key}_strip_del_{index}",
                    help="Descartar os frames deste trecho da analise",
                    width="stretch",
                ):
                    pending.append(index)

    if pending:
        cells = [sample_cell(result.run.time_range, timestamps, index) for index in pending]
        on_exclude(merge_ranges([*result.excluded_ranges, *cells]))


def _strip_hint() -> str:
    """Texto de ajuda da faixa (menciona o clique quando ele esta disponivel)."""

    hint = (
        "Confira se o esqueleto acompanha o mesmo atleta em todo o trecho. "
        "A cor ao lado do tempo mostra o kimono medido em cada amostra: se ela "
        "mudar, a identidade trocou."
    )
    if click_enabled():
        hint += " Clique numa miniatura para corrigir a pessoa naquele instante;"
    hint += " use o **X** para descartar os frames errados:"
    return hint


def pick_offset(
    time_range: TimeRange, timestamp: float, step: float = _OFFSET_STEP
) -> float:
    """Deslocamento (s) do instante clicado, no dominio do slider da correcao.

    O instante da miniatura vira segundos a partir do inicio do trecho, alinhado
    ao passo do slider e limitado as bordas do trecho.
    """

    offset = min(max(timestamp - time_range.start, 0.0), time_range.duration)
    aligned = round(round(offset / step) * step, 6)
    return round(min(max(aligned, 0.0), time_range.duration), 6)


def _render_thumbnail(
    image, index: int, time_range: TimeRange, sample, key: str
) -> None:
    """Miniatura clicavel: o clique aponta a correcao para este instante.

    Leva a secao "Corrigir pessoa" para o ponto da amostra, ja com a deteccao
    refeita nele (o usuario so precisa escolher o atleta).
    """

    if image is None:
        return
    component_key = f"{key}_strip_pick_{index}"
    value = render_clickable_image(image, component_key)
    if is_new_click(component_key, value):
        _mark_for_correction(key, time_range, sample.timestamp)


def _mark_for_correction(key: str, time_range: TimeRange, timestamp: float) -> None:
    """Aponta a secao "Corrigir pessoa" para o instante clicado na faixa."""

    st.session_state[f"{key}_offset"] = pick_offset(time_range, timestamp)
    st.session_state[f"{key}_pick_time"] = timestamp
    # O preview anterior e de outro instante: sera refeito no novo ponto.
    st.session_state.pop(f"{key}_preview", None)
    st.session_state[f"{key}_pending_detect"] = True


def _is_marked(key: str, timestamp: float) -> bool:
    """Indica se a amostra e a que esta apontada na secao de correcao."""

    picked = st.session_state.get(f"{key}_pick_time")
    return picked is not None and abs(float(picked) - timestamp) < 1e-6


def _sample_caption(sample, marked: bool = False) -> str:
    """Legenda da amostra: instante, cor do kimono e marca de correcao."""

    caption = format_duration_precise(sample.timestamp)
    color = sample.pose.gi_color if sample.pose is not None else None
    if color is not None:
        caption = f"{caption} - kimono {color.name()}"
    if marked:
        caption = f"{caption} ← corrigir"
    return caption


def _annotated_image(reader, annotator: FrameAnnotator, sample):
    if reader is None:
        return None
    try:
        _, _, frame = reader.frame_at(sample.timestamp)
        return _shrink(annotator.annotate(frame, sample.pose))
    except Exception:  # noqa: BLE001 - verificacao e opcional
        return None


def _render_correction(
    result: AnalysisResult,
    num_poses: int,
    on_correct: Callable[[PersonSelection, float], None],
    key: str,
) -> None:
    run = result.run
    duration = float(run.time_range.duration)
    offset_key = f"{key}_offset"
    if offset_key not in st.session_state:
        # Sem ``value=``: o instante pode ter sido definido pela miniatura.
        st.session_state[offset_key] = pick_offset(
            run.time_range, run.time_range.start + duration / 2.0
        )
    # O clique numa miniatura ja pede a deteccao no ponto escolhido.
    picked = bool(st.session_state.pop(f"{key}_pending_detect", False))
    correction_key = f"{key}_correction"
    if picked:
        # Abre a secao de correcao no instante escolhido na faixa.
        st.session_state[correction_key] = True

    with st.expander(
        "Corrigir pessoa (se o rastreamento pegou o atleta errado)",
        key=correction_key,
        on_change="rerun",
    ):
        offset = st.slider(
            "Instante para corrigir (s)",
            min_value=0.0,
            max_value=duration,
            step=_OFFSET_STEP,
            key=offset_key,
        )
        anchor_time = run.time_range.start + offset
        st.caption(f"Instante: {format_duration_precise(anchor_time)}")

        clicked = st.button("Detectar pessoas neste instante", key=f"{key}_detect")
        if clicked or picked:
            _detect_people(run, num_poses, anchor_time, key)

        preview = st.session_state.get(f"{key}_preview")
        if preview is None:
            return
        if not preview.poses:
            st.warning("Nenhuma pessoa detectada neste instante. Tente outro ponto.")
            return

        selected = render_pose_selector(
            preview.frame,
            preview.poses,
            key,
            caption="Clique sobre o atleta correto (ou escolha na lista).",
        )
        selected = st.selectbox(
            "Pessoa correta neste instante",
            options=list(range(len(preview.poses))),
            format_func=lambda index: pose_label(index, preview.poses[index]),
            key=f"{key}_person",
        )

        if st.button("Reanalisar com esta pessoa", key=f"{key}_apply", type="primary"):
            chosen = preview.poses[selected]
            on_correct(PersonSelection(person_id=chosen.person_id, pose=chosen), anchor_time)


def _detect_people(run, num_poses: int, anchor_time: float, key: str) -> None:
    """Detecta as pessoas em um instante e guarda a previa para a selecao."""

    try:
        detector = make_pose_detector(num_poses)
        with OpenCVVideoReader(run.video.path) as reader:
            preview = DetectPeopleUseCase(reader, detector).execute(anchor_time)
    except Exception as error:  # noqa: BLE001 - mostramos o erro ao usuario
        st.session_state.pop(f"{key}_preview", None)
        st.error(f"Falha ao detectar pessoas neste instante: {error}")
        return
    st.session_state[f"{key}_preview"] = preview


def _shrink(frame):
    scale = _THUMBNAIL_WIDTH / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (_THUMBNAIL_WIDTH, height), interpolation=cv2.INTER_AREA)
