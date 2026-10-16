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
from ...infrastructure.render import FrameAnnotator, to_rgb
from ...infrastructure.video import OpenCVVideoReader
from ..resources import make_pose_detector
from .exclusion import merge_ranges, render_frame_exclusion, sample_cell
from .pose_selection import pose_label, render_pose_selector

_THUMBNAIL_WIDTH = 360


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

    st.caption(
        "Confira se o esqueleto acompanha o mesmo atleta em todo o trecho. "
        "A cor ao lado do tempo mostra o kimono medido em cada amostra: se ela "
        "mudar, a identidade trocou. Clique no **X** para descartar os frames "
        "errados:"
    )
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
                caption = _sample_caption(sample)
                image = _annotated_image(reader, annotator, sample)
                if image is not None:
                    st.image(to_rgb(image), caption=caption, width="stretch")
                else:
                    st.caption(caption)

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


def _sample_caption(sample) -> str:
    """Legenda da amostra: instante e cor do kimono medida (quando houver)."""

    caption = format_duration_precise(sample.timestamp)
    color = sample.pose.gi_color if sample.pose is not None else None
    if color is not None:
        caption = f"{caption} - kimono {color.name()}"
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
    with st.expander("Corrigir pessoa (se o rastreamento pegou o atleta errado)"):
        run = result.run
        duration = run.time_range.duration
        offset = st.slider(
            "Instante para corrigir (s)",
            min_value=0.0,
            max_value=float(duration),
            value=float(duration) / 2.0,
            step=0.1,
            key=f"{key}_offset",
        )
        anchor_time = run.time_range.start + offset
        st.caption(f"Instante: {format_duration_precise(anchor_time)}")

        if st.button("Detectar pessoas neste instante", key=f"{key}_detect"):
            detector = make_pose_detector(num_poses)
            with OpenCVVideoReader(run.video.path) as reader:
                preview = DetectPeopleUseCase(reader, detector).execute(anchor_time)
            st.session_state[f"{key}_preview"] = preview

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


def _shrink(frame):
    scale = _THUMBNAIL_WIDTH / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (_THUMBNAIL_WIDTH, height), interpolation=cv2.INTER_AREA)
