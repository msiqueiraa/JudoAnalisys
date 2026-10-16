"""Componente de selecao da pessoa e do trecho a analisar."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from ...application.dto import PersonSelection
from ...application.use_cases import DetectPeopleUseCase
from ...domain.entities.video import TimeRange
from ...formatting import (
    format_duration_precise,
    parse_time,
    seconds_to_time,
    slider_time_format,
    time_to_seconds,
)
from ...infrastructure.video import OpenCVVideoReader
from ..resources import make_pose_detector
from .pose_selection import pose_label, render_pose_selector

# Janela sugerida (segundos) para o trecho inicial quando o video e longo.
_DEFAULT_WINDOW = 8.0
# Duracao minima de um trecho (evita slider/trecho degenerado).
_MIN_WINDOW = 0.5


def render_person_selection(
    label: str,
    key: str,
    video_path: str,
    duration: float,
    num_poses: int,
) -> tuple[PersonSelection | None, TimeRange | None, bool]:
    """Seleciona o trecho, a pessoa a acompanhar e a dica de cor do kimono.

    Returns:
        A pessoa escolhida, o trecho a analisar e se os kimonos dos dois atletas
        tem cores diferentes (pista opcional de identidade).
    """

    st.markdown(f"**{label}**")

    if duration <= 0:
        st.warning("Nao foi possivel determinar a duracao do video.")
        return None, None, False

    if duration < 1:
        st.warning("O video e curto demais para selecionar um trecho.")
        return None, None, False

    start, end = _render_time_range(key, duration)
    st.caption(
        f"Trecho: {format_duration_precise(start)} - {format_duration_precise(end)} "
        f"(duracao {format_duration_precise(max(0.0, end - start))})"
    )
    different_kimono = st.checkbox(
        "Os dois atletas tem kimonos de cores diferentes",
        value=False,
        key=f"{key}_different_kimono",
        help=(
            "Marque quando um atleta esta de branco e o outro de azul. A cor "
            "vira uma pista extra para manter a identidade quando os dois corpos "
            "ficam sobrepostos (nunca a unica forma de reconhecer o atleta). O "
            "app mede as duas cores no trecho e descarta a pista sozinho se elas "
            "nao forem claramente diferentes."
        ),
    )

    if st.button("Detectar pessoas neste trecho", key=f"{key}_detect"):
        detector = make_pose_detector(num_poses)
        with OpenCVVideoReader(video_path) as reader:
            preview = DetectPeopleUseCase(reader, detector).execute(start)
        st.session_state[f"{key}_preview"] = preview

    preview = st.session_state.get(f"{key}_preview")
    selection: PersonSelection | None = None

    if preview is not None and preview.poses:
        selected_option = render_pose_selector(
            preview.frame,
            preview.poses,
            key,
            caption="Clique sobre o atleta desejado (ou escolha na lista).",
        )
        selected_option = st.selectbox(
            "Pessoa",
            options=list(range(len(preview.poses))),
            format_func=lambda index: pose_label(index, preview.poses[index]),
            key=f"{key}_person",
        )

        chosen = preview.poses[selected_option]
        selection = PersonSelection(person_id=chosen.person_id, pose=chosen)
    elif preview is not None:
        st.warning("Nenhuma pessoa foi detectada neste frame. Tente outro trecho.")

    if end <= start:
        st.warning("O trecho selecionado e invalido.")
        return selection, None, different_kimono

    return selection, TimeRange(start=start, end=end), different_kimono


def _render_time_range(key: str, duration: float) -> tuple[float, float]:
    """Seleciona inicio e fim do trecho: slider grosso + caixas em ``mm:ss.cc``.

    O slider anda de segundo em segundo (ajuste rapido) e as caixas aceitam
    centesimos (ex.: ``00:01.55``) - util quando o golpe dura 2-3 s. Quando as
    caixas tem um valor valido **elas mandam**, porque o slider nao representa
    fracoes de segundo; sem caixas validas vale o slider. As duas formas ficam
    sincronizadas por ``on_change`` (que roda antes do proximo rerun) - e uma
    escrita programatica no slider nao dispara o callback.
    """

    slider_key = f"{key}_slider"
    start_key = f"{key}_start"
    end_key = f"{key}_end"
    default_end = min(duration, _DEFAULT_WINDOW)

    def _write_box(box_key: str, seconds: float) -> None:
        text = format_duration_precise(seconds)
        # Escreve so o que mudou: mantem o callback idempotente (sem loop).
        if st.session_state.get(box_key) != text:
            st.session_state[box_key] = text

    def _slider_to_boxes() -> None:
        """Arrastar o slider reescreve as caixas (ajuste grosso)."""

        chosen = st.session_state[slider_key]
        _write_box(start_key, time_to_seconds(chosen[0]))
        _write_box(end_key, time_to_seconds(chosen[1]))

    def _boxes_to_slider() -> None:
        """Digitar nas caixas reposiciona o slider, limitando ao video."""

        start = _box_seconds(start_key)
        end = _box_seconds(end_key)
        if start is None or end is None:
            return
        limited_start, limited_end = _clamp(start, duration), _clamp(end, duration)
        if limited_end <= limited_start:
            return
        # Corrige o texto apenas quando ele passa do fim do video; o que esta
        # sendo digitado fica intacto (reescrever a cada tecla atrapalharia).
        if limited_start != start:
            _write_box(start_key, limited_start)
        if limited_end != end:
            _write_box(end_key, limited_end)
        st.session_state[slider_key] = (
            seconds_to_time(limited_start),
            seconds_to_time(limited_end),
        )

    st.slider(
        "Trecho do golpe",
        min_value=dt.time(0, 0, 0),
        max_value=seconds_to_time(duration),
        value=(seconds_to_time(0.0), seconds_to_time(default_end)),
        format=slider_time_format(duration),
        key=slider_key,
        on_change=_slider_to_boxes,
    )
    start_column, end_column = st.columns(2)
    start_column.text_input(
        "Inicio (mm:ss)",
        value=format_duration_precise(0.0),
        key=start_key,
        on_change=_boxes_to_slider,
        help="Aceita centesimos para marcar o instante exato (ex.: 00:01.55).",
    )
    end_column.text_input(
        "Fim (mm:ss)",
        value=format_duration_precise(default_end),
        key=end_key,
        on_change=_boxes_to_slider,
        help="Aceita centesimos para marcar o instante exato (ex.: 00:04.20).",
    )

    return _effective_range(start_key, end_key, slider_key, duration, default_end)


def _effective_range(
    start_key: str,
    end_key: str,
    slider_key: str,
    duration: float,
    default_end: float,
) -> tuple[float, float]:
    """Trecho efetivo: as caixas mandam quando validas; senao vale o slider."""

    start = _box_seconds(start_key)
    end = _box_seconds(end_key)
    if start is not None and end is not None:
        start, end = _clamp(start, duration), _clamp(end, duration)
        if end > start:
            return start, end

    chosen = st.session_state.get(slider_key)
    if chosen is None:
        return 0.0, default_end

    start = _clamp(time_to_seconds(chosen[0]), duration)
    end = _clamp(time_to_seconds(chosen[1]), duration)
    if end <= start:
        end = min(duration, start + _MIN_WINDOW)
        if end <= start:
            start, end = max(0.0, duration - _MIN_WINDOW), duration
    return start, end


def _box_seconds(box_key: str) -> float | None:
    """Segundos digitados na caixa (``None`` quando o texto nao e um tempo)."""

    return parse_time(str(st.session_state.get(box_key, "")))


def _clamp(seconds: float, duration: float) -> float:
    return max(0.0, min(seconds, duration))