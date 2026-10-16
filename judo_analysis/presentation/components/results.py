"""Componentes de resultado: frame anotado, painel de metricas e comparacao."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...application.dto import AnalysisResult
from ...application.use_cases import SelectKeyFrameUseCase
from ...domain.entities.analysis import AnalysisRun, FrameMetrics
from ...domain.metrics_catalog import (
    METRICS,
    METRICS_BY_KEY,
    SPEED_METRICS,
    TIME_SERIES_METRICS,
)
from ...domain.services.statistics import frame_by_index
from ...domain.value_objects.comparison import ComparisonResult, MetricComparison
from ...formatting import (
    format_duration_precise,
    seconds_to_time,
    slider_time_format,
    time_to_seconds,
)
from ...infrastructure.render import FrameAnnotator, to_rgb
from ...infrastructure.video import OpenCVVideoReader

_REFERENCE_COLOR = "#1f77b4"
_ANALYZED_COLOR = "#d62728"
# Passo do slider do frame-chave quando o FPS do video nao e conhecido.
_DEFAULT_FRAME_STEP = 0.01
# Diferenca (s) abaixo da qual o slider nao mudou de posicao.
_TIME_EPSILON = 1e-6


def render_analysis_panel(
    result: AnalysisResult, key: str, ignore_speed: bool = False
) -> None:
    """Mostra o frame-chave anotado ao lado das metricas daquele instante.

    O slider acima da imagem percorre o trecho: onde o usuario parar vira o
    frame-chave do golpe (usado aqui e no PDF).
    """

    run = result.run
    _render_key_frame_slider(result, key)
    metrics = frame_by_index(run, run.key_frame_index)
    annotator = FrameAnnotator()
    annotated = annotator.annotate(
        result.key_frame,
        result.key_frame_pose,
        metrics.angles if metrics else None,
        metrics.hand_speed if metrics else None,
        title=run.label,
    )

    image_column, data_column = st.columns([2, 1])
    with image_column:
        st.image(
            to_rgb(annotated),
            caption="Foto analisada" if run.is_photo else "Frame-chave do golpe",
            width="stretch",
        )
    with data_column:
        st.markdown(f"**Metricas no golpe ({run.label})**")
        st.caption(_source_caption(run))
        rows = [
            {"Metrica": meta.label, "Valor": _format_value(_metric_value(metrics, meta.key), meta.unit)}
            for meta in METRICS
            if not (ignore_speed and meta.key in SPEED_METRICS)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not ignore_speed and run.stats.get("hand_speed"):
            peak = run.stats["hand_speed"].maximum
            st.metric("Pico de velocidade", f"{peak:.0f} px/s")


def _source_caption(run: AnalysisRun) -> str:
    """Legenda da origem da analise: trecho de video ou foto parada."""

    if run.is_photo:
        return "Foto analisada (quadro unico)."
    return (
        f"Trecho analisado: {format_duration_precise(run.time_range.start)} - "
        f"{format_duration_precise(run.time_range.end)}"
    )


def _render_key_frame_slider(result: AnalysisResult, key: str) -> None:
    """Slider que percorre o trecho e define o instante do golpe.

    Fica acima da imagem para a troca valer no mesmo rerun: o slider devolve o
    valor novo e, se ele mudou, o frame e relido do video antes de desenhar.
    """

    run = result.run
    if len(run.frames) < 2:
        return

    first = run.frames[0].timestamp
    last = run.frames[-1].timestamp
    if last <= first:
        return

    slider_key = f"{key}_keyframe_slider"
    current = min(max(_current_key_frame_time(result, first), first), last)
    # Depois do primeiro render o estado do widget manda. Uma analise nova limpa
    # a chave (``reset_key_frame_sliders``) para o slider voltar ao pico.
    if slider_key not in st.session_state:
        st.session_state[slider_key] = seconds_to_time(current)

    slider_column, reset_column = st.columns([4, 1], vertical_alignment="bottom")
    with slider_column:
        chosen = st.slider(
            "Instante do golpe (arraste para escolher o frame-chave)",
            min_value=seconds_to_time(first),
            max_value=seconds_to_time(last),
            step=_frame_step(run),
            format=slider_time_format(last, precise=True),
            key=slider_key,
        )
    with reset_column:
        reset = st.button(
            "Automatico",
            key=f"{key}_keyframe_auto",
            help="Voltar ao frame de maior velocidade da mao",
            width="stretch",
        )

    if reset:
        _select_key_frame(result, None)
        # O widget ja foi desenhado neste run: limpa o estado e redesenha.
        st.session_state.pop(slider_key, None)
        st.rerun()
    elif abs(time_to_seconds(chosen) - current) > _TIME_EPSILON:
        _select_key_frame(result, time_to_seconds(chosen))

    frame = frame_by_index(run, run.key_frame_index)
    instant = format_duration_precise(frame.timestamp) if frame else "--"
    if result.key_frame_time is None:
        st.caption(
            f"Frame-chave automatico (pico de velocidade): {instant}. "
            "Arraste o slider para escolher outro instante."
        )
    else:
        st.caption(f"Frame-chave escolhido: {instant}.")


def _current_key_frame_time(result: AnalysisResult, first: float) -> float:
    """Instante atual do frame-chave: o escolhido ou o pico de velocidade."""

    if result.key_frame_time is not None:
        return result.key_frame_time
    frame = frame_by_index(result.run, result.run.key_frame_index)
    return frame.timestamp if frame is not None else first


def _frame_step(run: AnalysisRun) -> dt.timedelta:
    """Passo do slider: um frame do video (10 ms se o FPS nao for conhecido)."""

    fps = run.video.fps
    step = 1.0 / fps if fps and fps > 0 else _DEFAULT_FRAME_STEP
    return dt.timedelta(seconds=min(max(step, 0.001), 0.5))


def _select_key_frame(result: AnalysisResult, timestamp: float | None) -> None:
    """Troca o frame-chave; ``None`` volta ao pico de velocidade."""

    try:
        with OpenCVVideoReader(result.run.video.path) as reader:
            SelectKeyFrameUseCase(reader).execute(result, timestamp)
    except Exception as error:  # noqa: BLE001 - a troca do frame e opcional
        st.warning(f"Nao foi possivel trocar o frame-chave: {error}")


def render_comparison(
    reference_run: AnalysisRun,
    analyzed_run: AnalysisRun,
    comparison: ComparisonResult,
) -> None:
    """Mostra o score geral, a tabela comparativa e os graficos temporais.

    Metricas sem valor em um dos videos sao descartadas da analise.
    """

    if comparison.overall_similarity is not None:
        st.metric("Similaridade geral", f"{comparison.overall_similarity:.1f}%")

    analyzed = comparison.analyzed_metrics()
    excluded = comparison.excluded_metrics()

    rows = [
        {
            "Metrica": metric.label,
            "Media ref.": _format_value(metric.reference_mean, metric.unit),
            "Media analisado": _format_value(metric.analyzed_mean, metric.unit),
            "Pico ref.": _format_value(metric.reference_peak, metric.unit),
            "Pico analisado": _format_value(metric.analyzed_peak, metric.unit),
            "Similaridade": f"{metric.similarity:.1f}%",
        }
        for metric in analyzed
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if excluded:
        names = ", ".join(metric.label for metric in excluded)
        st.caption(f"Nao analisadas (dados insuficientes em uma das analises): {names}.")

    ignored = comparison.ignored_metrics()
    if ignored:
        names = ", ".join(metric.label for metric in ignored)
        st.caption(f"Ignoradas nesta comparacao: {names}.")

    analyzed_keys = comparison.analyzed_keys()
    # Sem serie temporal (as duas fotos tem um quadro so) nao ha evolucao a mostrar.
    has_series = len(reference_run.frames) > 1 or len(analyzed_run.frames) > 1
    chart_keys = (
        [key for key in TIME_SERIES_METRICS if key in analyzed_keys] if has_series else []
    )

    if chart_keys:
        st.markdown("**Evolucao das metricas**")
        for key in chart_keys:
            figure = _line_chart(reference_run, analyzed_run, key)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")

    if analyzed:
        st.markdown("**Similaridade por metrica**")
        st.caption(
            "A similaridade geral e exatamente a media das barras abaixo "
            "(metricas nao analisadas ficam de fora)."
        )
        st.plotly_chart(
            _similarity_chart(analyzed, comparison.overall_similarity), width="stretch"
        )


def _line_chart(reference: AnalysisRun, analyzed: AnalysisRun, key: str) -> go.Figure | None:
    meta = METRICS_BY_KEY[key]
    figure = go.Figure()
    has_data = False
    for run, name, color in (
        (reference, f"Referencia ({reference.label})", _REFERENCE_COLOR),
        (analyzed, f"Analisado ({analyzed.label})", _ANALYZED_COLOR),
    ):
        points = run.series_with_time(key)
        if not points:
            continue
        has_data = True
        times = [timestamp - run.time_range.start for timestamp, _ in points]
        values = [value for _, value in points]
        # Uma foto tem um unico ponto: sem marcador a linha nao apareceria.
        single_point = len(points) == 1
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="markers" if single_point else "lines",
                name=name,
                line={"color": color},
                marker={"color": color, "size": 12, "symbol": "diamond"},
            )
        )

    if not has_data:
        return None

    figure.update_layout(
        title=meta.label,
        xaxis_title="Tempo no trecho (s)",
        yaxis_title=meta.unit,
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        height=320,
    )
    return figure


def _similarity_chart(
    analyzed: list[MetricComparison], overall: float | None
) -> go.Figure:
    labels = [metric.label for metric in analyzed]
    values = [metric.similarity or 0.0 for metric in analyzed]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": values, "colorscale": "RdYlGn", "cmin": 0, "cmax": 100},
            text=[f"{value:.1f}%" for value in values],
            textposition="auto",
        )
    )
    title = (
        f"Similaridade geral (media das barras): {overall:.1f}%"
        if overall is not None
        else "Similaridade por metrica"
    )
    if overall is not None:
        figure.add_vline(
            x=overall,
            line_dash="dash",
            line_color="#444444",
            annotation_text=f"Geral: {overall:.1f}%",
            annotation_position="top",
        )
    figure.update_layout(
        title=title,
        xaxis={"title": "Similaridade (%)", "range": [0, 100]},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        height=max(240, 60 * len(labels)),
    )
    return figure


def _metric_value(frame: FrameMetrics | None, key: str) -> float | None:
    if frame is None:
        return None
    if key == "hand_speed":
        return frame.hand_speed.pixels_per_second if frame.hand_speed else None
    return frame.angles.as_dict().get(key)


def _format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "--"
    if unit == "deg":
        return f"{value:.1f} deg"
    return f"{value:.0f} {unit}"
