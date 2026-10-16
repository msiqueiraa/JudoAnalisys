"""Exclusao manual de trechos (frames com erro) do calculo.

Duas formas de descartar frames:

- pelo **X** em cada amostra da faixa de verificacao (rapido, visual);
- pelo editor de trechos em ``mm:ss`` (preciso).

Ambos produzem uma lista de :class:`TimeRange` que e repassada a reanalise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import pandas as pd
import streamlit as st

from ...application.dto import AnalysisResult
from ...domain.entities.video import TimeRange
from ...formatting import format_duration_precise, parse_time

_START_COLUMN = "Inicio (mm:ss)"
_END_COLUMN = "Fim (mm:ss)"
_COLUMNS = (_START_COLUMN, _END_COLUMN)
_EPSILON = 1e-6


def render_frame_exclusion(
    result: AnalysisResult,
    on_exclude: Callable[[list[TimeRange]], None],
    key: str,
) -> None:
    """Permite descartar trechos (em tempo de video) do calculo das metricas."""

    with st.expander("Descartar frames com erro (opcional)"):
        st.caption(
            "Informe trechos problemáticos (na hora do vídeo, em `mm:ss`) ou use o "
            "**X** nas amostras acima. Os frames descartados saem do cálculo, dos "
            "gráficos e do score."
        )
        if result.excluded_ranges:
            excluded = ", ".join(
                f"{format_duration_precise(r.start)}–{format_duration_precise(r.end)}"
                for r in result.excluded_ranges
            )
            st.caption(f"Trechos descartados: {excluded}")

        edited = st.data_editor(
            _ranges_to_frame(result.excluded_ranges),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"{key}_exclude_editor",
        )
        ranges, invalid = parse_exclusion_rows(edited)
        if invalid:
            st.caption("Trechos inválidos foram ignorados. Use o formato `mm:ss` (ex.: 00:12).")

        if st.button("Reanalisar descartando esses trechos", key=f"{key}_exclude_apply"):
            on_exclude(merge_ranges(ranges))

        if result.excluded_ranges and st.button(
            "Restaurar todos os frames", key=f"{key}_exclude_reset"
        ):
            on_exclude([])

        if result.excluded_frame_count:
            st.caption(f"{result.excluded_frame_count} frames descartados neste vídeo.")


def sample_cell(
    time_range: TimeRange, timestamps: Sequence[float], index: int
) -> TimeRange:
    """Trecho representado por uma amostra da faixa de verificacao.

    Divide o intervalo analisado em celulas delimitadas pelos pontos medios
    entre amostras vizinhas: o X de uma amostra descarta exatamente os frames
    que ela representa.
    """

    if not 0 <= index < len(timestamps):
        raise IndexError("indice de amostra fora do intervalo.")

    current = timestamps[index]
    left = time_range.start if index == 0 else (timestamps[index - 1] + current) / 2
    right = (
        time_range.end
        if index == len(timestamps) - 1
        else (current + timestamps[index + 1]) / 2
    )

    left = max(time_range.start, min(left, current))
    right = min(time_range.end, max(right, current))
    if right <= left:  # amostras coincidentes: garante um intervalo valido
        right = min(time_range.end, left + 1.0)
    if right <= left:
        left = max(0.0, right - 1.0)
    return TimeRange(left, right)


def merge_ranges(ranges: Iterable[TimeRange]) -> list[TimeRange]:
    """Une trechos sobrepostos ou adjacentes em intervalos canonicos."""

    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    merged: list[TimeRange] = []
    for item in ordered:
        if merged and item.start <= merged[-1].end + _EPSILON:
            previous = merged[-1]
            merged[-1] = TimeRange(previous.start, max(previous.end, item.end))
        else:
            merged.append(item)
    return merged


def parse_exclusion_rows(data) -> tuple[list[TimeRange], bool]:
    """Le as linhas do editor e devolve ``(trechos, houve_invalido)``."""

    ranges: list[TimeRange] = []
    invalid = False
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for _, row in frame.iterrows():
        start_text = row.get(_START_COLUMN)
        end_text = row.get(_END_COLUMN)
        has_start = _has_text(start_text)
        has_end = _has_text(end_text)
        if not has_start and not has_end:
            continue

        start = parse_time(str(start_text)) if has_start else None
        end = parse_time(str(end_text)) if has_end else None
        if start is None or end is None or end <= start:
            invalid = True
            continue
        ranges.append(TimeRange(start, end))
    return merge_ranges(ranges), invalid


def _ranges_to_frame(ranges: list[TimeRange]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                _START_COLUMN: format_duration_precise(r.start),
                _END_COLUMN: format_duration_precise(r.end),
            }
            for r in ranges
        ],
        columns=list(_COLUMNS),
    )


def _has_text(value) -> bool:
    if value is None:
        return False
    try:
        if math.isnan(float(value)):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"<na>", "nan", "none"}
