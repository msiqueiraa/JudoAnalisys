"""Testes da leitura dos trechos descartados manualmente (exclusao de frames)."""

from __future__ import annotations

import pandas as pd
import pytest

from judo_analysis.domain.entities.video import TimeRange
from judo_analysis.presentation.components.exclusion import (
    _ranges_to_frame,
    merge_ranges,
    parse_exclusion_rows,
    sample_cell,
)


def test_parse_rows_ignores_empty_and_invalid() -> None:
    frame = pd.DataFrame(
        [
            {"Inicio (mm:ss)": "00:12", "Fim (mm:ss)": "00:18"},
            {"Inicio (mm:ss)": "", "Fim (mm:ss)": ""},
            {"Inicio (mm:ss)": "abc", "Fim (mm:ss)": "00:20"},
            {"Inicio (mm:ss)": "00:30", "Fim (mm:ss)": "00:20"},
        ]
    )

    ranges, invalid = parse_exclusion_rows(frame)

    assert ranges == [TimeRange(12.0, 18.0)]
    assert invalid is True


def test_parse_rows_accepts_hours_and_raw_seconds() -> None:
    frame = pd.DataFrame(
        [
            {"Inicio (mm:ss)": "1:00:00", "Fim (mm:ss)": "1:00:30"},
            {"Inicio (mm:ss)": "5", "Fim (mm:ss)": "8"},
        ]
    )

    ranges, invalid = parse_exclusion_rows(frame)

    # Os trechos saem em ordem canonica (por inicio).
    assert ranges == [TimeRange(5.0, 8.0), TimeRange(3600.0, 3630.0)]
    assert invalid is False


def test_ranges_to_frame_roundtrips() -> None:
    frame = _ranges_to_frame([TimeRange(12.0, 18.0)])

    ranges, invalid = parse_exclusion_rows(frame)

    assert ranges == [TimeRange(12.0, 18.0)]
    assert invalid is False


def test_empty_frame_has_no_ranges() -> None:
    ranges, invalid = parse_exclusion_rows(_ranges_to_frame([]))

    assert ranges == []
    assert invalid is False


def test_sample_cell_partitions_the_analyzed_range() -> None:
    time_range = TimeRange(10.0, 16.0)
    timestamps = [10.0, 12.0, 14.0, 16.0]

    assert sample_cell(time_range, timestamps, 0) == TimeRange(10.0, 11.0)
    assert sample_cell(time_range, timestamps, 1) == TimeRange(11.0, 13.0)
    assert sample_cell(time_range, timestamps, 2) == TimeRange(13.0, 15.0)
    assert sample_cell(time_range, timestamps, 3) == TimeRange(15.0, 16.0)


def test_sample_cell_covers_the_whole_range_when_all_are_removed() -> None:
    time_range = TimeRange(20.0, 23.0)
    timestamps = [20.0, 21.0, 22.0, 23.0]

    cells = [sample_cell(time_range, timestamps, index) for index in range(4)]
    merged = merge_ranges(cells)

    assert merged == [TimeRange(20.0, 23.0)]


def test_sample_cell_with_single_sample_covers_everything() -> None:
    assert sample_cell(TimeRange(5.0, 9.0), [7.0], 0) == TimeRange(5.0, 9.0)


def test_sample_cell_rejects_invalid_index() -> None:
    with pytest.raises(IndexError):
        sample_cell(TimeRange(0.0, 3.0), [1.0], 3)


def test_merge_ranges_unions_overlaps_and_adjacent() -> None:
    ranges = [TimeRange(1.0, 3.0), TimeRange(2.0, 5.0), TimeRange(5.0, 6.0), TimeRange(8.0, 9.0)]

    assert merge_ranges(ranges) == [TimeRange(1.0, 6.0), TimeRange(8.0, 9.0)]


def test_merge_ranges_keeps_disjoint_ranges() -> None:
    ranges = [TimeRange(1.0, 2.0), TimeRange(3.0, 4.0)]

    assert merge_ranges(ranges) == [TimeRange(1.0, 2.0), TimeRange(3.0, 4.0)]


def test_parse_rows_merges_overlapping_entries() -> None:
    frame = _ranges_to_frame([TimeRange(1.0, 5.0), TimeRange(3.0, 8.0)])

    ranges, invalid = parse_exclusion_rows(frame)

    assert ranges == [TimeRange(1.0, 8.0)]
    assert invalid is False


def test_ranges_to_frame_keeps_fractional_seconds() -> None:
    # A celula do X nao cai em segundos inteiros; a exibicao nao deve arredondar.
    frame = _ranges_to_frame([TimeRange(60.4583, 61.375)])

    ranges, invalid = parse_exclusion_rows(frame)

    # Duas casas decimais (centesimos), como na selecao do trecho.
    assert frame.iloc[0].tolist() == ["01:00.46", "01:01.38"]
    assert ranges == [TimeRange(60.46, 61.38)]
    assert invalid is False


def _render_script() -> None:
    import numpy as np

    from judo_analysis.application.dto import AnalysisResult
    from judo_analysis.domain.entities.analysis import AnalysisRun
    from judo_analysis.domain.entities.video import TimeRange, Video
    from judo_analysis.presentation.components.exclusion import render_frame_exclusion

    run = AnalysisRun(
        label="Referencia",
        video=Video(path="x.mp4", fps=25.0, frame_count=100, width=10, height=10),
        time_range=TimeRange(0.0, 4.0),
        person_id=0,
    )
    result = AnalysisResult(
        run=run,
        key_frame=np.zeros((4, 4, 3), np.uint8),
        excluded_ranges=[TimeRange(1.0, 2.0)],
        excluded_frame_count=25,
    )
    render_frame_exclusion(result, lambda ranges: None, "ref")


def test_exclusion_editor_renders_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_script, default_timeout=60).run()

    assert not app.exception
