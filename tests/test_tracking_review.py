"""Testes da faixa de verificacao do rastreamento (com o X para descartar frames)."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from judo_analysis.presentation.components.pose_selection import nearest_pose
from tests.conftest import make_pose


def _person(cx: float) -> object:
    return make_pose(
        LEFT_HIP=(cx - 0.02, 0.5),
        RIGHT_HIP=(cx + 0.02, 0.5),
        LEFT_SHOULDER=(cx - 0.04, 0.3),
        RIGHT_SHOULDER=(cx + 0.04, 0.3),
    )


def test_nearest_pose_picks_the_closest_center() -> None:
    left = _person(0.2)
    right = _person(0.7)

    assert nearest_pose([left, right], 0.72, 0.5) == 1
    assert nearest_pose([left, right], 0.18, 0.5) == 0


def test_nearest_pose_defaults_to_first_when_no_center() -> None:
    # Pose sem landmarks visiveis: nao ha centro, mantem o indice 0.
    empty = [make_pose(), make_pose()]
    assert nearest_pose(empty, 0.5, 0.5) == 0


def _strip_script() -> None:
    import numpy as np

    from judo_analysis.application.dto import AnalysisResult, TrackingSample
    from judo_analysis.domain.entities.analysis import AnalysisRun
    from judo_analysis.domain.entities.video import TimeRange, Video
    from judo_analysis.presentation.components.tracking_review import (
        render_tracking_review,
    )

    run = AnalysisRun(
        label="Referencia",
        video=Video(path="nao_existe.mp4", fps=25.0, frame_count=100, width=10, height=10),
        time_range=TimeRange(10.0, 14.0),
        person_id=0,
    )
    result = AnalysisResult(
        run=run,
        key_frame=np.zeros((4, 4, 3), np.uint8),
        tracking_samples=[
            TrackingSample(frame_index=0, timestamp=10.0),
            TrackingSample(frame_index=50, timestamp=12.0),
            TrackingSample(frame_index=99, timestamp=14.0),
        ],
    )
    render_tracking_review(
        result,
        num_poses=2,
        on_correct=lambda selection, anchor_time: None,
        on_exclude=lambda ranges: None,
        key="t",
    )


def test_strip_renders_one_remove_button_per_sample() -> None:
    app = AppTest.from_function(_strip_script, default_timeout=60).run()

    assert not app.exception
    labels = [button.label for button in app.button]
    assert labels.count("X") == 3


def _strip_click_script() -> None:
    import numpy as np
    import streamlit as st

    from judo_analysis.application.dto import AnalysisResult, TrackingSample
    from judo_analysis.domain.entities.analysis import AnalysisRun
    from judo_analysis.domain.entities.video import TimeRange, Video
    from judo_analysis.presentation.components.tracking_review import (
        render_tracking_review,
    )

    run = AnalysisRun(
        label="Referencia",
        video=Video(path="nao_existe.mp4", fps=25.0, frame_count=100, width=10, height=10),
        time_range=TimeRange(10.0, 14.0),
        person_id=0,
    )
    result = AnalysisResult(
        run=run,
        key_frame=np.zeros((4, 4, 3), np.uint8),
        tracking_samples=[
            TrackingSample(frame_index=0, timestamp=10.0),
            TrackingSample(frame_index=50, timestamp=12.0),
            TrackingSample(frame_index=99, timestamp=14.0),
        ],
    )

    def on_exclude(ranges) -> None:
        st.session_state["recebido"] = [(r.start, r.end) for r in ranges]

    render_tracking_review(
        result,
        num_poses=2,
        on_correct=lambda selection, anchor_time: None,
        on_exclude=on_exclude,
        key="t",
    )


def test_clicking_x_excludes_that_sample_cell() -> None:
    app = AppTest.from_function(_strip_click_script, default_timeout=60).run()
    assert not app.exception

    remove_buttons = [button for button in app.button if button.label == "X"]
    remove_buttons[1].click().run()

    assert not app.exception
    # Amostra de 12s em um trecho 10s-14s -> celula (11s, 13s).
    assert app.session_state["recebido"] == [(11.0, 13.0)]
