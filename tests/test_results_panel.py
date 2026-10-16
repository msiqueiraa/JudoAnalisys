"""Testes do painel de resultado: slider que escolhe o frame-chave do golpe."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from judo_analysis.application.dto import AnalysisResult
from judo_analysis.domain.entities.analysis import AnalysisRun, FrameMetrics
from judo_analysis.domain.entities.video import TimeRange, Video
from judo_analysis.domain.services.statistics import aggregate_run
from judo_analysis.domain.value_objects.joint_angles import JointAngles
from judo_analysis.domain.value_objects.speed import Speed
from tests.conftest import make_pose

_FPS = 10.0
_FRAMES = 10
_WIDTH = 64
_HEIGHT = 48
# Velocidade por frame: o pico (o "golpe") fica no meio do trecho.
_SPEEDS = (100.0, 150.0, 200.0, 250.0, 300.0, 900.0, 250.0, 200.0, 150.0, 100.0)
_PEAK_INDEX = 5

# Script minimo que renderiza o painel isolado, como a aplicacao faz.
_SCRIPT = """
import streamlit as st

from judo_analysis.presentation.components import render_analysis_panel
from tests.test_results_panel import build_sample_result

result = build_sample_result()
render_analysis_panel(result, "ref")
st.session_state["key_frame_index"] = result.run.key_frame_index
st.session_state["key_frame_time"] = result.key_frame_time
st.session_state["key_frame_sum"] = int(result.key_frame.sum())
"""


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_WIDTH, _HEIGHT)
    )
    assert writer.isOpened()
    try:
        for index in range(_FRAMES):
            writer.write(np.full((_HEIGHT, _WIDTH, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def build_sample_result() -> AnalysisResult:
    """Resultado de exemplo; o video existe em disco para reler o frame-chave."""

    path = Path(tempfile.mkdtemp()) / "sample.mp4"
    _write_video(path)
    frames = [
        FrameMetrics(
            frame_index=index,
            timestamp=index / _FPS,
            angles=JointAngles(knee_left=90.0, knee_right=95.0, spine=10.0),
            hand_speed=Speed(
                pixels_per_second=speed, body_heights_per_second=speed / 300.0
            ),
            pose=make_pose(LEFT_HIP=(0.4, 0.5), RIGHT_HIP=(0.6, 0.5)),
        )
        for index, speed in enumerate(_SPEEDS)
    ]
    run = AnalysisRun(
        label="Referencia",
        video=Video(
            path=str(path), fps=_FPS, frame_count=_FRAMES, width=_WIDTH, height=_HEIGHT
        ),
        time_range=TimeRange(0.0, _FRAMES / _FPS),
        person_id=0,
        frames=frames,
    )
    aggregate_run(run)
    return AnalysisResult(
        run=run,
        key_frame=np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8),
        key_frame_pose=frames[run.key_frame_index].pose,
    )


def _app() -> AppTest:
    return AppTest.from_string(_SCRIPT, default_timeout=120).run()


def test_panel_starts_on_the_automatic_key_frame() -> None:
    app = _app()

    assert not app.exception
    assert len(app.slider) == 1
    assert app.slider[0].label.startswith("Instante do golpe")
    # O slider abre no pico de velocidade (0,5 s a 10 fps).
    assert app.slider[0].value == dt.time(0, 0, 0, 500000)
    assert app.session_state["key_frame_index"] == _PEAK_INDEX
    assert app.session_state["key_frame_time"] is None
    assert any("automatico" in caption.value for caption in app.caption)


def test_moving_the_slider_moves_the_key_frame() -> None:
    app = _app()
    before = app.session_state["key_frame_sum"]

    app.slider[0].set_value(dt.time(0, 0, 0, 200000)).run()

    assert not app.exception
    assert app.session_state["key_frame_index"] == 2
    assert app.session_state["key_frame_time"] == pytest.approx(0.2)
    # A imagem do frame-chave foi relida do video naquele instante.
    assert app.session_state["key_frame_sum"] != before
    assert any("escolhido" in caption.value for caption in app.caption)


def test_automatic_button_returns_to_the_peak() -> None:
    app = _app()
    app.slider[0].set_value(dt.time(0, 0, 0, 200000)).run()

    next(button for button in app.button if button.key == "ref_keyframe_auto").click().run()

    assert not app.exception
    assert app.session_state["key_frame_index"] == _PEAK_INDEX
    assert app.session_state["key_frame_time"] is None
    assert app.slider[0].value == dt.time(0, 0, 0, 500000)
