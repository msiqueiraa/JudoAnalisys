"""Testes da faixa de verificacao do rastreamento (com o X para descartar frames)."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from judo_analysis.domain.entities.video import TimeRange
from judo_analysis.presentation.components.pose_selection import nearest_pose
from judo_analysis.presentation.components.tracking_review import pick_offset
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


def test_pick_offset_converts_the_instant_to_the_slider_domain() -> None:
    time_range = TimeRange(10.0, 14.0)

    # O instante da miniatura vira segundos a partir do inicio do trecho.
    assert pick_offset(time_range, 10.0) == 0.0
    assert pick_offset(time_range, 12.0) == 2.0
    assert pick_offset(time_range, 14.0) == 4.0


def test_pick_offset_is_aligned_and_clamped() -> None:
    time_range = TimeRange(10.0, 14.0)

    # Alinhado ao passo do slider (0.1s).
    assert pick_offset(time_range, 12.03) == 2.0
    assert pick_offset(time_range, 12.07) == 2.1

    # Fora do trecho: limitado as bordas.
    assert pick_offset(time_range, 9.0) == 0.0
    assert pick_offset(time_range, 20.0) == 4.0

    # O alinhamento nunca passa do fim do trecho (13.97s arredondaria para 4.0s).
    assert pick_offset(TimeRange(10.0, 13.97), 13.97) == 3.97


def _strip_pick_script() -> None:
    import numpy as np

    from judo_analysis.application.dto import AnalysisResult, TrackingSample
    from judo_analysis.domain.entities.analysis import AnalysisRun
    from judo_analysis.domain.entities.video import TimeRange, Video
    from judo_analysis.presentation.components import pose_selection, tracking_review
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

    class _FakeReader:
        """Leitor de video que devolve um quadro pequeno em qualquer instante."""

        def __init__(self, path: str) -> None:
            self._path = path

        def __enter__(self) -> "_FakeReader":
            return self

        def __exit__(self, *args) -> bool:
            return False

        def frame_at(self, timestamp: float):
            return 0, timestamp, np.zeros((4, 4, 3), np.uint8)

    class _FakeEstimator:
        """Estimador que nao ve ninguem: a previa fica sem poses."""

        def detect(self, frame, timestamp_ms=None):
            return []

        def close(self) -> None:
            return None

    # Simula o clique na segunda miniatura (instante de 12s).
    clique = {"x": 10, "y": 10, "width": 100, "height": 100, "unix_time": 111.0}
    original_coordinates = pose_selection.streamlit_image_coordinates
    original_has_click = pose_selection._HAS_CLICK
    original_reader = tracking_review.OpenCVVideoReader
    original_detector = tracking_review.make_pose_detector
    pose_selection._HAS_CLICK = True
    pose_selection.streamlit_image_coordinates = (
        lambda image, width=None, key=None: clique if key.endswith("_1") else None
    )
    tracking_review.OpenCVVideoReader = _FakeReader
    tracking_review.make_pose_detector = lambda num_poses: _FakeEstimator()
    try:
        render_tracking_review(
            result,
            num_poses=2,
            on_correct=lambda selection, anchor_time: None,
            on_exclude=lambda ranges: None,
            key="t",
        )
    finally:
        pose_selection.streamlit_image_coordinates = original_coordinates
        pose_selection._HAS_CLICK = original_has_click
        tracking_review.OpenCVVideoReader = original_reader
        tracking_review.make_pose_detector = original_detector


def test_clicking_a_thumbnail_points_the_correction_to_that_instant() -> None:
    app = AppTest.from_function(_strip_pick_script, default_timeout=60).run()

    assert not app.exception
    # O slider da correcao foi para o instante da miniatura: 12s - 10s = 2s.
    assert app.slider[0].value == 2.0
    # A secao de correcao abre no ponto escolhido.
    assert app.expander[0].label == "Corrigir pessoa (se o rastreamento pegou o atleta errado)"
    assert app.expander[0].proto.expanded is True
    # O clique ja dispara a deteccao no ponto escolhido (sem pessoas no video).
    assert [warning.value for warning in app.warning] == [
        "Nenhuma pessoa detectada neste instante. Tente outro ponto."
    ]
    # A miniatura clicada fica marcada na faixa (segundos inteiros saem sem decimais).
    assert [caption.value for caption in app.caption if caption.value.endswith("← corrigir")] == [
        "00:12 ← corrigir"
    ]


def _strip_without_click_script() -> None:
    import numpy as np

    from judo_analysis.application.dto import AnalysisResult, TrackingSample
    from judo_analysis.domain.entities.analysis import AnalysisRun
    from judo_analysis.domain.entities.video import TimeRange, Video
    from judo_analysis.presentation.components import pose_selection, tracking_review
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
        tracking_samples=[TrackingSample(frame_index=0, timestamp=10.0)],
    )

    original_has_click = pose_selection._HAS_CLICK
    pose_selection._HAS_CLICK = False
    try:
        render_tracking_review(
            result,
            num_poses=2,
            on_correct=lambda selection, anchor_time: None,
            on_exclude=lambda ranges: None,
            key="t",
        )
    finally:
        pose_selection._HAS_CLICK = original_has_click


def test_strip_hint_omits_the_click_without_the_component() -> None:
    app = AppTest.from_function(_strip_without_click_script, default_timeout=60).run()

    assert not app.exception
    hints = [caption.value for caption in app.caption if "X**" in caption.value]
    assert hints and "Clique numa miniatura" not in hints[0]
