"""Testes da selecao do trecho (slider + caixas em ``mm:ss.cc``)."""

from __future__ import annotations

import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

# Script minimo que renderiza o componente isolado, como a aplicacao faz.
_SCRIPT = """
import streamlit as st

from judo_analysis.presentation.components.person_selection import _render_time_range

start, end = _render_time_range("t", 30.0)
st.session_state["resultado"] = (start, end)
"""

# Script do componente completo, para conferir o texto do trecho na tela.
_SELECTION_SCRIPT = """
import streamlit as st

from judo_analysis.presentation.components.person_selection import render_person_selection

_selection, _range, _diff = render_person_selection("Referencia", "t", "nao_existe.mp4", 30.0, 2)
st.session_state["resultado"] = None if _range is None else (_range.start, _range.end)
"""


def _app() -> AppTest:
    return AppTest.from_string(_SCRIPT, default_timeout=60).run()


def _selection_app() -> AppTest:
    return AppTest.from_string(_SELECTION_SCRIPT, default_timeout=60).run()


def test_renders_a_slider_and_the_two_time_boxes() -> None:
    app = _app()
    assert not app.exception
    assert len(app.slider) == 1
    assert [box.label for box in app.text_input] == ["Inicio (mm:ss)", "Fim (mm:ss)"]
    assert [box.value for box in app.text_input] == ["00:00", "00:08"]
    assert app.session_state["resultado"] == (0.0, 8.0)


def test_slider_updates_the_time_boxes() -> None:
    app = _app()
    app.slider[0].set_value((dt.time(0, 0, 3), dt.time(0, 0, 9))).run()
    assert not app.exception
    assert [box.value for box in app.text_input] == ["00:03", "00:09"]
    assert app.session_state["resultado"] == (3.0, 9.0)


def test_time_boxes_move_the_slider() -> None:
    app = _app()
    app.text_input[0].set_value("00:05").run()
    assert not app.exception
    assert app.slider[0].value == (dt.time(0, 0, 5), dt.time(0, 0, 8))
    assert app.session_state["resultado"] == (5.0, 8.0)


def test_invalid_box_keeps_the_previous_range() -> None:
    app = _app()
    app.text_input[1].set_value("abc").run()
    assert not app.exception
    # "abc" nao e um tempo valido: o trecho anterior continua valendo.
    assert app.session_state["resultado"] == (0.0, 8.0)


def test_box_before_the_start_is_ignored() -> None:
    app = _app()
    # Inicio (00:10) depois do fim (00:08): o trecho anterior continua valendo.
    app.text_input[0].set_value("00:10").run()
    assert not app.exception
    assert app.session_state["resultado"] == (0.0, 8.0)


def test_box_beyond_the_video_is_clamped() -> None:
    app = _app()
    app.text_input[1].set_value("00:45").run()
    assert not app.exception
    assert app.session_state["resultado"] == (0.0, 30.0)
    # O texto tambem e normalizado para o fim real do video.
    assert app.text_input[1].value == "00:30"


def test_centiseconds_in_the_box_are_kept() -> None:
    app = _app()
    app.text_input[0].set_value("00:01.55").run()
    assert not app.exception
    # O slider nao representa fracoes de segundo: as caixas mandam.
    assert app.session_state["resultado"] == pytest.approx((1.55, 8.0))
    assert app.text_input[0].value == "00:01.55"


def test_short_precise_range_is_accepted() -> None:
    app = _app()
    app.text_input[0].set_value("00:01.55").run()
    app.text_input[1].set_value("00:04.20").run()
    assert not app.exception
    assert app.session_state["resultado"] == pytest.approx((1.55, 4.20))


def test_slider_keeps_the_precision_and_the_boxes_stay_intact() -> None:
    app = _app()
    app.text_input[0].set_value("00:01.55").run()
    # O slider guarda o valor preciso (so o passo da interface e de 1 s)...
    assert app.slider[0].value == (dt.time(0, 0, 1, 550000), dt.time(0, 0, 8))
    # ...e o texto digitado nao e reescrito pelo callback do slider.
    assert app.text_input[0].value == "00:01.55"
    assert app.session_state["resultado"] == pytest.approx((1.55, 8.0))


def test_typing_is_not_reformatted() -> None:
    app = _app()
    app.text_input[0].set_value("00:06.5").run()
    assert not app.exception
    # O texto digitado e preservado (nao vira "00:06.50" no meio da digitacao).
    assert app.text_input[0].value == "00:06.5"
    assert app.session_state["resultado"] == pytest.approx((6.5, 8.0))


def test_dragging_the_slider_overwrites_the_centiseconds() -> None:
    app = _app()
    app.text_input[0].set_value("00:01.55").run()
    app.slider[0].set_value((dt.time(0, 0, 5), dt.time(0, 0, 9))).run()
    assert not app.exception
    assert [box.value for box in app.text_input] == ["00:05", "00:09"]
    assert app.session_state["resultado"] == (5.0, 9.0)


def test_caption_shows_the_precise_range() -> None:
    app = _selection_app()
    assert not app.exception
    assert any("Trecho: 00:00 - 00:08" in caption.value for caption in app.caption)

    app.text_input[0].set_value("00:01.55").run()
    assert not app.exception
    assert app.session_state["resultado"] == pytest.approx((1.55, 8.0))
    captions = [caption.value for caption in app.caption]
    assert any("Trecho: 00:01.55 - 00:08 (duracao 00:06.45)" in text for text in captions)
