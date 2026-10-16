"""Testes da aba de foto (duas fotos comparadas e uma foto so).

A deteccao de pessoas nao roda aqui: a previa (com a pose) entra pronta na
sessao, com a assinatura que a aba usa para nao refazer o trabalho.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from judo_analysis.application.dto import PersonPreview
from tests.conftest import make_pose

_FRAME = np.full((48, 64, 3), 120, dtype=np.uint8)

_POINTS = {
    "LEFT_SHOULDER": (0.35, 0.30),
    "RIGHT_SHOULDER": (0.65, 0.30),
    "LEFT_HIP": (0.40, 0.50),
    "RIGHT_HIP": (0.60, 0.50),
    "LEFT_KNEE": (0.35, 0.65),
    "RIGHT_KNEE": (0.65, 0.65),
    "LEFT_ANKLE": (0.30, 0.85),
    "RIGHT_ANKLE": (0.70, 0.85),
    "LEFT_EAR": (0.42, 0.20),
    "RIGHT_EAR": (0.58, 0.20),
}


def _photo(name: str) -> str:
    path = Path(tempfile.mkdtemp()) / name
    assert cv2.imwrite(str(path), _FRAME)
    return str(path)


REFERENCE_PHOTO = _photo("ref.png")
ANALYZED_PHOTO = _photo("ana.png")


def build_preview(path: str) -> PersonPreview:
    """Previa com uma pessoa detectada (substitui a deteccao real)."""

    del path
    return PersonPreview(
        frame_index=0,
        timestamp=0.0,
        frame=_FRAME,
        poses=[make_pose(**_POINTS)],
    )


def _script(photos: str) -> str:
    """Script que monta a aba de foto com as fotos ja "detectadas"."""

    return f"""
import streamlit as st

from judo_analysis.presentation.components.photo_analysis import render_photo_mode
from judo_analysis.presentation.components.photo_input import photo_signature
from tests.test_photo_analysis import ANALYZED_PHOTO, REFERENCE_PHOTO, build_preview

for key, path in {photos}:
    st.session_state[f"{{key}}_photo_path"] = path
    st.session_state[f"{{key}}_photo_preview"] = build_preview(path)
    st.session_state[f"{{key}}_photo_preview_signature"] = f"{{photo_signature(path)}}:4"

render_photo_mode(4)
"""


def _app(photos: str) -> AppTest:
    return AppTest.from_string(_script(photos), default_timeout=120).run()


def _click_analyze(app: AppTest) -> None:
    next(button for button in app.button if button.label == "Analisar foto").click().run()


def test_two_photos_are_compared() -> None:
    app = _app('(("fref", REFERENCE_PHOTO), ("fana", ANALYZED_PHOTO))')

    assert not app.exception
    # Uma lista de pessoas por foto, sem resultado ainda.
    assert len(app.selectbox) == 2
    assert not app.get("plotly_chart")

    _click_analyze(app)

    assert not app.exception
    reference = app.session_state["photo_reference_result"]
    analyzed = app.session_state["photo_analyzed_result"]
    assert reference.run.is_photo
    assert analyzed.run.is_photo
    assert len(reference.run.frames) == 1
    assert len(analyzed.run.frames) == 1
    # Foto nao tem velocidade nem serie temporal: so as barras de similaridade.
    assert "hand_speed" not in reference.run.stats
    assert len(app.get("plotly_chart")) == 1
    assert any("Foto analisada" in caption.value for caption in app.caption)
    assert any("Velocidade do golpe" in caption.value for caption in app.caption)


def test_single_photo_shows_the_angles_without_comparison() -> None:
    app = _app('(("fana", ANALYZED_PHOTO),)')

    _click_analyze(app)

    assert not app.exception
    assert app.session_state["photo_reference_result"] is None
    assert app.session_state["photo_analyzed_result"] is not None
    assert not app.get("plotly_chart")
    assert any(
        "Carregue a foto de referencia" in info.value for info in app.info
    )
