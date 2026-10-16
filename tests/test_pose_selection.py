"""Teste de fumaca do seletor de pessoa clicavel."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _script() -> None:
    import numpy as np
    import streamlit as st

    from judo_analysis.domain.entities.landmark import Landmark
    from judo_analysis.domain.entities.pose import Pose, PoseLandmark
    from judo_analysis.presentation.components.pose_selection import render_pose_selector

    poses = []
    for offset in (0.2, 0.7):
        landmarks = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0)
            for _ in range(len(PoseLandmark))
        ]
        landmarks[int(PoseLandmark.LEFT_HIP)] = Landmark(offset - 0.02, 0.5, 0.0, 1.0)
        landmarks[int(PoseLandmark.RIGHT_HIP)] = Landmark(offset + 0.02, 0.5, 0.0, 1.0)
        landmarks[int(PoseLandmark.LEFT_SHOULDER)] = Landmark(offset - 0.04, 0.3, 0.0, 1.0)
        landmarks[int(PoseLandmark.RIGHT_SHOULDER)] = Landmark(offset + 0.04, 0.3, 0.0, 1.0)
        poses.append(Pose(person_id=0, landmarks=landmarks))

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    selected = render_pose_selector(frame, poses, "t")
    st.write(f"selected={selected}")
    st.selectbox("Pessoa", options=[0, 1], key="t_person")


def test_pose_selector_renders_and_feeds_the_selectbox() -> None:
    # No AppTest o componente de clique nao executa (retorna None), entao o teste
    # garante que o caminho sem clique funciona e alimenta o selectbox.
    app = AppTest.from_function(_script, default_timeout=60).run()

    assert not app.exception
    assert app.selectbox[0].value == 0
