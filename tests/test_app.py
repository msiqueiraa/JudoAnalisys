"""Teste de fumaca da aplicacao Streamlit (renderizacao inicial sem excecoes)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_without_exceptions() -> None:
    app = AppTest.from_file(_APP, default_timeout=120).run()
    assert not app.exception
    # Sem videos carregados, a aplicacao orienta o usuario.
    assert any("Carregue os dois videos" in info.value for info in app.info)

    labels = [checkbox.label for checkbox in app.sidebar.checkbox]
    assert "Manter identidade na sobreposicao" in labels
    assert "Ignorar velocidade do golpe" in labels

    buttons = [button.label for button in app.sidebar.button]
    assert "Carregar videos de exemplo" in buttons


def test_app_has_a_tab_for_each_kind_of_analysis() -> None:
    app = AppTest.from_file(_APP, default_timeout=120).run()

    assert [tab.proto.label for tab in app.tabs] == ["Video", "Foto", "Foto x video"]
    # A primeira aba abre selecionada.
    assert app.session_state["analysis_mode"] == "Video"


def test_photo_tab_guides_the_user_without_photos() -> None:
    app = AppTest.from_file(_APP, default_timeout=120).run()
    app.session_state["analysis_mode"] = "Foto"
    app.run()

    assert not app.exception
    assert any("Carregue a foto analisada" in info.value for info in app.info)
