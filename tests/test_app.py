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
