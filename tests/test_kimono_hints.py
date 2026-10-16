"""Testes da dica de cor do kimono guardada entre analise e reanalise.

O erro original: a analise escrevia em ``ref_different_kimono`` /
``ana_different_kimono``, que sao as chaves dos checkboxes ja instanciados no
mesmo run, e o Streamlit recusa isso (``StreamlitWidgetAlreadyInstantiatedError``).
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

# Reproduz a ordem real da aplicacao: os checkboxes dos paineis sao criados
# antes de o botao "Analisar" rodar.
_SCRIPT = """
import streamlit as st

from judo_analysis.presentation.analysis import kimono_hint, remember_kimono_hint

st.checkbox("Os dois atletas tem kimonos de cores diferentes", key="ref_different_kimono")
st.checkbox("Os dois atletas tem kimonos de cores diferentes", key="ana_different_kimono")

st.session_state["antes"] = (kimono_hint("ref"), kimono_hint("ana"))
remember_kimono_hint("ref", True)
remember_kimono_hint("ana", False)
st.session_state["depois"] = (kimono_hint("ref"), kimono_hint("ana"))
"""


def _app() -> AppTest:
    return AppTest.from_string(_SCRIPT, default_timeout=60).run()


def test_remembering_hints_after_the_checkboxes_exist_does_not_raise() -> None:
    app = _app()
    assert not app.exception


def test_hints_start_disabled_and_are_stored_under_dedicated_keys() -> None:
    app = _app()
    assert app.session_state["antes"] == (False, False)
    assert app.session_state["depois"] == (True, False)
    # As chaves dos widgets nao sao reescritas.
    assert app.session_state["ref_different_kimono"] is False
    assert app.session_state["ana_different_kimono"] is False
