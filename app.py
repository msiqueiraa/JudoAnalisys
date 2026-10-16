"""Entrypoint da aplicacao Streamlit.

Execute com:

    streamlit run app.py
"""

# Deve rodar antes de importar o pacote para silenciar os logs do MediaPipe.
from judo_analysis.logging_setup import configure_logging

configure_logging()

from judo_analysis.presentation.app import main  # noqa: E402

main()
