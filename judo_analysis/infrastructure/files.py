"""Utilitarios de arquivos compartilhados pelos adaptadores."""

from __future__ import annotations

import re
from pathlib import Path


def safe_name(filename: str, default: str) -> str:
    """Nome de arquivo sem caracteres perigosos (nunca vazio).

    Recebe o nome original enviado pelo usuario e devolve apenas o nome base,
    com tudo que nao for letra/numero/ponto/hifen/underscore trocado por ``_``.
    """

    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    return name or default
