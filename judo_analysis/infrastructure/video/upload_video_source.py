"""Fonte de video a partir de um upload do usuario."""

from __future__ import annotations

import re
from pathlib import Path

from ...application.ports.video_source import IVideoSource
from ...config import TEMP_DIR


class UploadVideoSource(IVideoSource):
    """Grava o conteudo enviado em disco e devolve o caminho local."""

    def __init__(self, filename: str, data: bytes) -> None:
        self._filename = filename
        self._data = data

    def resolve(self) -> str:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_name(self._filename)
        destination = TEMP_DIR / safe_name
        destination.write_bytes(self._data)
        return str(destination)


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload.mp4"
