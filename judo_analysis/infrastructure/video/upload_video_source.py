"""Fonte de video a partir de um upload do usuario."""

from __future__ import annotations

from ...application.ports.video_source import IVideoSource
from ...config import TEMP_DIR
from ..files import safe_name


class UploadVideoSource(IVideoSource):
    """Grava o conteudo enviado em disco e devolve o caminho local."""

    def __init__(self, filename: str, data: bytes) -> None:
        self._filename = filename
        self._data = data

    def resolve(self) -> str:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        destination = TEMP_DIR / safe_name(self._filename, "upload.mp4")
        destination.write_bytes(self._data)
        return str(destination)
