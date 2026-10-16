"""Fonte de foto a partir de um upload do usuario."""

from __future__ import annotations

from ...application.ports.video_source import IVideoSource
from ...config import PHOTO_DIR
from ..files import safe_name


class UploadPhotoSource(IVideoSource):
    """Grava a imagem enviada em disco e devolve o caminho local.

    O contrato reaproveitado (:class:`IVideoSource`) e o mesmo do video: uma
    origem enviada pelo usuario vira um caminho local para a analise.
    """

    def __init__(self, filename: str, data: bytes) -> None:
        self._filename = filename
        self._data = data

    def resolve(self) -> str:
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        destination = PHOTO_DIR / safe_name(self._filename, "foto.png")
        destination.write_bytes(self._data)
        return str(destination)
