"""Leitura de imagens paradas (analise de foto)."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from ...application.ports.video_reader import IVideoReader
from ...config import PHOTO_FPS
from ...domain.entities.video import Video


def read_image(path: str) -> np.ndarray:
    """Le uma imagem do disco em BGR, com erro claro quando o arquivo nao abre."""

    image = cv2.imread(path)
    if image is None:
        raise IOError(f"Nao foi possivel ler a imagem: {path}")
    return image


class SingleImageReader(IVideoReader):
    """Trata uma foto como um "video" de um unico quadro.

    Com isso a deteccao de pessoas (deduplicacao, segunda tentativa e cor do
    kimono) e reaproveitada inteira, sem um caminho paralelo so para fotos.
    O FPS nominal de :data:`PHOTO_FPS` nao representa duracao real: serve apenas
    para as conversoes de tempo continuarem validas.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._image = read_image(path)

    @property
    def image(self) -> np.ndarray:
        """A imagem lida (BGR)."""

        return self._image

    def metadata(self) -> Video:
        height, width = self._image.shape[:2]
        return Video(
            path=self._path,
            fps=PHOTO_FPS,
            frame_count=1,
            width=width,
            height=height,
        )

    def frame_at(self, timestamp: float) -> tuple[int, float, np.ndarray]:
        return 0, 0.0, self._image

    def frames(self, start: float, end: float) -> Iterator[tuple[int, float, np.ndarray]]:
        yield 0, 0.0, self._image

    def close(self) -> None:
        """Nada a liberar: a imagem fica em memoria."""
