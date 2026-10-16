"""Porta para leitura de frames de video."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from ...domain.entities.video import Video


class IVideoReader(ABC):
    """Contrato de leitura de metadados e frames de um video."""

    @abstractmethod
    def metadata(self) -> Video:
        """Retorna os metadados do video."""

    @abstractmethod
    def frame_at(self, timestamp: float) -> tuple[int, float, np.ndarray]:
        """Retorna ``(indice, timestamp, frame)`` do frame mais proximo."""

    @abstractmethod
    def frames(self, start: float, end: float) -> Iterator[tuple[int, float, np.ndarray]]:
        """Itera pelos frames no intervalo ``[start, end)`` em segundos."""

    @abstractmethod
    def close(self) -> None:
        """Libera o video."""

    def __enter__(self) -> "IVideoReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
