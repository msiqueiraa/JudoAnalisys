"""Porta para estimacao de pose a partir de um frame."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...domain.entities.pose import Pose


class IPoseEstimator(ABC):
    """Contrato de um estimador de pose multi-pessoa."""

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[Pose]:
        """Detecta as poses presentes em um frame BGR.

        Args:
            frame: Imagem no formato BGR (padrao OpenCV).
            timestamp_ms: Instante do frame em milissegundos. Obrigatorio quando
                o estimador opera em modo VIDEO (rastreamento temporal).

        Returns:
            Lista de poses detectadas, ordenada da esquerda para a direita.
        """

    @abstractmethod
    def close(self) -> None:
        """Libera os recursos do estimador."""
