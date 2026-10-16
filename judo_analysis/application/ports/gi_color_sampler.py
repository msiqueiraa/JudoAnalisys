"""Porta para medir a cor do kimono de uma pessoa em um frame."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...domain.entities.pose import Pose
from ...domain.value_objects.gi_color import GiColor


class IGiColorSampler(ABC):
    """Contrato de medicao da cor media do kimono."""

    @abstractmethod
    def sample(self, frame: np.ndarray, pose: Pose) -> GiColor | None:
        """Mede a cor do kimono da pessoa na pose indicada.

        Args:
            frame: Imagem no formato BGR (padrao OpenCV).
            pose: Pose da pessoa no mesmo frame.

        Returns:
            A cor medida, ou ``None`` quando nao ha como medir com confianca
            (kimono ocluso, patch heterogeneo, corpo sem landmarks suficientes).
        """
