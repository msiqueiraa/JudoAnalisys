"""Value object que representa um ponto do corpo detectado."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Landmark:
    """Ponto corporal em coordenadas normalizadas (0..1).

    Attributes:
        x: Coordenada horizontal normalizada pela largura do frame.
        y: Coordenada vertical normalizada pela altura do frame.
        z: Profundidade relativa (quanto menor, mais perto da camera).
        visibility: Confianca de que o ponto esta visivel (0..1).
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple(self) -> tuple[float, float]:
        """Retorna ``(x, y)`` para calculos geometricos."""

        return (self.x, self.y)

    def as_pixel(self, width: int, height: int) -> tuple[float, float]:
        """Converte as coordenadas normalizadas para pixels do frame."""

        return (self.x * width, self.y * height)
