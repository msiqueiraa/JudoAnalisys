"""Value object que representa uma velocidade medida."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Speed:
    """Velocidade em pixels/s e normalizada por altura de corpo.

    A velocidade normalizada (alturas de corpo por segundo) permite comparar
    videos com resolucoes e enquadramentos diferentes.
    """

    pixels_per_second: float
    body_heights_per_second: float

    def __add__(self, other: "Speed") -> "Speed":
        return Speed(
            self.pixels_per_second + other.pixels_per_second,
            self.body_heights_per_second + other.body_heights_per_second,
        )

    def __mul__(self, factor: float) -> "Speed":
        return Speed(
            self.pixels_per_second * factor,
            self.body_heights_per_second * factor,
        )
