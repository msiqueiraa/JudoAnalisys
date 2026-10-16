"""Entidades relacionadas a video e recortes de tempo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Intervalo de tempo (em segundos) a ser analisado."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("O inicio do intervalo nao pode ser negativo.")
        if self.end <= self.start:
            raise ValueError("O fim do intervalo deve ser maior que o inicio.")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Video:
    """Metadados de um arquivo de video."""

    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps
