"""Value object com os angulos articulares medidos em um frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JointAngles:
    """Angulos (em graus) medidos em um frame.

    Cada campo pode ser ``None`` quando o landmark necessario nao esta visivel
    o suficiente para um calculo confiavel.
    """

    knee_left: float | None = None
    knee_right: float | None = None
    spine: float | None = None
    neck: float | None = None
    leg_opening: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "knee_left": self.knee_left,
            "knee_right": self.knee_right,
            "spine": self.spine,
            "neck": self.neck,
            "leg_opening": self.leg_opening,
        }
