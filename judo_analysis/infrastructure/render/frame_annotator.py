"""Anotacao visual de frames: esqueleto, selecao de pessoa e metricas."""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.entities.pose import Pose, PoseLandmark
from ...domain.value_objects.gi_color import GiColor
from ...domain.value_objects.joint_angles import JointAngles
from ...domain.value_objects.speed import Speed

# Conexoes do esqueleto (BlazePose) desenhadas sobre o frame.
POSE_CONNECTIONS: tuple[tuple[PoseLandmark, PoseLandmark], ...] = (
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.LEFT_EAR, PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_EAR, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR),
)

_VISIBLE = 0.2

# Altura (px) da primeira linha de texto das metricas no frame.
_TEXT_TOP = 28


class FrameAnnotator:
    """Desenha esqueleto e metricas em frames BGR."""

    def annotate(
        self,
        frame: np.ndarray,
        pose: Pose | None,
        angles: JointAngles | None = None,
        speed: Speed | None = None,
        title: str | None = None,
        text_top: int = _TEXT_TOP,
    ) -> np.ndarray:
        canvas = frame.copy()
        if pose is not None:
            self._draw_skeleton(canvas, pose)
        self._draw_metrics(canvas, angles, speed, title, text_top)
        return canvas

    def annotate_selection(
        self,
        frame: np.ndarray,
        poses: list[Pose],
        selected_id: int | None = None,
    ) -> np.ndarray:
        """Desenha todas as pessoas numeradas; destaca a selecionada.

        Quando a cor do kimono foi medida, ela aparece ao lado do numero - o que
        ajuda a escolher a pessoa certa ("Pessoa 2 (azul)").
        """

        canvas = frame.copy()
        for pose in poses:
            selected = selected_id is not None and pose.person_id == selected_id
            color = (0, 200, 0) if selected else (0, 140, 255)
            self._draw_skeleton(canvas, pose, color=color, thickness=3 if selected else 1)
            centroid = pose.centroid
            if centroid is not None:
                height, width = canvas.shape[:2]
                anchor = (int(centroid[0] * width), int(centroid[1] * height))
                self._draw_person_label(canvas, pose, anchor, color)
        return canvas

    def _draw_person_label(
        self,
        canvas: np.ndarray,
        pose: Pose,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Escreve o numero da pessoa e, se houver, a cor do kimono."""

        label = f"Pessoa {pose.person_id + 1}"
        if pose.gi_color is not None:
            label = f"{label} ({pose.gi_color.name()})"

        left = anchor[0] - 40
        baseline = anchor[1] - 12
        text_left = left
        if pose.gi_color is not None:
            swatch = _gi_color_bgr(pose.gi_color)
            top_left = (left, baseline - 14)
            bottom_right = (left + 16, baseline + 2)
            cv2.rectangle(canvas, top_left, bottom_right, swatch, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 1, cv2.LINE_AA)
            text_left = left + 22

        cv2.putText(
            canvas,
            label,
            (text_left, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        pose: Pose,
        color: tuple[int, int, int] = (0, 200, 0),
        thickness: int = 2,
    ) -> None:
        height, width = canvas.shape[:2]
        for start, end in POSE_CONNECTIONS:
            point_a = pose.get(start)
            point_b = pose.get(end)
            if point_a is None or point_b is None:
                continue
            a = (int(point_a.x * width), int(point_a.y * height))
            b = (int(point_b.x * width), int(point_b.y * height))
            cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)

        for landmark in pose.landmarks:
            if landmark.visibility < _VISIBLE:
                continue
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(canvas, point, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_metrics(
        self,
        canvas: np.ndarray,
        angles: JointAngles | None,
        speed: Speed | None,
        title: str | None,
        text_top: int = _TEXT_TOP,
    ) -> None:
        lines: list[str] = []
        if title:
            lines.append(title)
        if angles is not None:
            lines.extend(
                [
                    f"Joelho E: {_fmt(angles.knee_left)}  Joelho D: {_fmt(angles.knee_right)}",
                    f"Coluna: {_fmt(angles.spine)}  Pescoco: {_fmt(angles.neck)}",
                    f"Abertura pernas: {_fmt(angles.leg_opening)}",
                ]
            )
        if speed is not None:
            lines.append(
                f"Velocidade golpe: {speed.pixels_per_second:.0f} px/s "
                f"({speed.body_heights_per_second:.2f} alturas/s)"
            )

        y = text_top
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 26


def _fmt(value: float | None) -> str:
    return f"{value:.0f}deg" if value is not None else "--"


def _gi_color_bgr(color: GiColor) -> tuple[int, int, int]:
    """Converte a cor do kimono (LAB) para BGR, so para desenhar a amostra."""

    lab = np.uint8([[[color.l * 255.0 / 100.0, color.a + 128.0, color.b + 128.0]]])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Converte um frame BGR para RGB (formato esperado pelo Streamlit)."""

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
