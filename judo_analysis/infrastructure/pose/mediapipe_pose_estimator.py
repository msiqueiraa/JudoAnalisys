"""Estimador de pose multi-pessoa baseado no MediaPipe Pose Landmarker.

Usa a Tasks API (arquivo ``pose_landmarker_full.task``) e detecta ate
``num_poses`` pessoas por frame.

O modo de execucao pode ser ``IMAGE`` (padrao) ou ``VIDEO``. Em testes com dois
atletas de judo, o modo ``VIDEO`` (que ativa o rastreamento temporal interno)
tende a fundir os dois corpos em uma unica deteccao quando eles se sobrepoem no
golpe; o modo ``IMAGE``, por tratar cada frame de forma independente, acabou
detectando os dois com mais frequencia. Por isso o padrao e ``IMAGE``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from ...application.ports.pose_estimator import IPoseEstimator
from ...config import DEFAULT_MODEL_PATH, DEFAULT_NUM_POSES, MIN_DETECTION_CONFIDENCE
from ...domain.entities.landmark import Landmark
from ...domain.entities.pose import Pose
from ...logging_setup import suppress_native_stderr

_MIN_TRACKING_CONFIDENCE = 0.5


class MediaPipePoseEstimator(IPoseEstimator):
    """Implementacao de :class:`IPoseEstimator` com MediaPipe Tasks."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_poses: int = DEFAULT_NUM_POSES,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        running_mode: str = "IMAGE",
        min_tracking_confidence: float = _MIN_TRACKING_CONFIDENCE,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo do MediaPipe nao encontrado em '{model_path}'. "
                "Baixe 'pose_landmarker_full.task' para a raiz do projeto."
            )

        self._mode = (
            vision.RunningMode.VIDEO
            if str(running_mode).upper() == "VIDEO"
            else vision.RunningMode.IMAGE
        )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self._mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # O ``create_from_options`` emite avisos nativos (XNNPACK/absl) no fd 2.
        with suppress_native_stderr():
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[Pose]:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with suppress_native_stderr():
            if self._mode == vision.RunningMode.VIDEO:
                if timestamp_ms is None:
                    raise ValueError("O modo VIDEO exige o timestamp do frame.")
                result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
            else:
                result = self._landmarker.detect(mp_image)

        poses: list[Pose] = []
        for landmarks in result.pose_landmarks:
            poses.append(
                Pose(
                    person_id=0,
                    landmarks=[
                        Landmark(
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(getattr(lm, "z", 0.0)),
                            visibility=float(getattr(lm, "visibility", 1.0)),
                        )
                        for lm in landmarks
                    ],
                )
            )

        return _sort_and_index(poses)

    def close(self) -> None:
        with suppress_native_stderr():
            self._landmarker.close()

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _sort_and_index(poses: list[Pose]) -> list[Pose]:
    """Ordena as poses da esquerda para a direita e reindexa de 0..n-1."""

    def key(pose: Pose) -> float:
        anchor = pose.pelvis or pose.centroid
        return anchor[0] if anchor else 0.0

    ordered = sorted(poses, key=key)
    for index, pose in enumerate(ordered):
        pose.person_id = index
    return ordered
