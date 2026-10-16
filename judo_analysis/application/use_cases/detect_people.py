"""Caso de uso: detectar as pessoas presentes em um frame de previa."""

from __future__ import annotations

from ..dto import PersonPreview
from ..pose_detection import PoseDetector
from ..ports.gi_color_sampler import IGiColorSampler
from ..ports.pose_estimator import IPoseEstimator
from ..ports.video_reader import IVideoReader


class DetectPeopleUseCase:
    """Extrai um frame e detecta todas as pessoas nele.

    Usado para montar a tela de selecao: o usuario escolhe qual pessoa sera a
    referencia ou a pessoa analisada. As poses ja vem deduplicadas, com a cor do
    kimono medida e - quando a cena tem dois atletas - com uma segunda tentativa
    de deteccao caso um deles tenha "desaparecido".
    """

    def __init__(
        self,
        video_reader: IVideoReader,
        pose_estimator: IPoseEstimator,
        color_sampler: IGiColorSampler | None = None,
    ) -> None:
        self._reader = video_reader
        self._detector = (
            pose_estimator
            if isinstance(pose_estimator, PoseDetector)
            else PoseDetector(pose_estimator, color_sampler=color_sampler)
        )

    def execute(self, timestamp: float) -> PersonPreview:
        frame_index, actual_ts, frame = self._reader.frame_at(timestamp)
        poses = self._detector.detect(frame, timestamp_ms=int(round(actual_ts * 1000)))
        return PersonPreview(
            frame_index=frame_index,
            timestamp=actual_ts,
            frame=frame,
            poses=poses,
        )
