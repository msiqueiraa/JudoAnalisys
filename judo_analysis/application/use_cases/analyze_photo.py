"""Caso de uso: medir os angulos de um atleta em uma foto (quadro unico).

Uma foto nao tem dimensao temporal: nao existe velocidade nem serie para
desenhar. O resultado sai no mesmo formato da analise de video - um
``AnalysisRun`` com um unico frame -, para a comparacao, o painel e o relatorio
funcionarem sem um caminho paralelo.
"""

from __future__ import annotations

import numpy as np

from ...config import PHOTO_FPS
from ...domain.entities.analysis import AnalysisRun, FrameMetrics
from ...domain.entities.video import TimeRange, Video
from ...domain.services.angle_calculator import AngleCalculator
from ...domain.services.statistics import aggregate_run
from ..dto import AnalysisResult, PersonSelection
from ..pose_detection import is_measurable_pose

_INVALID_POSE = (
    "A pessoa escolhida tem poucos pontos visiveis nesta foto. Escolha outro "
    "atleta ou use uma foto com o corpo mais inteiro."
)
_NO_ANGLES = (
    "Nao foi possivel medir nenhum angulo nesta foto. Escolha outro atleta ou "
    "use uma foto com o corpo mais inteiro."
)


class AnalyzePhotoUseCase:
    """Mede os angulos da pessoa escolhida em uma foto."""

    def __init__(self, angle_calculator: AngleCalculator | None = None) -> None:
        self._angles = angle_calculator or AngleCalculator()

    def execute(
        self,
        label: str,
        image: np.ndarray,
        selection: PersonSelection,
        path: str = "",
    ) -> AnalysisResult:
        """Mede os angulos da pose escolhida e monta o resultado de uma foto.

        Raises:
            ValueError: Quando a pose escolhida tem pontos visiveis de menos ou
                quando nenhum angulo pode ser medido (a foto nao serve).
        """

        pose = selection.pose
        if not is_measurable_pose(pose):
            raise ValueError(_INVALID_POSE)

        angles = self._angles.compute(pose)
        if all(value is None for value in angles.as_dict().values()):
            raise ValueError(_NO_ANGLES)

        height, width = image.shape[:2]
        video = Video(
            path=path,
            fps=PHOTO_FPS,
            frame_count=1,
            width=width,
            height=height,
        )
        run = AnalysisRun(
            label=label,
            video=video,
            time_range=TimeRange(start=0.0, end=video.duration),
            person_id=selection.person_id,
            frames=[
                FrameMetrics(
                    frame_index=0,
                    timestamp=0.0,
                    angles=angles,
                    hand_speed=None,
                    pose=pose,
                )
            ],
            is_photo=True,
        )
        aggregate_run(run)
        return AnalysisResult(
            run=run,
            key_frame=image,
            key_frame_pose=pose,
            selection=selection,
        )
