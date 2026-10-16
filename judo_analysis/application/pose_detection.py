"""Deteccao de pessoas em um frame: deduplicacao, segunda tentativa e cor.

Reune, em um unico ponto, o que os casos de uso precisam antes de rastrear:

1. **deduplicar** poses que descrevem o mesmo corpo - o detector as vezes devolve
   duas poses sobre o mesmo atleta e nenhuma para o adversario, o que engana o
   rastreador;
2. **tentar de novo** nas metades do quadro sempre que vier menos de dois atletas;
3. **medir a cor do kimono** de cada pessoa detectada.
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_SPLIT_RETRY, MIN_VISIBLE_LANDMARKS
from ..domain.entities.landmark import Landmark
from ..domain.entities.pose import Pose
from ..domain.services.pose_dedup import deduplicate
from .ports.gi_color_sampler import IGiColorSampler
from .ports.pose_estimator import IPoseEstimator

# Abaixo de duas pessoas distintas a segunda tentativa entra em acao.
_TWO_PEOPLE = 2
# Largura minima (px) de uma metade do quadro para valer a pena tentar de novo.
_MIN_HALF_WIDTH = 16


class PoseDetector(IPoseEstimator):
    """Detector de pose com deduplicacao, segunda tentativa e cor do kimono."""

    def __init__(
        self,
        estimator: IPoseEstimator,
        color_sampler: IGiColorSampler | None = None,
        retry: bool = True,
        split_retry: bool = DEFAULT_SPLIT_RETRY,
    ) -> None:
        self._estimator = estimator
        self._sampler = color_sampler
        self._retry = retry
        self._split_retry = split_retry
        self._max_seen = 0

    @property
    def max_people_seen(self) -> int:
        """Maior numero de pessoas distintas ja vistas no video."""

        return self._max_seen

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[Pose]:
        poses = deduplicate(self._estimator.detect(frame, timestamp_ms=timestamp_ms))

        # Sempre que o quadro traz menos de duas pessoas, tenta de novo nas
        # metades. Nao da para condicionar isso a "ja vimos dois atletas": e
        # exatamente quando o detector funde os dois corpos que ele NUNCA ve dois
        # e o rastreamento passa a alternar entre os atletas.
        if self._retry and len(poses) < _TWO_PEOPLE:
            extra = self._extra_detections(frame, timestamp_ms)
            if extra:
                poses = deduplicate([*poses, *extra])

        self._max_seen = max(self._max_seen, len(poses))

        if self._sampler is not None:
            for pose in poses:
                pose.gi_color = self._sampler.sample(frame, pose)
        return poses

    def close(self) -> None:
        """Nao fecha nada: os estimadores recebidos sao compartilhados/cacheados."""

    def _extra_detections(self, frame: np.ndarray, timestamp_ms: int | None) -> list[Pose]:
        """Segunda tentativa de encontrar o atleta que o detector perdeu."""

        if not self._split_retry:
            return []
        return detect_in_halves(self._estimator, frame, timestamp_ms)


def detect_in_halves(
    estimator: IPoseEstimator, frame: np.ndarray, timestamp_ms: int | None = None
) -> list[Pose]:
    """Roda o detector nas metades esquerda e direita do frame.

    Ajuda quando o detector funde os dois atletas em uma unica caixa: em cada
    metade so um deles costuma ficar inteiro, o que facilita a deteccao. As
    poses voltam para o sistema de coordenadas do frame inteiro e deteccoes
    incompletas (poucos landmarks) sao descartadas.
    """

    height, width = frame.shape[:2]
    split = width // 2
    if split < _MIN_HALF_WIDTH or width - split < _MIN_HALF_WIDTH:
        return []

    poses: list[Pose] = []
    for left in (0.0, 0.5):
        start = int(left * width)
        end = width if left > 0 else split
        half = np.ascontiguousarray(frame[:, start:end])
        fraction = (end - start) / width
        for pose in estimator.detect(half, timestamp_ms=timestamp_ms):
            if not is_measurable_pose(pose):
                continue
            poses.append(_shift_pose(pose, left, fraction))
    return poses


def is_measurable_pose(pose: Pose | None) -> bool:
    """Pose com landmarks visiveis suficientes para medir os angulos.

    Vale para um frame de video rastreado e para a pose escolhida em uma foto:
    abaixo do minimo os angulos saem de pontos inventados pelo detector.
    """

    return pose is not None and pose.visible_count >= MIN_VISIBLE_LANDMARKS


def _shift_pose(pose: Pose, left: float, fraction: float) -> Pose:
    """Move os landmarks de uma pose de um recorte para o frame inteiro.

    O detector devolve coordenadas normalizadas ao **recorte** (0..1 dentro da
    metade). Para voltar ao quadro inteiro a posicao precisa ser comprimida pela
    fracao que a metade ocupa e depois deslocada - somar o deslocamento direto
    dobrava a posicao horizontal das deteccoes da metade esquerda.
    """

    return Pose(
        person_id=pose.person_id,
        landmarks=[
            Landmark(
                x=left + point.x * fraction,
                y=point.y,
                z=point.z,
                visibility=point.visibility,
            )
            for point in pose.landmarks
        ],
        score=pose.score,
    )
