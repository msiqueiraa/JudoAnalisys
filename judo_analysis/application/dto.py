"""Objetos de transferencia (DTO) que trafegam entre aplicacao e apresentacao."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..domain.entities.analysis import AnalysisRun
from ..domain.entities.pose import Pose
from ..domain.entities.video import TimeRange


@dataclass(slots=True)
class PersonPreview:
    """Frame de previa com as pessoas detectadas, para o usuario escolher."""

    frame_index: int
    timestamp: float
    frame: np.ndarray
    poses: list[Pose] = field(default_factory=list)


@dataclass(slots=True)
class PersonSelection:
    """Pessoa escolhida pelo usuario, ancorada em sua pose no frame de previa."""

    person_id: int
    pose: Pose


@dataclass(slots=True)
class TrackingSample:
    """Amostra do rastreamento para conferencia visual (sem imagem).

    A imagem e relida pela apresentacao a partir do video, evitando manter
    frames em memoria.
    """

    frame_index: int
    timestamp: float
    pose: Pose | None = None


@dataclass(slots=True)
class AnalysisResult:
    """Resultado da analise de um video (ou foto) pronto para exibicao/relatorio."""

    run: AnalysisRun
    key_frame: np.ndarray
    key_frame_pose: Pose | None = None
    tracking_samples: list[TrackingSample] = field(default_factory=list)
    # Pessoa usada na analise (permite reanalisar mantendo a mesma identidade).
    selection: PersonSelection | None = None
    # Trechos (tempo do video) descartados manualmente pelo usuario.
    excluded_ranges: list[TimeRange] = field(default_factory=list)
    excluded_frame_count: int = 0
    # Instante escolhido pelo usuario para o frame-chave do golpe.
    # ``None`` = automatico (frame de maior velocidade da mao).
    key_frame_time: float | None = None
