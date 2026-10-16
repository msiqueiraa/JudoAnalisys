"""Recursos compartilhados e cacheados pela interface."""

from __future__ import annotations

import streamlit as st

from ..application.pose_detection import PoseDetector
from ..config import DEFAULT_MODEL_PATH
from ..domain.entities.video import Video
from ..infrastructure.color import OpenCVGiColorSampler
from ..infrastructure.pose import MediaPipePoseEstimator
from ..infrastructure.video import OpenCVVideoReader


@st.cache_resource(show_spinner=False)
def get_pose_estimator(num_poses: int) -> MediaPipePoseEstimator:
    """Estimador de pose no modo IMAGE, criado uma vez por quantidade de pessoas.

    Carregar o modelo do MediaPipe e caro, por isso ele e reaproveitado entre a
    previa (selecao da pessoa) e a analise. O modo IMAGE trata cada frame de
    forma independente, o que se mostrou melhor para videos com dois atletas:
    no modo VIDEO o rastreamento interno tende a fundir os dois corpos em uma
    unica deteccao quando eles se sobrepoem no golpe.
    """

    return MediaPipePoseEstimator(
        DEFAULT_MODEL_PATH, num_poses=num_poses, running_mode="IMAGE"
    )


def make_pose_detector(num_poses: int) -> PoseDetector:
    """Detector completo: deduplicacao, segunda tentativa e cor do kimono.

    O objeto guarda o estado da analise (quantas pessoas a cena ja teve), por
    isso e criado a cada uso, enquanto o estimador do MediaPipe continua
    cacheado e compartilhado.
    """

    return PoseDetector(
        get_pose_estimator(num_poses), color_sampler=OpenCVGiColorSampler()
    )


def read_metadata(path: str) -> Video:
    """Le os metadados de um video abrindo e fechando o arquivo."""

    with OpenCVVideoReader(path) as reader:
        return reader.metadata()
