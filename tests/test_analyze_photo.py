"""Testes do caso de uso de analise de foto (um quadro, sem velocidade)."""

from __future__ import annotations

import numpy as np
import pytest

from judo_analysis.application.dto import PersonSelection
from judo_analysis.application.use_cases import AnalyzePhotoUseCase
from judo_analysis.config import PHOTO_FPS
from tests.conftest import make_pose

_IMAGE = np.zeros((48, 64, 3), dtype=np.uint8)

# Pose com todos os landmarks que os angulos precisam.
_ANGLED = {
    "LEFT_SHOULDER": (0.35, 0.30),
    "RIGHT_SHOULDER": (0.65, 0.30),
    "LEFT_HIP": (0.40, 0.50),
    "RIGHT_HIP": (0.60, 0.50),
    "LEFT_KNEE": (0.35, 0.65),
    "RIGHT_KNEE": (0.65, 0.65),
    "LEFT_ANKLE": (0.30, 0.85),
    "RIGHT_ANKLE": (0.70, 0.85),
    "LEFT_EAR": (0.42, 0.20),
    "RIGHT_EAR": (0.58, 0.20),
}


def _selection(**points: tuple[float, float]) -> PersonSelection:
    pose = make_pose(**points)
    return PersonSelection(person_id=pose.person_id, pose=pose)


def test_photo_run_has_a_single_frame_and_no_speed() -> None:
    result = AnalyzePhotoUseCase().execute(
        "Analisada", _IMAGE, _selection(**_ANGLED), "foto.png"
    )

    run = result.run
    assert run.is_photo
    assert run.label == "Analisada"
    assert len(run.frames) == 1
    assert run.frames[0].frame_index == 0
    assert run.frames[0].timestamp == 0.0
    assert run.frames[0].hand_speed is None
    assert run.person_id == 0
    assert run.video.path == "foto.png"
    assert run.video.frame_count == 1
    assert run.video.fps == PHOTO_FPS
    assert run.video.width == 64
    assert run.video.height == 48
    # A foto nao tem duracao real, mas o intervalo precisa ser valido.
    assert run.time_range.start == 0.0
    assert run.time_range.end > run.time_range.start

    # Os angulos foram medidos e agregados; a velocidade nao existe em foto.
    assert run.stats["knee_left"].count == 1
    assert run.stats["spine"].count == 1
    assert "hand_speed" not in run.stats
    assert run.key_frame_index == 0

    assert result.key_frame is _IMAGE
    assert result.key_frame_pose is run.frames[0].pose
    assert result.selection is not None


def test_pose_with_too_few_landmarks_is_rejected() -> None:
    selection = _selection(LEFT_HIP=(0.4, 0.5), RIGHT_HIP=(0.6, 0.5))

    with pytest.raises(ValueError, match="poucos pontos"):
        AnalyzePhotoUseCase().execute("Analisada", _IMAGE, selection)


def test_pose_without_any_measurable_angle_is_rejected() -> None:
    # Sete landmarks visiveis, mas nenhum deles forma um dos angulos medidos.
    selection = _selection(
        NOSE=(0.50, 0.10),
        LEFT_EYE=(0.45, 0.10),
        RIGHT_EYE=(0.55, 0.10),
        LEFT_WRIST=(0.30, 0.60),
        RIGHT_WRIST=(0.70, 0.60),
        LEFT_ELBOW=(0.35, 0.50),
        RIGHT_ELBOW=(0.65, 0.50),
    )

    with pytest.raises(ValueError, match="nenhum angulo"):
        AnalyzePhotoUseCase().execute("Analisada", _IMAGE, selection)
