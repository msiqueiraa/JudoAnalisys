"""Testes da escolha do frame-chave do golpe (automatico ou pelo usuario)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from judo_analysis.application.dto import AnalysisResult
from judo_analysis.application.ports.video_reader import IVideoReader
from judo_analysis.application.use_cases import SelectKeyFrameUseCase
from judo_analysis.domain.entities.analysis import AnalysisRun, FrameMetrics
from judo_analysis.domain.entities.video import TimeRange, Video
from judo_analysis.domain.services.statistics import aggregate_run
from judo_analysis.domain.value_objects.joint_angles import JointAngles
from judo_analysis.domain.value_objects.speed import Speed
from tests.conftest import make_pose

_FPS = 10.0
_FRAMES = 10
_VIDEO = Video(path="fake.mp4", fps=_FPS, frame_count=_FRAMES, width=100, height=100)


class _FakeVideoReader(IVideoReader):
    """Leitor que devolve um frame identificado pelo proprio indice."""

    def metadata(self) -> Video:
        return _VIDEO

    def frame_at(self, timestamp: float) -> tuple[int, float, np.ndarray]:
        index = max(0, min(_FRAMES - 1, int(round(timestamp * _FPS))))
        return index, index / _FPS, np.full((4, 4, 3), index, dtype=np.uint8)

    def frames(self, start: float, end: float) -> Iterator[tuple[int, float, np.ndarray]]:
        for index in range(_FRAMES):
            timestamp = index / _FPS
            if start <= timestamp < end:
                yield index, timestamp, np.full((4, 4, 3), index, dtype=np.uint8)

    def close(self) -> None:
        pass


def _result() -> AnalysisResult:
    """Analise de 1 s (10 fps) com a velocidade crescendo: pico no ultimo frame."""

    frames = [
        FrameMetrics(
            frame_index=index,
            timestamp=index / _FPS,
            angles=JointAngles(),
            hand_speed=Speed(pixels_per_second=100.0 * index, body_heights_per_second=0.1 * index),
            pose=make_pose(LEFT_HIP=(0.4, 0.5), RIGHT_HIP=(0.6, 0.5)),
        )
        for index in range(_FRAMES)
    ]
    run = AnalysisRun(
        label="Referencia",
        video=_VIDEO,
        time_range=TimeRange(0.0, _FRAMES / _FPS),
        person_id=0,
        frames=frames,
    )
    aggregate_run(run)
    return AnalysisResult(run=run, key_frame=np.zeros((4, 4, 3), np.uint8))


def test_automatic_key_frame_is_the_peak_speed() -> None:
    result = _result()

    assert result.run.key_frame_index == _FRAMES - 1
    assert result.key_frame_time is None


def test_manual_instant_moves_the_key_frame_and_rereads_the_image() -> None:
    result = _result()

    SelectKeyFrameUseCase(_FakeVideoReader()).execute(result, 0.3)

    assert result.run.key_frame_index == 3
    assert result.key_frame_time == 0.3
    # A imagem do frame-chave e relida do video naquele instante.
    assert int(result.key_frame[0, 0, 0]) == 3
    assert result.key_frame_pose is result.run.frames[3].pose


def test_instant_between_two_frames_uses_the_closest() -> None:
    result = _result()

    SelectKeyFrameUseCase(_FakeVideoReader()).execute(result, 0.36)

    assert result.run.key_frame_index == 4


def test_automatic_returns_to_the_peak_speed_frame() -> None:
    result = _result()
    SelectKeyFrameUseCase(_FakeVideoReader()).execute(result, 0.3)

    SelectKeyFrameUseCase(_FakeVideoReader()).execute(result, None)

    assert result.run.key_frame_index == _FRAMES - 1
    assert result.key_frame_time is None


def test_run_without_frames_is_kept_untouched() -> None:
    result = _result()
    result.run.frames = []

    SelectKeyFrameUseCase(_FakeVideoReader()).execute(result, 0.3)

    assert result.key_frame_time is None
