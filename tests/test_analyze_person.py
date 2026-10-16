"""Testes do caso de uso de analise: rastreamento da pessoa e correcao ancorada.

Reproduz, sem o MediaPipe, o cenario do bug relatado: dois atletas que se
cruzam no golpe. O rastreamento deve manter a identidade da pessoa escolhida.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from judo_analysis.application.dto import PersonSelection
from judo_analysis.application.ports.pose_estimator import IPoseEstimator
from judo_analysis.application.ports.video_reader import IVideoReader
from judo_analysis.application.use_cases.analyze_person import AnalyzePersonUseCase
from judo_analysis.domain.entities.pose import Pose
from judo_analysis.domain.entities.video import TimeRange, Video
from tests.conftest import make_pose

_FPS = 10
_FRAMES = 60
_STEP = 0.01  # deslocamento por frame de cada atleta
_VIDEO = Video(path="fake.mp4", fps=_FPS, frame_count=_FRAMES, width=100, height=100)


def _pose(cx: float) -> Pose:
    """Pose sintetica com pelve em ``(cx, 0.5)`` e altura de corpo constante."""

    return make_pose(
        LEFT_HIP=(cx - 0.03, 0.5),
        RIGHT_HIP=(cx + 0.03, 0.5),
        LEFT_SHOULDER=(cx - 0.05, 0.3),
        RIGHT_SHOULDER=(cx + 0.05, 0.3),
        LEFT_ANKLE=(cx - 0.02, 0.7),
        RIGHT_ANKLE=(cx + 0.02, 0.7),
    )


def _position_a(index: int) -> float:
    return 0.2 + _STEP * index


def _position_b(index: int) -> float:
    return 0.8 - _STEP * index


class _FakeVideoReader(IVideoReader):
    """Leitor de video que entrega frames vazios em instantes fixos."""

    def __init__(self) -> None:
        self._times = [index / _FPS for index in range(_FRAMES)]

    def metadata(self) -> Video:
        return _VIDEO

    def frame_at(self, timestamp: float) -> tuple[int, float, np.ndarray]:
        index = min(range(_FRAMES), key=lambda i: abs(self._times[i] - timestamp))
        return index, self._times[index], np.zeros((4, 4, 3), np.uint8)

    def frames(self, start: float, end: float) -> Iterator[tuple[int, float, np.ndarray]]:
        for index, timestamp in enumerate(self._times):
            if start <= timestamp < end:
                yield index, timestamp, np.zeros((4, 4, 3), np.uint8)

    def close(self) -> None:
        pass


class _CrossingEstimator(IPoseEstimator):
    """Estima dois atletas que trocam de lado ao longo do trecho."""

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[Pose]:
        index = int(round((timestamp_ms or 0) / 1000 * _FPS))
        return [_pose(_position_a(index)), _pose(_position_b(index))]

    def close(self) -> None:
        pass


def _analyze(selection: PersonSelection, anchor: float | None = None):
    with _FakeVideoReader() as reader:
        return AnalyzePersonUseCase(reader, _CrossingEstimator()).execute(
            "Referencia", TimeRange(0.0, _FRAMES / _FPS), selection, anchor
        )


def _sample_positions(result) -> list[float]:
    return [sample.pose.pelvis[0] for sample in result.tracking_samples if sample.pose is not None]


def test_follows_the_chosen_athlete_through_the_crossing() -> None:
    selection = PersonSelection(person_id=0, pose=_pose(_position_a(0)))
    result = _analyze(selection)

    positions = _sample_positions(result)
    assert positions, "o rastreador nao acompanhou a pessoa escolhida"
    # Todas as amostras devem estar sobre o atleta A (que cruza com B no meio).
    for sample, position in zip(result.tracking_samples, positions):
        index = int(round(sample.timestamp * _FPS))
        assert position == pytest.approx(_position_a(index), abs=0.05)
        # E nao sobre o atleta B.
        if index != _FRAMES // 2:  # no cruzamento os dois coincidem
            assert abs(position - _position_b(index)) > 0.05


def test_follows_the_other_athlete_when_selected() -> None:
    selection = PersonSelection(person_id=1, pose=_pose(_position_b(0)))
    result = _analyze(selection)

    for sample in result.tracking_samples:
        if sample.pose is None:
            continue
        index = int(round(sample.timestamp * _FPS))
        assert sample.pose.pelvis[0] == pytest.approx(_position_b(index), abs=0.05)


def test_anchor_tracks_forward_and_backward() -> None:
    """Correcao ancorada no meio do trecho deve valer para antes e depois."""

    anchor_index = 40
    anchor_time = anchor_index / _FPS
    selection = PersonSelection(person_id=0, pose=_pose(_position_a(anchor_index)))
    result = _analyze(selection, anchor=anchor_time)

    samples = [sample for sample in result.tracking_samples if sample.pose is not None]
    assert len(samples) >= 5
    for sample in samples:
        index = int(round(sample.timestamp * _FPS))
        assert sample.pose.pelvis[0] == pytest.approx(_position_a(index), abs=0.05)


def test_key_frame_pose_belongs_to_the_chosen_athlete() -> None:
    selection = PersonSelection(person_id=0, pose=_pose(_position_a(0)))
    result = _analyze(selection)
    assert result.key_frame_pose is not None
    assert result.key_frame_pose.pelvis is not None


def test_chosen_key_frame_is_used_and_kept_for_reanalysis() -> None:
    """O instante escolhido pelo usuario vale para o painel e para o PDF."""

    selection = PersonSelection(person_id=0, pose=_pose(_position_a(0)))
    with _FakeVideoReader() as reader:
        result = AnalyzePersonUseCase(reader, _CrossingEstimator()).execute(
            "Referencia",
            TimeRange(0.0, _FRAMES / _FPS),
            selection,
            None,
            None,
            1.02,
        )

    # 1,02 s a 10 fps fica no frame 10 (o pico de velocidade seria outro).
    assert result.run.key_frame_index == 10
    assert result.key_frame_time == 1.02
    assert result.key_frame_pose is not None


class _MissingHalfEstimator(IPoseEstimator):
    """Detecta a pessoa apenas na primeira metade (segunda metade "bugada")."""

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[Pose]:
        index = int(round((timestamp_ms or 0) / 1000 * _FPS))
        if index >= _FRAMES // 2:
            return []
        return [_pose(_position_a(index))]

    def close(self) -> None:
        pass


def test_frames_without_tracking_are_removed() -> None:
    selection = PersonSelection(person_id=0, pose=_pose(_position_a(0)))
    with _FakeVideoReader() as reader:
        result = AnalyzePersonUseCase(reader, _MissingHalfEstimator()).execute(
            "Referencia", TimeRange(0.0, _FRAMES / _FPS), selection
        )

    assert result.run.frames, "os frames validos devem ser mantidos"
    # Frames em que o rastreamento falhou nao entram na analise.
    assert len(result.run.frames) < _FRAMES
    assert all(frame.pose is not None for frame in result.run.frames)


def test_manual_exclusion_removes_the_chosen_frames() -> None:
    selection = PersonSelection(person_id=0, pose=_pose(_position_a(0)))
    with _FakeVideoReader() as reader:
        result = AnalyzePersonUseCase(reader, _CrossingEstimator()).execute(
            "Referencia",
            TimeRange(0.0, _FRAMES / _FPS),
            selection,
            None,
            [TimeRange(2.0, 4.0)],
        )

    # 2s a 4s, a 10 fps, sao 20 frames descartados manualmente.
    assert result.excluded_frame_count == 20
    assert len(result.run.frames) == _FRAMES - 20
    assert all(not 2.0 <= frame.timestamp < 4.0 for frame in result.run.frames)
    assert result.excluded_ranges == [TimeRange(2.0, 4.0)]
