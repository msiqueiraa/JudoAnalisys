"""Teste de fumaca do recorte lado a lado com o esqueleto desenhado."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from judo_analysis.config import CLIP_END_CARD_SECONDS
from judo_analysis.domain.entities.analysis import AnalysisRun, FrameMetrics
from judo_analysis.domain.entities.video import TimeRange, Video
from judo_analysis.domain.value_objects.joint_angles import JointAngles
from judo_analysis.domain.value_objects.speed import Speed
from judo_analysis.infrastructure.render import FrameAnnotator
from judo_analysis.infrastructure.video import FfmpegClipComposer
from judo_analysis.infrastructure.video.ffmpeg_clip_composer import _annotate, _end_card
from tests.conftest import make_pose

_WIDTH = 320
_HEIGHT = 240
_FPS = 10.0
_FRAMES = 10


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_WIDTH, _HEIGHT)
    )
    assert writer.isOpened()
    try:
        for index in range(_FRAMES):
            frame = np.full((_HEIGHT, _WIDTH, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _run(label: str, path: Path) -> AnalysisRun:
    frames = [
        FrameMetrics(
            frame_index=index,
            timestamp=index / _FPS,
            angles=JointAngles(),
            pose=make_pose(
                LEFT_HIP=(0.45, 0.5),
                RIGHT_HIP=(0.55, 0.5),
                LEFT_SHOULDER=(0.42, 0.25),
                RIGHT_SHOULDER=(0.58, 0.25),
                LEFT_ANKLE=(0.45, 0.85),
                RIGHT_ANKLE=(0.55, 0.85),
            ),
        )
        for index in range(_FRAMES)
    ]
    return AnalysisRun(
        label=label,
        video=Video(
            path=str(path), fps=_FPS, frame_count=_FRAMES, width=_WIDTH, height=_HEIGHT
        ),
        time_range=TimeRange(0.0, _FRAMES / _FPS),
        person_id=0,
        frames=frames,
    )


def test_compose_draws_two_annotated_halves(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source)
    output = tmp_path / "clip.mp4"

    FfmpegClipComposer(height=_HEIGHT).compose(
        _run("Referencia", source), _run("Analisado", source), output
    )

    assert output.exists()
    assert output.stat().st_size > 0

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        ok, frame = capture.read()
        assert ok
        # Largura = metade esquerda + faixa + metade direita.
        assert frame.shape[1] == _WIDTH * 2 + 4
        assert frame.shape[0] == _HEIGHT
    finally:
        capture.release()


def test_compose_adds_the_similarity_end_card(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source)
    output = tmp_path / "clip.mp4"

    FfmpegClipComposer(height=_HEIGHT).compose(
        _run("Referencia", source),
        _run("Analisado", source),
        output,
        similarity=72.5,
    )

    capture = cv2.VideoCapture(str(output))
    try:
        # 10 frames do trecho + 2 s de cartao final (a 10 fps) = 30 frames.
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        assert frame_count >= _FRAMES + int(CLIP_END_CARD_SECONDS * _FPS)
        # O ultimo frame e o cartao (escurecido, com o texto da similaridade).
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count) - 1)
        ok, card = capture.read()
        assert ok
        assert card.mean() < 128
    finally:
        capture.release()


def test_end_card_darkens_and_writes_the_similarity() -> None:
    canvas = np.full((_HEIGHT, 640, 3), 200, dtype=np.uint8)

    card = _end_card(canvas, 80.0, "Referencia", "Analisado")

    assert card.shape == canvas.shape
    # Escurecido, mas com o texto claro no centro.
    assert card.mean() < canvas.mean()
    center = card[_HEIGHT // 2 - 60 : _HEIGHT // 2 + 20, :]
    assert center.max() > 200


def test_end_card_without_similarity_still_renders() -> None:
    canvas = np.full((_HEIGHT, 640, 3), 200, dtype=np.uint8)

    card = _end_card(canvas, None, "Referencia", "Analisado")

    assert card.shape == canvas.shape
    assert card.mean() < canvas.mean()


def test_annotate_writes_the_metrics_of_the_instant() -> None:
    frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    metrics = FrameMetrics(
        frame_index=3,
        timestamp=1.55,
        angles=JointAngles(knee_left=95.0, spine=12.0),
        hand_speed=Speed(pixels_per_second=1200.0, body_heights_per_second=3.2),
    )

    annotated = _annotate(FrameAnnotator(), frame, metrics)

    assert annotated.shape == frame.shape
    # O frame comeca preto; o texto das metricas (e o instante) deixa pixels.
    assert annotated.max() > 0
    assert frame.max() == 0  # o frame original nao e alterado


def test_annotate_without_metrics_keeps_the_frame() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert _annotate(FrameAnnotator(), frame, None) is frame
