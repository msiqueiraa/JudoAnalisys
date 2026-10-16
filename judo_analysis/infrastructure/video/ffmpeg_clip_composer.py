"""Compositor de recorte lado a lado com OpenCV + FFmpeg.

Le os dois recortes, redimensiona para a mesma altura, junta lado a lado e
codifica em H.264 (compativel com navegadores) usando o FFmpeg empacotado pelo
``imageio-ffmpeg``. Sobre cada lado sao desenhados o esqueleto e as metricas do
instante que esta tocando (os mesmos dados do frame-chave do golpe). No fim
entra um cartao com a similaridade geral. O audio e descartado.
"""

from __future__ import annotations

import bisect
import math
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ...application.ports.clip_composer import IClipComposer
from ...config import CLIP_END_CARD_SECONDS, CLIP_GAP_PX, CLIP_OUTPUT_HEIGHT
from ...domain.entities.analysis import AnalysisRun, FrameMetrics
from ...domain.entities.video import TimeRange
from ...formatting import format_duration_precise
from ..render.frame_annotator import FrameAnnotator

_DEFAULT_FPS = 25.0
_LABEL_COLOR = (255, 255, 255)
_HIGHLIGHT_COLOR = (0, 255, 255)
# Primeira linha das metricas: abaixo do rotulo grande ("Referencia"/"Analisado").
_METRICS_TOP = 66
# Escurecimento do ultimo quadro no cartao final (0..1).
_END_CARD_DARKENING = 0.35


class FfmpegClipComposer(IClipComposer):
    """Implementacao de :class:`IClipComposer` com OpenCV e FFmpeg."""

    def __init__(
        self,
        height: int = CLIP_OUTPUT_HEIGHT,
        gap: int = CLIP_GAP_PX,
        output_fps: float | None = None,
    ) -> None:
        self._height = height
        self._gap = gap
        self._output_fps = output_fps
        self._annotator = FrameAnnotator()

    def compose(
        self,
        reference: AnalysisRun,
        analyzed: AnalysisRun,
        output_path: Path,
        similarity: float | None = None,
    ) -> Path:
        reference_capture = cv2.VideoCapture(reference.video.path)
        analyzed_capture = cv2.VideoCapture(analyzed.video.path)
        if not reference_capture.isOpened() or not analyzed_capture.isOpened():
            reference_capture.release()
            analyzed_capture.release()
            raise IOError("Nao foi possivel abrir um dos videos para o recorte.")

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            reference_fps = _fps_of(reference_capture)
            analyzed_fps = _fps_of(analyzed_capture)
            output_fps = self._output_fps or min(reference_fps, analyzed_fps)

            reference_width = _scaled_width(reference_capture, self._height)
            analyzed_width = _scaled_width(analyzed_capture, self._height)
            composite_width = reference_width + self._gap + analyzed_width
            if composite_width % 2:
                composite_width += 1

            reference_range = reference.time_range
            analyzed_range = analyzed.time_range
            duration = max(reference_range.duration, analyzed_range.duration)
            frame_count = max(1, int(math.ceil(duration * output_fps)))
            card_frame_count = max(1, int(round(CLIP_END_CARD_SECONDS * output_fps)))

            reference_frames = _iter_frames(
                reference_capture, reference_fps, reference_range, output_fps, frame_count
            )
            analyzed_frames = _iter_frames(
                analyzed_capture, analyzed_fps, analyzed_range, output_fps, frame_count
            )
            reference_metrics = _MetricsProvider(reference)
            analyzed_metrics = _MetricsProvider(analyzed)

            command = _ffmpeg_command(
                composite_width, self._height, output_fps, output_path
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None

            try:
                canvas: np.ndarray | None = None
                for index in range(frame_count):
                    elapsed = index / output_fps
                    left = _prepare(next(reference_frames), reference_width, self._height)
                    right = _prepare(next(analyzed_frames), analyzed_width, self._height)
                    left = _annotate(
                        self._annotator,
                        left,
                        reference_metrics.at(reference_range.start + elapsed),
                    )
                    right = _annotate(
                        self._annotator,
                        right,
                        analyzed_metrics.at(analyzed_range.start + elapsed),
                    )
                    canvas = _compose(left, right, self._gap, composite_width, self._height)
                    process.stdin.write(canvas.tobytes())

                # Cartao final: ultimo quadro escurecido com a similaridade geral.
                assert canvas is not None
                card = _end_card(canvas, similarity, reference.label, analyzed.label)
                for _ in range(card_frame_count):
                    process.stdin.write(card.tobytes())
            finally:
                process.stdin.close()
                return_code = process.wait()
                error_output = process.stderr.read().decode(errors="replace") if process.stderr else ""

            if return_code != 0:
                raise RuntimeError(f"FFmpeg falhou ao gerar o recorte: {error_output.strip()}")

            return output_path
        finally:
            reference_capture.release()
            analyzed_capture.release()


class _MetricsProvider:
    """Devolve as metricas rastreadas mais proximas de um instante."""

    def __init__(self, run: AnalysisRun) -> None:
        self._frames = sorted(run.frames, key=lambda frame: frame.timestamp)
        self._times = [frame.timestamp for frame in self._frames]

    def at(self, timestamp: float) -> FrameMetrics | None:
        if not self._frames:
            return None
        index = bisect.bisect_left(self._times, timestamp)
        if index <= 0:
            return self._frames[0]
        if index >= len(self._frames):
            return self._frames[-1]
        before = self._frames[index - 1]
        after = self._frames[index]
        return before if abs(timestamp - before.timestamp) <= abs(after.timestamp - timestamp) else after


def _annotate(
    annotator: FrameAnnotator, frame: np.ndarray, metrics: FrameMetrics | None
) -> np.ndarray:
    """Desenha esqueleto e metricas do instante, como no frame-chave do golpe."""

    if metrics is None:
        return frame
    return annotator.annotate(
        frame,
        metrics.pose,
        metrics.angles,
        metrics.hand_speed,
        title=format_duration_precise(metrics.timestamp),
        text_top=_METRICS_TOP,
    )


def _fps_of(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else _DEFAULT_FPS


def _scaled_width(capture: cv2.VideoCapture, height: int) -> int:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or source_height <= 0:
        return height
    scaled = int(round(width * (height / source_height)))
    return max(2, scaled + (scaled % 2))


def _iter_frames(
    capture: cv2.VideoCapture,
    source_fps: float,
    time_range: TimeRange,
    output_fps: float,
    frame_count: int,
) -> Iterator[np.ndarray | None]:
    """Gera um frame (ou ``None``) para cada instante do recorte composto.

    Le os frames sequencialmente. Quando a fonte termina antes do fim do
    recorte, repete o ultimo frame (congelamento).
    """

    start_frame = max(0, int(round(time_range.start * source_fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    last_frame: np.ndarray | None = None
    next_index = 0
    for index in range(frame_count):
        needed = int(round((index / output_fps) * source_fps))
        while next_index <= needed:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
            next_index += 1
        yield last_frame


def _prepare(frame: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if frame is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _compose(
    left: np.ndarray,
    right: np.ndarray,
    gap: int,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    right_start = left.shape[1] + gap
    canvas[:, right_start : right_start + right.shape[1]] = right
    _draw_label(canvas, "Referencia", 10)
    _draw_label(canvas, "Analisado", right_start + 10)
    return canvas


def _end_card(
    canvas: np.ndarray,
    similarity: float | None,
    reference_label: str,
    analyzed_label: str,
) -> np.ndarray:
    """Ultimo quadro escurecido com a similaridade geral em destaque."""

    card = (canvas.astype(np.float32) * _END_CARD_DARKENING).astype(np.uint8)
    center = card.shape[0] // 2
    _draw_centered(card, "Similaridade geral", center - int(card.shape[0] * 0.18), 1.0, _LABEL_COLOR)
    _draw_centered(
        card,
        f"{similarity:.1f}%" if similarity is not None else "--",
        center + int(card.shape[0] * 0.06),
        2.4,
        _HIGHLIGHT_COLOR,
    )
    _draw_centered(
        card,
        f"{reference_label} x {analyzed_label}",
        center + int(card.shape[0] * 0.26),
        0.7,
        _LABEL_COLOR,
    )
    return card


def _draw_label(canvas: np.ndarray, text: str, x: int) -> None:
    cv2.putText(canvas, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _LABEL_COLOR, 2, cv2.LINE_AA)


def _draw_centered(
    canvas: np.ndarray, text: str, y: int, scale: float, color: tuple[int, int, int]
) -> None:
    """Escreve o texto centralizado horizontalmente na altura ``y``."""

    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(0, (canvas.shape[1] - text_width) // 2)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _ffmpeg_command(width: int, height: int, fps: float, output_path: Path) -> list[str]:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
