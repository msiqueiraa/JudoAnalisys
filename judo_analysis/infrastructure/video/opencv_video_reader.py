"""Leitor de frames de video com OpenCV."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from ...application.ports.video_reader import IVideoReader
from ...domain.entities.video import Video

_DEFAULT_FPS = 30.0


class OpenCVVideoReader(IVideoReader):
    """Implementacao de :class:`IVideoReader` com ``cv2.VideoCapture``."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Nao foi possivel abrir o video: {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else _DEFAULT_FPS
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def metadata(self) -> Video:
        return Video(
            path=self._path,
            fps=self._fps,
            frame_count=self._frame_count,
            width=self._width,
            height=self._height,
        )

    def frame_at(self, timestamp: float) -> tuple[int, float, np.ndarray]:
        index = max(0, int(round(timestamp * self._fps)))
        if self._frame_count:
            index = min(index, max(0, self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise IOError(f"Nao foi possivel ler o frame {index} do video.")
        return index, index / self._fps, frame

    def frames(self, start: float, end: float) -> Iterator[tuple[int, float, np.ndarray]]:
        first = max(0, int(start * self._fps))
        last = int(end * self._fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        index = first
        while index < last:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield index, index / self._fps, frame
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
