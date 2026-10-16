"""Testes dos adaptadores de imagem (leitor de um quadro e upload de foto)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from judo_analysis.config import PHOTO_FPS
from judo_analysis.infrastructure.files import safe_name
from judo_analysis.infrastructure.image import (
    SingleImageReader,
    UploadPhotoSource,
    read_image,
)


def _write_image(path: Path, height: int = 48, width: int = 64) -> Path:
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_reader_treats_the_photo_as_a_single_frame_video() -> None:
    path = _write_image(Path(tempfile.mkdtemp()) / "foto.png")

    with SingleImageReader(str(path)) as reader:
        video = reader.metadata()
        assert video.path == str(path)
        assert video.fps == PHOTO_FPS
        assert video.frame_count == 1
        assert video.width == 64
        assert video.height == 48

        index, timestamp, frame = reader.frame_at(0.0)
        assert (index, timestamp) == (0, 0.0)
        assert frame.shape == (48, 64, 3)

        frames = list(reader.frames(0.0, 1.0))
        assert len(frames) == 1
        assert frames[0][0] == 0


def test_read_image_reports_a_missing_file() -> None:
    missing = Path(tempfile.mkdtemp()) / "nao_existe.png"

    with pytest.raises(IOError, match="Nao foi possivel ler a imagem"):
        read_image(str(missing))


def test_upload_photo_source_writes_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    destination = Path(tempfile.mkdtemp())
    monkeypatch.setattr(
        "judo_analysis.infrastructure.image.photo_source.PHOTO_DIR", destination
    )

    resolved = UploadPhotoSource("Minha Foto!.jpg", b"conteudo").resolve()

    assert Path(resolved).parent == destination
    assert Path(resolved).name == "Minha_Foto_.jpg"
    assert Path(resolved).read_bytes() == b"conteudo"


def test_safe_name_keeps_the_base_name_only() -> None:
    assert safe_name("../../etc/passwd", "foto.png") == "passwd"
    assert safe_name("", "foto.png") == "foto.png"
    assert safe_name("a b c.png", "foto.png") == "a_b_c.png"
