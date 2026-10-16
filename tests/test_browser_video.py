"""Testes da previa do video compativel com o player do navegador.

O ``st.video`` serve o arquivo como ``video/mp4``; containers como ``.mov`` nao
tocam e a previa fica em branco. Aqui verificamos a decisao de converter e a
conversao real com o FFmpeg empacotado.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from judo_analysis.infrastructure.video import browser_video
from judo_analysis.infrastructure.video.browser_video import (
    BrowserVideoConverter,
    needs_conversion,
    video_codec,
)

_WIDTH = 64
_HEIGHT = 48
_FRAMES = 5


def _write_video(path: Path, fourcc: str = "mp4v", fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (_WIDTH, _HEIGHT)
    )
    assert writer.isOpened(), f"nao foi possivel criar {path}"
    try:
        for index in range(_FRAMES):
            frame = np.full((_HEIGHT, _WIDTH, 3), index * 40, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


# --- Decisao de converter ---------------------------------------------------


def test_webm_never_needs_conversion() -> None:
    assert not needs_conversion("video.webm")


def test_other_containers_always_need_conversion() -> None:
    for name in ("video.mov", "video.MOV", "video.avi", "video.mkv"):
        assert needs_conversion(name), name


def test_mp4_with_unsupported_codec_needs_conversion(monkeypatch) -> None:
    monkeypatch.setattr(browser_video, "video_codec", lambda path: "hevc")
    assert needs_conversion("video.mp4")


def test_mp4_with_supported_codec_is_kept(monkeypatch) -> None:
    for codec in ("h264", "avc1", "av01"):
        monkeypatch.setattr(browser_video, "video_codec", lambda path, codec=codec: codec)
        assert not needs_conversion("video.mp4"), codec


def test_mp4_with_unknown_codec_is_kept(monkeypatch) -> None:
    # Sem informacao de codec nao vale a pena recodificar: assume compativel.
    monkeypatch.setattr(browser_video, "video_codec", lambda path: "")
    assert not needs_conversion("video.mp4")


# --- Conversao real ---------------------------------------------------------


@pytest.fixture(scope="module")
def mov_video(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("previa")
    return _write_video(directory / "atleta.mov")


def test_mov_is_converted_to_a_playable_mp4(tmp_path, mov_video) -> None:
    converter = BrowserVideoConverter(cache_dir=tmp_path)

    prepared = Path(converter.prepare(str(mov_video)))

    assert prepared != mov_video
    assert prepared.suffix == ".mp4"
    assert prepared.exists()
    # A saida precisa ser H.264 (o que o navegador toca em MP4).
    assert video_codec(prepared) in {"h264", "avc1"}


def test_conversion_result_is_reused_from_the_cache(tmp_path, mov_video, monkeypatch) -> None:
    converter = BrowserVideoConverter(cache_dir=tmp_path)
    first = converter.prepare(str(mov_video))

    def _explode(*args, **kwargs):
        raise AssertionError("nao deveria chamar o FFmpeg de novo")

    monkeypatch.setattr(browser_video.subprocess, "run", _explode)
    assert converter.prepare(str(mov_video)) == first


def test_converted_preview_is_invalidated_when_the_file_changes(tmp_path) -> None:
    directory = tmp_path / "entrada"
    directory.mkdir()
    video = _write_video(directory / "atleta.mov")
    converter = BrowserVideoConverter(cache_dir=tmp_path / "previa")

    first = converter.prepare(str(video))
    video.write_bytes(video.read_bytes() + b"\0")  # simula outro arquivo no mesmo caminho
    second = converter.prepare(str(video))

    assert first != second


def test_playable_video_is_returned_untouched(tmp_path, monkeypatch) -> None:
    video = _write_video(tmp_path / "atleta.mp4")
    monkeypatch.setattr(browser_video, "needs_conversion", lambda path: False)
    assert BrowserVideoConverter(cache_dir=tmp_path).prepare(str(video)) == str(video)


def test_missing_file_is_returned_untouched(tmp_path) -> None:
    missing = str(tmp_path / "nao_existe.mov")
    assert BrowserVideoConverter(cache_dir=tmp_path).prepare(missing) == missing


def test_failure_keeps_the_original_video(tmp_path, mov_video, monkeypatch) -> None:
    # Sem FFmpeg a previa nao e convertida, mas a analise continua funcionando.
    monkeypatch.setattr(browser_video, "ffmpeg_executable", lambda: None)
    converter = BrowserVideoConverter(cache_dir=tmp_path)
    assert converter.prepare(str(mov_video)) == str(mov_video)


def test_cached_name_is_stable_for_the_same_file(tmp_path, mov_video) -> None:
    converter = BrowserVideoConverter(cache_dir=tmp_path)
    assert converter.prepare(str(mov_video)) == converter.prepare(str(mov_video))
