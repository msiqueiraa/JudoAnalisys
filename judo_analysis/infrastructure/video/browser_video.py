"""Previa do video em um formato que o player do navegador toca.

O ``st.video`` serve o arquivo como ``video/mp4`` e o navegador recusa
containers que nao sejam MP4 (``.mov``, ``.avi``, ``.mkv``): a analise funciona,
mas a previa fica em branco. Aqui a previa e convertida para MP4 com o FFmpeg
empacotado pelo ``imageio-ffmpeg``:

- **remux** (``-c copy``) quando o codec ja e H.264: troca so o container,
  rapido e sem perda;
- **recodificacao** (``libx264``) nos demais casos (HEVC, ProRes, MPEG-4...).

O resultado fica em cache em disco e a **analise continua lendo o arquivo
original** - a conversao serve apenas a previa.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import cv2

from ...config import PREVIEW_DIR
from .ffmpeg import ffmpeg_executable

# Containers que o navegador toca com qualquer codec que ele entenda.
_UNIVERSAL_SUFFIXES = {".webm", ".ogv", ".ogg"}
# Containers MP4 (o que o ``st.video`` anuncia no cabecalho da resposta).
_MP4_SUFFIXES = {".mp4", ".m4v"}
# Codecs que o navegador NAO toca dentro de um MP4 (exigem recodificacao).
_NON_BROWSER_CODECS = {
    "hevc",
    "hvc1",
    "hev1",
    "dvh1",
    "dvhe",  # HEVC / Dolby Vision
    "mp4v",
    "fmp4",
    "mpeg",
    "xvid",
    "divx",  # MPEG-4 Parte 2 / MPEG-2
    "apch",
    "apcn",
    "ap4h",
    "ap4x",
    "prores",  # ProRes
}
# Codecs H.264: basta trocar o container (remux, sem recodificar).
_H264_CODECS = {"h264", "avc1", "avc3", "x264"}
# Altura maxima da previa recodificada (acelera videos grandes).
_MAX_PREVIEW_HEIGHT = 720


def video_codec(path: Path | str) -> str:
    """Codec de video reportado pelo OpenCV (ex.: ``h264``, ``hevc``, ``av01``)."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return ""
        raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    return "".join(chr((raw >> (8 * index)) & 0xFF) for index in range(4)).strip().lower()


def needs_conversion(path: Path | str) -> bool:
    """Indica se o arquivo precisa de conversao para tocar no navegador."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in _UNIVERSAL_SUFFIXES:
        return False
    if suffix in _MP4_SUFFIXES:
        # Dentro de MP4, H.264/AV1/VP9 tocam; HEVC/ProRes/MPEG-4 nao. Codec
        # desconhecido e mantido como esta para nao recodificar sem motivo.
        return video_codec(source) in _NON_BROWSER_CODECS
    # Qualquer outro container (.mov, .avi, .mkv...) nao e MP4.
    return True


class BrowserVideoConverter:
    """Converte um video para MP4 jogavel no navegador (quando necessario)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else PREVIEW_DIR

    def prepare(self, path: str) -> str:
        """Devolve um caminho que o player do navegador toca.

        Devolve o proprio arquivo quando ele ja e compativel ou quando a
        conversao nao for possivel: a previa e opcional, a analise nunca
        depende dela.
        """

        source = Path(path)
        if not source.exists() or not needs_conversion(source):
            return str(source)

        destination = self._cached_path(source)
        if destination.exists():
            return str(destination)

        try:
            self._convert(source, destination)
        except Exception:  # noqa: BLE001 - previa opcional: mantem o original
            return str(source)
        return str(destination)

    def _cached_path(self, source: Path) -> Path:
        """Nome do cache: muda quando o arquivo (ou a estrategia) muda."""

        stat = source.stat()
        mode = "remux" if video_codec(source) in _H264_CODECS else "x264"
        key = hashlib.sha1(
            f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{mode}".encode("utf-8")
        ).hexdigest()[:16]
        return self._cache_dir / f"{_safe_stem(source.stem)}-{key}.mp4"

    def _convert(self, source: Path, destination: Path) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            raise RuntimeError("FFmpeg indisponivel para preparar a previa.")

        destination.parent.mkdir(parents=True, exist_ok=True)

        commands: list[list[str]] = []
        if video_codec(source) in _H264_CODECS:
            commands.append(_remux_command(ffmpeg, source, destination))
        commands.append(_transcode_command(ffmpeg, source, destination))

        last_error = ""
        for command in commands:
            result = subprocess.run(command, capture_output=True)
            if result.returncode == 0 and destination.exists():
                return
            last_error = result.stderr.decode(errors="replace").strip()
            destination.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg falhou ao preparar a previa: {last_error}")


def _remux_command(ffmpeg: str, source: Path, destination: Path) -> list[str]:
    """Troca apenas o container (rapido e sem perda)."""

    return [
        ffmpeg,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _transcode_command(ffmpeg: str, source: Path, destination: Path) -> list[str]:
    """Recodifica em H.264/AAC, limitando a altura da previa."""

    return [
        ffmpeg,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale=-2:'min({_MAX_PREVIEW_HEIGHT},ih)'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return cleaned[:40] or "previa"
