"""Localizacao do FFmpeg empacotado pelo ``imageio-ffmpeg``.

O projeto nao depende de um FFmpeg instalado no sistema: o binario vem do pacote
``imageio-ffmpeg``. Este modulo centraliza a busca desse binario para os
adaptadores que precisam chamar o FFmpeg (download de URLs e previa do video).
"""

from __future__ import annotations


def ffmpeg_executable() -> str | None:
    """Caminho do executavel do FFmpeg (``None`` quando indisponivel)."""

    try:
        import imageio_ffmpeg
    except ImportError:  # pragma: no cover - depende do ambiente
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - sem ffmpeg empacotado utilizavel
        return None
