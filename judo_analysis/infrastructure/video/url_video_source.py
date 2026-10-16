"""Fonte de video a partir de uma URL (YouTube, Instagram, TikTok, etc.)."""

from __future__ import annotations

from pathlib import Path

from ...application.ports.video_source import IVideoSource
from ...config import TEMP_DIR
from .ffmpeg import ffmpeg_executable

# Formatos progressivos (video+audio no mesmo arquivo): evitam a etapa de merge,
# que exigiria ``ffprobe`` instalado no sistema.
_FORMAT_SELECTOR = (
    "22/18/b[ext=mp4][vcodec!=none][acodec!=none]/b[ext=mp4]/bv*[ext=mp4]+ba[ext=m4a]/b"
)


class UrlVideoSource(IVideoSource):
    """Baixa o video da URL com ``yt-dlp`` e devolve o caminho local.

    O ``yt-dlp`` tambem resolve URLs diretas de arquivos ``.mp4`` ou ``.mov``.
    """

    def __init__(self, url: str) -> None:
        self._url = url.strip()

    def resolve(self) -> str:
        if not self._url:
            raise ValueError("Informe uma URL de video valida.")

        try:
            import yt_dlp
        except ImportError as error:  # pragma: no cover - depende do ambiente
            raise RuntimeError(
                "A dependencia 'yt-dlp' nao esta instalada. "
                "Instale com 'pip install yt-dlp' para usar URLs."
            ) from error

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_template = str(TEMP_DIR / "%(id)s.%(ext)s")

        options = {
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": _FORMAT_SELECTOR,
            "merge_output_format": "mp4",
        }
        ffmpeg_location = ffmpeg_executable()
        if ffmpeg_location:
            options["ffmpeg_location"] = ffmpeg_location

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(self._url, download=True)
            path = Path(ydl.prepare_filename(info))

        if not path.exists():
            # O merge pode gerar outro container; procura pelo id.
            matches = list(TEMP_DIR.glob(f"{info.get('id', '*')}.*"))
            if not matches:
                raise IOError("Falha ao baixar o video da URL informada.")
            path = matches[0]

        return str(path)
