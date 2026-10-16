"""Adaptadores de video (leitura e obtencao de arquivos)."""

from .browser_video import BrowserVideoConverter, needs_conversion, video_codec
from .ffmpeg import ffmpeg_executable
from .ffmpeg_clip_composer import FfmpegClipComposer
from .opencv_video_reader import OpenCVVideoReader
from .upload_video_source import UploadVideoSource
from .url_video_source import UrlVideoSource

__all__ = [
    "BrowserVideoConverter",
    "FfmpegClipComposer",
    "OpenCVVideoReader",
    "UploadVideoSource",
    "UrlVideoSource",
    "ffmpeg_executable",
    "needs_conversion",
    "video_codec",
]
