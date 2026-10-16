"""Portas de entrada/saida da aplicacao (interfaces)."""

from .clip_composer import IClipComposer
from .gi_color_sampler import IGiColorSampler
from .pose_estimator import IPoseEstimator
from .report_writer import IReportWriter
from .video_reader import IVideoReader
from .video_source import IVideoSource

__all__ = [
    "IClipComposer",
    "IGiColorSampler",
    "IPoseEstimator",
    "IReportWriter",
    "IVideoReader",
    "IVideoSource",
]
