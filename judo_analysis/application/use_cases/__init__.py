"""Casos de uso da aplicacao."""

from .analyze_person import AnalyzePersonUseCase
from .analyze_photo import AnalyzePhotoUseCase
from .build_comparison_clip import BuildComparisonClipUseCase
from .compare_analyses import CompareAnalysesUseCase
from .detect_people import DetectPeopleUseCase
from .select_key_frame import SelectKeyFrameUseCase

__all__ = [
    "AnalyzePersonUseCase",
    "AnalyzePhotoUseCase",
    "BuildComparisonClipUseCase",
    "CompareAnalysesUseCase",
    "DetectPeopleUseCase",
    "SelectKeyFrameUseCase",
]
