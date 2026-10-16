"""Componentes de interface da apresentacao."""

from .comparison_clip import render_comparison_clip
from .person_selection import render_person_selection
from .report import render_report_download
from .results import render_analysis_panel, render_comparison, reset_key_frame_sliders
from .tracking_review import render_tracking_review
from .video_input import render_example_loader, render_video_input

__all__ = [
    "render_analysis_panel",
    "render_comparison",
    "render_comparison_clip",
    "render_example_loader",
    "render_person_selection",
    "render_report_download",
    "render_tracking_review",
    "render_video_input",
    "reset_key_frame_sliders",
]
