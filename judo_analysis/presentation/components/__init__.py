"""Componentes de interface da apresentacao."""

from .comparison_clip import render_comparison_clip
from .mixed_analysis import render_mixed_mode
from .person_selection import render_person_selection
from .photo_analysis import render_photo_mode
from .photo_input import render_photo_input
from .report import render_report_download
from .results import render_analysis_panel, render_comparison
from .results_view import SideSpec, render_results_view
from .tracking_review import render_tracking_review
from .video_input import render_example_loader, render_video_input

__all__ = [
    "SideSpec",
    "render_analysis_panel",
    "render_comparison",
    "render_comparison_clip",
    "render_example_loader",
    "render_mixed_mode",
    "render_person_selection",
    "render_photo_input",
    "render_photo_mode",
    "render_report_download",
    "render_results_view",
    "render_tracking_review",
    "render_video_input",
]
