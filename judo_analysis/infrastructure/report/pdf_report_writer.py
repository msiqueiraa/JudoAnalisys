"""Geracao do relatorio em PDF com ReportLab."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ...application.dto import AnalysisResult  # noqa: E402
from ...application.ports.report_writer import IReportWriter  # noqa: E402
from ...domain.entities.analysis import AnalysisRun  # noqa: E402
from ...domain.metrics_catalog import METRICS_BY_KEY, TIME_SERIES_METRICS  # noqa: E402
from ...domain.services.statistics import frame_by_index  # noqa: E402
from ...domain.value_objects.comparison import ComparisonResult  # noqa: E402
from ...formatting import format_duration_precise  # noqa: E402
from ..render.frame_annotator import FrameAnnotator  # noqa: E402

_CHART_WIDTH_CM = 16.0
_CHART_FIGSIZE = (10.0, 3.0)  # Largura x altura da figura (polegadas).


class PdfReportWriter(IReportWriter):
    """Monta um PDF com a comparacao das metricas e o frame-chave anotado."""

    def __init__(self) -> None:
        self._annotator = FrameAnnotator()

    def write(
        self,
        reference: tuple[AnalysisRun, AnalysisResult],
        analyzed: tuple[AnalysisRun, AnalysisResult],
        comparison: ComparisonResult,
        output_path: Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        document = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            title="Relatorio de Analise de Judo",
        )

        story: list = [Paragraph("Relatorio de Analise de Judo", styles["Title"])]
        story.append(Spacer(1, 0.4 * cm))
        story.append(self._metadata_table(reference[0], analyzed[0]))
        story.append(Spacer(1, 0.4 * cm))

        if comparison.overall_similarity is not None:
            story.append(
                Paragraph(
                    f"Similaridade geral: <b>{comparison.overall_similarity:.1f}%</b>",
                    styles["Heading2"],
                )
            )
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Metricas comparadas", styles["Heading2"]))
        story.append(self._metrics_table(comparison))
        story.extend(self._excluded_note(comparison, styles))
        story.extend(self._ignored_note(comparison, styles))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("Frame-chave do golpe", styles["Heading2"]))
        story.append(self._frames_row(reference, analyzed))
        story.append(Spacer(1, 0.5 * cm))

        chart_paths = self._build_charts(reference[0], analyzed[0], comparison)
        if chart_paths:
            story.append(Paragraph("Evolucao das metricas", styles["Heading2"]))
            story.append(Spacer(1, 0.2 * cm))
            height_cm = _CHART_WIDTH_CM * _CHART_FIGSIZE[1] / _CHART_FIGSIZE[0]
            for path in chart_paths:
                story.append(Image(str(path), width=_CHART_WIDTH_CM * cm, height=height_cm * cm))
                story.append(Spacer(1, 0.25 * cm))

        document.build(story)
        return output_path

    def _metadata_table(self, reference: AnalysisRun, analyzed: AnalysisRun):
        rows = [
            ["", "Referencia", "Analisado"],
            ["Pessoa", str(reference.person_id + 1), str(analyzed.person_id + 1)],
            [
                "Trecho",
                _trecho(reference),
                _trecho(analyzed),
            ],
            ["FPS", f"{reference.video.fps:.1f}", f"{analyzed.video.fps:.1f}"],
        ]
        table = Table(rows, colWidths=[3 * cm, 7 * cm, 7 * cm])
        table.setStyle(_table_style())
        return table

    def _metrics_table(self, comparison: ComparisonResult):
        header = [
            "Metrica",
            "Media ref.",
            "Media analisado",
            "Pico ref.",
            "Pico analisado",
            "Similaridade",
        ]
        rows = [header]
        for metric in comparison.analyzed_metrics():
            unit = metric.unit
            rows.append(
                [
                    metric.label,
                    _fmt(metric.reference_mean, unit),
                    _fmt(metric.analyzed_mean, unit),
                    _fmt(metric.reference_peak, unit),
                    _fmt(metric.analyzed_peak, unit),
                    f"{metric.similarity:.1f}%" if metric.similarity is not None else "--",
                ]
            )
        if len(rows) == 1:
            rows.append(["Nenhuma metrica pode ser comparada", "", "", "", "", ""])
        table = Table(rows, repeatRows=1)
        table.setStyle(_table_style())
        return table

    def _excluded_note(self, comparison: ComparisonResult, styles):
        excluded = comparison.excluded_metrics()
        if not excluded:
            return []
        names = ", ".join(metric.label for metric in excluded)
        return [
            Spacer(1, 0.2 * cm),
            Paragraph(
                f"<i>Nao analisadas (dados insuficientes em um dos videos): {names}.</i>",
                styles["BodyText"],
            ),
        ]

    def _ignored_note(self, comparison: ComparisonResult, styles):
        ignored = comparison.ignored_metrics()
        if not ignored:
            return []
        names = ", ".join(metric.label for metric in ignored)
        return [
            Spacer(1, 0.2 * cm),
            Paragraph(
                f"<i>Ignoradas nesta comparacao: {names}.</i>",
                styles["BodyText"],
            ),
        ]

    def _frames_row(self, reference, analyzed):
        images = []
        for run, result in (reference, analyzed):
            frame = self._annotate_key_frame(run, result)
            images.append(Image(str(frame), width=8 * cm, height=6 * cm))

        table = Table([images], colWidths=[8.5 * cm, 8.5 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return table

    def _annotate_key_frame(self, run: AnalysisRun, result: AnalysisResult) -> Path:
        metrics = frame_by_index(run, run.key_frame_index)
        angles = metrics.angles if metrics else None
        speed = metrics.hand_speed if metrics else None
        annotated = self._annotator.annotate(
            result.key_frame, result.key_frame_pose, angles, speed, title=run.label
        )
        path = Path(tempfile.mkdtemp()) / f"key_{run.label}.png"
        cv2.imwrite(str(path), annotated)
        return path

    def _build_charts(
        self,
        reference: AnalysisRun,
        analyzed: AnalysisRun,
        comparison: ComparisonResult,
    ) -> list[Path]:
        analyzed_keys = comparison.analyzed_keys()
        keys = [key for key in TIME_SERIES_METRICS if key in analyzed_keys]

        paths: list[Path] = []
        for key in keys:
            path = self._build_chart(reference, analyzed, key)
            if path is not None:
                paths.append(path)
        return paths

    def _build_chart(
        self, reference: AnalysisRun, analyzed: AnalysisRun, key: str
    ) -> Path | None:
        meta = METRICS_BY_KEY[key]
        figure, axis = plt.subplots(figsize=_CHART_FIGSIZE)
        has_data = False
        for run, color, name in (
            (reference, "#1f77b4", "Referencia"),
            (analyzed, "#d62728", "Analisado"),
        ):
            points = run.series_with_time(key)
            if not points:
                continue
            has_data = True
            times = [timestamp - run.time_range.start for timestamp, _ in points]
            values = [value for _, value in points]
            axis.plot(times, values, color=color, label=name)

        if not has_data:
            plt.close(figure)
            return None

        axis.set_title(meta.label)
        axis.set_xlabel("Tempo no trecho (s)")
        axis.set_ylabel(meta.unit)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
        figure.tight_layout()

        path = Path(tempfile.mkdtemp()) / f"chart_{key}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        return path


def _trecho(run: AnalysisRun) -> str:
    return (
        f"{format_duration_precise(run.time_range.start)} - "
        f"{format_duration_precise(run.time_range.end)}"
    )


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "--"
    if unit == "deg":
        return f"{value:.1f} deg"
    return f"{value:.0f} {unit}"


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
    )
