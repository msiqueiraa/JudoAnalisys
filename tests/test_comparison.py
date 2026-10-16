"""Testes da agregacao de estatisticas e da comparacao entre execucoes."""

from __future__ import annotations

from judo_analysis.domain.entities.analysis import AnalysisRun, FrameMetrics, MetricStats
from judo_analysis.domain.entities.video import TimeRange, Video
from judo_analysis.domain.services.pose_comparator import PoseComparator
from judo_analysis.domain.services.statistics import aggregate_run, compute_stats
from judo_analysis.domain.value_objects.joint_angles import JointAngles
from judo_analysis.domain.value_objects.speed import Speed

_VIDEO = Video(path="x.mp4", fps=10, frame_count=100, width=100, height=100)


def _run(label: str, knee: float, speed: float, frames: int = 5) -> AnalysisRun:
    frame_metrics = [
        FrameMetrics(
            frame_index=index,
            timestamp=index * 0.1,
            angles=JointAngles(knee_left=knee, spine=20.0),
            hand_speed=Speed(pixels_per_second=speed, body_heights_per_second=speed / 60),
        )
        for index in range(frames)
    ]
    run = AnalysisRun(
        label=label,
        video=_VIDEO,
        time_range=TimeRange(0.0, 1.0),
        person_id=0,
        frames=frame_metrics,
    )
    return aggregate_run(run)


def test_compute_stats() -> None:
    stats = compute_stats([10.0, 20.0, 30.0])
    assert stats is not None
    assert stats.mean == 20.0
    assert stats.maximum == 30.0
    assert stats.minimum == 10.0


def test_compute_stats_empty_returns_none() -> None:
    assert compute_stats([]) is None


def test_aggregate_sets_key_frame_at_peak_speed() -> None:
    run = _run("analisado", knee=90.0, speed=100.0)
    # Aumenta a velocidade no ultimo frame.
    run.frames[-1].hand_speed = Speed(500.0, 8.3)
    aggregate_run(run)
    assert run.key_frame_index == run.frames[-1].frame_index
    assert run.stats["hand_speed"].maximum == 500.0


def test_identical_runs_have_full_similarity() -> None:
    reference = _run("ref", knee=90.0, speed=200.0)
    analyzed = _run("ana", knee=90.0, speed=200.0)
    result = PoseComparator().compare(reference, analyzed)
    assert result.overall_similarity == 100.0


def test_knee_outside_scale_yields_zero_similarity() -> None:
    reference = _run("ref", knee=90.0, speed=200.0)
    analyzed = _run("ana", knee=180.0, speed=200.0)
    result = PoseComparator().compare(reference, analyzed)
    knee = result.by_key("knee_left")
    assert knee is not None
    assert knee.similarity == 0.0


def test_speed_similarity_uses_peak() -> None:
    reference = _run("ref", knee=90.0, speed=400.0)
    analyzed = _run("ana", knee=90.0, speed=400.0)
    # Pico do analisado fica proximo; a media permanece igual.
    analyzed.frames[-1].hand_speed = Speed(500.0, 8.3)
    aggregate_run(analyzed)
    result = PoseComparator().compare(reference, analyzed)
    speed = result.by_key("hand_speed")
    assert speed is not None
    assert speed.reference_peak == 400.0
    assert speed.analyzed_peak == 500.0
    assert speed.similarity == 75.0


def test_metric_stats_type() -> None:
    stats = compute_stats([1.0])
    assert isinstance(stats, MetricStats)
    assert stats.count == 1


def test_metric_measured_in_too_few_frames_is_excluded() -> None:
    reference = _run("ref", knee=90.0, speed=200.0, frames=20)
    analyzed = _run("ana", knee=90.0, speed=200.0, frames=20)
    # A abertura das pernas so aparece em 2 frames (dados espurios).
    for run in (reference, analyzed):
        for frame in run.frames[:2]:
            frame.angles = JointAngles(knee_left=90.0, spine=20.0, leg_opening=40.0)
        aggregate_run(run)

    result = PoseComparator().compare(reference, analyzed)
    assert "leg_opening" not in result.analyzed_keys()
    assert "leg_opening" in {metric.key for metric in result.excluded_metrics()}


def test_metric_missing_in_one_video_is_excluded() -> None:
    reference = _run("ref", knee=90.0, speed=200.0)
    analyzed = _run("ana", knee=90.0, speed=200.0)
    # Remove o joelho esquerdo do video analisado (nunca foi medido).
    for frame in analyzed.frames:
        frame.angles = JointAngles(knee_left=None, spine=frame.angles.spine)
    aggregate_run(analyzed)

    result = PoseComparator().compare(reference, analyzed)
    assert "knee_left" not in result.analyzed_keys()
    assert "knee_left" in {metric.key for metric in result.excluded_metrics()}
    # O joelho esquerdo nao pode influenciar a similaridade geral.
    assert result.overall_similarity is not None
    assert result.overall_similarity > 0.0


def test_all_metrics_excluded_returns_none_overall() -> None:
    reference = _run("ref", knee=90.0, speed=200.0)
    analyzed = _run("ana", knee=90.0, speed=200.0)
    for frame in analyzed.frames:
        frame.angles = JointAngles()
        frame.hand_speed = None
    aggregate_run(analyzed)

    result = PoseComparator().compare(reference, analyzed)
    assert result.analyzed_metrics() == []
    assert result.overall_similarity is None


def test_ignored_metric_leaves_score_and_is_listed_separately() -> None:
    reference = _run("ref", knee=90.0, speed=100.0)
    analyzed = _run("ana", knee=90.0, speed=900.0)
    result = PoseComparator().compare(reference, analyzed, {"hand_speed"})

    assert "hand_speed" not in result.analyzed_keys()
    assert "hand_speed" not in {metric.key for metric in result.excluded_metrics()}
    assert "hand_speed" in {metric.key for metric in result.ignored_metrics()}
    # A velocidade nao pode derrubar a similaridade geral.
    assert result.overall_similarity == 100.0
