"""Testes da cor do kimono (value object)."""

from __future__ import annotations

from judo_analysis.domain.value_objects.gi_color import GiColor

WHITE = GiColor(l=72.0, a=2.0, b=4.0)
BLUE = GiColor(l=25.0, a=8.0, b=-20.0)
WHITE_SHADOW = GiColor(l=58.0, a=3.0, b=5.0)


def test_distance_is_zero_for_the_same_color() -> None:
    assert WHITE.distance(GiColor(l=72.0, a=2.0, b=4.0)) == 0.0


def test_clearly_different_separates_white_from_blue() -> None:
    assert WHITE.is_clearly_different_from(BLUE)
    assert not WHITE.is_clearly_different_from(WHITE_SHADOW)


def test_name_labels_white_blue_and_dark() -> None:
    assert WHITE.name() == "branco"
    assert BLUE.name() == "azul"
    assert GiColor(l=12.0, a=1.0, b=1.0).name() == "preto"
    assert GiColor(l=50.0, a=30.0, b=10.0).name() == "colorido"


def test_ema_moves_the_reference_towards_the_new_sample() -> None:
    moved = WHITE.ema(BLUE, factor=0.5)
    assert moved.l == 48.5
    assert moved.b == -8.0


def test_median_ignores_a_bad_sample() -> None:
    samples = [WHITE, WHITE, WHITE, GiColor(l=5.0, a=0.0, b=0.0)]
    median = GiColor.median(samples)
    assert median is not None
    assert median.distance(WHITE) == 0.0


def test_median_of_empty_is_none() -> None:
    assert GiColor.median([]) is None


def test_cluster_two_separates_the_two_kimonos() -> None:
    samples = [WHITE, WHITE, WHITE_SHADOW, BLUE, BLUE, GiColor(l=30.0, a=7.0, b=-18.0)]
    clusters = GiColor.cluster_two(samples)
    assert clusters is not None
    first, second = clusters
    assert first.distance(second) > 35.0
    assert min(first.distance(WHITE), second.distance(WHITE)) < 10.0
    assert min(first.distance(BLUE), second.distance(BLUE)) < 10.0


def test_cluster_two_needs_enough_samples() -> None:
    assert GiColor.cluster_two([WHITE, BLUE]) is None


def test_cluster_two_of_a_single_color_is_none() -> None:
    assert GiColor.cluster_two([WHITE] * 8) is None
