"""Testes do rastreador em par (os dois atletas ao mesmo tempo)."""

from __future__ import annotations

import pytest

from judo_analysis.domain.services.pair_tracker import PairTracker
from judo_analysis.domain.value_objects.gi_color import GiColor
from tests.conftest import make_pose

WHITE = GiColor(l=72.0, a=2.0, b=4.0)
BLUE = GiColor(l=25.0, a=8.0, b=-20.0)


def _pose(cx: float, cy: float = 0.5, color: GiColor | None = None) -> object:
    pose = make_pose(
        LEFT_HIP=(cx - 0.03, cy),
        RIGHT_HIP=(cx + 0.03, cy),
        LEFT_SHOULDER=(cx - 0.05, cy - 0.2),
        RIGHT_SHOULDER=(cx + 0.05, cy - 0.2),
        LEFT_ANKLE=(cx - 0.02, cy + 0.2),
        RIGHT_ANKLE=(cx + 0.02, cy + 0.2),
    )
    pose.gi_color = color
    return pose


def test_returns_the_pose_of_the_primary_person() -> None:
    primary = _pose(0.2)
    tracker = PairTracker(primary, _pose(0.8))
    chosen = tracker.select([_pose(0.8), _pose(0.2)])
    assert chosen is not None
    assert chosen.pelvis[0] == pytest.approx(0.2, abs=0.02)


def test_each_athlete_keeps_its_own_detection_when_they_cross() -> None:
    """No cruzamento, cada deteccao vai para o atleta certo (sem roubo)."""

    tracker = PairTracker(_pose(0.2), _pose(0.8))
    for first, second in ((0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)):
        chosen = tracker.select([_pose(first), _pose(second)])
        assert chosen is not None
        # O atleta principal continua sendo o que veio da esquerda.
        assert chosen.pelvis[0] == pytest.approx(first, abs=0.06)


def test_primary_is_not_stolen_when_the_other_athlete_is_alone() -> None:
    """Se so o adversario aparece, a pessoa principal fica sem pose."""

    tracker = PairTracker(_pose(0.2), _pose(0.8))
    assert tracker.select([_pose(0.8)]) is None
    assert tracker.select([_pose(0.2)]) is not None


def test_color_reference_keeps_identity_when_positions_overlap() -> None:
    """Corrige o caso em que os dois corpos estao praticamente no mesmo lugar."""

    tracker = PairTracker(
        _pose(0.5, color=WHITE), _pose(0.52, color=BLUE), reference_color=WHITE, other_color=BLUE
    )
    # As deteccoes trocam de ordem no quadro, mas as cores seguem certas.
    chosen = tracker.select([_pose(0.5, color=BLUE), _pose(0.52, color=WHITE)])
    assert chosen is not None
    assert chosen.gi_color == WHITE


def test_coast_cost_decides_whether_a_distant_candidate_is_accepted() -> None:
    strict = PairTracker(_pose(0.2), _pose(0.8), coast_cost=0.2)
    permissive = PairTracker(_pose(0.2), _pose(0.8), coast_cost=2.0)
    far = _pose(0.5)
    assert strict.select([far]) is None
    assert permissive.select([far]) is not None


def test_without_a_secondary_it_behaves_like_a_single_tracker() -> None:
    tracker = PairTracker(_pose(0.2))
    assert tracker.secondary_anchor is None
    assert tracker.select([_pose(0.22)]) is not None


def test_missing_frames_do_not_move_the_anchor_wildly() -> None:
    tracker = PairTracker(_pose(0.2), _pose(0.8))
    for _ in range(3):
        tracker.select([])
    anchor = tracker.anchor
    assert anchor is not None
    assert abs(anchor[0] - 0.2) < 0.05
