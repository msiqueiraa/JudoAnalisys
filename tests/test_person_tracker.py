"""Testes do rastreador de pessoa (caso critico: atletas se cruzando)."""

from __future__ import annotations

import pytest

from judo_analysis.domain.services.person_tracker import PersonTracker
from tests.conftest import make_pose


def _pose(cx: float, cy: float = 0.5) -> object:
    """Pose sintetica com pelve em ``(cx, cy)`` e altura de corpo 0.4."""

    return make_pose(
        LEFT_HIP=(cx - 0.03, cy),
        RIGHT_HIP=(cx + 0.03, cy),
        LEFT_SHOULDER=(cx - 0.05, cy - 0.2),
        RIGHT_SHOULDER=(cx + 0.05, cy - 0.2),
        LEFT_ANKLE=(cx - 0.02, cy + 0.2),
        RIGHT_ANKLE=(cx + 0.02, cy + 0.2),
    )


def test_selects_the_seeded_person() -> None:
    tracker = PersonTracker(seed=_pose(0.2))
    near = _pose(0.22)
    far = _pose(0.8)
    assert tracker.select([far, near]) is near


def test_rejects_jump_beyond_gate() -> None:
    tracker = PersonTracker(seed=_pose(0.1), max_distance=0.1)
    assert tracker.select([_pose(0.9)]) is None


def test_rejects_candidate_with_very_different_scale() -> None:
    tracker = PersonTracker(seed=_pose(0.5), max_scale_ratio=1.5)
    big = make_pose(
        LEFT_HIP=(0.47, 0.5),
        RIGHT_HIP=(0.53, 0.5),
        LEFT_SHOULDER=(0.45, 0.1),
        RIGHT_SHOULDER=(0.55, 0.1),
        LEFT_ANKLE=(0.48, 0.9),
        RIGHT_ANKLE=(0.52, 0.9),
    )
    assert tracker.select([big]) is None


def test_follows_a_person_moving_with_constant_velocity() -> None:
    tracker = PersonTracker(seed=_pose(0.1))
    for cx in (0.15, 0.2, 0.25, 0.3):
        selected = tracker.select([_pose(cx)])
        assert selected is not None
        assert selected.pelvis[0] == pytest.approx(cx, abs=0.02)


def test_keeps_identity_when_two_athletes_cross() -> None:
    """Regressao: A (executa o golpe) e B (recebe) trocam de lado.

    O rastreador semeado com A deve continuar seguindo A mesmo depois do
    cruzamento, sem trocar para B.
    """

    tracker = PersonTracker(seed=_pose(0.2))
    a_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    b_positions = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    for expected_a, b in zip(a_positions[1:], b_positions[1:]):
        selected = tracker.select([_pose(expected_a), _pose(b)])
        assert selected is not None
        assert selected.pelvis[0] == pytest.approx(expected_a, abs=0.05)


def test_recovers_after_a_missing_frame() -> None:
    tracker = PersonTracker(seed=_pose(0.2))
    # Frame sem a pessoa rastreada.
    assert tracker.select([]) is None
    # Ela reaparece proxima da posicao prevista.
    selected = tracker.select([_pose(0.22)])
    assert selected is not None


def test_keep_identity_rejects_far_single_detection_after_overlap() -> None:
    """Apos haver duas pessoas, uma unica deteccao distante nao troca a identidade."""

    tracker = PersonTracker(
        seed=_pose(0.2), max_distance=0.4, keep_identity=True, strict_factor=0.5
    )
    # Primeiro frame da cena com dois atletas: ativa a exigencia estrita.
    tracker.select([_pose(0.2), _pose(0.8)])
    # O atleta rastreado desaparece e so a outra pessoa (distante) e detectada.
    assert tracker.select([_pose(0.55)]) is None


def test_without_keep_identity_a_far_single_detection_is_adopted() -> None:
    tracker = PersonTracker(seed=_pose(0.2), max_distance=0.4, keep_identity=False)
    tracker.select([_pose(0.2), _pose(0.8)])
    selected = tracker.select([_pose(0.55)])
    assert selected is not None
    assert selected.pelvis[0] == pytest.approx(0.55, abs=0.02)
