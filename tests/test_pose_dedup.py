"""Testes da deduplicacao de poses do mesmo corpo."""

from __future__ import annotations

from judo_analysis.domain.services.pose_dedup import (
    deduplicate,
    is_same_body,
    relative_landmark_distance,
)
from tests.conftest import make_pose


def _pose(cx: float, cy: float = 0.5, scale: float = 0.2) -> object:
    """Pose sintetica com pelve em ``(cx, cy)`` e altura de corpo 2x ``scale``."""

    return make_pose(
        LEFT_HIP=(cx - scale * 0.15, cy),
        RIGHT_HIP=(cx + scale * 0.15, cy),
        LEFT_SHOULDER=(cx - scale * 0.25, cy - scale),
        RIGHT_SHOULDER=(cx + scale * 0.25, cy - scale),
        LEFT_ANKLE=(cx - scale * 0.1, cy + scale),
        RIGHT_ANKLE=(cx + scale * 0.1, cy + scale),
    )


def test_identical_detections_are_the_same_body() -> None:
    assert is_same_body(_pose(0.5), _pose(0.5))


def test_distant_athletes_are_not_the_same_body() -> None:
    assert not is_same_body(_pose(0.2), _pose(0.8))


def test_nearby_but_distinct_athletes_are_not_merged() -> None:
    """Dois atletas encostados continuam sendo duas pessoas."""

    assert not is_same_body(_pose(0.40), _pose(0.62))


def test_deduplicate_keeps_one_of_the_duplicates() -> None:
    duplicate = _pose(0.5)
    other = _pose(0.8)
    result = deduplicate([duplicate, _pose(0.5), other])
    assert len(result) == 2
    assert [pose.pelvis[0] for pose in result] == [0.5, 0.8]


def test_deduplicate_reindexes_from_left_to_right() -> None:
    result = deduplicate([_pose(0.7), _pose(0.3)])
    assert [pose.person_id for pose in result] == [0, 1]
    assert result[0].pelvis[0] == 0.3


def test_deduplicate_keeps_the_most_complete_detection() -> None:
    complete = make_pose(
        LEFT_HIP=(0.5, 0.5),
        RIGHT_HIP=(0.5, 0.5),
        LEFT_SHOULDER=(0.5, 0.3),
        RIGHT_SHOULDER=(0.5, 0.3),
        LEFT_ANKLE=(0.5, 0.7),
        RIGHT_ANKLE=(0.5, 0.7),
        NOSE=(0.5, 0.25),
    )
    partial = _pose(0.5)
    result = deduplicate([partial, complete])
    assert result == [complete]


def test_anchor_rule_merges_two_poses_on_the_same_spot() -> None:
    """Mesmo com landmarks divergentes, pelves praticamente iguais = um corpo."""

    first = make_pose(
        LEFT_HIP=(0.5, 0.5),
        RIGHT_HIP=(0.5, 0.5),
        LEFT_SHOULDER=(0.5, 0.3),
        RIGHT_SHOULDER=(0.5, 0.3),
        LEFT_ANKLE=(0.5, 0.7),
        RIGHT_ANKLE=(0.5, 0.7),
    )
    second = make_pose(
        LEFT_HIP=(0.51, 0.5),
        RIGHT_HIP=(0.51, 0.5),
        LEFT_SHOULDER=(0.44, 0.3),
        RIGHT_SHOULDER=(0.58, 0.3),
        LEFT_ANKLE=(0.45, 0.7),
        RIGHT_ANKLE=(0.57, 0.7),
    )
    assert is_same_body(first, second)


def test_relative_landmark_distance_is_none_without_common_landmarks() -> None:
    first = make_pose(LEFT_HIP=(0.5, 0.5))
    second = make_pose(RIGHT_HIP=(0.5, 0.5))
    assert relative_landmark_distance(first, second) is None


def test_relative_landmark_distance_is_none_without_scale() -> None:
    first = make_pose(NOSE=(0.5, 0.2))
    second = make_pose(NOSE=(0.5, 0.2))
    assert relative_landmark_distance(first, second) is None


def test_relative_landmark_distance_grows_with_separation() -> None:
    near = relative_landmark_distance(_pose(0.40), _pose(0.42))
    far = relative_landmark_distance(_pose(0.40), _pose(0.60))
    assert near is not None and far is not None
    assert near < far
