"""Testes das constantes de configuracao."""

from __future__ import annotations

from judo_analysis.config import (
    DEFAULT_KEEP_IDENTITY,
    EXAMPLE_ANALYZED_URL,
    EXAMPLE_REFERENCE_URL,
)


def test_example_urls_are_youtube_and_distinct() -> None:
    assert "youtube.com/watch" in EXAMPLE_REFERENCE_URL
    assert "youtube.com/watch" in EXAMPLE_ANALYZED_URL
    assert EXAMPLE_REFERENCE_URL != EXAMPLE_ANALYZED_URL


def test_keep_identity_default_is_enabled() -> None:
    assert DEFAULT_KEEP_IDENTITY is True
