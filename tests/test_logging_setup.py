"""Testes do silenciamento dos avisos nativos do MediaPipe/TFLite no stderr."""

from __future__ import annotations

import io
import os
import sys

import pytest

from judo_analysis.logging_setup import native_stderr_suppressed, suppress_native_stderr

_ENV = "JUDO_SUPPRESS_NATIVE_STDERR"


def test_suppression_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert native_stderr_suppressed() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_suppression_can_be_disabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_ENV, value)
    assert native_stderr_suppressed() is False


def test_restores_the_stderr_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    before = os.fstat(2)

    with suppress_native_stderr():
        os.write(2, b"aviso nativo que deve sumir\n")

    after = os.fstat(2)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_disabled_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "0")
    before = os.fstat(2)

    with suppress_native_stderr():
        pass

    after = os.fstat(2)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_reraises_and_dumps_captured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    fake_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    with pytest.raises(RuntimeError):
        with suppress_native_stderr():
            os.write(2, b"detalhe nativo do erro\n")
            raise RuntimeError("falhou de proposito")

    # Nada e escondido: o que foi capturado aparece quando ha erro.
    assert "detalhe nativo do erro" in fake_stderr.getvalue()
