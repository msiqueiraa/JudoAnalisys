"""Configuracao de logging para reduzir o ruido das bibliotecas nativas.

O MediaPipe/TensorFlow Lite emitem muitos avisos internos (XNNPACK,
``inference_feedback_manager``, ``NORM_RECT``...) que nao indicam problema na
aplicacao.

Esses avisos vem do C++ e sao escritos **direto no descritor stderr (fd 2)**,
fora do ``logging`` do Python — por isso nao basta configurar o logger. A
estrategia e dupla:

1. :func:`configure_logging` ajusta variaveis de ambiente e o ``absl`` (reduz
   parte do ruido); deve ser chamada **antes** de importar o MediaPipe.
2. :func:`suppress_native_stderr` redireciona o fd 2 para um arquivo temporario
   durante as chamadas nativas (criacao do landmarker e deteccao), capturando o
   que sobrar. Em caso de excecao o conteudo capturado e reimpresso, para nao
   esconder erros reais.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
from collections.abc import Iterator

# Desligue com ``JUDO_SUPPRESS_NATIVE_STDERR=0`` para depurar os avisos nativos.
_SUPPRESS_ENV = "JUDO_SUPPRESS_NATIVE_STDERR"
_FALSE_VALUES = {"0", "false", "no", "off"}

_native_lock = threading.Lock()
_native_sink: "tempfile._TemporaryFileWrapper | None" = None


def configure_logging() -> None:
    """Silencia avisos internos do TensorFlow Lite, glog e absl."""

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:  # pragma: no cover - absl deve existir via mediapipe
        pass


def native_stderr_suppressed() -> bool:
    """Indica se os avisos nativos no stderr devem ser capturados."""

    return os.environ.get(_SUPPRESS_ENV, "1").strip().lower() not in _FALSE_VALUES


@contextlib.contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """Captura o stderr nativo (fd 2) durante chamadas ruidosas do MediaPipe.

    E seguro para uso concorrente (as chamadas nativas sao serializadas por um
    lock) e restaura o descritor original sempre. Se uma excecao escapar do
    bloco, o que foi capturado e reimpresso em ``sys.stderr``.
    """

    if not native_stderr_suppressed():
        yield
        return

    with _native_lock:
        _flush_stderr()
        try:
            saved_fd = os.dup(2)
        except OSError:  # pragma: no cover - stderr indisponivel (ex.: pythonw)
            yield
            return

        sink = _sink()
        os.dup2(sink.fileno(), 2)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            _flush_stderr()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            if failed:
                sink.seek(0)
                captured = sink.read().decode("utf-8", "replace")
                if captured.strip():
                    sys.stderr.write(captured)


def _sink() -> "tempfile._TemporaryFileWrapper":
    """Arquivo temporario reutilizado entre chamadas (ja truncado)."""

    global _native_sink
    if _native_sink is None or _native_sink.closed:
        _native_sink = tempfile.TemporaryFile()
    _native_sink.seek(0)
    _native_sink.truncate()
    return _native_sink


def _flush_stderr() -> None:
    try:
        if sys.stderr is not None:
            sys.stderr.flush()
    except Exception:  # pragma: no cover - stderr pode estar fechado
        pass
