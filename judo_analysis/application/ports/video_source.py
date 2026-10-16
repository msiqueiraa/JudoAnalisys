"""Porta para obtencao de um video local (upload ou URL)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class IVideoSource(ABC):
    """Contrato que resolve uma origem (upload/URL) em um caminho local."""

    @abstractmethod
    def resolve(self) -> str:
        """Retorna o caminho local do arquivo de video."""
