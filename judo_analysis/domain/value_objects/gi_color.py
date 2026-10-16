"""Assinatura de cor do kimono (value object).

A cor do kimono e um sinal de identidade util no judo: em competicao um atleta
veste branco e o outro azul. Diferente da posicao, ela nao muda quando os dois
atletas se encostam, o que ajuda exatamente nos momentos em que o rastreamento
por posicao erra.

A cor e guardada no espaco **LAB** (canal ``L`` de 0 a 100 e canais ``a``/``b``
de -128 a 127), que aproxima a percepcao humana e separa bem o branco do azul
mesmo com variacao de iluminacao.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from ...config import GI_COLOR_DIFFERENT_DELTA_E, GI_COLOR_SAME_DELTA_E

# Abaixo deste croma a cor e praticamente acromatica (branco/cinza).
_ACHROMATIC_CHROMA = 18.0
# Brilho minimo para uma cor acromatica ser chamada de branca.
_WHITE_LIGHTNESS = 60.0
# Amostras minimas para tentar separar as cores dos dois kimonos.
_MIN_CLUSTER_SAMPLES = 6
# Iteracoes do agrupamento de cores (converge bem antes disso).
_CLUSTER_ITERATIONS = 12


@dataclass(frozen=True, slots=True)
class GiColor:
    """Cor media do kimono de uma pessoa, no espaco LAB.

    Attributes:
        l: Luminosidade (0 = preto, 100 = branco).
        a: Eixo verde (-) / vermelho (+).
        b: Eixo azul (-) / amarelo (+).
    """

    l: float
    a: float
    b: float

    @property
    def chroma(self) -> float:
        """Saturacao da cor: 0 em tons de cinza, alta em cores vivas."""

        return math.hypot(self.a, self.b)

    def distance(self, other: "GiColor") -> float:
        """Distancia perceptual entre duas cores (Delta E, CIE76)."""

        return math.dist((self.l, self.a, self.b), (other.l, other.a, other.b))

    def is_same_as(self, other: "GiColor", threshold: float | None = None) -> bool:
        """Indica se as duas medidas descrevem a mesma cor (mesmo kimono)."""

        limit = GI_COLOR_SAME_DELTA_E if threshold is None else threshold
        return self.distance(other) <= limit

    def is_clearly_different_from(
        self, other: "GiColor", threshold: float | None = None
    ) -> bool:
        """Indica se as cores sao diferentes o bastante para serem de outra pessoa."""

        limit = GI_COLOR_DIFFERENT_DELTA_E if threshold is None else threshold
        return self.distance(other) > limit

    def name(self) -> str:
        """Rotulo curto para a interface (ex.: ``branco``, ``azul``)."""

        if self.chroma < _ACHROMATIC_CHROMA:
            return "branco" if self.l >= _WHITE_LIGHTNESS else "preto"
        if self.b <= -8.0:
            return "azul"
        return "colorido"

    def ema(self, other: "GiColor", factor: float) -> "GiColor":
        """Media exponencial (aproxima esta referencia da cor recem-medida)."""

        weight = min(max(factor, 0.0), 1.0)
        return GiColor(
            l=(1 - weight) * self.l + weight * other.l,
            a=(1 - weight) * self.a + weight * other.a,
            b=(1 - weight) * self.b + weight * other.b,
        )

    @staticmethod
    def median(colors: "list[GiColor] | tuple[GiColor, ...]") -> "GiColor | None":
        """Cor mediana de uma serie de medidas (robusta a amostras ruins).

        O rastreamento de um trecho inteiro rende centenas de medidas; a mediana
        descreve o kimono muito melhor do que uma unica amostra.
        """

        if not colors:
            return None
        return GiColor(
            l=statistics.median(color.l for color in colors),
            a=statistics.median(color.a for color in colors),
            b=statistics.median(color.b for color in colors),
        )

    @classmethod
    def cluster_two(
        cls, colors: "list[GiColor] | tuple[GiColor, ...]"
    ) -> "tuple[GiColor, GiColor] | None":
        """Separa as cores medidas nos dois kimonos presentes no trecho.

        Agrupa as medidas em dois grupos (k-medias) usando a mediana de cada
        grupo, que e robusta a amostras ruins do fundo. A inicializacao usa as
        cores de luminosidade extrema (kimono claro x escuro), o que torna o
        resultado reproduzivel e costuma cair exatamente nos dois atletas.

        Devolve ``None`` quando ha amostras de menos ou quando todos os grupos
        ficam vazios.

        Diferente de :meth:`median`, este metodo NAO depende do rastreamento
        estar certo: as duas cores sao deduzidas do conjunto todo de medidas.
        """

        samples = [color for color in colors if color is not None]
        if len(samples) < _MIN_CLUSTER_SAMPLES:
            return None

        ordered = sorted(samples, key=lambda color: color.l)
        centers = [ordered[0], ordered[-1]]
        if centers[0].is_same_as(centers[1], threshold=1e-6):
            return None

        for _ in range(_CLUSTER_ITERATIONS):
            groups: tuple[list[GiColor], list[GiColor]] = ([], [])
            for color in samples:
                nearest = (
                    0
                    if color.distance(centers[0]) <= color.distance(centers[1])
                    else 1
                )
                groups[nearest].append(color)
            updated: list[GiColor] = []
            for index, group in enumerate(groups):
                median = cls.median(group)
                updated.append(median if median is not None else centers[index])
            if updated == centers:
                break
            centers = updated

        return centers[0], centers[1]
