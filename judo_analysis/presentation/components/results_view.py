"""Desenho do resultado comparativo (paineis, comparacao e relatorio).

As abas de video, de foto e de foto x video terminam da mesma forma: dois
paineis lado a lado, a tabela/graficos da comparacao e o botao do PDF. O que
muda em cada aba e o que cada lado tem (video tem rastreamento para conferir,
foto nao; video tem recorte comparativo, foto nao), e isso vem em :class:`SideSpec`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import streamlit as st

from ...application.dto import AnalysisResult
from ...domain.value_objects.comparison import ComparisonResult
from ..analysis import VIDEO_OUTPUTS, correction_callback, exclusion_callback, kimono_hint
from .comparison_clip import render_comparison_clip
from .report import render_report_download
from .results import render_analysis_panel, render_comparison
from .tracking_review import render_tracking_review


@dataclass(slots=True)
class SideSpec:
    """Um lado da comparacao: resultado, chaves de estado e recursos.

    Attributes:
        result: Resultado da analise deste lado.
        panel_key: Prefixo das chaves dos widgets do painel (ex.: ``"ref"``).
        state_key: Chave do resultado na sessao, usada nas reanalises.
        review: Se a faixa de verificacao do rastreamento deve ser desenhada
            (so faz sentido quando o lado veio de um video).
        hint_slot: Chave da dica de cor do kimono usada na analise deste lado.
        ignore_speed: Se a velocidade do golpe sai do painel deste lado.
    """

    result: AnalysisResult
    panel_key: str
    state_key: str = ""
    review: bool = False
    hint_slot: str = ""
    ignore_speed: bool = False


def render_results_view(
    reference: SideSpec,
    analyzed: SideSpec,
    comparison: ComparisonResult,
    *,
    clip: bool = False,
    report_key: str = "report",
    outputs: Sequence[str] = VIDEO_OUTPUTS,
    num_poses: int = 0,
    keep_identity: bool = True,
) -> None:
    """Desenha o resultado completo da comparacao.

    Args:
        clip: Se o recorte comparativo dos dois videos deve ser oferecido
            (so existe quando os dois lados sao videos).
        report_key: Chave do botao/estado do PDF (uma por aba).
        outputs: Prefixos de PDF/recorte a esquecer quando uma reanalise roda.
        num_poses: Pessoas por frame, usada na reanalise de um lado em video.
        keep_identity: Idem, para a reanalise de um lado em video.
    """

    st.divider()
    st.header("Resultado da analise")

    if clip:
        render_comparison_clip(reference.result, analyzed.result, comparison)

    st.divider()
    panels = (reference.panel_key, analyzed.panel_key)
    left, right = st.columns(2)
    with left:
        _render_side(reference, panels, outputs, num_poses, keep_identity)
    with right:
        _render_side(analyzed, panels, outputs, num_poses, keep_identity)

    st.divider()
    st.subheader("Comparacao")
    render_comparison(reference.result.run, analyzed.result.run, comparison)

    st.divider()
    render_report_download(reference.result, analyzed.result, comparison, key=report_key)


def _render_side(
    side: SideSpec,
    panels: Sequence[str],
    outputs: Sequence[str],
    num_poses: int,
    keep_identity: bool,
) -> None:
    render_analysis_panel(side.result, side.panel_key, side.ignore_speed)
    if not side.review:
        return

    different_kimono = kimono_hint(side.hint_slot)
    render_tracking_review(
        side.result,
        num_poses,
        on_correct=correction_callback(
            side.state_key,
            side.result,
            num_poses,
            keep_identity,
            different_kimono,
            panels=panels,
            outputs=outputs,
        ),
        on_exclude=exclusion_callback(
            side.state_key,
            side.result,
            num_poses,
            keep_identity,
            different_kimono,
            panels=panels,
            outputs=outputs,
        ),
        key=f"{side.panel_key}_review",
    )
