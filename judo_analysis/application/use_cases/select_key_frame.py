"""Caso de uso: escolher o frame-chave do golpe (automatico ou manual)."""

from __future__ import annotations

from ...domain.services.statistics import nearest_frame, peak_speed_frame
from ..dto import AnalysisResult
from ..ports.video_reader import IVideoReader


class SelectKeyFrameUseCase:
    """Troca o frame-chave usado no painel e no relatorio.

    O frame-chave padrao e o de maior velocidade da mao (o "golpe"), mas o pico
    de velocidade nem sempre cai no instante que o usuario considera o golpe.
    Com um instante informado, vale o frame valido mais proximo dele.
    """

    def __init__(self, video_reader: IVideoReader) -> None:
        self._reader = video_reader

    def execute(
        self, result: AnalysisResult, timestamp: float | None = None
    ) -> AnalysisResult:
        """Define o frame-chave; ``timestamp`` ``None`` volta ao automatico."""

        run = result.run
        frame = (
            peak_speed_frame(run) if timestamp is None else nearest_frame(run, timestamp)
        )
        if frame is None:
            return result

        run.key_frame_index = frame.frame_index
        result.key_frame_time = timestamp
        # A imagem do frame-chave nao fica guardada na analise (so as metricas):
        # ela e relida do video no instante escolhido.
        _, _, image = self._reader.frame_at(frame.timestamp)
        result.key_frame = image
        result.key_frame_pose = frame.pose
        return result
