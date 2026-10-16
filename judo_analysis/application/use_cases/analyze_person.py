"""Caso de uso: analisar uma pessoa escolhida em um intervalo de tempo.

O rastreamento e feito sobre as deteccoes ja obtidas, o que permite:

- ancorar a identidade da pessoa em um instante escolhido (``anchor_time``);
- rastrear para frente e para tras a partir dessa ancora, corrigindo trocas
  de identidade que ocorram antes ou depois do golpe;
- reexecutar apenas o rastreamento quando o usuario corrige a pessoa.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ...config import PAIR_SEED_SEARCH_FRAMES
from ...domain.entities.analysis import AnalysisRun, FrameMetrics
from ...domain.entities.pose import Pose
from ...domain.entities.video import TimeRange, Video
from ...domain.services.angle_calculator import AngleCalculator
from ...domain.services.pair_tracker import PairTracker
from ...domain.services.person_tracker import PersonTracker
from ...domain.services.speed_calculator import SpeedCalculator
from ...domain.services.statistics import aggregate_run, nearest_frame
from ...domain.value_objects.gi_color import GiColor
from ..dto import AnalysisResult, PersonSelection, TrackingSample
from ..pose_detection import is_measurable_pose
from ..ports.pose_estimator import IPoseEstimator
from ..ports.video_reader import IVideoReader


@dataclass(slots=True)
class _FrameDetections:
    """Poses detectadas em um frame (candidatas ao rastreamento)."""

    frame_index: int
    timestamp: float
    poses: list[Pose]


class AnalyzePersonUseCase:
    """Rastreia a pessoa escolhida e calcula as metricas ao longo do trecho."""

    def __init__(
        self,
        video_reader: IVideoReader,
        pose_estimator: IPoseEstimator,
        angle_calculator: AngleCalculator | None = None,
        sample_count: int = 6,
        keep_identity: bool = True,
        different_kimono: bool = False,
        pair_tracking: bool = True,
    ) -> None:
        self._reader = video_reader
        self._estimator = pose_estimator
        self._angles = angle_calculator or AngleCalculator()
        self._sample_count = sample_count
        self._keep_identity = keep_identity
        # Quando os kimonos tem cores diferentes, a cor entra como pista extra
        # (nunca como unica forma de identidade).
        self._different_kimono = different_kimono
        self._pair_tracking = pair_tracking

    def execute(
        self,
        label: str,
        time_range: TimeRange,
        selection: PersonSelection,
        anchor_time: float | None = None,
        exclude: Sequence[TimeRange] | None = None,
        key_frame_time: float | None = None,
    ) -> AnalysisResult:
        video = self._reader.metadata()
        detections = self._detect(time_range)
        anchor = time_range.start if anchor_time is None else anchor_time
        tracked = self._track(detections, selection.pose, anchor, self._keep_identity)

        # A cor do kimono e usada em uma segunda passagem: a primeira mede, ao
        # longo de todo o trecho, a cor de cada atleta. As duas cores sao
        # deduzidas do conjunto de medidas (o rastreio inicial pode estar errado),
        # nunca de uma unica amostra.
        if self._different_kimono:
            colors = self._color_references(detections, selection.pose)
            if colors is not None:
                tracked = self._track(
                    detections, selection.pose, anchor, self._keep_identity, colors
                )

        excluded = list(exclude or [])
        frames, excluded_count = self._metrics(video, detections, tracked, excluded)
        run = AnalysisRun(
            label=label,
            video=video,
            time_range=time_range,
            person_id=selection.person_id,
            frames=frames,
        )
        if not run.frames:
            raise ValueError("Nenhum frame foi processado no trecho selecionado.")
        aggregate_run(run)
        # O usuario pode ter escolhido o instante do golpe em uma analise
        # anterior (o pico de velocidade nem sempre e o frame que ele quer).
        _apply_key_frame_time(run, key_frame_time)

        key_frame, key_pose = self._read_key_frame(run, tracked)
        samples = self._build_samples(run, tracked)
        return AnalysisResult(
            run=run,
            key_frame=key_frame,
            key_frame_pose=key_pose,
            tracking_samples=samples,
            selection=selection,
            excluded_ranges=excluded,
            excluded_frame_count=excluded_count,
            key_frame_time=key_frame_time,
        )

    def _detect(self, time_range: TimeRange) -> list[_FrameDetections]:
        detections: list[_FrameDetections] = []
        for frame_index, timestamp, image in self._reader.frames(
            time_range.start, time_range.end
        ):
            poses = self._estimator.detect(image, timestamp_ms=int(round(timestamp * 1000)))
            detections.append(
                _FrameDetections(frame_index=frame_index, timestamp=timestamp, poses=poses)
            )
        return detections

    def _track(
        self,
        detections: list[_FrameDetections],
        seed: Pose,
        anchor: float,
        keep_identity: bool = True,
        colors: tuple[GiColor | None, GiColor | None] | None = None,
    ) -> dict[int, Pose]:
        tracked: dict[int, Pose] = {}
        split = next(
            (index for index, det in enumerate(detections) if det.timestamp >= anchor),
            len(detections),
        )

        # Semente do adversario: a pessoa detectada mais distante da escolhida no
        # frame da ancora. Rastrear as duas juntas evita "roubar" a deteccao uma
        # da outra quando os corpos se encostam.
        secondary = self._secondary_seed(detections, split, seed)

        forward = self._make_tracker(seed, secondary, keep_identity, colors)
        for detection in detections[split:]:
            pose = forward.select(detection.poses)
            if pose is not None:
                tracked[detection.frame_index] = pose

        backward = self._make_tracker(seed, secondary, keep_identity, colors)
        for detection in reversed(detections[:split]):
            pose = backward.select(detection.poses)
            if pose is not None:
                tracked[detection.frame_index] = pose

        return tracked

    def _make_tracker(
        self,
        seed: Pose,
        secondary: Pose | None,
        keep_identity: bool,
        colors: tuple[GiColor | None, GiColor | None] | None,
    ) -> PairTracker | PersonTracker:
        reference, other = colors if colors is not None else (None, None)
        if self._pair_tracking and secondary is not None:
            return PairTracker(
                seed, secondary, reference_color=reference, other_color=other
            )
        return PersonTracker(
            seed,
            keep_identity=keep_identity,
            reference_color=reference,
            other_color=other,
        )

    @staticmethod
    def _color_references(
        detections: list[_FrameDetections], seed: Pose
    ) -> tuple[GiColor, GiColor] | None:
        """Cores dos dois kimonos do trecho, identificadas sem depender do rastreio.

        As medidas de todas as poses do trecho sao agrupadas em dois grupos (os
        dois kimonos). O grupo mais proximo da cor da pessoa escolhida pelo
        usuario vira a referencia dela; o outro, a do adversario.

        Depender do rastreamento aqui era um problema circular: quando a
        identidade alternava, as duas medianas saiam misturadas e a pista de cor
        era descartada justamente nos trechos em que ela era mais necessaria.
        """

        mine = seed.gi_color
        if mine is None:
            return None

        clusters = GiColor.cluster_two(
            [pose.gi_color for det in detections for pose in det.poses if pose.gi_color]
        )
        if clusters is None:
            return None
        first, second = clusters
        # Cores parecidas: a cor nao separa os atletas e nao deve ser usada.
        if not first.is_clearly_different_from(second):
            return None
        if mine.distance(first) <= mine.distance(second):
            return first, second
        return second, first

    @staticmethod
    def _secondary_seed(
        detections: list[_FrameDetections], split: int, seed: Pose
    ) -> Pose | None:
        """Pose do adversario no frame da ancora (ou no mais proximo com duas).

        O frame escolhido pelo usuario pode ter apenas uma pessoa detectavel
        (oclusao, duplicata removida). Nesse caso procura-se por perto um frame
        com as duas, ate ``PAIR_SEED_SEARCH_FRAMES`` de distancia.
        """

        position = seed.pelvis or seed.centroid
        if position is None:
            return None

        for index in _nearby_indexes(split, len(detections), PAIR_SEED_SEARCH_FRAMES):
            poses = [pose for pose in detections[index].poses if pose.pelvis is not None]
            if len(poses) >= 2:
                return max(poses, key=lambda pose: math.dist(pose.pelvis, position))
        return None

    def _metrics(
        self,
        video: Video,
        detections: list[_FrameDetections],
        tracked: dict[int, Pose],
        exclude: list[TimeRange],
    ) -> tuple[list[FrameMetrics], int]:
        speed_calculator = SpeedCalculator(video.width, video.height, video.fps)
        frames: list[FrameMetrics] = []
        excluded_count = 0
        for detection in detections:
            pose = tracked.get(detection.frame_index)
            # Atualiza a velocidade sempre (inclusive para resetar a base quando o
            # rastreamento falha), mas so guarda os frames validos.
            speed = speed_calculator.update(pose)
            if _is_excluded(detection.timestamp, exclude):
                excluded_count += 1
                continue
            if not is_measurable_pose(pose):
                continue

            angles = self._angles.compute(pose)
            if speed is None and all(value is None for value in angles.as_dict().values()):
                continue

            frames.append(
                FrameMetrics(
                    frame_index=detection.frame_index,
                    timestamp=detection.timestamp,
                    angles=angles,
                    hand_speed=speed,
                    pose=pose,
                )
            )
        return frames, excluded_count

    def _read_key_frame(self, run: AnalysisRun, tracked: dict[int, Pose]):
        timestamp = _timestamp_of(run, run.key_frame_index)
        _, _, image = self._reader.frame_at(timestamp)
        return image, tracked.get(run.key_frame_index)

    def _build_samples(
        self, run: AnalysisRun, tracked: dict[int, Pose]
    ) -> list[TrackingSample]:
        total = len(run.frames)
        count = min(self._sample_count, total)
        if count == 0:
            return []
        step = max(1, total // count)
        selected = run.frames[::step][:count]
        return [
            TrackingSample(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                pose=tracked.get(frame.frame_index),
            )
            for frame in selected
        ]


def _timestamp_of(run: AnalysisRun, frame_index: int) -> float:
    for frame in run.frames:
        if frame.frame_index == frame_index:
            return frame.timestamp
    return run.frames[0].timestamp if run.frames else run.time_range.start


def _apply_key_frame_time(run: AnalysisRun, key_frame_time: float | None) -> None:
    """Move o frame-chave para o instante escolhido pelo usuario.

    O instante cai entre dois frames: vale o frame valido mais proximo. Um
    instante que nao existe mais (frame descartado depois) tambem e acomodado.
    """

    if key_frame_time is None:
        return
    frame = nearest_frame(run, key_frame_time)
    if frame is not None:
        run.key_frame_index = frame.frame_index


def _nearby_indexes(center: int, count: int, radius: int) -> Iterator[int]:
    """Indices de ``detections`` do centro para fora (centro, +1, -1, +2, ...)."""

    if count <= 0:
        return
    center = min(max(center, 0), count - 1)
    yield center
    for offset in range(1, radius + 1):
        after, before = center + offset, center - offset
        if after < count:
            yield after
        if before >= 0:
            yield before


def _is_excluded(timestamp: float, ranges: Sequence[TimeRange]) -> bool:
    """Indica se o instante cai em algum dos trechos descartados pelo usuario."""

    return any(r.start <= timestamp < r.end for r in ranges)
