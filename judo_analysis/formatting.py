"""Utilitarios de formatacao compartilhados entre apresentacao e relatorio."""

from __future__ import annotations

import datetime as dt

_MAX_TIME_SECONDS = 23 * 3600 + 59 * 60 + 59


def format_duration(seconds: float) -> str:
    """Formata uma duracao em ``mm:ss`` (ou ``h:mm:ss`` se passar de 1 hora)."""

    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration_precise(seconds: float, decimals: int = 2) -> str:
    """Como :func:`format_duration`, mas preserva os decimais quando existem.

    Usado para exibir trechos que nao caem em segundos inteiros (ex.: as caixas
    de selecao do trecho e a faixa de verificacao) sem arredondar na exibicao.
    O padrao de duas casas casa com ``00:01.55``; valores inteiros continuam
    saindo limpos (``00:02``).
    """

    if abs(seconds - round(seconds)) < 10 ** (-decimals) / 2:
        return format_duration(seconds)

    total = max(0.0, seconds)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs_text = f"{total % 60:0{3 + decimals}.{decimals}f}"
    if hours:
        return f"{hours}:{minutes:02d}:{secs_text}"
    return f"{minutes:02d}:{secs_text}"


def parse_time(text: str) -> float | None:
    """Interpreta ``mm:ss`` (ou ``h:mm:ss``, com centesimos, ou segundos puros).

    Aceita ``00:01.55`` e ``1:02.5``; retorna ``None`` quando o texto nao puder
    ser interpretado.
    """

    cleaned = text.strip()
    if not cleaned:
        return None

    parts = cleaned.split(":")
    if len(parts) > 3:
        return None

    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None

    if any(number < 0 for number in numbers):
        return None

    if len(parts) == 1:
        return numbers[0]

    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def slider_time_format(duration: float, precise: bool = False) -> str:
    """Formato (padrao momentJS) do slider conforme a duracao do video.

    Com ``precise=True`` acrescenta os centesimos (``mm:ss.SS``) - usado no
    slider que escolhe o frame-chave do golpe.
    """

    suffix = ".SS" if precise else ""
    return f"h:mm:ss{suffix}" if duration >= 3600 else f"mm:ss{suffix}"


def seconds_to_time(seconds: float) -> dt.time:
    """Converte segundos em ``datetime.time`` para uso em sliders.

    Preserva os microssegundos: as caixas de selecao aceitam centesimos
    (ex.: ``00:01.55``) e o valor nao pode ser arredondado para o segundo
    inteiro ao passar pelo slider.
    """

    micros = int(round(max(0.0, min(float(seconds), float(_MAX_TIME_SECONDS))) * 1_000_000))
    micros = min(micros, _MAX_TIME_SECONDS * 1_000_000)
    hours, remainder = divmod(micros, 3_600_000_000)
    minutes, remainder = divmod(remainder, 60_000_000)
    secs, micros_rest = divmod(remainder, 1_000_000)
    return dt.time(hours, minutes, secs, micros_rest)


def time_to_seconds(value: dt.time) -> float:
    """Converte ``datetime.time`` de volta para segundos."""

    return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
