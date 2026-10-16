"""Judo Analysis - analise biomecanica de videos de judo.

Pacote organizado em Clean Architecture:

- ``domain``: regras de negocio puras (entidades, value objects, servicos).
- ``application``: casos de uso e portas (interfaces) que orquestram o dominio.
- ``infrastructure``: adaptadores concretos (MediaPipe, OpenCV, yt-dlp, ReportLab).
- ``presentation``: interface Streamlit.
"""

__version__ = "1.0.0"
