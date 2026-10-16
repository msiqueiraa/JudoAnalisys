"""Adaptadores de imagem (analise de foto)."""

from .photo_source import UploadPhotoSource
from .single_image_reader import SingleImageReader, read_image

__all__ = [
    "SingleImageReader",
    "UploadPhotoSource",
    "read_image",
]
