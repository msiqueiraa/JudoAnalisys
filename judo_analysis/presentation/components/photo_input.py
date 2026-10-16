"""Componente de entrada de foto (upload de imagem)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ...infrastructure.image import UploadPhotoSource, read_image
from ...infrastructure.render import to_rgb

_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "bmp"]


def render_photo_input(label: str, key: str) -> str | None:
    """Renderiza o upload da foto e devolve o caminho local da imagem."""

    path_key = f"{key}_photo_path"
    st.markdown(f"**{label}**")

    uploaded = st.file_uploader(
        "Envie a foto", type=_EXTENSIONS, key=f"{key}_photo_upload"
    )
    if uploaded is not None:
        signature = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get(f"{key}_photo_signature") != signature:
            try:
                with st.spinner("Preparando a foto..."):
                    resolved = UploadPhotoSource(
                        uploaded.name, uploaded.getvalue()
                    ).resolve()
            except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
                st.error(f"Nao foi possivel ler a foto: {error}")
                return None
            st.session_state[path_key] = resolved
            st.session_state[f"{key}_photo_signature"] = signature
            reset_photo_state(key)

    path = st.session_state.get(path_key)
    if not path:
        return None

    st.success("Foto carregada.")
    try:
        st.image(to_rgb(read_image(path)), width="stretch")
    except Exception:  # noqa: BLE001 - a previa e opcional
        st.caption("Previa indisponivel para este arquivo.")
    return path


def reset_photo_state(key: str) -> None:
    """Esquece deteccao, escolha do atleta e resultados de uma foto trocada."""

    for suffix in (
        "_photo_preview",
        "_photo_preview_signature",
        "_person",
        "_click",
        "_last_click",
    ):
        st.session_state.pop(f"{key}{suffix}", None)


def photo_signature(path: str) -> str:
    """Assinatura do arquivo da foto (muda quando a imagem muda)."""

    try:
        stat = Path(path).stat()
    except OSError:
        return path
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
