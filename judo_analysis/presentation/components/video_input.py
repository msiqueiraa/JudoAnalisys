"""Componente de entrada de video (upload ou URL)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ...config import EXAMPLE_ANALYZED_URL, EXAMPLE_REFERENCE_URL
from ...infrastructure.video import BrowserVideoConverter, UploadVideoSource, UrlVideoSource

_EXAMPLE_URLS = {
    "ref": EXAMPLE_REFERENCE_URL,
    "ana": EXAMPLE_ANALYZED_URL,
}


def render_video_input(label: str, key: str) -> str | None:
    """Renderiza os controles de entrada e devolve o caminho local do video."""

    path_key = f"{key}_path"
    st.markdown(f"**{label}**")

    url_key = f"{key}_url"
    if url_key not in st.session_state:
        st.session_state[url_key] = _EXAMPLE_URLS.get(key, "")

    mode = st.radio(
        "Origem",
        ["Upload", "URL"],
        key=f"{key}_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Upload":
        uploaded = st.file_uploader(
            "Envie o video",
            type=["mp4", "avi", "mov", "mkv"],
            key=f"{key}_upload",
        )
        if uploaded is not None:
            signature = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get(f"{key}_signature") != signature:
                with st.spinner("Preparando video..."):
                    resolved = UploadVideoSource(uploaded.name, uploaded.getvalue()).resolve()
                st.session_state[path_key] = resolved
                st.session_state[f"{key}_signature"] = signature
                _reset_downstream(key)
    else:
        url = st.text_input(
            "URL do video (YouTube, Instagram, TikTok ou .mp4/.mov)", key=url_key
        )
        if st.button("Carregar URL", key=f"{key}_load"):
            if not url.strip():
                st.warning("Informe uma URL valida.")
            else:
                try:
                    with st.spinner("Baixando video..."):
                        resolved = UrlVideoSource(url).resolve()
                    st.session_state[path_key] = resolved
                    _reset_downstream(key)
                except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
                    st.error(f"Falha ao carregar a URL: {error}")

    path = st.session_state.get(path_key)
    if path:
        st.success("Video carregado.")
        preview = _preview_source(path)
        if preview != path:
            st.caption(
                "Previa convertida para MP4 para tocar no navegador; a analise "
                "usa o arquivo original."
            )
        try:
            st.video(preview)
        except Exception:  # noqa: BLE001
            st.caption("Previa indisponivel para este formato.")
    return path


def _preview_source(path: str) -> str:
    """Previa tocavel no navegador, invalidada quando o arquivo muda."""

    try:
        stat = Path(path).stat()
        signature = f"{stat.st_size}-{stat.st_mtime_ns}"
    except OSError:
        signature = "0"
    return _convert_for_preview(path, signature)


@st.cache_data(show_spinner=False)
def _convert_for_preview(path: str, signature: str) -> str:
    """Converte .mov/.avi/.mkv (e codecs sem suporte) para a previa em MP4."""

    with st.spinner("Preparando a previa do video..."):
        return BrowserVideoConverter().prepare(path)


def render_example_loader() -> None:
    """Botao (na barra lateral) que baixa e carrega os dois videos de exemplo."""

    if not st.sidebar.button("Carregar videos de exemplo"):
        return

    try:
        with st.spinner("Baixando os videos de exemplo (pode levar um instante)..."):
            reference = UrlVideoSource(EXAMPLE_REFERENCE_URL).resolve()
            analyzed = UrlVideoSource(EXAMPLE_ANALYZED_URL).resolve()
    except Exception as error:  # noqa: BLE001 - exibimos o erro ao usuario
        st.sidebar.error(f"Falha ao baixar os exemplos: {error}")
        return

    for key, url, path in (
        ("ref", EXAMPLE_REFERENCE_URL, reference),
        ("ana", EXAMPLE_ANALYZED_URL, analyzed),
    ):
        st.session_state[f"{key}_mode"] = "URL"
        st.session_state[f"{key}_url"] = url
        st.session_state[f"{key}_path"] = path
        _reset_downstream(key)
    st.rerun()


def _reset_downstream(key: str) -> None:
    for suffix in ("_preview", "_selection", "_result"):
        st.session_state.pop(f"{key}{suffix}", None)
