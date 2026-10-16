"""Selecao da pessoa em uma imagem: lista + clique opcional sobre o atleta.

O componente ``streamlit-image-coordinates`` e opcional: sem ele, a selecao
continua disponivel pela lista (``selectbox``).
"""

from __future__ import annotations

import streamlit as st

from ...domain.entities.pose import Pose
from ...infrastructure.render import FrameAnnotator, to_rgb

# Largura (px) da imagem clicavel. Precisa ser um inteiro para que as
# coordenadas do clique se relacionem de forma previsivel com a imagem.
SELECT_WIDTH = 560

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    _HAS_CLICK = True
except Exception:  # noqa: BLE001 - ambiente sem a dependencia
    streamlit_image_coordinates = None  # type: ignore[assignment]
    _HAS_CLICK = False


def render_pose_selector(
    frame,
    poses: list[Pose],
    key: str,
    caption: str | None = None,
) -> int:
    """Desenha as poses numeradas e devolve o indice da pessoa escolhida.

    Se o componente de clique estiver disponivel, um clique sobre o atleta
    atualiza a escolha (guardada em ``st.session_state[f"{key}_person"]``) antes
    de a lista ser criada, de modo que o ``selectbox`` do chamador reflita o
    clique. Sem o componente, apenas a imagem e exibida.
    """

    person_key = f"{key}_person"
    current = st.session_state.get(person_key)
    if not isinstance(current, int) or not 0 <= current < len(poses):
        current = 0

    annotated = FrameAnnotator().annotate_selection(frame, poses, selected_id=current)

    if not _HAS_CLICK:
        st.image(to_rgb(annotated), width="stretch")
        st.session_state[person_key] = current
        return current

    st.caption(caption or "Clique sobre o atleta desejado (ou use a lista).")
    value = streamlit_image_coordinates(
        to_rgb(annotated), width=SELECT_WIDTH, key=f"{key}_click"
    )
    if is_new_click(key, value):
        width = value.get("width") or SELECT_WIDTH
        height = value.get("height") or SELECT_WIDTH
        current = nearest_pose(poses, value["x"] / float(width), value["y"] / float(height))

    # Define a escolha antes de o ``selectbox`` do chamador ser criado, para que
    # o clique se reflita na lista e no destaque da imagem.
    st.session_state[person_key] = current
    return current


def pose_label(index: int, pose: Pose) -> str:
    """Rotulo da pessoa na lista de selecao, com a cor do kimono quando medida."""

    if pose.gi_color is None:
        return f"Pessoa {index + 1}"
    return f"Pessoa {index + 1} ({pose.gi_color.name()})"


def click_enabled() -> bool:
    """Indica se o clique sobre a imagem esta disponivel neste ambiente."""

    return _HAS_CLICK


def render_clickable_image(image, key: str):
    """Desenha uma imagem clicavel e devolve o ultimo clique (ou ``None``).

    O retorno e o dicionario do componente (com ``unix_time``), o mesmo tratado
    por :func:`is_new_click`; sem o componente instalado apenas a imagem e
    desenhada e o retorno e ``None``. A legenda fica a cargo de quem chama, para
    poder refletir o clique no mesmo rerun. A largura acompanha a coluna.
    """

    if not _HAS_CLICK:
        st.image(to_rgb(image), width="stretch")
        return None
    return streamlit_image_coordinates(to_rgb(image), width="stretch", key=key)


def nearest_pose(poses: list[Pose], x: float, y: float) -> int:
    """Indice da pose cujo centro esta mais proximo do ponto clicado (0..1)."""

    best_index = 0
    best_distance = float("inf")
    for index, pose in enumerate(poses):
        center = pose.centroid or pose.pelvis
        if center is None:
            continue
        distance = (center[0] - x) ** 2 + (center[1] - y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def is_new_click(key: str, value) -> bool:
    """Detecta um clique novo (evita reprocessar o mesmo evento a cada rerun)."""

    if not value or "unix_time" not in value:
        return False
    if st.session_state.get(f"{key}_last_click") == value["unix_time"]:
        return False
    st.session_state[f"{key}_last_click"] = value["unix_time"]
    return True
