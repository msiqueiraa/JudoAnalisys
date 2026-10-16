"""Configuracoes globais da aplicacao."""

from __future__ import annotations

from pathlib import Path

# Raiz do repositorio (pasta que contem o pacote `judo_analysis`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modelo do MediaPipe Pose Landmarker (Tasks API).
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pose_landmarker_full.task"

# Numero maximo de pessoas detectadas simultaneamente por frame.
DEFAULT_NUM_POSES = 4

# Visibilidade minima de um landmark para considera-lo valido (0..1).
MIN_VISIBILITY = 0.2

# Confianca minima de deteccao de pose.
MIN_DETECTION_CONFIDENCE = 0.5

# Pasta temporaria para videos baixados de URLs.
TEMP_DIR = PROJECT_ROOT / ".cache" / "videos"

# Pasta de cache das previas convertidas para o player do navegador (videos
# .mov/.avi/.mkv e codecs que o navegador nao toca).
PREVIEW_DIR = PROJECT_ROOT / ".cache" / "preview"

# Pasta onde as fotos enviadas para a analise de foto sao gravadas.
PHOTO_DIR = PROJECT_ROOT / ".cache" / "photos"

# FPS nominal de uma foto (quadro unico). Nao representa duracao real: existe
# para as conversoes de tempo (slider, formatacao) continuarem validas.
PHOTO_FPS = 1.0

# Numero minimo de frames validos para uma metrica entrar na comparacao.
# Metricas medidas em pouquissimos frames (dados espurios) sao descartadas.
MIN_METRIC_SAMPLES = 3

# Fracao minima do trecho em que a metrica precisa ter valor em cada video.
MIN_METRIC_COVERAGE = 0.1

# Numero minimo de landmarks visiveis para considerar uma pose mensuravel.
MIN_VISIBLE_LANDMARKS = 6

# Recorte lado a lado (player comparativo).
CLIP_OUTPUT_HEIGHT = 480  # Altura (px) de cada metade do recorte.
CLIP_GAP_PX = 4  # Faixa preta entre as duas metades.
# Duracao (s) do cartao final com a similaridade geral.
CLIP_END_CARD_SECONDS = 2.0

# Videos de exemplo (mock) carregados pelo botao "Carregar videos de exemplo".
EXAMPLE_REFERENCE_URL = "https://www.youtube.com/watch?v=FQnOlCxo4oI"
EXAMPLE_ANALYZED_URL = "https://www.youtube.com/watch?v=fS6RavQRknM"

# Mantem a identidade da pessoa quando so uma pose e detectada (sobreposicao).
DEFAULT_KEEP_IDENTITY = True

# --- Deduplicacao de poses do MESMO corpo ----------------------------------
# O detector as vezes devolve duas poses sobre o mesmo atleta (e nenhuma para o
# adversario). Contar essas poses como "duas pessoas" engana o rastreador.
#
# Distancia media entre landmarks (relativa a altura do corpo) abaixo da qual
# as duas deteccoes sao consideradas o mesmo corpo.
DEDUP_LANDMARK_DISTANCE = 0.15
# Sobreposicao de caixa (IoU) a partir da qual as deteccoes sao suspeitas de
# serem o mesmo corpo.
DEDUP_IOU = 0.6
# Com caixas sobrepostas, distancia media (relativa) que confirma o mesmo corpo.
DEDUP_IOU_LANDMARK_DISTANCE = 0.35
# Duas deteccoes praticamente no MESMO ponto do quadro sao o mesmo corpo, mesmo
# que os landmarks divirjam (acontece quando o detector "ve" o mesmo atleta duas
# vezes). Distancia entre pelve/centroide, em fracao do quadro.
DEDUP_ANCHOR_DISTANCE = 0.04

# --- Segunda tentativa de deteccao -----------------------------------------
# Sempre que o quadro traz menos de dois atletas, o detector procura de novo nas
# metades esquerda e direita do quadro. Nao da para condicionar isso a "a cena ja
# mostrou dois atletas": quando o detector funde os dois corpos ele NUNCA ve dois,
# e a condicao nunca se cumpria - o rastreamento alternava entre os atletas.
# Medido em videos reais: recupera o segundo atleta na maioria dos frames
# (baixar a confianca de deteccao quase nao ajuda, porque o detector funde os
# dois corpos na mesma caixa antes de decidir a confianca).
DEFAULT_SPLIT_RETRY = True

# --- Custo de associacao do rastreador --------------------------------------
# Pesos das pistas somadas no custo entre a pessoa rastreada e um candidato.
TRACKER_WEIGHT_POSITION = 1.0
# O peso do IoU e baixo de proposito: em luta de solo as caixas se deformam e o
# IoU cai a zero mesmo no casamento correto. Com peso alto ele inflava o custo
# de TODO casamento bom, e a atribuicao passava a preferir descartar frames.
TRACKER_WEIGHT_IOU = 0.3
TRACKER_WEIGHT_SCALE = 0.25
TRACKER_WEIGHT_COLOR = 1.0
# Margem de custo exigida para trocar de identidade (evita alternancia).
TRACKER_SWITCH_MARGIN = 0.15
# Custo acima do qual o casamento e incerto: a velocidade nao e atualizada.
TRACKER_TRUST_COST = 0.75
# Custo de deixar uma pessoa sem correspondencia (para a atribuicao em par).
# Medido em videos reais: abaixo de 0.85 a atribuicao descarta frames bons
# (perde metade do trecho); acima disso ela aceita candidatos duvidosos e a
# identidade volta a alternar. 0.85 fica entre os dois.
TRACKER_COAST_COST = 0.85
# Quantos frames, para cada lado, procurar as duas pessoas ao definir a semente
# do adversario (o frame escolhido pelo usuario pode ter so uma detectavel).
PAIR_SEED_SEARCH_FRAMES = 25

# --- Cor do kimono ----------------------------------------------------------
# Raio dos patches de amostragem, em fracao da altura do corpo.
GI_COLOR_PATCH_RADIUS = 0.06
# Distancia (Delta E, CIE76) ate a qual a cor e tratada como a mesma pessoa.
GI_COLOR_SAME_DELTA_E = 25.0
# Acima desta distancia a cor prova que a pose e de OUTRO atleta.
GI_COLOR_DIFFERENT_DELTA_E = 35.0
# Quantas rejeicoes seguidas desligam o gate rigido (mantem o peso da cor).
# Evita perder a analise inteira caso a cor de referencia esteja ruim.
GI_COLOR_MAX_REJECTIONS = 120
# Suavizacao (media exponencial) da cor de referencia em matches confiaveis.
GI_COLOR_SMOOTHING = 0.1
# Heterogeneidade maxima (desvio absoluto mediano do canal L) de um patch.
# Patches que misturam kimono com fundo/pele sao descartados.
GI_COLOR_PATCH_MAX_MAD = 45.0
# Minimo de patches validos para aceitar a cor medida.
GI_COLOR_MIN_PATCHES = 2
# Margem (Delta E) pela qual um candidato precisa parecer MAIS com o adversario
# do que com o atleta rastreado para ser rejeitado pelo gate rigido.
GI_COLOR_REJECT_MARGIN = 8.0
