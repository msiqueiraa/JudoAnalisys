# Arquitetura

Documento técnico da arquitetura do **Judo Analysis**. Explica as camadas, o
fluxo de dados, as portas/adaptadores e as decisões de projeto.

## Visão geral

O projeto segue **Clean Architecture**: as regras de negócio (biomecânica e
comparação) ficam no centro, isoladas de frameworks, bibliotecas de vídeo e
interface. As dependências sempre apontam **para dentro**.

```
┌──────────────────────────────────────────────────────────────┐
│                      presentation (Streamlit)                 │
│   app.py (abas Vídeo/Foto/Foto x vídeo) · analysis.py ·       │
│   components/ (video_input, photo_input, photo_analysis,      │
│                mixed_analysis, person_selection,              │
│                tracking_review, exclusion, results_view, ...) │
└───────────────────────────────┬──────────────────────────────┘
                                │ usa
┌───────────────────────────────▼──────────────────────────────┐
│                         application                           │
│   use_cases/ (DetectPeople, AnalyzePerson, AnalyzePhoto,      │
│               CompareAnalyses)                                │
│   ports/  (IPoseEstimator, IVideoReader, IVideoSource,        │
│            IReportWriter) · dto.py                            │
└───────────────────────────────┬──────────────────────────────┘
        │ implementa portas                │ usa regras
┌───────▼───────────────────┐   ┌─────────▼────────────────────┐
│      infrastructure       │   │           domain             │
│  MediaPipe, OpenCV,       │   │  entities · value_objects ·  │
│  yt-dlp, ReportLab,       │   │  services (Angle, Speed,     │
│  FrameAnnotator,          │   │  PersonTracker, Comparator)  │
│  SingleImageReader        │   │                              │
└───────────────────────────┘   └──────────────────────────────┘
```

### Regra de dependência

- `domain` **não** importa nada das outras camadas nem de frameworks.
- `application` depende de `domain` e define **portas** (interfaces).
- `infrastructure` implementa as portas; depende de `domain`/`application`.
- `presentation` orquestra casos de uso; depende de todas as camadas.

## Camadas

### `domain` — regras de negócio

| Módulo | Responsabilidade |
| --- | --- |
| `entities/landmark.py` | `Landmark`: ponto corporal normalizado |
| `entities/pose.py` | `Pose` (33 landmarks) e enum `PoseLandmark` |
| `entities/video.py` | `Video` (metadados) e `TimeRange` |
| `entities/analysis.py` | `FrameMetrics` (inclui a pose rastreada), `MetricStats`, `AnalysisRun` (com a flag `is_photo`) |
| `value_objects/joint_angles.py` | `JointAngles`: joelho, coluna, pescoço, abertura |
| `value_objects/speed.py` | `Speed`: px/s e alturas de corpo/s |
| `value_objects/gi_color.py` | `GiColor`: cor do kimono em LAB (`cluster_two`, `median`) |
| `value_objects/comparison.py` | `MetricComparison`, `ComparisonResult` |
| `services/angle_calculator.py` | Cálculo dos ângulos (com `clip` no `arccos`) |
| `services/speed_calculator.py` | Velocidade do golpe normalizada |
| `services/track.py` | `Track` (estado de uma pessoa) e `ColorGate` (cor do kimono) |
| `services/person_tracker.py` | Rastreamento da pessoa escolhida (pelve + predição + IoU + escala) |
| `services/pair_tracker.py` | Rastreamento simultâneo dos dois atletas (atribuição global) |
| `services/pose_dedup.py` | Deduplicação de poses do mesmo corpo |
| `services/pose_comparator.py` | Similaridade entre referência e analisado (nunca exige mais amostras do que a análise tem — uma foto passa) |
| `services/statistics.py` | Agregação (média, pico, desvio) e frame-chave |
| `metrics_catalog.py` | Catálogo central (rótulo, unidade, escala) e `SPEED_METRICS` |

### `application` — casos de uso e portas

Portas (interfaces) em `application/ports/`:

- `IPoseEstimator` — detecta poses em um frame.
- `IVideoReader` — lê metadados e frames de um vídeo.
- `IVideoSource` — resolve upload/URL em um arquivo local.
- `IReportWriter` — gera o relatório.
- `IClipComposer` — gera o recorte dos dois vídeos lado a lado.
- `IGiColorSampler` — mede a cor do kimono de uma pose.

`application/pose_detection.py` (`PoseDetector`) decora um `IPoseEstimator` com
deduplicação, segunda tentativa nas metades do quadro e medição de cor, de modo
que os casos de uso recebam detecções já limpas. O módulo também expõe
`is_measurable_pose(pose)`, usado tanto pelo rastreamento quanto pela análise de
foto (pose com landmarks visíveis suficientes para medir ângulos).

Casos de uso em `application/use_cases/`:

- `DetectPeopleUseCase` — detecta pessoas no frame inicial do trecho (seleção).
- `AnalyzePersonUseCase` — rastreia a pessoa e calcula as métricas por frame
  (descartando frames sem rastreamento/pose válida e os **trechos descartados
  manualmente** pelo usuário, via `exclude`). Quando os kimonos têm cores
  diferentes, faz uma segunda passagem de rastreamento com a cor como pista.
  Recebe `key_frame_time` para reaproveitar o frame-chave escolhido pelo usuário.
- `CompareAnalysesUseCase` — compara referência × analisado.
- `AnalyzePhotoUseCase` — mede os ângulos do atleta escolhido em uma **foto** e
  monta um `AnalysisRun` com um **único frame** (`is_photo=True`,
  `hand_speed=None`): assim o painel, a comparação e o PDF funcionam sem um
  caminho paralelo. Rejeita a foto quando a pose tem pontos visíveis de menos ou
  quando nenhum ângulo pode ser medido.
- `BuildComparisonClipUseCase` — monta o recorte lado a lado para o player,
  passando a similaridade geral para o cartão final.
- `SelectKeyFrameUseCase` — troca o frame-chave do golpe (pico de velocidade ou
  instante escolhido) e relê a imagem do vídeo.

DTOs (`application/dto.py`): `PersonPreview`, `PersonSelection`, `TrackingSample`,
`AnalysisResult`.

### `infrastructure` — adaptadores

- `pose/MediaPipePoseEstimator` — Tasks API (`pose_landmarker_full.task`),
  multi-pessoa, ordena as poses da esquerda para a direita. Modo `IMAGE` por
  padrão (cada frame tratado de forma independente).
- `video/OpenCVVideoReader` — leitura de frames com OpenCV (context manager).
- `video/UploadVideoSource` / `video/UrlVideoSource` — upload e download (yt-dlp).
- `video/BrowserVideoConverter` — **prévia tocável no navegador**: o `st.video`
  sempre serve o arquivo como `video/mp4` e o navegador recusa containers que não
  sejam MP4 (`.mov`, `.avi`, `.mkv` — a análise funciona, mas o player fica em
  branco). Converte para MP4 com o FFmpeg embutido: *remux* (`-c copy`) quando o
  codec já é H.264 (rápido e sem perda) e recodificação (`libx264`, altura máxima
  de 720 px) nos demais casos (HEVC, ProRes, MPEG-4…). O resultado é cacheado em
  `.cache/preview/` e invalidado por `tamanho + mtime`; a **análise continua lendo
  o arquivo original**.
- `video/ffmpeg.py` — localiza o binário do FFmpeg empacotado pelo
  `imageio-ffmpeg` (compartilhado pela prévia e pelo download de URLs).
- `video/FfmpegClipComposer` — recorte lado a lado em H.264 (OpenCV + o FFmpeg
  embutido do `imageio-ffmpeg`; sem áudio), **com o esqueleto da pessoa
  escolhida** desenhado sobre cada lado (a pose vem de `FrameMetrics.pose`).
- `video/UrlVideoSource` — prioriza formatos **progressivos** (evita `ffprobe`)
  e aponta o `ffmpeg_location` para o binário do `imageio-ffmpeg`.
- `image/SingleImageReader` — trata uma **foto** como um "vídeo" de um único
  quadro (implementa `IVideoReader`): `frame_at`/`frames` devolvem a imagem e os
  metadados saem com `frame_count=1` e `fps=PHOTO_FPS`. É o que permite
  reaproveitar `DetectPeopleUseCase` inteiro (deduplicação, segunda tentativa e
  cor do kimono) na análise de foto.
- `image/UploadPhotoSource` — grava a imagem enviada em `.cache/photos/`.
- `files.safe_name` — nome de arquivo seguro, compartilhado pelos uploads de
  vídeo e de foto.
- `render/FrameAnnotator` — desenha esqueleto, seleção e métricas.
- `report/PdfReportWriter` — PDF com tabela, frame-chave e gráficos empilhados
  (ReportLab + Matplotlib).

### `presentation` — interface

- `app.py` — orquestra o fluxo em Streamlit: cria as **três abas**
  (`st.tabs(["Vídeo", "Foto", "Foto x vídeo"], on_change="rerun", key=...)`) e
  desenha a aba de vídeo. Só a aba **aberta** (`tab.open`) roda o trabalho
  pesado (tabelas, gráficos, PDF e recorte); os controles das outras continuam
  sendo criados, para o Streamlit não descartar o que o usuário já escolheu.
- `analysis.py` — execução de vídeo compartilhada pelas abas: `analyze_video`,
  `reanalyze_video` (correção de pessoa/descarte de frames), os *callbacks* da
  faixa de verificação, as **dicas de cor do kimono** por lado
  (`remember_kimono_hint`/`kimono_hint`) e `reset_key_frame_sliders`. Não importa
  componentes (evita ciclo de importação).
- `components/` — `video_input` (upload/URL + **vídeos de exemplo**),
  `photo_input` (upload de imagem), `person_selection` (slider +
  **caixas `mm:ss.cc`**), `photo_analysis` (aba **Foto**: detecção cacheada por
  assinatura do arquivo, seleção do atleta e comparação das duas fotos),
  `mixed_analysis` (aba **Foto x vídeo**), `pose_selection`
  (**seleção clicável** compartilhada, usada também na faixa), `results`
  (painel + tabela/gráficos), `results_view` (`SideSpec` + `render_results_view`,
  o desenho do resultado comum às três abas), `comparison_clip`,
  `tracking_review` (faixa de verificação com **miniaturas clicáveis** — o clique
  leva a correção para aquele instante — e **X para descartar frames**),
  `exclusion` (**descarte manual por `mm:ss`**) e `report`.
- `resources.py` — estimador cacheado (`st.cache_resource`) e leitura de metadados.
- `../formatting.py` — formatação e **parse** de tempo (`mm:ss`, `mm:ss.cc`)
  compartilhados com
  o relatório.
- `../logging_setup.py` — silencia os avisos internos do MediaPipe/TFLite em duas
  frentes: `configure_logging()` (variáveis de ambiente + `absl`, chamada pelo
  `app.py` **antes** de importar o pacote) e `suppress_native_stderr()`, um
  context manager que redireciona o **fd 2** durante as chamadas nativas (os
  avisos do C++ não passam pelo `logging` do Python). Em caso de exceção o
  conteúdo capturado é reimpresso, para não esconder erros. Pode ser desligado com
  `JUDO_SUPPRESS_NATIVE_STDERR=0`.

## Fluxo de dados

```
URL/Upload ──► IVideoSource ──► caminho local
                                     │
                 ┌───────────────────┴────────────────────┐
                 ▼                                        ▼
   DetectPeopleUseCase                        AnalyzePersonUseCase
 (frame do trecho)                           (trecho completo)
     │ IVideoReader.frame_at                      │ IVideoReader.frames
     ▼                                           ▼
   IPoseEstimator.detect                  IPoseEstimator.detect (por frame)
     │                                           │
     ▼                                           ▼ PairTracker.select
   PersonPreview ──► usuário escolhe      AngleCalculator.compute
                     a pessoa             SpeedCalculator.update
                                                 │
                                                 ▼
                                          AnalysisRun (+ stats, frame-chave)
                                                 │
                          CompareAnalysesUseCase │
                                                 ▼
                                         ComparisonResult
                                                 │
                        ┌────────────────────────┼───────────────────┐
                        ▼                        ▼                   ▼
              results (tela + Plotly)   BuildComparisonClip   PdfReportWriter
                                        (player lado a lado)       (PDF)
```

O `IPoseEstimator` usado pelos dois casos de uso é o `PoseDetector`, que
acrescenta deduplicação, a segunda tentativa nas metades do quadro e a medição da
cor do kimono antes de entregar as poses.

Na **análise de foto** o caminho é o mesmo até `PersonPreview`: o que muda é o
leitor — `SingleImageReader` devolve a mesma imagem em `frame_at`/`frames`, então
a detecção e a escolha do atleta são idênticas às do vídeo. A partir da pose
escolhida, `AnalyzePhotoUseCase` mede os ângulos e monta um `AnalysisRun` de um
quadro, que segue para `CompareAnalysesUseCase` e `PdfReportWriter` como qualquer
outro resultado — sem recorte comparativo (não há dois vídeos) e sem série
temporal (não há evolução a mostrar).

## Decisões de projeto

- **Multi-pessoa** via `num_poses` do MediaPipe Tasks, permitindo escolher o
  atleta dentro de uma luta.
- **Rastreamento robusto da pessoa** (`PersonTracker`): âncora na **pelve**,
  **predição de movimento** (velocidade constante), custo combinando distância,
  **IoU** com a caixa anterior e **escala do corpo**, com *gate* de
  distância/escala e continuidade. Evita a troca de identidade do antigo
  rastreamento por centroide quando os atletas se cruzam ou se sobrepõem.
- **Rastreamento dos dois atletas em par** (`PairTracker`): as detecções do frame
  são distribuídas entre as duas pessoas de uma vez (**atribuição global**, menor
  custo total, `_assignments`). A troca de identidade deixa de ser uma decisão
  isolada: ela só acontece se os **dois** casamentos piorarem ao mesmo tempo.
  Quando o atleta acompanhado não é o melhor encaixe, o frame é descartado
  (*coast*, custo `TRACKER_COAST_COST`) em vez de adotar o corpo do adversário.
  O adversário é semeado no frame da âncora (`_secondary_seed`); se esse frame
  tem só uma pessoa detectável, o app procura por perto (`PAIR_SEED_SEARCH_FRAMES`).
- **Deduplicação de poses** (`pose_dedup.deduplicate`): o detector às vezes
  devolve duas poses sobre o **mesmo** corpo (e nenhuma para o adversário), o que
  engana todo o pipeline. Duas detecções são consideradas o mesmo corpo quando a
  pelve/centroide está a `DEDUP_ANCHOR_DISTANCE`, ou quando a distância média
  entre landmarks comuns é pequena, ou ainda quando há muita sobreposição de
  caixa com landmarks próximos. A detecção mantida é a mais completa.
- **Segunda tentativa de detecção nas metades** (`PoseDetector`): sempre que o
  frame traz **menos de dois atletas**, o detector roda de novo na metade
  esquerda e na direita. É o caso mais comum em judô — o detector funde os dois
  corpos em uma única caixa. Condicionar a segunda tentativa a "a cena já mostrou
  dois atletas" não funcionava: quando o detector funde, ele **nunca** vê dois, e
  a condição nunca se cumpria (o rastreamento alternava). As poses do recorte
  voltam para o sistema de coordenadas do quadro inteiro
  (`x = left + x_recorte * fracao`); somar o deslocamento direto dobrava a
  posição horizontal e criava detecções "fantasma" fora dos atletas.
- **Cor do kimono como pista extra** (opcional): com a opção ligada, as medidas
  de cor de **todas** as poses do trecho são agrupadas em dois grupos
  (`GiColor.cluster_two`, k-médias com mediana) — um por kimono. O grupo mais
  próximo da cor da pessoa escolhida é a referência dela e o outro é a do
  adversário; se os centros não forem claramente diferentes
  (`GI_COLOR_DIFFERENT_DELTA_E`), a pista é **descartada**. Deduzir as cores do
  conjunto de medidas (em vez de medianas do rastreamento) evita um problema
  circular: quando a identidade alternava, as duas medianas saíam misturadas e a
  cor era descartada justamente onde era mais útil. A cor entra como **peso
  relativo** no custo (só penaliza quem parece mais com o adversário) e só
  **rejeita** um candidato quando o gate rígido está ligado, com desligamento
  automático após muitas rejeições seguidas.
- **Manter identidade na sobreposição**: quando a cena já teve duas pessoas e
  passa a ter **uma única detecção**, o `PersonTracker` exige uma correspondência
  mais próxima (`strict_factor`) antes de adotá-la; caso contrário, mantém a
  identidade anterior (*coast*). Isso reduz o "pulo" para o corpo errado quando
  os atletas se fundem em uma detecção. Pode ser desligado na barra lateral.
- **Modo IMAGE do MediaPipe**: cada frame é detectado de forma independente. Em
  vídeos de luta com dois atletas, o modo VIDEO (rastreamento temporal interno)
  tende a fundir os dois corpos em uma única detecção durante o golpe; o modo
  IMAGE detectou os dois com mais frequência. O parâmetro `running_mode` segue
  disponível para experimentação.
- **Identidade ancorada e bidirecional**: ao escolher a pessoa em um instante
  (a seleção inicial ou uma correção), o rastreamento parte desse ponto e segue
  **para frente e para trás**, de modo que a identidade vale para todo o trecho.
- **Faixa de verificação + correção manual**: como a detecção de pose tem
  limitações em oclusões severas, o resultado mostra a pessoa rastreada em
  vários instantes (`TrackingSample`) e permite **corrigir a pessoa** e
  reanalisar ancorando no instante escolhido.
- **Detecção no início do trecho**: alinha a seleção com o rastreamento.
- **Trecho selecionável**: analisar só o momento do golpe é mais rápido e mais
  preciso do que processar o vídeo inteiro.
- **Velocidade normalizada**: compara vídeos com resoluções diferentes sem
  exigir calibração.
- **Similaridade interpretável**: ângulos pela média, velocidade pelo pico, com
  escala por métrica definida em `metrics_catalog.py`.
- **Cálculo defensivo**: `arccos` com `clip`, verificação de visibilidade dos
  landmarks e contexto/uso de `with` para liberar recursos.
- **Frame "bugado" é descartado**: `AnalyzePersonUseCase` só registra frames em
  que o rastreamento encontrou a pessoa **e** a pose tem landmarks visíveis
  suficientes (`MIN_VISIBLE_LANDMARKS`). Frames sem rastreamento não entram em
  `AnalysisRun.frames`, então não contaminam médias, gráficos nem o score.
- **Correção pela faixa**: as miniaturas da faixa de verificação são clicáveis
  (`render_clickable_image`, largura `stretch`). O clique converte o instante da
  amostra para o domínio do slider (`pick_offset`: segundos a partir do início do
  trecho, alinhados ao passo e limitados às bordas) e marca a amostra; a seção
  **"Corrigir pessoa"** já abre no ponto com as pessoas detectadas nele.
- **Descarte manual de frames**: quando um trecho fica "bugado" (o rastreamento
  troca de atleta — ex.: *uke* de costas marcado como *tori*), o usuário pode
  clicar no **X** abaixo da amostra da faixa de verificação (o X descarta a
  "célula" daquela amostra: `sample_cell` divide o trecho pelos pontos médios
  entre amostras vizinhas) ou informar intervalos em `mm:ss` no componente
  `exclusion`. Os trechos são unidos/canonizados (`merge_ranges`) e repassados
  como `exclude` para `AnalyzePersonUseCase`, que descarta esses frames
  (contados em `AnalysisResult.excluded_frame_count`). A reanálise reusa
  `AnalysisResult.selection`, mantendo a mesma identidade. O rastreamento
  continua processando o trecho todo (só o cálculo ignora os frames excluídos),
  então a exclusão não desestabiliza a identidade.
- **Métrica só entra com dados suficientes**: além de precisar de valor nos dois
  lados, a métrica precisa de um mínimo de amostras em cada execução
  (`MIN_METRIC_SAMPLES` e `MIN_METRIC_COVERAGE`); dados medidos em pouquíssimos
  frames (espúrios) são descartados. O mínimo nunca passa do total de frames da
  análise (`min(required, total)`), então uma **foto** (um quadro) é comparável
  sem regra especial. As descartadas ficam em `excluded_metrics()` e nunca
  influenciam o score.
- **Similaridade geral = média das barras**: `overall_similarity` é exatamente a
  média das similaridades de `analyzed_metrics()`; o gráfico "Similaridade por
  métrica" exibe essa média como linha de referência, garantindo que o número do
  topo coincida com o gráfico (e com o PDF).
- **Player comparativo em H.264**: o recorte é composto em um único MP4 lado a
  lado (duração = a do maior recorte, o menor congela), garantindo um único
  controle de play e compatibilidade com o navegador; o áudio é removido.
- **Dados ao vivo no recorte**: cada metade mostra o **esqueleto e as métricas do
  instante** que está tocando (os mesmos dados do frame-chave), buscadas em
  `FrameMetrics` pelo timestamp mais próximo; no fim entra um **cartão de ~2 s**
  (`CLIP_END_CARD_SECONDS`) com a similaridade geral — o `compose` recebe o valor
  por parâmetro, então o recorte em cache é invalidado quando a similaridade muda
  (ex.: ao ligar "Ignorar velocidade").
- **Tempo em min:seg com centésimos**: o slider usa `datetime.time` com
  `format="mm:ss"` (ajuste grosso, de segundo em segundo) e há **caixas
  `mm:ss.cc`** sincronizadas — a caixa é a **fonte de verdade** (mover o slider
  reescreve as caixas). `seconds_to_time` preserva microssegundos, então
  `00:01.55` sobrevive ao slider; `format_duration_precise` (2 casas) exibe o
  valor exato na seleção, nos resultados, no PDF e na faixa de verificação.
  Essencial para golpes de 2-3 s em vídeos longos.
- **Seleção/correção por clique**: a pessoa pode ser escolhida clicando sobre o
  atleta na imagem (`streamlit-image-coordinates`); a pose mais próxima do
  clique é selecionada. Sem a dependência, a seleção continua pela lista.
- **Ignorar velocidade (opcional)**: quando ligada, `hand_speed` entra em
  `SPEED_METRICS` e é marcada como `ignored` na comparação — sai da tabela, dos
  gráficos e da nota, e é listada em uma nota própria. Útil quando os vídeos têm
  ritmos diferentes (*slow-motion* x normal).
- **Recorte anotado**: o player comparativo reutiliza `FrameAnnotator` (com
  `text_top` para as métricas ficarem abaixo do rótulo grande do lado) e a pose
  de cada instante vem de `FrameMetrics.pose` (busca do frame mais próximo por
  timestamp).
- **Frame-chave escolhido pelo usuário**: `AnalysisRun.key_frame_index` aponta
  para o frame do "golpe"; `aggregate_run` define o **pico de velocidade**
  (`peak_speed_frame`), mas o slider "Instante do golpe" chama
  `SelectKeyFrameUseCase`, que grava `AnalysisResult.key_frame_time` e **relê o
  quadro** do vídeo naquele instante (`nearest_frame` acomoda o instante entre
  dois frames). A escolha sobrevive à reanálise (é passada de volta ao
  `AnalyzePersonUseCase`) e vale para o painel e para o PDF. O estado do slider é
  limpo por `reset_key_frame_sliders()` (`presentation/analysis.py`) a cada
  (re)análise.
- **Análise de foto reaproveita o vídeo**: em vez de um pipeline paralelo, uma
  foto vira um `AnalysisRun` de **um quadro** (`is_photo=True`, `hand_speed=None`,
  `Video(frame_count=1)`) e o `SingleImageReader` faz a foto passar pelo mesmo
  `DetectPeopleUseCase`. Com isso, painel, comparação, notas de métricas
  descartadas e PDF valem sem código novo; `is_photo` serve só para rotular a
  interface/PDF ("Foto analisada", sem FPS/trecho) e para pular a série temporal.
  A **velocidade** sai da comparação sozinha (não há valor na foto) e a nota
  correspondente é a mesma das métricas sem dados suficientes.
- **Abas independentes com estado próprio**: as três abas usam `st.tabs(...,
  on_change="rerun", key=...)`, que guarda a aba ativa na sessão e expõe
  `tab.open`. Cada aba tem prefixos de estado próprios (`ref_*`/`ana_*` no vídeo,
  `fref_*`/`fana_*` na foto, `mvideo_*`/`mphoto_*` na mista) e só a aba aberta
  desenha o resultado pesado; os controles continuam sendo criados em todas as
  abas, porque o Streamlit descarta o estado de widgets que não foram desenhados.
- **Vídeos de exemplo**: botão na barra lateral que baixa e carrega dois vídeos
  de demonstração (mesmo caminho de URL); os campos vêm pré-preenchidos.
- **Download robusto de URL**: prioriza formatos progressivos (sem merge) e
  informa o `ffmpeg_location` do `imageio-ffmpeg`, evitando depender de
  `ffprobe` no PATH.
- **Relatório sem distorção**: um gráfico por métrica analisada, empilhados
  verticalmente com a proporção da figura preservada.

## Extensão

Para adicionar um novo backend de pose (ex.: YOLO-Pose), basta criar uma classe
que implemente `IPoseEstimator` em `infrastructure/pose/` e injetá-la nos casos
de uso — o domínio e a aplicação permanecem inalterados.

Para uma nova métrica: adicione o campo em `JointAngles`, calcule em
`AngleCalculator` e registre a entrada correspondente em `metrics_catalog.py`
(rótulo, unidade e escala de similaridade).
