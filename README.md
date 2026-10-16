# 🥋 Judo Analysis

Aplicação que compara a biomecânica de um golpe de judô entre um vídeo de
**referência** (gabarito) e um vídeo **analisado**, medindo:

- **Ângulo do joelho** (esquerdo e direito)
- **Ângulo da coluna** (inclinação do tronco)
- **Ângulo do pescoço** (inclinação da cabeça)
- **Abertura das pernas** (ângulo entre as coxas)
- **Velocidade do golpe** (velocidade da mão no pico do movimento)

O resultado é exibido na tela (frame-chave anotado + painel de métricas +
gráficos interativos), pode ser assistido em um **player comparativo** (os dois
recortes lado a lado, tocando ao mesmo tempo, com as métricas ao vivo e a
similaridade no fim) e **baixado em PDF**.

---

## Sumário

- [Como funciona](#como-funciona)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Métricas](#métricas)
- [Arquitetura](#arquitetura)
- [Testes](#testes)
- [Solução de problemas](#solução-de-problemas)

---

## Como funciona

1. Você carrega **dois vídeos** — um de referência e um para analisar — por
   **upload** ou por **URL** (YouTube, Instagram, TikTok ou link direto). Para
   testar rapidamente, clique em **"Carregar vídeos de exemplo"** (barra
   lateral): os dois campos vêm pré-preenchidos e os arquivos são baixados só
   quando você clica.
2. Escolhe o **trecho** que contém o golpe. Em vídeos longos, use as caixas
   **Início/Fim em `mm:ss`** (aceitam centésimos, ex.: `00:01.55`; o slider
   continua disponível para ajuste rápido).
3. O app detecta as pessoas no frame inicial do trecho e você **seleciona qual
   pessoa** deve ser acompanhada em cada vídeo — pela lista ou **clicando sobre
   o atleta** na imagem.
4. O app rastreia essa pessoa quadro a quadro, calcula as métricas e compara a
   execução analisada com a referência, gerando um **score de similaridade**.
   - O rastreamento usa a **pelve** como âncora, **predição de movimento**,
     sobreposição de caixa (**IoU**) e **escala do corpo**, para não trocar de
     atleta quando os dois se cruzam ou se sobrepõem no golpe.
   - Os **dois atletas são rastreados juntos** (atribuição global por frame): a
     detecção só vai para a pessoa acompanhada se ela for o melhor encaixe
     **considerando o adversário também**. Quando o atleta acompanhado não
     aparece, o frame é descartado em vez de adotar o corpo do adversário.
   - O detector **tenta de novo nas metades** do quadro sempre que o frame traz
     menos de dois atletas. Isso resolve o caso mais comum em judô: o detector
     funde os dois corpos em uma única caixa e o rastreamento passa a alternar
     entre os atletas. O custo é maior (duas detecções extras nos frames
     incompletos), mas recupera o segundo atleta na maioria deles.
   - A opção **"Os dois atletas têm kimonos de cores diferentes"** acrescenta a
     **cor do kimono** como pista de identidade (útil quando os corpos ficam
     sobrepostos e a posição não distingue). As duas cores são deduzidas do
     conjunto de medidas do trecho, nunca de uma amostra isolada, e a pista é
     **descartada** sozinha se as cores medidas não forem claramente diferentes.
     A cor nunca é a única forma de identidade.
   - A opção **"Manter identidade na sobreposição"** evita "pular" para o outro
     atleta quando os corpos se fundem em uma única detecção.
   - Uma **faixa de verificação** mostra a pessoa rastreada em vários instantes
     do trecho. Se o rastreamento pegou o atleta errado, use
     **"Corrigir pessoa"**: escolha um instante, indique a pessoa correta (lista
     ou clique) e o app reanalisa ancorando a identidade nesse ponto (para frente
     e para trás).
   - As métricas são medidas sobre o trecho escolhido.
   - O **frame-chave** (imagem anotada e PDF) é, por padrão, o de **maior
     velocidade da mão**; o slider **"Instante do golpe"** permite parar em
     outro instante.
   - **Frames "bugados"** (em que o rastreamento não encontrou a pessoa ou a pose
     ficou incompleta) são **removidos** da análise — as métricas e o score usam
     apenas os frames válidos.
   - Uma métrica só entra na análise se tiver **dados suficientes** nos dois
     vídeos; se aparecer em pouquíssimos frames (dados espúrios), é **descartada**
     (não entra na tabela, nos gráficos nem no score).
   - Se algum trecho ficar **"bugado"** (o rastreamento perdeu ou trocou o
     atleta — ex.: quando o *uke* fica de costas e o app marca só o *uke*),
     clique no **X** abaixo da amostra errada na faixa de verificação: os frames
     daquele trecho saem do cálculo, dos gráficos e do score, e o app reanalisa.
     Para um ajuste fino, use **"Descartar frames com erro"** e informe um ou
     mais intervalos em `mm:ss`.
   - A **similaridade geral** é exatamente a **média das barras** do gráfico
     "Similaridade por métrica" (que mostra a linha da média).
   - Se os vídeos tiverem **ritmos diferentes** (ex.: um em *slow-motion* e outro
     normal), ligue **"Ignorar velocidade do golpe"**: a velocidade sai da
     tabela, dos gráficos e da nota.
5. O frame do golpe é exibido anotado, com os dados ao lado. Use o **slider
   "Instante do golpe"** (acima da imagem) para percorrer o trecho e parar no
   frame que você considera o golpe — o botão **"Automático"** volta para o pico
   de velocidade. O instante escolhido vale para o painel e para o PDF. Você
   também pode:
   - gerar e assistir o **player comparativo** — recorte dos dois vídeos lado a
     lado, tocando simultaneamente, com o **esqueleto e as métricas do instante**
     desenhados sobre cada vídeo; no fim, um cartão mostra a **similaridade
     geral**;
   - baixar um **relatório em PDF** (com os gráficos empilhados e as notas).

## Requisitos

- **Python 3.12** (compatível com 3.10/3.11)
- Webcam/câmera não é necessária; apenas os arquivos de vídeo
- O modelo `pose_landmarker_full.task` já está incluído na raiz do projeto

## Instalação

### Opção 1 — pip + requirements.txt

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Importante:** rode o app sempre **com o mesmo ambiente** em que instalou as
> dependências (ative o venv antes do `streamlit run`). Se você rodar com um
> Python global "por fora", podem faltar pacotes (ex.: `imageio-ffmpeg`) e a
> versão do Streamlit pode ser diferente da esperada (`streamlit>=1.50`).

### Opção 2 — pipenv (Pipfile)

```bash
pipenv install
pipenv shell
```

## Como usar

```bash
streamlit run app.py
```

O navegador abre em `http://localhost:8501`.

1. (Opcional) Na barra lateral, clique em **"Carregar vídeos de exemplo"** para
   baixar automaticamente dois vídeos de judô de demonstração.
2. Na coluna **"Vídeo de referência (gabarito)"**, carregue o vídeo com a
   técnica correta (upload ou URL). Formatos aceitos: **`.mp4`, `.mov`, `.avi`,
   `.mkv`**.
3. Na coluna **"Vídeo analisado"**, carregue o vídeo a ser avaliado.
4. Em cada coluna, ajuste o **trecho analisado** pelo slider ou digitando
   **Início/Fim em `mm:ss`** (aceita centésimos, ex.: `00:01.55`) e clique em
   **"Detectar pessoas neste trecho"**.
5. Escolha a **pessoa** no seletor ou **clique sobre o atleta** na imagem e
   confira o destaque verde no frame.
6. Clique em **Analisar**.
7. Veja o resultado, compare as métricas e clique em
   **"Gerar relatório PDF"** para baixar o laudo.
8. Confira a **faixa de verificação** do rastreamento. Se o esqueleto tiver
   pulado para o outro atleta (ou marcado só o *uke* nas costas), você tem três
   opções:
   - clique no **X** abaixo do frame errado — os frames daquele trecho saem do
     cálculo, dos gráficos e do score;
   - abra **"Corrigir pessoa"**, escolha o instante e a pessoa correta (lista ou
     clique) e clique em **"Reanalisar com esta pessoa"**; ou
   - abra **"Descartar frames com erro"**, informe um ou mais intervalos em
     `mm:ss` (ex.: `01:05`–`01:09`) e clique em
     **"Reanalisar descartando esses trechos"**.

> Barra lateral: **"Máximo de pessoas por frame"** (detecção),
> **"Os dois atletas têm kimonos de cores diferentes"** (usa a cor do kimono como
> pista extra de identidade; o app descarta a pista sozinho se as cores medidas
> não forem claramente diferentes), **"Manter identidade na sobreposição"**
> (evita trocar de atleta quando os corpos se fundem) e
> **"Ignorar velocidade do golpe"** (para vídeos com ritmos diferentes).

> Dica: o slider usa o formato **min:seg** (`mm:ss`), igual ao YouTube/TikTok;
> as caixas ao lado aceitam **centésimos** (`00:01.55`) — úteis quando o golpe
> dura 2-3 s — ou `mm:ss` (ex.: `00:12`) ou `h:mm:ss`. As caixas mandam no valor
> final (o slider continua sendo o ajuste rápido). Mantenha o frame
> de detecção dentro do trecho escolhido — a seleção de pessoa é feita no início
> do trecho para o rastreamento começar já correto.

> Dica: no **player comparativo**, o recorte mais longo define a duração; quando
> o mais curto acaba, a imagem dele congela. O áudio é removido e, sobre cada
> recorte, são desenhados o esqueleto e as **métricas do instante** (os mesmos
> dados do frame-chave). No fim entra um cartão de ~2 s com a **similaridade
> geral**.

## Métricas

| Métrica | Como é calculada | Unidade |
| --- | --- | --- |
| Joelho (E/D) | Ângulo no joelho entre quadril e tornozelo | graus |
| Coluna | Inclinação do tronco (ombros → quadris) em relação à vertical | graus |
| Pescoço | Inclinação da cabeça (ombros → orelhas/nariz) em relação à vertical | graus |
| Abertura das pernas | Ângulo entre as duas coxas, medido no quadril | graus |
| Velocidade do golpe | Deslocamento quadro a quadro dos punhos ÷ tempo | px/s e alturas de corpo/s |

A **similaridade** de cada métrica vai de 0 a 100% e é calculada a partir da
diferença entre referência e analisado:

- ângulos usam a **média** do trecho;
- velocidade usa o **pico** (força do golpe).

A velocidade é **relativa/normalizada** (não exige calibração): além de px/s, ela
é expressa em "alturas de corpo por segundo", dividindo pela distância
ombro→tornozelo. Assim é possível comparar vídeos com resoluções diferentes.

Se uma métrica não tiver **dados suficientes** em um dos vídeos (ex.: a abertura
das pernas só apareceu em 2 frames), ela é **descartada da análise** — não
aparece na tabela, nos gráficos nem no score, e é listada em uma nota
("Não analisadas: ...") na tela e no PDF. Isso evita que dados espúrios de
frames "bugados" poluam o resultado.

Se você marcar **"Ignorar velocidade do golpe"**, a velocidade é removida da
comparação **por escolha sua** — útil quando os vídeos têm ritmos diferentes
(*slow-motion* x normal) — e aparece em uma nota separada
("Ignoradas nesta comparação: ..."). Por padrão o frame-chave é o instante de
maior movimento, mas você pode escolher outro pelo slider **"Instante do golpe"**.

Além do descarte automático de frames inválidos, você pode **descartar frames
manualmente**:
- clique no **X** abaixo de uma amostra da faixa de verificação — o X descarta
  exatamente os frames que aquela amostra representa (a célula entre as amostras
  vizinhas);
- ou abra **"Descartar frames com erro"** (em cada vídeo) e informe intervalos em
  `mm:ss` (aceita centésimos, ex.: `00:01.55`).

Os frames descartados saem do cálculo, dos gráficos e do score; o total aparece
no painel e o botão **"Restaurar todos os frames"** desfaz os descartes. Isso é
útil, por exemplo, quando o *uke* fica de costas e o rastreador marca só o *uke*
como *tori*.

## Arquitetura

O projeto segue **Clean Architecture** e **Clean Code**, em quatro camadas com
dependências apontando para dentro (domínio no centro):

```
judo_analysis/
├── domain/          # Regras de negócio puras (sem frameworks)
│   ├── entities/    Landmark, Pose, Video, AnalysisRun, FrameMetrics
│   ├── value_objects/ JointAngles, Speed, ComparisonResult
│   ├── services/    AngleCalculator, SpeedCalculator, PersonTracker, PoseComparator
│   └── metrics_catalog.py
├── application/     # Casos de uso + portas (interfaces)
│   ├── ports/       IPoseEstimator, IVideoReader, IVideoSource,
│   │                IReportWriter, IClipComposer
│   └── use_cases/   DetectPeople, AnalyzePerson, CompareAnalyses,
│                    BuildComparisonClip
├── infrastructure/  # Adaptadores concretos
│   ├── pose/        MediaPipePoseEstimator (Tasks API)
│   ├── video/       OpenCVVideoReader, UploadVideoSource, UrlVideoSource,
│   │                FfmpegClipComposer (recorte lado a lado)
│   ├── render/      FrameAnnotator
│   └── report/      PdfReportWriter
└── presentation/    # UI Streamlit
    ├── app.py
    └── components/  (..., tracking_review, ...)
```

Detalhes em [`docs/arquitetura.md`](docs/arquitetura.md).

## Testes

```bash
python -m pytest
```

Os testes cobrem o domínio (ângulos, velocidade, rastreamento simples e em par,
deduplicação de poses, cor do kimono, estatística e comparação) e a aplicação
(caso de uso de análise com rastreamento de dois atletas que se cruzam, incluindo
a correção ancorada), sem depender de vídeos reais. Há também um teste de fumaça
que renderiza o app Streamlit.

## Solução de problemas

| Problema | Causa provável | Solução |
| --- | --- | --- |
| "Modelo do MediaPipe não encontrado" | `pose_landmarker_full.task` ausente | Mantenha o arquivo na raiz do projeto |
| "A dependência 'yt-dlp' não está instalada" | URL sem a lib | `pip install yt-dlp` |
| Nenhuma pessoa detectada | Iluminação/enquadramento ruim | Ajuste o trecho ou o vídeo |
| Pessoa errada rastreada | Oclusão/sobreposição no golpe (ex.: *uke* de costas marcado como *tori*) | Confira a **faixa de verificação** e clique no **X** do frame errado; use **"Corrigir pessoa"** para trocar a identidade; mantenha **"Manter identidade na sobreposição"** ligado; se os kimonos tiverem cores diferentes, ligue a opção de cor do kimono |
| Rastreamento alterna entre os dois atletas | O detector funde os dois corpos em uma caixa só | Já tratado: o app refaz a detecção nas metades do quadro quando o frame traz menos de dois atletas; ligue a opção de **cor do kimono** se eles tiverem cores diferentes |
| Vídeo de URL não baixa | Link privado/bloqueado | Use upload do arquivo |
| Clique na imagem não funciona | `streamlit-image-coordinates` ausente | `pip install streamlit-image-coordinates`; sem ele a correção continua pela lista |
| Recorte não toca no navegador | Codec/codificação | O recorte já é gerado em H.264; aguarde a geração terminar |
| Prévia do vídeo fica em branco (`.mov`/`.avi`/`.mkv`) | O `st.video` serve o arquivo como `video/mp4` e o navegador recusa o container QuickTime/AVI/Matroska | **Já tratado**: a prévia é convertida para MP4 (remux quando o codec é H.264; recodificação nos demais) e cacheada em `.cache/preview/` — a análise continua usando o arquivo original |
| Aviso `use_container_width` no console | Versão antiga do Streamlit | Garanta `streamlit>=1.50` (o código já usa a API nova `width="stretch"`) |
| `ConnectionResetError [WinError 10054]` / `Exception in callback _ProactorBasePipeTransport...` | Ruído do asyncio no Windows | **Inofensivo** — ocorre quando o navegador fecha/reconecta; pode ignorar |
| Logs do MediaPipe (XNNPACK, `inference_feedback_manager`, NORM_RECT…) | Avisos internos do TFLite/absl escritos direto no `stderr` | **Já silenciados**: `configure_logging()` + `suppress_native_stderr()` capturam o fd 2 durante as chamadas nativas. Para ver os avisos (depuração), rode com `JUDO_SUPPRESS_NATIVE_STDERR=0` |
