# Reconhecimento_Facial

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Controle de acesso por reconhecimento facial com **OpenCV** e **DeepFace**.

Cada rosto encontrado em uma foto ou na webcam é comparado com as pessoas autorizadas:

- **caixa verde** `ACESSO LIBERADO: <nome>`: rosto real de uma pessoa autorizada;
- **caixa vermelha** `ACESSO NEGADO`: qualquer outro caso;
- **caixa amarela** `VERIFICANDO...` (só na webcam): reconhecido, aguardando confirmação.

## Começando: teste com você mesmo na webcam

```bash
cd Reconhecimento_Facial
python -m uv sync
```

1. Tire suas **3 fotos de cadastro** pela própria webcam e cadastre-se:
   ```bash
   python -m module_reconhecimento_facial.main capturar gustavo
   ```
   Abre uma janela com a câmera. Olhe para ela, com o rosto bem iluminado, e aperte
   **ESPAÇO** para cada foto. Entre uma foto e outra, vire um pouco o rosto ou mude a
   expressão. Uma foto só é aceita se tiver exatamente um rosto nítido. As fotos vão
   para `data/autorizados/gustavo/` e o cadastro é feito em seguida.

   Também dá para usar fotos do celular: coloque as 3 em `data/autorizados/<nome>/`
   e rode `python -m module_reconhecimento_facial.main cadastrar`.
   ```
   data/autorizados/
   └── gustavo/
       ├── 01.jpg
       ├── 02.jpg
       └── 03.jpg
   ```
2. Abra a webcam:
   ```bash
   python -m module_reconhecimento_facial.main webcam
   ```
   A caixa fica amarela por um instante e depois verde. Para testar a segurança:
   - peça para outra pessoa aparecer na câmera: a caixa fica vermelha;
   - mostre uma foto sua na tela do celular: a caixa fica vermelha com `rosto falso`;
   - afaste-se bastante da câmera: a caixa fica vermelha com `aproxime-se`.

   Para sair, aperte **q** ou **Esc**. Se o PC tiver mais de uma câmera, use `--camera 1`.

Na primeira execução, o DeepFace baixa os pesos dos modelos (cerca de 100 MB) para `models/.deepface`.

### Se algo não sair como esperado

| Sintoma | O que fazer |
|---|---|
| Cadastro avisa `não se parece com as outras fotos` | Uma das fotos está muito diferente (outra pessoa, óculos escuros, muito de lado). Troque a foto. |
| Fica em `VERIFICANDO...` sem liberar | Olhe de frente para a câmera, com o rosto bem iluminado e sem contraluz. |
| Você aparece como `rosto falso` | Luz fraca ou contraluz confundem o anti-spoofing. Acenda a luz de frente para o rosto. |
| `Não foi possível abrir a câmera` | Feche outros programas que usam a câmera (Teams, Zoom, navegador) ou tente `--camera 1`. |
| Libera com dificuldade mesmo com boa luz | Cadastre fotos tiradas na própria webcam, que são mais parecidas com o que o sistema vai ver. |

## Fotos de teste

Coloque fotos em `data/teste` e rode:

```bash
python -m module_reconhecimento_facial.main reconhecer
```

As fotos com as caixas são gravadas em `reports/resultados`, junto de um
`relatorio.csv` com a decisão, a distância e o motivo de cada rosto. Também dá para
passar uma foto ou pasta específica: `... reconhecer caminho/da/foto.jpg`.

## Como funciona

```
foto/quadro ──> YuNet (OpenCV) ──> rostos ──> Facenet512 ──> embedding ──┐
                                      │                                  ├──> decisão ──> caixa
                                      └──> MiniFASNet ──> real ou falso ─┘
```

1. **Detecção**: o YuNet, detector de rostos por rede neural do OpenCV, encontra os rostos.
2. **Embedding**: o Facenet512 transforma cada rosto em um vetor de 512 números.
   Fotos da mesma pessoa geram vetores próximos.
3. **Anti-spoofing**: o MiniFASNet verifica se o rosto é real ou uma reprodução
   (foto impressa, tela de celular).
4. **Decisão**: cada pessoa autorizada é representada pela média dos vetores das
   suas fotos (o *template*). O acesso só é liberado se **todas** as regras passarem:

   | Regra | Se falhar |
   |---|---|
   | Rosto com pelo menos 60 px | `ACESSO NEGADO: aproxime-se` |
   | Rosto real (anti-spoofing) | `ACESSO NEGADO: rosto falso` |
   | Distância até o template ≤ limiar | `ACESSO NEGADO` |
   | Mais parecido ganha do 2º por ≥ 0,05 | `ACESSO NEGADO: nao identificado` |
   | Webcam: reconhecido em 3 análises seguidas | `VERIFICANDO...` |

## Segurança

| Risco | Proteção |
|---|---|
| Foto ou vídeo do autorizado mostrado para a câmera | Anti-spoofing (MiniFASNet) em cada rosto |
| Um quadro ruim liberando a pessoa errada | Na webcam, exige 3 análises seguidas reconhecendo a mesma pessoa |
| Pessoa parecida com dois autorizados | Margem mínima entre o 1º e o 2º mais parecidos |
| Rosto longe ou pequeno, com pouco detalhe | Rostos menores que 60 px são negados |
| Foto errada no cadastro (de outra pessoa, ou com duas pessoas) | Cadastro recusa fotos com mais de um rosto e remove a foto que não se parece com as outras da mesma pessoa |
| Cache adulterado executando código | Base salva em `.npz` e lida com `allow_pickle=False` (nunca `pickle`) |
| Erro de ambiente parecer "sem rosto" | Só a exceção `FaceNotDetected` é tratada como foto sem rosto |
| Auditoria | A webcam registra acessos liberados e rostos falsos em `reports/registro_acessos.csv` |
| Vazamento de dados biométricos | O `.gitignore` nunca versiona fotos, embeddings nem resultados |

**Limites.** Este é um projeto de estudo. O anti-spoofing por imagem barra fotos e
telas comuns, mas não máscaras 3D nem ataques feitos sob medida. Sistemas
comerciais usam câmeras de profundidade ou infravermelho para isso.

## Calibração do limiar

O limiar padrão do DeepFace (0,30) é calibrado para comparar duas fotos soltas.
Aqui a comparação é com a média de 3 fotos, então o limiar foi medido de novo com o
comando `avaliar`, usando a base pública [LFW](https://vis-www.cs.umass.edu/lfw/):

- 100 pessoas cadastradas com **3 fotos cada**; as outras fotos delas são tentativas legítimas;
- 300 pessoas nunca cadastradas tentando entrar (impostores);
- FAR = falsos aceites (impostor liberado, ou autorizado liberado como outra pessoa);
- FRR = falsas rejeições (autorizado barrado).

RESULTADOS_CALIBRACAO

Para repetir com outra base rotulada (uma subpasta por pessoa):

```bash
python -m module_reconhecimento_facial.main avaliar caminho/da/base --fotos-cadastro 3
```

## Comandos

```bash
python -m module_reconhecimento_facial.main capturar NOME [--fotos 3]
python -m module_reconhecimento_facial.main cadastrar [--forcar]
python -m module_reconhecimento_facial.main reconhecer [FOTO_OU_PASTA] [--limiar 0.25] [--sem-anti-spoofing]
python -m module_reconhecimento_facial.main webcam [--camera 1] [--confirmacoes 3]
python -m module_reconhecimento_facial.main avaliar PASTA_ROTULADA
python -m module_reconhecimento_facial.main --help

python -m pytest                  # testes unitários (rápidos, sem baixar nada)
python -m pytest -m integracao    # ponta a ponta com o modelo real e fotos de exemplo
```

Ou pelo `make`: `make capturar NOME=...`, `make cadastrar`, `make reconhecer`, `make webcam`, `make test`,
`make test-integracao`, `make avaliar PASTA=...`.

Os parâmetros ficam em [config.py](module_reconhecimento_facial/config.py): modelo,
detector, limiar, margem, tamanho mínimo do rosto, anti-spoofing e confirmações da webcam.

## Privacidade

Fotos de rosto são dados biométricos. O `.gitignore` versiona só a estrutura das pastas
(`.gitkeep`); fotos, embeddings, resultados e o registro de acessos nunca vão para o git.

## Project Organization

```
├── Makefile           <- make cadastrar | reconhecer | webcam | avaliar | test
├── README.md
├── data
│   ├── autorizados    <- Fotos das pessoas autorizadas (uma subpasta por pessoa)
│   ├── teste          <- Fotos a verificar
│   ├── external       <- Fotos de exemplo baixadas pelo teste de integração
│   └── processed      <- Embeddings dos autorizados (cache)
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Pesos baixados pelo DeepFace (models/.deepface)
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         module_reconhecimento_facial and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   ├── resultados     <- Fotos com as caixas de acesso e relatorio.csv
│   └── registro_acessos.csv  <- Eventos da webcam (gerado em uso)
│
├── tests
│   ├── test_reconhecimento.py  <- Testes unitários (sem modelo real)
│   └── test_integracao.py      <- Ponta a ponta com o modelo real
│
└── module_reconhecimento_facial   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_reconhecimento_facial a Python module
    ├── config.py               <- Caminhos, modelo, limiar, regras de segurança e cores
    ├── dataset.py              <- Listagem, leitura e gravação das fotos
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── faces.py            <- Detecção, embeddings e anti-spoofing (DeepFace)
    │   ├── enroll.py           <- Cadastro com controle de qualidade e cache
    │   ├── recognize.py        <- Regras de decisão e confirmação da webcam
    │   └── evaluate.py         <- FAR/FRR em base rotulada para calibrar o limiar
    │
    ├── draw.py                 <- Bounding boxes verde/vermelha/amarela com OpenCV
    ├── webcam.py               <- Vídeo ao vivo e captura das fotos de cadastro
    └── main.py                 <- Linha de comando: capturar, cadastrar, reconhecer, webcam, avaliar
```

--------
