import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Uma foto por arquivo. O nome da pessoa vem do nome do arquivo (joao.jpg) ou,
# para cadastrar várias fotos da mesma pessoa, do nome da subpasta (joao/1.jpg).
AUTHORIZED_DIR = DATA_DIR / "autorizados"

# Fotos que serão submetidas ao controle de acesso.
TEST_DIR = DATA_DIR / "teste"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "resultados"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Ambiente do DeepFace
# ---------------------------------------------------------------------------

# Precisa ser definido antes de qualquer import do deepface. O __init__ do pacote
# importa este módulo primeiro, então basta importar o deepface depois dele.

# Os pesos baixados pelo DeepFace ficam em models/.deepface, dentro do projeto,
# e não espalhados na pasta do usuário.
os.environ.setdefault("DEEPFACE_HOME", str(MODELS_DIR))

# Esconde os avisos de inicialização do TensorFlow (oneDNN, CPU etc.).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# O DeepFace imprime emojis nos logs, e o console do Windows (cp1252) derruba o
# programa com UnicodeEncodeError ao tentar exibi-los. Com errors="replace" o
# caractere vira "?" em vez de quebrar a execução.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")

# ---------------------------------------------------------------------------
# Configuração do reconhecimento
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Modelo que transforma cada rosto em um vetor (embedding). O Facenet512 tem boa
# precisão e pesos bem menores que o VGG-Face, padrão do DeepFace.
MODEL_NAME = "Facenet512"

# Detector de rostos. "yunet" é o detector por rede neural do próprio OpenCV
# (cv2.FaceDetectorYN): nas fotos de teste foi mais rápido e mais preciso que o
# Haar Cascade ("opencv"), que marcou um pedaço do fundo como rosto.
# Outras opções do DeepFace: "opencv", "mtcnn", "retinaface", "mediapipe".
DETECTOR_BACKEND = "yunet"

# ---------------------------------------------------------------------------
# Decisão de acesso
# ---------------------------------------------------------------------------

# Cada pessoa é representada pela média dos embeddings das suas fotos de cadastro
# (template). A decisão compara o rosto com esse template por distância de cosseno:
# 0 = idênticos, 1 = sem relação.

# Distância máxima até o template para liberar o acesso. Calibrado na LFW com 3
# fotos de cadastro por pessoa (ver README): é o maior limiar que manteve os
# falsos aceites em zero. Diminuir deixa o sistema mais rígido.
THRESHOLD = 0.30

# O autorizado mais parecido precisa ganhar do segundo por pelo menos esta
# distância. Sem isso, um rosto entre duas pessoas parecidas seria liberado como
# a que ficou um pouco mais perto.
MIN_MARGIN = 0.05

# Rostos menores que isso (em pixels) não têm detalhe suficiente para decidir com
# segurança e são negados como baixa qualidade. Na webcam 640x480, um rosto a
# cerca de 1,5 m da câmera tem ~100 px.
MIN_FACE_SIZE = 60

# Detecta rostos falsos (foto impressa, foto na tela do celular, vídeo) com o
# modelo MiniFASNet do DeepFace. Sem isso, mostrar uma foto do autorizado para a
# câmera liberaria o acesso.
ANTI_SPOOFING = True

# Quantidade de fotos recomendada por pessoa no cadastro.
RECOMMENDED_PHOTOS = 3

# Embeddings dos autorizados já calculados, para não reprocessar as fotos a cada
# execução. Salvo em .npz (só números e textos): um .pkl adulterado executaria
# código ao ser carregado.
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "embeddings_autorizados.npz"

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------

WEBCAM_INDEX = 0

# Na webcam o acesso só é liberado depois que a mesma pessoa é reconhecida em
# tantas análises seguidas. Um quadro isolado (borrado, de lado, no meio de um
# movimento) não basta para abrir a porta.
WEBCAM_CONFIRMATIONS = 3

# Registro de eventos da webcam: acessos confirmados e rostos falsos detectados.
ACCESS_LOG_PATH = REPORTS_DIR / "registro_acessos.csv"

# ---------------------------------------------------------------------------
# Desenho das caixas
# ---------------------------------------------------------------------------

# O OpenCV trabalha com cores em BGR, não RGB.
COLOR_GRANTED = (0, 200, 0)  # verde
COLOR_DENIED = (0, 0, 255)  # vermelho
COLOR_VERIFYING = (0, 200, 255)  # amarelo: reconhecido, aguardando confirmação
TEXT_COLOR = (255, 255, 255)

LABEL_GRANTED = "ACESSO LIBERADO"
LABEL_DENIED = "ACESSO NEGADO"
LABEL_VERIFYING = "VERIFICANDO"
