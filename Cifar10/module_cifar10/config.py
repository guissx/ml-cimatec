from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Configuração da base
# ---------------------------------------------------------------------------

# Arquivo onde a base baixada pelo Keras é guardada já normalizada, para não
# depender do cache do Keras nem refazer a conversão a cada execução.
DATASET_PATH = RAW_DATA_DIR / "cifar10.npz"

# Ordem idêntica aos rótulos 0-9 da CIFAR-10.
CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]

NUM_CLASSES = len(CLASS_NAMES)

# Imagens 32x32 coloridas (RGB).
INPUT_SHAPE = (32, 32, 3)

# ---------------------------------------------------------------------------
# Configuração de modelagem
# ---------------------------------------------------------------------------

# Semente única para todo o projeto: pesos iniciais, augmentation e embaralhamento.
RANDOM_STATE = 42

# Filtros e dropout de cada bloco convolucional: 32x32 -> 16x16 -> 8x8 -> 4x4.
CONV_BLOCKS = [(32, 0.2), (64, 0.3), (128, 0.4)]

DENSE_UNITS = 128
DENSE_DROPOUT = 0.5

EPOCHS = 60
BATCH_SIZE = 64

# Fração do treino separada para validação durante o fit.
VALIDATION_SPLIT = 0.1

# EarlyStopping olha a acurácia de validação e devolve os melhores pesos.
EARLY_STOPPING_PATIENCE = 10

# ReduceLROnPlateau corta a taxa de aprendizado pela metade quando a loss estabiliza.
LR_PATIENCE = 4
LR_FACTOR = 0.5
MIN_LR = 1e-5

MODEL_PATH = MODELS_DIR / "cnn_cifar10.keras"

# O history do Keras não vai junto do .keras, então é gravado à parte para
# permitir gerar as curvas de treino depois sem treinar de novo.
HISTORY_PATH = MODELS_DIR / "history.json"
