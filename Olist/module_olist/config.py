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
# Configuração de modelagem
# ---------------------------------------------------------------------------

# Semente única para todo o projeto: split, folds da validação cruzada e os
# próprios modelos. Centralizar evita que uma parte use 42 e outra use outro
# valor, o que tornaria os resultados irreproduzíveis sem ninguém perceber.
RANDOM_STATE = 42

# Proporção da base reservada para o teste final.
TEST_SIZE = 0.2

# Quantidade de divisões usadas na validação cruzada.
N_SPLITS = 5

TARGET = "is_late"

# customer_state é a única categórica usada. customer_city ficou de fora de
# propósito: são 4.085 cidades distintas, e a codificação one-hot criaria mais
# colunas do que o modelo consegue aproveitar com 77 mil linhas de treino.
CATEGORICAL_FEATURES = [
    "customer_state",
]

NUMERIC_FEATURES = [
    "purchase_hour",
    "purchase_day",
    "purchase_weekday",
    "purchase_month",
    "promised_days",
    "item_count",
    "seller_count",
    "total_freight",
    "total_price",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Arquivo onde o melhor modelo é gravado, junto do threshold escolhido.
MODEL_PATH = MODELS_DIR / "model.joblib"
