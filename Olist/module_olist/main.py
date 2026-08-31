"""Etapa de preparação dos dados: dos CSVs brutos até a base intermediária.

O treino fica em module_olist/modeling/train.py, e não aqui. Separar as duas
etapas evita reprocessar os arquivos brutos toda vez que se quer apenas treinar
os modelos de novo com outros hiperparâmetros.
"""

from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import load_data, save_data
from module_olist.features import create_dataset, create_features

ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"


def main() -> None:
    """Carrega os dados brutos, aplica os tratamentos e salva a base intermediária."""
    orders, items, customers = load_data(ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH)

    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)

    # save_data espera o diretório de saída, não o caminho do arquivo: o nome
    # dataset.csv é definido dentro da própria função.
    save_data(dataset, INTERIM_DATA_DIR)


if __name__ == "__main__":
    main()
