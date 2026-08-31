from loguru import logger

from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import load_data, save_data
from module_olist.features import create_dataset, create_features
from module_olist.modeling.evaluate import evaluate_models
from module_olist.modeling.split import split_data
from module_olist.modeling.train import train_models

ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"


def main() -> None:
    """Carrega os dados brutos, treina os modelos e avalia o resultado."""
    orders, items, customers = load_data(ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH)

    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)

    # save_data espera o diretorio de saida, nao o caminho do arquivo:
    # o nome dataset.csv e definido dentro da propria funcao.
    save_data(dataset, INTERIM_DATA_DIR)

    X_train, X_test, y_train, y_test = split_data(dataset)
    logger.info("Treino: {} linhas | Teste: {} linhas", len(X_train), len(X_test))

    trained_models = train_models(X_train, y_train, X_test, y_test)

    # train_models devolve {"nome": {"model": ..., "score": ...}} e o
    # evaluate_models espera {"nome": modelo}.
    evaluate_models(
        X_test,
        y_test,
        {name: info["model"] for name, info in trained_models.items()},
    )


if __name__ == "__main__":
    main()
