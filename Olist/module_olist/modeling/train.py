"""Treino dos modelos candidatos e persistência do vencedor.

Este módulo é o ponto de entrada da etapa de modelagem. Ele lê a base
intermediária já tratada, e não os CSVs brutos: o pré-processamento é
responsabilidade do main.py e não precisa ser refeito a cada novo treino.
"""

from joblib import dump
from loguru import logger
import pandas as pd

from module_olist.config import FEATURES, INTERIM_DATA_DIR, MODEL_PATH, TARGET
from module_olist.modeling.cross_validation import (
    cross_validate_models,
    select_best_model,
)
from module_olist.modeling.evaluate import evaluate_models
from module_olist.modeling.pipeline import build_pipelines, compute_scale_pos_weight
from module_olist.modeling.split import split_data

DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Treina cada modelo candidato no conjunto de treino completo.

    Args:
        X_train (pd.DataFrame): Features de treino.
        y_train (pd.Series): Target de treino.

    Returns:
        dict: Mapeamento entre o nome do modelo e o pipeline já treinado.

    """
    # O peso da classe positiva sai apenas do conjunto de treino: usar a base
    # completa aqui vazaria informação do teste para o modelo.
    pipelines = build_pipelines(compute_scale_pos_weight(y_train))

    for name, pipeline in pipelines.items():
        logger.info("Treinando {} no conjunto de treino completo...", name)
        pipeline.fit(X_train, y_train)

    return pipelines


def save_model(model, name: str, threshold: float, metrics: dict, path=MODEL_PATH) -> None:
    """Grava o modelo escolhido junto do que é necessário para usá-lo.

    O threshold vai junto de propósito: um modelo salvo sozinho é inútil aqui,
    porque a decisão "atrasa ou não" depende do ponto de corte escolhido na
    validação. Salvar os dois separados é a receita para usá-los desencontrados.

    Args:
        model: Pipeline treinado.
        name (str): Nome do modelo.
        threshold (float): Ponto de corte escolhido na validação cruzada.
        metrics (dict): Métricas obtidas no conjunto de teste.
        path (Path): Arquivo de destino.

    """
    path.parent.mkdir(parents=True, exist_ok=True)

    dump(
        {
            "model": model,
            "name": name,
            "threshold": threshold,
            "features": FEATURES,
            "metrics": metrics,
        },
        path,
    )

    logger.success("Modelo '{}' salvo em {}", name, path)


def main() -> None:
    """Executa a etapa de modelagem de ponta a ponta."""
    logger.info("Lendo a base intermediária de {}", DATASET_PATH)
    dataset = pd.read_csv(DATASET_PATH, usecols=FEATURES + [TARGET])

    X_train, X_test, y_train, y_test = split_data(dataset)
    logger.info("Treino: {} linhas | Teste: {} linhas", len(X_train), len(X_test))

    # A validação cruzada roda apenas sobre o treino, e é dela que saem tanto a
    # escolha do modelo quanto o ponto de corte. O conjunto de teste fica
    # intocado até a avaliação final.
    cv_results, thresholds = cross_validate_models(X_train, y_train)
    best_name = select_best_model(cv_results)

    models = train_models(X_train, y_train)

    logger.info("Avaliação final no conjunto de teste:")
    metrics = evaluate_models(X_test, y_test, models, thresholds)

    save_model(models[best_name], best_name, thresholds[best_name], metrics[best_name])


if __name__ == "__main__":
    main()
