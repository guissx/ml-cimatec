"""Escolha do ponto de corte e avaliação dos modelos.

A separação entre as duas funções deste módulo é proposital.

find_best_threshold varre vários pontos de corte e devolve o melhor. Ela deve
receber probabilidades de validação, nunca do teste: escolher o corte olhando o
teste e depois reportar o resultado nesse mesmo teste produz uma métrica
otimista, porque o número reportado já foi maximizado naqueles dados.

evaluate_at_threshold aplica um corte já decidido e apenas mede. É essa que
recebe o conjunto de teste, uma única vez.
"""

from loguru import logger
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Faixa de busca do ponto de corte. Precisa cobrir valores acima de 0.5 porque
# os modelos com scale_pos_weight deslocam as probabilidades para cima, e o
# corte ótimo deles costuma ficar entre 0.6 e 0.7.
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.01)


def find_best_threshold(y_true, y_proba) -> dict:
    """Encontra o ponto de corte que maximiza o F1.

    Args:
        y_true: Rótulos verdadeiros do conjunto de validação.
        y_proba: Probabilidade da classe positiva no conjunto de validação.

    Returns:
        dict: Threshold escolhido e as métricas obtidas com ele na validação.

    """
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    for threshold in THRESHOLD_GRID:
        y_pred = (y_proba >= threshold).astype(int)

        # zero_division=0 evita o UndefinedMetricWarning nos cortes altos, em
        # que o modelo pode não classificar nenhum pedido como atrasado.
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }

    return best


def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    """Mede o desempenho usando um ponto de corte já definido.

    Args:
        y_true: Rótulos verdadeiros do conjunto avaliado.
        y_proba: Probabilidade da classe positiva.
        threshold (float): Ponto de corte escolhido na validação.

    Returns:
        dict: Métricas no corte informado, mais as duas métricas que independem
            de threshold (pr_auc e roc_auc).

    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_models(X_test, y_test, models: dict, thresholds: dict) -> dict:
    """Avalia cada modelo treinado no conjunto de teste.

    Args:
        X_test: Features do conjunto de teste.
        y_test: Rótulos verdadeiros do conjunto de teste.
        models (dict): Mapeamento entre nome e modelo treinado.
        thresholds (dict): Ponto de corte de cada modelo, escolhido na validação.

    Returns:
        dict: Mapeamento entre o nome do modelo e suas métricas de teste.

    """
    metrics = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate_at_threshold(y_test, y_proba, thresholds[name])

        logger.info(
            "{} | threshold: {:.2f} | precision: {:.4f} | recall: {:.4f} | "
            "f1: {:.4f} | pr_auc: {:.4f} | roc_auc: {:.4f}",
            name,
            metrics[name]["threshold"],
            metrics[name]["precision"],
            metrics[name]["recall"],
            metrics[name]["f1"],
            metrics[name]["pr_auc"],
            metrics[name]["roc_auc"],
        )

    return metrics
