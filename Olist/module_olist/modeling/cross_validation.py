"""Validação cruzada dos modelos candidatos.

O split simples treino/teste dá uma única estimativa de desempenho, que depende
de quais linhas caíram em cada lado do sorteio. A validação cruzada repete a
medição em k divisões diferentes e devolve média e desvio-padrão, mostrando não
só quanto o modelo acerta, mas o quanto esse número é estável.

Além das métricas, este módulo devolve as probabilidades out-of-fold: a
previsão de cada linha feita pelo fold em que ela ficou de fora do treino.
Isso dá uma previsão honesta para a base de treino inteira, e é sobre ela que
o ponto de corte é escolhido — sem encostar no conjunto de teste.
"""

from time import perf_counter

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from module_olist.config import N_SPLITS, RANDOM_STATE
from module_olist.modeling.evaluate import find_best_threshold
from module_olist.modeling.pipeline import build_pipelines, compute_scale_pos_weight


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
) -> tuple[pd.DataFrame, dict]:
    """Avalia cada modelo candidato com validação cruzada estratificada.

    Args:
        X (pd.DataFrame): Features do conjunto de treino.
        y (pd.Series): Variável-alvo do conjunto de treino.
        n_splits (int): Quantidade de divisões (folds).

    Returns:
        tuple: DataFrame com uma linha por modelo, ordenado da maior para a
            menor pr_auc, e dicionário com o ponto de corte escolhido para
            cada modelo.

    """
    # StratifiedKFold, e não KFold: com 8% de positivos, um sorteio sem
    # estratificação pode produzir folds com proporções bem diferentes de
    # atraso, e a variação entre folds passaria a medir o sorteio, não o modelo.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # O peso é calculado uma vez sobre o conjunto de treino inteiro. Como o
    # StratifiedKFold mantém a mesma proporção de classes em todos os folds,
    # esse valor é praticamente idêntico ao que seria calculado dentro de cada
    # fold — não há vazamento relevante aqui.
    pipelines = build_pipelines(compute_scale_pos_weight(y))

    # Um modelo que chuta ao acaso tem pr_auc igual à prevalência da classe
    # positiva. É essa a régua para julgar se os números abaixo são bons.
    logger.info(
        "Prevalência da classe positiva: {:.2%} (é a pr_auc de um modelo aleatório)",
        y.mean(),
    )

    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    rows = []
    thresholds = {}

    for name, pipeline in pipelines.items():
        logger.info("Validando {} em {} folds...", name, n_splits)

        # Guarda a previsão de cada linha feita pelo fold que não a treinou.
        oof_proba = np.zeros(len(y))

        fold_pr_auc = []
        fold_roc_auc = []
        fold_train_pr_auc = []
        fold_times = []

        for train_idx, valid_idx in cv.split(X, y):
            inicio = perf_counter()
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_times.append(perf_counter() - inicio)

            valid_proba = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
            oof_proba[valid_idx] = valid_proba

            fold_pr_auc.append(average_precision_score(y.iloc[valid_idx], valid_proba))
            fold_roc_auc.append(roc_auc_score(y.iloc[valid_idx], valid_proba))

            # O desempenho no próprio fold de treino serve de comparação: uma
            # diferença grande em relação à validação indica overfitting.
            train_proba = pipeline.predict_proba(X.iloc[train_idx])[:, 1]
            fold_train_pr_auc.append(
                average_precision_score(y.iloc[train_idx], train_proba)
            )

        # O ponto de corte sai das probabilidades out-of-fold, que cobrem a base
        # de treino inteira sem que nenhuma linha tenha sido vista pelo modelo
        # que a previu.
        melhor = find_best_threshold(y, oof_proba)
        thresholds[name] = melhor["threshold"]

        rows.append(
            {
                "modelo": name,
                "pr_auc": float(np.mean(fold_pr_auc)),
                "pr_auc_std": float(np.std(fold_pr_auc)),
                "roc_auc": float(np.mean(fold_roc_auc)),
                "pr_auc_treino": float(np.mean(fold_train_pr_auc)),
                "overfit_gap": float(np.mean(fold_train_pr_auc) - np.mean(fold_pr_auc)),
                "threshold": melhor["threshold"],
                "f1": melhor["f1"],
                "precision": melhor["precision"],
                "recall": melhor["recall"],
                "fit_time_s": float(np.mean(fold_times)),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )

    formato = "{:.4f}".format

    logger.info(
        "Validação cruzada ({} folds) — métricas independentes de threshold:\n{}",
        n_splits,
        results[
            ["modelo", "pr_auc", "pr_auc_std", "roc_auc", "pr_auc_treino",
             "overfit_gap", "fit_time_s"]
        ].to_string(index=False, float_format=formato),
    )

    # Agora estes três números são comparáveis entre modelos: cada um está no
    # seu próprio ponto de corte ótimo, medido out-of-fold.
    logger.info(
        "Ponto de corte escolhido out-of-fold:\n{}",
        results[["modelo", "threshold", "f1", "precision", "recall"]].to_string(
            index=False, float_format=formato
        ),
    )

    return results, thresholds


def select_best_model(results: pd.DataFrame) -> str:
    """Retorna o nome do modelo com a maior pr_auc média.

    Args:
        results (pd.DataFrame): Saída de cross_validate_models.

    Returns:
        str: Nome do modelo vencedor.

    """
    best = results.iloc[0]

    # Quando a diferença entre o primeiro e o segundo colocado é menor que o
    # desvio-padrão entre folds, os dois estão empatados: a ordem é ruído do
    # sorteio, não superioridade de um sobre o outro.
    if len(results) > 1:
        diferenca = best["pr_auc"] - results.iloc[1]["pr_auc"]

        if diferenca < best["pr_auc_std"]:
            logger.warning(
                "{} e {} estão empatados: diferença de {:.4f} contra desvio de "
                "{:.4f} entre folds.",
                best["modelo"],
                results.iloc[1]["modelo"],
                diferenca,
                best["pr_auc_std"],
            )

    logger.success(
        "Melhor modelo na validação cruzada: {} (pr_auc {:.4f} +/- {:.4f})",
        best["modelo"],
        best["pr_auc"],
        best["pr_auc_std"],
    )

    return str(best["modelo"])
