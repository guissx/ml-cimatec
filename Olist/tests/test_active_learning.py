"""Testes da simulação de active learning."""

import numpy as np
import pandas as pd
import pytest

from module_olist.config import NUMERIC_FEATURES, TARGET
from module_olist.modeling.active_learning import run_active_learning
from module_olist.modeling.split import split_data


@pytest.fixture
def dataset() -> pd.DataFrame:
    """Base sintética com sinal: atrasos concentrados em promised_days alto."""
    rng = np.random.default_rng(0)
    n = 600

    data = pd.DataFrame({feature: rng.normal(size=n) for feature in NUMERIC_FEATURES})
    data["customer_state"] = rng.choice(["SP", "RJ", "MG"], size=n)
    data[TARGET] = (data["promised_days"] + rng.normal(scale=0.5, size=n) > 1.4).astype(int)

    # Um valor ausente, como pode acontecer em pedidos sem itens na base real.
    data.loc[0, "total_price"] = np.nan

    return data


@pytest.mark.parametrize("strategy", ["uncertainty", "random"])
def test_cada_rodada_adiciona_o_numero_de_consultas(dataset, strategy):
    X_train, X_test, y_train, y_test = split_data(dataset)

    history = run_active_learning(
        X_train, y_train, X_test, y_test,
        strategy=strategy, n_initial_labeled=40, n_queries=10, max_iterations=3,
    )

    assert history["n_labeled"].tolist() == [40, 50, 60, 70]
    assert history["pr_auc"].between(0, 1).all()


def test_estrategia_desconhecida_falha(dataset):
    X_train, X_test, y_train, y_test = split_data(dataset)

    with pytest.raises(ValueError):
        run_active_learning(
            X_train, y_train, X_test, y_test,
            strategy="foo", n_initial_labeled=40, max_iterations=1,
        )
