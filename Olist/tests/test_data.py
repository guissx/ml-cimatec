"""Testes das etapas de preparação e modelagem."""

import numpy as np
import pandas as pd
import pytest

from module_olist.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from module_olist.modeling.evaluate import evaluate_at_threshold, find_best_threshold
from module_olist.modeling.pipeline import compute_scale_pos_weight, create_preprocessor
from module_olist.modeling.split import split_data


@pytest.fixture
def dataset() -> pd.DataFrame:
    """Base sintética com a mesma estrutura e desbalanceamento da base real."""
    rng = np.random.default_rng(0)
    n = 500

    data = pd.DataFrame({feature: rng.normal(size=n) for feature in NUMERIC_FEATURES})
    data["customer_state"] = rng.choice(["SP", "RJ", "MG"], size=n)

    # ~8% de positivos, como na base da Olist.
    data[TARGET] = (rng.random(n) < 0.08).astype(int)

    return data


def test_features_nao_contem_o_target():
    assert TARGET not in FEATURES


def test_features_e_a_uniao_de_numericas_e_categoricas():
    assert set(FEATURES) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_split_preserva_a_proporcao_de_classes(dataset):
    X_train, X_test, y_train, y_test = split_data(dataset)

    assert len(X_train) + len(X_test) == len(dataset)
    assert list(X_train.columns) == FEATURES

    # A estratificação deve manter a prevalência praticamente igual nos dois.
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


def test_preprocessor_converte_a_categorica_em_numerica(dataset):
    transformado = create_preprocessor().fit_transform(dataset[FEATURES])

    assert "customer_state" not in transformado.columns
    assert "customer_state_SP" in transformado.columns

    # Nenhuma coluna de texto pode sobrar: o modelo não aceitaria.
    assert transformado.select_dtypes(include="object").empty


def test_preprocessor_ignora_categoria_nova(dataset):
    preprocessor = create_preprocessor().fit(dataset[FEATURES])

    novo = dataset[FEATURES].head(1).copy()
    novo["customer_state"] = "BA"

    # handle_unknown="ignore": vira uma linha de zeros em vez de erro.
    transformado = preprocessor.transform(novo)
    assert transformado.filter(like="customer_state_").to_numpy().sum() == 0


def test_scale_pos_weight_e_a_razao_entre_as_classes():
    y = pd.Series([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_scale_pos_weight_falha_sem_classe_positiva():
    with pytest.raises(ValueError):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))


def test_find_best_threshold_acha_o_corte_de_um_modelo_perfeito():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.2, 0.9, 0.8])

    melhor = find_best_threshold(y_true, y_proba)

    assert melhor["f1"] == pytest.approx(1.0)
    assert 0.2 < melhor["threshold"] <= 0.8


def test_evaluate_at_threshold_usa_o_corte_informado():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.7, 0.9])

    # Em 0.65 o segundo pedido deixa de ser apontado como atraso.
    metricas = evaluate_at_threshold(y_true, y_proba, threshold=0.65)

    assert metricas["threshold"] == 0.65
    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["recall"] == pytest.approx(1.0)
