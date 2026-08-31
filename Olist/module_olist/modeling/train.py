import pandas as pd

from module_olist.modeling.pipeline import (
    compute_scale_pos_weight,
    create_gradient_boosting_pipeline,
    create_lgbm_pipeline,
    create_xgb_pipeline,
)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Treina e avalia os modelos de previsão de atraso na entrega.

    Args:
        X_train (pd.DataFrame): Features de treino.
        y_train (pd.Series): Target de treino.
        X_test (pd.DataFrame): Features de teste.
        y_test (pd.Series): Target de teste.

    Returns:
        dict: Dicionário com os modelos treinados e suas métricas de avaliação.

    """
    # O peso da classe positiva sai apenas do conjunto de treino: usar a base
    # completa aqui vazaria informação do teste para o modelo.
    scale_pos_weight = compute_scale_pos_weight(y_train)

    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "LightGBM": create_lgbm_pipeline(scale_pos_weight),
        "XGBoost": create_xgb_pipeline(scale_pos_weight),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        trained_models[name] = {
            "model": model,
            "score": score,
        }

    return trained_models
