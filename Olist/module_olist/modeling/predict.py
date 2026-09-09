"""Aplicação do modelo treinado a pedidos com as features já calculadas.

Este módulo é o ponto mais baixo da cadeia de inferência: recebe as features no
formato em que o modelo foi treinado e devolve a probabilidade e a decisão.
Quem parte de um pedido cru — datas, valores e estado — usa o inference.py, que
deriva as features e chama estas funções.
"""

from pathlib import Path

from joblib import load
from loguru import logger
import pandas as pd
import typer

from module_olist.config import MODEL_PATH, PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)


def load_model(path: Path = MODEL_PATH) -> dict:
    """Carrega o modelo salvo junto do seu ponto de corte.

    Args:
        path (Path): Arquivo gravado por train.save_model.

    Returns:
        dict: Modelo, nome, threshold, lista de features e métricas de teste.

    """
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. "
            "Rode 'python -m module_olist.modeling.train' antes de prever."
        )

    artefato = load(path)

    logger.info(
        "Modelo '{}' carregado (threshold {:.2f})",
        artefato["name"],
        artefato["threshold"],
    )

    return artefato


def predict(data: pd.DataFrame, artefato: dict | None = None) -> pd.DataFrame:
    """Prevê a probabilidade de atraso de cada pedido.

    Args:
        data (pd.DataFrame): Pedidos a prever, com as colunas de features.
        artefato (dict | None): Saída de load_model. Carregado do disco se omitido.

    Returns:
        pd.DataFrame: Probabilidade de atraso e a decisão no ponto de corte salvo.

    """
    artefato = artefato or load_model()

    features = artefato["features"]
    faltando = [column for column in features if column not in data.columns]

    if faltando:
        raise ValueError(f"Colunas ausentes na base de entrada: {faltando}")

    # O threshold vem junto do modelo de propósito: a decisão "atrasa ou não"
    # depende do ponto de corte escolhido na validação cruzada, e usar o padrão
    # de 0.5 aqui produziria uma decisão diferente da que foi avaliada.
    proba = artefato["model"].predict_proba(data[features])[:, 1]

    return pd.DataFrame(
        {
            "late_probability": proba,
            "is_late_predicted": (proba >= artefato["threshold"]).astype(int),
        },
        index=data.index,
    )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    output_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    model_path: Path = MODEL_PATH,
) -> None:
    """Lê uma base de pedidos, prevê o atraso e grava o resultado em CSV."""
    data = pd.read_csv(input_path)
    predictions = predict(data, load_model(model_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    logger.success(
        "{} previsões gravadas em {} ({} apontadas como atraso)",
        len(predictions),
        output_path,
        int(predictions["is_late_predicted"].sum()),
    )


if __name__ == "__main__":
    app()
