"""Avaliação do modelo treinado no conjunto de teste."""

from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow import keras
import typer

from module_cifar10.config import CLASS_NAMES, MODEL_PATH
from module_cifar10.dataset import load_data
from module_cifar10.modeling.predict import load_model, predict_classes

app = typer.Typer(add_completion=False)


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcula loss e acurácia no teste.

    Args:
        model (keras.Model): Modelo treinado.
        X_test (np.ndarray): Imagens de teste normalizadas.
        y_test (np.ndarray): Rótulos de teste.

    Returns:
        dict: Chaves "loss" e "accuracy".

    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def accuracy_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Acurácia de cada classe, para ver onde o modelo mais erra.

    Args:
        y_true (np.ndarray): Rótulos reais.
        y_pred (np.ndarray): Rótulos previstos.

    Returns:
        dict[str, float]: Nome da classe -> fração de acertos naquela classe.

    """
    return {
        name: float((y_pred[y_true == label] == label).mean())
        for label, name in enumerate(CLASS_NAMES)
        if (y_true == label).any()
    }


@app.command()
def main(model_path: Path = MODEL_PATH) -> None:
    """Avalia o modelo salvo no conjunto de teste da CIFAR-10."""
    _, (X_test, y_test) = load_data()
    model = load_model(model_path)

    metrics = evaluate_model(model, X_test, y_test)
    logger.info("Acurácia no conjunto de teste: {:.2f}%", metrics["accuracy"] * 100)
    logger.info("Loss no teste: {:.4f}", metrics["loss"])

    per_class = accuracy_per_class(y_test, predict_classes(model, X_test))
    for name, acc in sorted(per_class.items(), key=lambda item: item[1]):
        logger.info("{:<10} {:.2f}%", name, acc * 100)


if __name__ == "__main__":
    app()
