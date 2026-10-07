"""Treino da CNN e persistência do modelo e do histórico."""

import json
from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import typer

from module_cifar10.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HISTORY_PATH,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    MODEL_PATH,
    RANDOM_STATE,
    VALIDATION_SPLIT,
)
from module_cifar10.dataset import load_data
from module_cifar10.modeling.model import build_model

app = typer.Typer(add_completion=False)


def build_callbacks() -> list[keras.callbacks.Callback]:
    """Callbacks que controlam o fim do treino e a taxa de aprendizado.

    Returns:
        list[keras.callbacks.Callback]: EarlyStopping e ReduceLROnPlateau.

    """
    return [
        # Para quando a acurácia de validação não melhora por várias épocas e
        # devolve os pesos da melhor época, não os da última.
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        # Reduz o passo do otimizador quando a loss estaciona, para refinar os pesos.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
        ),
    ]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[keras.Sequential, dict]:
    """Treina a CNN do zero.

    Args:
        X_train (np.ndarray): Imagens de treino normalizadas.
        y_train (np.ndarray): Rótulos de treino (0-9).
        epochs (int): Máximo de épocas; o EarlyStopping pode parar antes.
        batch_size (int): Tamanho do lote.

    Returns:
        tuple[keras.Sequential, dict]: Modelo treinado e o history do fit.

    """
    keras.utils.set_random_seed(RANDOM_STATE)

    model = build_model()
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_callbacks(),
    )

    return model, history.history


def save_model(
    model: keras.Sequential,
    history: dict,
    model_path: Path = MODEL_PATH,
    history_path: Path = HISTORY_PATH,
) -> None:
    """Grava o modelo em .keras e o histórico de treino em JSON.

    Args:
        model (keras.Sequential): Modelo treinado.
        history (dict): Métricas por época devolvidas pelo fit.
        model_path (Path): Destino do modelo.
        history_path (Path): Destino do histórico.

    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    # Os valores vêm como float32 do NumPy, que o json não sabe serializar.
    serializable = {metric: [float(v) for v in values] for metric, values in history.items()}
    history_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    logger.success("Modelo salvo em {} e histórico em {}", model_path, history_path)


@app.command()
def main(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
    """Treina a CNN na CIFAR-10 e grava o resultado em models/."""
    (X_train, y_train), _ = load_data()

    model, history = train_model(X_train, y_train, epochs, batch_size)

    best = int(np.argmax(history["val_accuracy"]))
    logger.info("Melhor época: {} | val_accuracy {:.4f}", best + 1, history["val_accuracy"][best])

    save_model(model, history)


if __name__ == "__main__":
    app()
