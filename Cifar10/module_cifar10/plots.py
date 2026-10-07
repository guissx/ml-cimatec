"""Figuras do projeto: exemplos da base, predições e curvas de treino."""

import json
from pathlib import Path

from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer

from module_cifar10.config import CLASS_NAMES, FIGURES_DIR, HISTORY_PATH, MODEL_PATH

app = typer.Typer(add_completion=False)


def plot_samples(images: np.ndarray, labels: np.ndarray, n: int = 12) -> plt.Figure:
    """Grade com as primeiras imagens da base e seus rótulos."""
    fig = plt.figure(figsize=(10, 10))

    for i in range(n):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        plt.title(CLASS_NAMES[labels[i]])
        plt.axis("off")

    plt.suptitle("Exemplos da base CIFAR-10.")
    plt.tight_layout()
    return fig


def plot_predictions(
    images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n: int = 9
) -> plt.Figure:
    """Grade com real x predito: título verde quando acerta, vermelho quando erra."""
    fig = plt.figure(figsize=(10, 10))

    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        real_class = CLASS_NAMES[y_true[i]]
        pred_class = CLASS_NAMES[y_pred[i]]
        cor = "green" if real_class == pred_class else "red"
        plt.title(f"Real: {real_class}\nPredito: {pred_class}", color=cor)
        plt.axis("off")

    plt.tight_layout()
    return fig


def plot_history(history: dict) -> tuple[plt.Figure, plt.Figure]:
    """Curvas de acurácia e de loss por época, treino x validação."""
    fig_acc = plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Treino")
    plt.plot(history["val_accuracy"], label="Validação")
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    fig_loss = plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Treino")
    plt.plot(history["val_loss"], label="Validação")
    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    return fig_acc, fig_loss


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Grava a figura em disco e libera a memória."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva em {}", path)


@app.command()
def main(output_dir: Path = FIGURES_DIR) -> None:
    """Gera todas as figuras a partir da base, do modelo e do histórico salvos."""
    # Pela linha de comando só gravamos arquivos, sem abrir janelas.
    matplotlib.use("Agg")

    # Imports locais: carregam o TensorFlow, desnecessário para quem só quer as funções.
    from module_cifar10.dataset import load_data
    from module_cifar10.modeling.predict import load_model, predict_classes

    (X_train, y_train), (X_test, y_test) = load_data()
    save_figure(plot_samples(X_train, y_train), output_dir / "samples.png")

    if MODEL_PATH.exists():
        y_pred = predict_classes(load_model(), X_test[:9])
        save_figure(plot_predictions(X_test, y_test, y_pred), output_dir / "predictions.png")
    else:
        logger.warning("Modelo não encontrado; pulando a figura de predições.")

    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        fig_acc, fig_loss = plot_history(history)
        save_figure(fig_acc, output_dir / "accuracy.png")
        save_figure(fig_loss, output_dir / "loss.png")
    else:
        logger.warning("Histórico não encontrado; pulando as curvas de treino.")

    logger.success("Figuras geradas em {}", output_dir)


if __name__ == "__main__":
    app()
