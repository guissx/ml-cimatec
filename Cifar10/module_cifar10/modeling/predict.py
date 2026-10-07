"""Inferência com o modelo treinado."""

from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img
import typer

from module_cifar10.config import CLASS_NAMES, INPUT_SHAPE, MODEL_PATH

app = typer.Typer(add_completion=False)


def load_model(path: Path = MODEL_PATH) -> keras.Model:
    """Carrega o modelo salvo pelo treino.

    Args:
        path (Path): Arquivo .keras gravado por train.save_model.

    Returns:
        keras.Model: Modelo pronto para prever.

    """
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. "
            "Rode 'python -m module_cifar10.modeling.train' antes de prever."
        )

    logger.info("Carregando modelo de {}", path)
    return keras.models.load_model(path)


def predict_classes(model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Prevê o rótulo (0-9) de cada imagem.

    Args:
        model (keras.Model): Modelo treinado.
        images (np.ndarray): Imagens normalizadas, formato (n, 32, 32, 3).

    Returns:
        np.ndarray: Índice da classe mais provável de cada imagem.

    """
    proba = model.predict(images, verbose=0)
    return np.argmax(proba, axis=1)


def load_image(path: Path) -> np.ndarray:
    """Lê uma imagem qualquer do disco no formato que a rede espera.

    Args:
        path (Path): Caminho da imagem (jpg, png...).

    Returns:
        np.ndarray: Lote de uma imagem, (1, 32, 32, 3), normalizado em [0, 1].

    """
    # A rede só conhece imagens 32x32, então qualquer foto é reduzida a esse tamanho.
    image = load_img(path, target_size=INPUT_SHAPE[:2])
    return img_to_array(image)[np.newaxis] / 255.0


@app.command()
def main(image_path: Path, model_path: Path = MODEL_PATH) -> None:
    """Classifica uma imagem do disco e mostra as três classes mais prováveis."""
    model = load_model(model_path)
    proba = model.predict(load_image(image_path), verbose=0)[0]

    for label in np.argsort(proba)[::-1][:3]:
        logger.info("{:<10} {:.2f}%", CLASS_NAMES[label], proba[label] * 100)


if __name__ == "__main__":
    app()
