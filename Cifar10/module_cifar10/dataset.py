"""Carregamento da base CIFAR-10.

A primeira execução baixa a base pelo Keras e grava uma cópia em data/raw. As
seguintes leem direto desse arquivo, sem depender da internet nem do cache do Keras.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import typer

from module_cifar10.config import DATASET_PATH

app = typer.Typer(add_completion=False)

Split = tuple[np.ndarray, np.ndarray]


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Converte os pixels de 0-255 (uint8) para o intervalo [0, 1] em float32.

    Args:
        images (np.ndarray): Imagens no formato original da base.

    Returns:
        np.ndarray: Imagens normalizadas, prontas para a rede.

    """
    return images.astype("float32") / 255.0


def download_data() -> tuple[Split, Split]:
    """Baixa a CIFAR-10 pelo Keras, sem nenhum tratamento.

    Returns:
        tuple[Split, Split]: (X_train, y_train), (X_test, y_test) em uint8.

    """
    # Import local: o TensorFlow demora a carregar e só é necessário no download.
    from tensorflow import keras

    logger.info("Baixando a CIFAR-10 pelo Keras...")
    return keras.datasets.cifar10.load_data()


def save_data(train: Split, test: Split, path: Path = DATASET_PATH) -> None:
    """Grava a base crua (uint8) em um único .npz comprimido.

    Args:
        train (Split): Imagens e rótulos de treino.
        test (Split): Imagens e rótulos de teste.
        path (Path): Arquivo de destino.

    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Salva em uint8, e não já normalizado: ocupa 4x menos espaço em disco.
    np.savez_compressed(path, X_train=train[0], y_train=train[1], X_test=test[0], y_test=test[1])

    logger.success("Base salva em {}", path)


def load_data(path: Path = DATASET_PATH) -> tuple[Split, Split]:
    """Carrega a CIFAR-10 normalizada, baixando-a se ainda não estiver em disco.

    Args:
        path (Path): Arquivo .npz com a base.

    Returns:
        tuple[Split, Split]: (X_train, y_train), (X_test, y_test), com imagens em
        [0, 1] e rótulos como vetor 1D.

    """
    if not path.exists():
        train, test = download_data()
        save_data(train, test, path)

    with np.load(path) as data:
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

    # O Keras entrega os rótulos como coluna (n, 1); a loss esparsa e a indexação
    # de CLASS_NAMES esperam um vetor (n,).
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info("Treino: {} | Teste: {}", X_train.shape, X_test.shape)

    return (normalize_images(X_train), y_train), (normalize_images(X_test), y_test)


@app.command()
def main(output_path: Path = DATASET_PATH) -> None:
    """Baixa a CIFAR-10 e grava em data/raw."""
    train, test = download_data()
    save_data(train, test, output_path)


if __name__ == "__main__":
    app()
