"""Leitura e gravação das fotos.

As funções de imagem usam np.fromfile + cv2.imdecode em vez de cv2.imread: no
Windows o cv2.imread não abre caminhos com acento (ex.: fotos/joão.jpg) e devolve
None sem avisar.
"""

from pathlib import Path

import cv2
from loguru import logger
import numpy as np

from module_reconhecimento_facial.config import AUTHORIZED_DIR, IMAGE_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    """Lista as imagens de uma pasta e das subpastas, em ordem alfabética.

    Args:
        directory (Path): Pasta a percorrer.

    Returns:
        list[Path]: Caminhos das imagens encontradas.

    """
    if not directory.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def person_name(path: Path, root: Path = AUTHORIZED_DIR) -> str:
    """Nome da pessoa dona da foto, a partir da organização das pastas.

    autorizados/maria.jpg     -> "maria"
    autorizados/joao/01.jpg   -> "joao"

    Args:
        path (Path): Foto de um autorizado.
        root (Path): Pasta de autorizados.

    Returns:
        str: Nome da pessoa.

    """
    relative = path.relative_to(root)

    # Foto dentro de subpasta: a subpasta é a pessoa, e o arquivo é só uma das fotos dela.
    if len(relative.parts) > 1:
        return relative.parts[0]

    return path.stem


def list_authorized(directory: Path = AUTHORIZED_DIR) -> list[tuple[str, Path]]:
    """Lista as fotos dos autorizados com o nome de cada pessoa.

    Args:
        directory (Path): Pasta de autorizados.

    Returns:
        list[tuple[str, Path]]: Pares (nome, caminho da foto).

    """
    return [(person_name(path, directory), path) for path in list_images(directory)]


def read_image(path: Path) -> np.ndarray:
    """Lê uma imagem do disco em BGR, o formato usado por OpenCV e DeepFace.

    Args:
        path (Path): Caminho da imagem.

    Returns:
        np.ndarray: Imagem (altura, largura, 3) em BGR.

    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")

    return image


def write_image(image: np.ndarray, path: Path) -> None:
    """Grava uma imagem em disco, no formato indicado pela extensão.

    Args:
        image (np.ndarray): Imagem em BGR.
        path (Path): Destino.

    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Não foi possível codificar a imagem como {path.suffix}")

    buffer.tofile(path)
    logger.debug("Imagem salva em {}", path)
