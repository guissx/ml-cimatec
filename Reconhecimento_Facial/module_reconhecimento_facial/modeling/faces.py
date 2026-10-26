"""Detecção de rostos, embeddings e anti-spoofing com o DeepFace.

Embedding é o vetor numérico que o modelo gera para cada rosto. Fotos da mesma
pessoa geram vetores próximos, e é essa proximidade que decide o acesso.
"""

from dataclasses import dataclass

from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
from deepface.modules.modeling import build_model
import numpy as np

from module_reconhecimento_facial.config import DETECTOR_BACKEND, MODEL_NAME


@dataclass
class Face:
    """Um rosto encontrado na imagem."""

    # Caixa do rosto em pixels: canto superior esquerdo (x, y), largura e altura.
    x: int
    y: int
    w: int
    h: int
    embedding: np.ndarray
    # Resultado do anti-spoofing: None quando ele não foi executado.
    is_real: bool | None = None
    spoof_score: float | None = None

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def size(self) -> int:
        """Menor lado da caixa: é o que limita o detalhe disponível do rosto."""
        return min(self.w, self.h)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Deixa cada vetor (ou cada linha de uma matriz) com comprimento 1."""
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def cosine_distances(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Distância de cosseno entre um vetor e cada linha da matriz.

    0 = mesma direção (rostos idênticos), 1 = sem relação, 2 = opostos.
    """
    return 1.0 - l2_normalize(matrix) @ l2_normalize(vector)


def check_liveness(image: np.ndarray, face: Face) -> None:
    """Preenche is_real e spoof_score do rosto com o modelo MiniFASNet.

    O modelo olha a região ao redor do rosto em busca de sinais de reprodução:
    borda de celular, reflexo de tela, textura de papel, moiré.
    """
    # build_model guarda o modelo em memória: só a primeira chamada carrega os pesos.
    model = build_model(task="spoofing", model_name="Fasnet")
    is_real, score = model.analyze(img=image, facial_area=(face.x, face.y, face.w, face.h))
    face.is_real = bool(is_real)
    face.spoof_score = float(score)


def detect_faces(
    image: np.ndarray,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    anti_spoofing: bool = False,
) -> list[Face]:
    """Encontra todos os rostos da imagem e calcula o embedding de cada um.

    Args:
        image (np.ndarray): Imagem em BGR.
        model_name (str): Modelo de reconhecimento do DeepFace.
        detector_backend (str): Detector de rostos do DeepFace.
        anti_spoofing (bool): Verifica também se cada rosto é real.

    Returns:
        list[Face]: Rostos encontrados; lista vazia se não houver nenhum.

    """
    try:
        # enforce_detection=True faz o DeepFace lançar FaceNotDetected quando não
        # acha rosto. Com False ele devolveria a imagem inteira como se fosse um
        # rosto, e o sistema desenharia uma caixa na foto toda.
        #
        # O anti-spoofing não é ligado aqui de propósito: dentro do represent, um
        # único rosto falso derruba a imagem inteira. Ele roda depois, rosto a rosto.
        representations = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
        )
    except FaceNotDetected:
        # Só "nenhum rosto" vira lista vazia. Outros erros (detector mal instalado,
        # imagem inválida) precisam aparecer, senão parecem foto sem rosto.
        return []

    faces = [
        Face(
            x=rep["facial_area"]["x"],
            y=rep["facial_area"]["y"],
            w=rep["facial_area"]["w"],
            h=rep["facial_area"]["h"],
            embedding=l2_normalize(rep["embedding"]),
        )
        for rep in representations
    ]

    if anti_spoofing:
        for face in faces:
            check_liveness(image, face)

    return faces
