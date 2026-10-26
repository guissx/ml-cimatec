"""Desenho das bounding boxes: verde para acesso liberado, vermelho para negado.

Na webcam existe também o amarelo: a pessoa foi reconhecida, mas o acesso ainda
aguarda a confirmação em análises seguidas.
"""

import unicodedata

import cv2
import numpy as np

from module_reconhecimento_facial.config import (
    COLOR_DENIED,
    COLOR_GRANTED,
    COLOR_VERIFYING,
    LABEL_DENIED,
    LABEL_GRANTED,
    LABEL_VERIFYING,
    TEXT_COLOR,
)
from module_reconhecimento_facial.modeling.recognize import Recognition, Status

FONT = cv2.FONT_HERSHEY_SIMPLEX


def ascii_text(text: str) -> str:
    """Remove acentos: as fontes do OpenCV só desenham ASCII ("João" viraria "Jo??o")."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


# Motivo exibido junto do ACESSO NEGADO. Desconhecido fica sem motivo de propósito:
# dizer "não cadastrado" a um estranho não ajuda ninguém além dele.
DENIAL_REASONS = {
    Status.UNKNOWN: "",
    Status.SPOOF: "rosto falso",
    Status.AMBIGUOUS: "nao identificado",
    Status.LOW_QUALITY: "aproxime-se",
}


def label_for(recognition: Recognition) -> tuple[str, tuple[int, int, int]]:
    """Texto e cor da caixa de um rosto."""
    if recognition.status is Status.GRANTED:
        return f"{LABEL_GRANTED}: {recognition.name}", COLOR_GRANTED

    if recognition.status is Status.VERIFYING:
        return f"{LABEL_VERIFYING}...", COLOR_VERIFYING

    reason = DENIAL_REASONS[recognition.status]
    return (f"{LABEL_DENIED}: {reason}" if reason else LABEL_DENIED), COLOR_DENIED


def draw_recognition(image: np.ndarray, recognition: Recognition) -> None:
    """Desenha a caixa e a etiqueta de um rosto, alterando a própria imagem.

    Args:
        image (np.ndarray): Imagem em BGR.
        recognition (Recognition): Rosto e decisão de acesso.

    """
    text, color = label_for(recognition)
    text = ascii_text(text)
    face = recognition.face
    width = image.shape[1]

    # Linha e fonte crescem com a imagem: em uma foto 4000px, a caixa fina e o
    # texto pequeno de uma foto 640px ficariam invisíveis.
    scale = max(0.6, width / 1000)
    thickness = max(2, round(width / 350))
    text_thickness = max(2, thickness // 2)

    cv2.rectangle(image, (face.x, face.y), (face.x + face.w, face.y + face.h), color, thickness)

    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, text_thickness)
    pad = thickness * 2
    label_h = text_h + baseline + pad

    # Etiqueta logo acima da caixa; se o rosto encostar no topo da foto, vai para dentro.
    top = face.y - label_h if face.y - label_h >= 0 else face.y
    # Etiquetas compridas não podem sair pela direita da imagem.
    left = max(0, min(face.x, width - text_w - pad))

    cv2.rectangle(image, (left, top), (left + text_w + pad, top + label_h), color, cv2.FILLED)
    cv2.putText(
        image,
        text,
        (left + pad // 2, top + text_h + pad // 2),
        FONT,
        scale,
        TEXT_COLOR,
        text_thickness,
        cv2.LINE_AA,
    )


def annotate(image: np.ndarray, recognitions: list[Recognition]) -> np.ndarray:
    """Cópia da imagem com a caixa de todos os rostos.

    Args:
        image (np.ndarray): Imagem original em BGR (não é alterada).
        recognitions (list[Recognition]): Decisões de acesso.

    Returns:
        np.ndarray: Nova imagem anotada.

    """
    annotated = image.copy()

    for recognition in recognitions:
        draw_recognition(annotated, recognition)

    return annotated
