"""Decisão de acesso: compara cada rosto encontrado com a base de autorizados.

A regra é "na dúvida, nega". O acesso só é liberado quando todas as checagens
passam: rosto com qualidade suficiente, rosto real (não foto de foto), parecido
o bastante com um autorizado e claramente mais parecido com ele do que com
qualquer outro.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
import math

import numpy as np

from module_reconhecimento_facial.config import (
    ANTI_SPOOFING,
    DETECTOR_BACKEND,
    MIN_FACE_SIZE,
    MIN_MARGIN,
    MODEL_NAME,
    THRESHOLD,
    WEBCAM_CONFIRMATIONS,
)
from module_reconhecimento_facial.modeling.enroll import AuthorizedBase
from module_reconhecimento_facial.modeling.faces import Face, cosine_distances, detect_faces


class Status(StrEnum):
    """Resultado do controle de acesso de um rosto."""

    GRANTED = "liberado"
    # Reconhecido, mas a webcam ainda não confirmou em análises seguidas.
    VERIFYING = "verificando"
    UNKNOWN = "desconhecido"
    AMBIGUOUS = "ambiguo"
    SPOOF = "rosto_falso"
    LOW_QUALITY = "baixa_qualidade"


@dataclass
class Recognition:
    """Decisão de acesso para um rosto."""

    face: Face
    status: Status
    # Autorizado mais parecido, mesmo quando o acesso é negado: ajuda a entender
    # por que o sistema decidiu assim.
    closest: str | None = None
    distance: float = math.nan
    # Diferença de distância entre o mais parecido e o segundo mais parecido.
    margin: float = math.nan
    threshold: float = THRESHOLD

    @property
    def granted(self) -> bool:
        return self.status is Status.GRANTED

    @property
    def name(self) -> str | None:
        """Nome da pessoa reconhecida, ou None se o acesso não foi liberado."""
        return self.closest if self.granted else None


def decide(
    face: Face,
    base: AuthorizedBase,
    threshold: float = THRESHOLD,
    min_margin: float = MIN_MARGIN,
    min_face_size: int = MIN_FACE_SIZE,
) -> Recognition:
    """Compara o rosto com os templates dos autorizados e decide o acesso.

    Args:
        face (Face): Rosto a verificar.
        base (AuthorizedBase): Base de autorizados.
        threshold (float): Distância máxima até o template para liberar.
        min_margin (float): Vantagem mínima do mais parecido sobre o segundo.
        min_face_size (int): Menor lado aceito para a caixa do rosto, em pixels.

    Returns:
        Recognition: Decisão, com a pessoa mais próxima e as distâncias.

    """
    distances = cosine_distances(face.embedding, base.templates)
    order = np.argsort(distances)
    best = int(order[0])
    margin = float(distances[order[1]] - distances[best]) if len(order) > 1 else math.inf

    result = Recognition(
        face=face,
        status=Status.GRANTED,
        closest=base.people[best],
        distance=float(distances[best]),
        margin=margin,
        threshold=threshold,
    )

    # A ordem importa só para o motivo exibido: qualquer falha nega o acesso.
    if face.size < min_face_size:
        return replace(result, status=Status.LOW_QUALITY)
    if face.is_real is False:
        return replace(result, status=Status.SPOOF)
    if result.distance > threshold:
        return replace(result, status=Status.UNKNOWN)
    if margin < min_margin:
        return replace(result, status=Status.AMBIGUOUS)

    return result


def recognize(
    image: np.ndarray,
    base: AuthorizedBase,
    threshold: float = THRESHOLD,
    anti_spoofing: bool = ANTI_SPOOFING,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
) -> list[Recognition]:
    """Detecta todos os rostos da imagem e decide o acesso de cada um.

    Args:
        image (np.ndarray): Imagem em BGR.
        base (AuthorizedBase): Base de autorizados.
        threshold (float): Distância máxima até o template para liberar.
        anti_spoofing (bool): Nega rostos identificados como falsos.
        model_name (str): Precisa ser o mesmo modelo usado no cadastro.
        detector_backend (str): Detector de rostos do DeepFace.

    Returns:
        list[Recognition]: Uma decisão por rosto; vazia se não houver rostos.

    """
    faces = detect_faces(image, model_name, detector_backend, anti_spoofing=anti_spoofing)
    return [decide(face, base, threshold) for face in faces]


class ConfirmationTracker:
    """Na webcam, só libera quem foi reconhecido em várias análises seguidas.

    Um único quadro bom não basta: o acesso fica em VERIFICANDO até a mesma pessoa
    ser liberada em `required` análises consecutivas. Qualquer análise em que ela
    não é liberada (rosto falso, distância alta, saiu da câmera) zera a contagem.
    """

    def __init__(self, required: int = WEBCAM_CONFIRMATIONS) -> None:
        self.required = required
        self.streaks: dict[str, int] = {}
        # Pessoas que atingiram a confirmação na última análise (para o registro).
        self.newly_confirmed: list[str] = []

    def update(self, recognitions: list[Recognition]) -> list[Recognition]:
        """Atualiza as contagens e rebaixa para VERIFICANDO quem ainda não confirmou."""
        granted_now = {rec.closest for rec in recognitions if rec.granted}
        self.streaks = {person: self.streaks.get(person, 0) + 1 for person in granted_now}
        self.newly_confirmed = sorted(
            person for person, streak in self.streaks.items() if streak == self.required
        )

        return [
            rec
            if not rec.granted or self.streaks[rec.closest] >= self.required
            else replace(rec, status=Status.VERIFYING)
            for rec in recognitions
        ]
