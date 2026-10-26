"""Cadastro dos autorizados: transforma as fotos em uma base de embeddings.

É o equivalente ao "treino" deste projeto. Nenhuma rede é treinada: o modelo do
DeepFace já vem pronto, e cadastrar uma pessoa é guardar os embeddings das fotos
dela. Como um cadastro errado libera a porta para a pessoa errada, cada foto
passa por um controle de qualidade antes de entrar na base.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path

from loguru import logger
import numpy as np

from module_reconhecimento_facial.config import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    EMBEDDINGS_PATH,
    MIN_FACE_SIZE,
    MODEL_NAME,
    RECOMMENDED_PHOTOS,
    THRESHOLD,
)
from module_reconhecimento_facial.dataset import list_authorized, read_image
from module_reconhecimento_facial.modeling.faces import (
    Face,
    cosine_distances,
    detect_faces,
    l2_normalize,
)


@dataclass
class AuthorizedBase:
    """Embeddings dos autorizados: a linha i de embeddings pertence a names[i]."""

    names: list[str]
    embeddings: np.ndarray
    files: list[str] = field(default_factory=list)

    # Calculados a partir dos campos acima.
    people: list[str] = field(init=False)
    templates: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.embeddings = l2_normalize(self.embeddings)
        self.people = sorted(set(self.names))

        # Template = média das fotos da pessoa. Comparar com a média é mais estável
        # que comparar com cada foto: ruídos de uma foto (luz, ângulo) se diluem, e
        # uma foto ruim isolada não vira uma porta de entrada para outra pessoa.
        names = np.array(self.names)
        self.templates = l2_normalize(
            np.vstack([self.embeddings[names == person].mean(axis=0) for person in self.people])
        )

    def photo_count(self, person: str) -> int:
        return self.names.count(person)


def pick_enrollment_face(faces: list[Face]) -> tuple[Face | None, str]:
    """Escolhe o rosto de uma foto de cadastro, ou explica por que a foto não serve.

    A foto precisa ter exatamente um rosto nítido. Com dois ou mais não há como
    saber com certeza quem é o dono da foto, e cadastrar o rosto errado daria
    acesso a outra pessoa.

    Returns:
        tuple[Face | None, str]: O rosto e "" se a foto serve; None e o motivo se não.

    """
    # Rostos minúsculos ao fundo não contam como "outra pessoa na foto".
    usable = [face for face in faces if face.size >= MIN_FACE_SIZE]

    if not usable:
        return None, "rosto pequeno demais" if faces else "nenhum rosto encontrado"

    if len(usable) > 1:
        return None, f"{len(usable)} rostos na foto; use fotos só com a pessoa"

    return usable[0], ""


def read_face(path: Path, model_name: str, detector_backend: str) -> np.ndarray | None:
    """Embedding do rosto de uma foto de cadastro, ou None se a foto não serve."""
    face, problem = pick_enrollment_face(
        detect_faces(read_image(path), model_name, detector_backend)
    )

    if face is None:
        logger.warning("{}: {}. Foto ignorada.", path.name, problem)
        return None

    return face.embedding


def consistent_photos(
    person: str, photos: list[tuple[Path, np.ndarray]], threshold: float
) -> list[tuple[Path, np.ndarray]]:
    """Remove as fotos que não se parecem com as demais fotos da mesma pessoa.

    Pega foto trocada de pasta, foto de outra pessoa com o mesmo nome de arquivo
    etc. Cada foto é comparada com a média das outras.

    Returns:
        list: Fotos aprovadas. Vazia quando não dá para saber quais estão certas.

    """
    if len(photos) < 2:
        return photos

    embeddings = np.vstack([embedding for _, embedding in photos])
    outliers = []

    for i, (path, embedding) in enumerate(photos):
        others = np.delete(embeddings, i, axis=0).mean(axis=0, keepdims=True)
        distance = float(cosine_distances(embedding, others)[0])
        if distance > threshold:
            outliers.append(path)
            logger.warning(
                "{}: {} não se parece com as outras fotos da pessoa (distância {:.3f}).",
                person,
                path.name,
                distance,
            )

    # Com 2 fotos diferentes entre si, ou com a maioria discordando, não há como
    # saber qual é a certa. Na dúvida, a pessoa fica fora até as fotos serem revistas.
    if outliers and (len(photos) == 2 or len(outliers) * 2 >= len(photos)):
        logger.error("{}: fotos inconsistentes; pessoa NÃO cadastrada. Revise as fotos.", person)
        return []

    return [(path, embedding) for path, embedding in photos if path not in outliers]


def build_base(
    directory: Path = AUTHORIZED_DIR,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    threshold: float = THRESHOLD,
) -> AuthorizedBase:
    """Calcula o embedding de cada foto aprovada no controle de qualidade.

    Args:
        directory (Path): Pasta de autorizados.
        model_name (str): Modelo de reconhecimento do DeepFace.
        detector_backend (str): Detector de rostos do DeepFace.
        threshold (float): Limiar usado para checar se as fotos de uma pessoa batem.

    Returns:
        AuthorizedBase: Base pronta para a comparação.

    """
    by_person: dict[str, list[tuple[Path, np.ndarray]]] = defaultdict(list)

    for name, path in list_authorized(directory):
        embedding = read_face(path, model_name, detector_backend)
        if embedding is not None:
            by_person[name].append((path, embedding))

    names, embeddings, files = [], [], []

    for person, photos in sorted(by_person.items()):
        approved = consistent_photos(person, photos, threshold)
        if not approved:
            continue

        if len(approved) < RECOMMENDED_PHOTOS:
            logger.warning(
                "{}: só {} foto(s) válida(s). Use pelo menos {} para um reconhecimento confiável.",
                person,
                len(approved),
                RECOMMENDED_PHOTOS,
            )

        for path, embedding in approved:
            names.append(person)
            embeddings.append(embedding)
            files.append(str(path))

        logger.info("Cadastrado: {} ({} foto(s))", person, len(approved))

    if not names:
        raise ValueError(
            f"Nenhuma pessoa cadastrada. Coloque fotos dos autorizados em {directory} "
            "(ex.: maria.jpg ou joao/01.jpg), uma pessoa por foto, rosto de frente."
        )

    return AuthorizedBase(names=names, embeddings=np.vstack(embeddings), files=files)


def base_signature(
    directory: Path,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    threshold: float = THRESHOLD,
) -> str:
    """Identifica o estado da pasta de autorizados e da configuração.

    Se qualquer foto for adicionada, removida ou alterada, se o modelo ou o limiar
    mudarem, ou se outra pasta de autorizados for usada, a assinatura muda e o
    cache deixa de valer.
    """
    photos = [
        [path.relative_to(directory).as_posix(), path.stat().st_mtime_ns, path.stat().st_size]
        for _, path in list_authorized(directory)
    ]
    return json.dumps(
        {
            "directory": str(directory.resolve()),
            "model": model_name,
            "detector": detector_backend,
            "threshold": threshold,
            "min_face_size": MIN_FACE_SIZE,
            "photos": photos,
        },
        sort_keys=True,
    )


def save_base(base: AuthorizedBase, signature: str, path: Path = EMBEDDINGS_PATH) -> None:
    """Grava a base em .npz junto da assinatura que a gerou."""
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        signature=np.array(signature),
        names=np.array(base.names),
        files=np.array(base.files),
        embeddings=base.embeddings,
    )

    logger.success("Base com {} pessoa(s) salva em {}", len(base.people), path)


def read_cache(path: Path) -> tuple[str, AuthorizedBase] | None:
    """Lê a base em cache, ou None se o arquivo não existir ou estiver corrompido."""
    if not path.exists():
        return None

    try:
        # allow_pickle=False: o arquivo só pode conter números e textos, nunca
        # objetos Python, que executariam código ao ser carregados.
        with np.load(path, allow_pickle=False) as data:
            signature = str(data["signature"])
            base = AuthorizedBase(
                names=data["names"].tolist(),
                embeddings=data["embeddings"],
                files=data["files"].tolist(),
            )
    except (OSError, KeyError, ValueError) as error:
        logger.warning("Cache inválido ({}); recalculando a base.", error)
        return None

    return signature, base


def load_base(
    directory: Path = AUTHORIZED_DIR,
    cache_path: Path = EMBEDDINGS_PATH,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    threshold: float = THRESHOLD,
    force: bool = False,
) -> AuthorizedBase:
    """Carrega a base do cache, recalculando só quando as fotos ou a configuração mudaram.

    Args:
        directory (Path): Pasta de autorizados.
        cache_path (Path): Arquivo da base em cache.
        model_name (str): Modelo de reconhecimento do DeepFace.
        detector_backend (str): Detector de rostos do DeepFace.
        threshold (float): Limiar usado no controle de qualidade do cadastro.
        force (bool): Recalcula a base mesmo com o cache válido.

    Returns:
        AuthorizedBase: Base de autorizados atualizada.

    """
    signature = base_signature(directory, model_name, detector_backend, threshold)
    cached = None if force else read_cache(cache_path)

    if cached is not None:
        cached_signature, base = cached
        if cached_signature == signature:
            logger.info("Base de autorizados carregada do cache ({})", cache_path.name)
            return base
        logger.info("Fotos de autorizados ou configuração mudaram; recalculando a base.")

    base = build_base(directory, model_name, detector_backend, threshold)
    save_base(base, signature, cache_path)
    return base
