"""Avaliação do reconhecimento em uma base rotulada, para calibrar o limiar.

A pasta precisa ter uma subpasta por pessoa (o formato da LFW):

    base/
    ├── Maria/  01.jpg 02.jpg 03.jpg 04.jpg
    └── Joao/   01.jpg

Protocolo, que simula o uso real do sistema:
- sorteia `n_people` pessoas com fotos suficientes e cadastra `n_enroll` fotos de cada;
- as fotos restantes delas são tentativas legítimas (o certo é liberar para a pessoa certa);
- uma foto de cada uma de `n_impostors` outras pessoas, nunca cadastradas, são
  tentativas de impostores (o certo é negar).

Métricas, para cada limiar:
- FRR (falsa rejeição): tentativas legítimas negadas. Incomoda, mas não é perigoso.
- FAR (falso aceite): impostores liberados + legítimos liberados como outra pessoa.
  É o erro perigoso, e o limiar é escolhido para mantê-lo em zero.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
from tqdm import tqdm

from module_reconhecimento_facial.config import (
    DETECTOR_BACKEND,
    MIN_MARGIN,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
)
from module_reconhecimento_facial.dataset import list_images, read_image
from module_reconhecimento_facial.modeling.enroll import AuthorizedBase
from module_reconhecimento_facial.modeling.faces import Face, detect_faces
from module_reconhecimento_facial.modeling.recognize import Status, decide


@dataclass
class Split:
    """Fotos sorteadas para a avaliação."""

    enroll: dict[str, list[Path]]
    genuine: list[tuple[str, Path]]
    impostor: list[tuple[str, Path]]


def split_people(
    directory: Path,
    n_enroll: int = 3,
    n_people: int = 100,
    n_impostors: int = 300,
    max_probes: int = 5,
    seed: int = 42,
) -> Split:
    """Sorteia quem é cadastrado, quais fotos são de cadastro e quem é impostor."""
    rng = np.random.default_rng(seed)
    photos = {
        person.name: list_images(person)
        for person in sorted(directory.iterdir())
        if person.is_dir()
    }

    eligible = [person for person, files in photos.items() if len(files) > n_enroll]
    if len(eligible) < n_people:
        raise ValueError(
            f"Só {len(eligible)} pessoas têm mais de {n_enroll} fotos; reduza --pessoas."
        )

    enrolled = set(rng.choice(eligible, size=n_people, replace=False).tolist())
    others = [person for person in photos if person not in enrolled]

    enroll, genuine = {}, []
    for person in sorted(enrolled):
        files = rng.permutation(photos[person]).tolist()
        enroll[person] = files[:n_enroll]
        genuine += [(person, path) for path in files[n_enroll : n_enroll + max_probes]]

    impostor = [
        (person, Path(rng.choice(photos[person])))
        for person in rng.choice(others, size=min(n_impostors, len(others)), replace=False)
    ]

    return Split(enroll=enroll, genuine=genuine, impostor=impostor)


def main_face(path: Path, model_name: str, detector_backend: str) -> Face | None:
    """Maior rosto da foto: nas bases públicas, a pessoa rotulada é a do centro e
    em primeiro plano, mas outras pessoas podem aparecer ao fundo."""
    faces = detect_faces(read_image(path), model_name, detector_backend)
    return max(faces, key=lambda face: face.area) if faces else None


def embed_paths(
    paths: list[Path], cache_path: Path, model_name: str, detector_backend: str
) -> dict[str, Face | None]:
    """Detecta o rosto de cada foto, reaproveitando o que já foi calculado antes."""
    faces: dict[str, Face | None] = {}

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            for key, box, embedding in zip(data["keys"], data["boxes"], data["embeddings"]):
                faces[str(key)] = None if box[2] == 0 else Face(*map(int, box), embedding)

    missing = [path for path in paths if str(path) not in faces]
    for path in tqdm(missing, desc="Calculando embeddings"):
        faces[str(path)] = main_face(path, model_name, detector_backend)

    if missing:
        keys = list(faces)
        dim = next(face.embedding.size for face in faces.values() if face is not None)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            keys=np.array(keys),
            boxes=np.array(
                [[f.x, f.y, f.w, f.h] if f else [0, 0, 0, 0] for f in faces.values()],
                dtype="int32",
            ),
            embeddings=np.vstack(
                [f.embedding if f else np.zeros(dim, "float32") for f in faces.values()]
            ),
        )

    return faces


def evaluate(
    split: Split,
    faces: dict[str, Face | None],
    thresholds: np.ndarray,
    min_margin: float = MIN_MARGIN,
) -> list[dict]:
    """FAR e FRR de cada limiar, usando a mesma função de decisão do sistema."""
    names, embeddings = [], []
    for person, paths in split.enroll.items():
        for path in paths:
            face = faces[str(path)]
            if face is not None:
                names.append(person)
                embeddings.append(face.embedding)

    base = AuthorizedBase(names=names, embeddings=np.vstack(embeddings))
    rows = []

    def outcome(path: Path, threshold: float) -> tuple[Status, str | None]:
        face = faces[str(path)]
        if face is None:
            return Status.LOW_QUALITY, None
        rec = decide(face, base, threshold=threshold, min_margin=min_margin)
        return rec.status, rec.closest

    for threshold in map(float, thresholds):
        genuine = [(person, *outcome(path, threshold)) for person, path in split.genuine]
        impostor = [outcome(path, threshold) for _, path in split.impostor]

        correct = sum(status is Status.GRANTED and closest == p for p, status, closest in genuine)
        wrong_person = sum(
            status is Status.GRANTED and closest != p for p, status, closest in genuine
        )
        impostors_in = sum(status is Status.GRANTED for status, _ in impostor)
        attempts = len(genuine) + len(impostor)

        rows.append(
            {
                "limiar": round(threshold, 3),
                "FRR": 1 - correct / len(genuine),
                "FAR": (wrong_person + impostors_in) / attempts,
                "impostores_liberados": impostors_in,
                "liberados_como_outra_pessoa": wrong_person,
            }
        )

    return rows


def run_evaluation(
    directory: Path,
    n_enroll: int = 3,
    n_people: int = 100,
    n_impostors: int = 300,
    seed: int = 42,
    min_margin: float = MIN_MARGIN,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
) -> list[dict]:
    """Sorteia, calcula os embeddings e avalia uma faixa de limiares."""
    split = split_people(directory, n_enroll, n_people, n_impostors, seed=seed)
    paths = [p for files in split.enroll.values() for p in files]
    paths += [p for _, p in split.genuine] + [p for _, p in split.impostor]

    cache = PROCESSED_DATA_DIR / f"avaliacao_{directory.name}_{model_name}_{detector_backend}.npz"
    faces = embed_paths(paths, cache, model_name, detector_backend)

    undetected = sum(faces[str(p)] is None for p in paths)
    logger.info(
        "{} pessoas cadastradas com {} foto(s), {} tentativas legítimas, {} de impostores "
        "({} fotos sem rosto detectado)",
        len(split.enroll),
        n_enroll,
        len(split.genuine),
        len(split.impostor),
        undetected,
    )

    return evaluate(split, faces, np.arange(0.15, 0.46, 0.01), min_margin)
