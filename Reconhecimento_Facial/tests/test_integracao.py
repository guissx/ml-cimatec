"""Teste ponta a ponta com o modelo real e fotos de verdade.

Usa as fotos públicas do repositório do DeepFace (baixadas uma vez para
data/external/amostras_deepface) e os pesos reais do Facenet512, YuNet e
MiniFASNet. Por isso fica fora da execução padrão:

    python -m pytest -m integracao
"""

from pathlib import Path
import shutil
from urllib.request import urlopen

import pytest

from module_reconhecimento_facial.config import DATA_DIR
from module_reconhecimento_facial.dataset import read_image
from module_reconhecimento_facial.modeling.enroll import load_base
from module_reconhecimento_facial.modeling.recognize import Status, recognize

pytestmark = pytest.mark.integracao

SAMPLES_URL = "https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset"
SAMPLES_DIR = DATA_DIR / "external" / "amostras_deepface"

# img1, img2 e img11 são da mesma pessoa; img3 e img9 são de outras pessoas;
# couple.jpg tem a pessoa cadastrada e um homem não cadastrado.
ENROLL = ["img1.jpg", "img2.jpg", "img11.jpg"]
STRANGERS = ["img3.jpg", "img9.jpg"]
COUPLE = "couple.jpg"


def sample(name: str) -> Path:
    path = SAMPLES_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(f"{SAMPLES_URL}/{name}", timeout=30) as response:
                path.write_bytes(response.read())
        except OSError as error:
            pytest.skip(f"Sem acesso às fotos de exemplo: {error}")
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    authorized = tmp_path_factory.mktemp("autorizados") / "pessoa_a"
    authorized.mkdir()
    for name in ENROLL:
        shutil.copy(sample(name), authorized / name)

    return load_base(authorized.parent, tmp_path_factory.mktemp("cache") / "base.npz")


def test_cadastro_aceita_as_tres_fotos(base):
    assert base.people == ["pessoa_a"]
    assert base.photo_count("pessoa_a") == 3


@pytest.mark.parametrize("name", STRANGERS)
def test_estranho_e_negado(base, name):
    recognitions = recognize(read_image(sample(name)), base)

    assert recognitions, "nenhum rosto detectado"
    assert not any(rec.granted for rec in recognitions)


def test_foto_com_autorizado_e_estranho(base):
    recognitions = recognize(read_image(sample(COUPLE)), base)
    statuses = sorted(rec.status for rec in recognitions)

    assert len(recognitions) == 2
    # Um liberado (a pessoa cadastrada) e um negado (o homem ao lado).
    assert statuses.count(Status.GRANTED) == 1
    granted = next(rec for rec in recognitions if rec.granted)
    assert granted.name == "pessoa_a"


def test_fotos_reais_passam_no_anti_spoofing(base):
    recognitions = recognize(read_image(sample(COUPLE)), base, anti_spoofing=True)

    assert all(rec.face.is_real for rec in recognitions)
