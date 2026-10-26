"""Testes unitários do cadastro, da decisão de acesso, do desenho e da webcam.

Nenhum teste aqui baixa pesos nem usa fotos reais: os rostos são embeddings
sintéticos e o DeepFace é substituído por funções falsas quando necessário.
O teste com o modelo de verdade fica em test_integracao.py.
"""

import csv
import math
from pathlib import Path
import time

from deepface.modules.exceptions import FaceNotDetected
import numpy as np
import pytest
import typer

from module_reconhecimento_facial.config import (
    COLOR_DENIED,
    COLOR_GRANTED,
    COLOR_VERIFYING,
    MIN_FACE_SIZE,
)
from module_reconhecimento_facial.dataset import (
    list_authorized,
    list_images,
    person_name,
    read_image,
    write_image,
)
from module_reconhecimento_facial.draw import annotate, ascii_text, label_for
from module_reconhecimento_facial.main import capturar, report_rows
from module_reconhecimento_facial.modeling import enroll, evaluate, faces
from module_reconhecimento_facial.modeling.enroll import (
    AuthorizedBase,
    consistent_photos,
    load_base,
    read_face,
)
from module_reconhecimento_facial.modeling.faces import (
    Face,
    cosine_distances,
    detect_faces,
    l2_normalize,
)
from module_reconhecimento_facial.modeling.recognize import (
    ConfirmationTracker,
    Recognition,
    Status,
    decide,
)
from module_reconhecimento_facial import webcam

DIM = 8


def unit(*values) -> np.ndarray:
    """Vetor de DIM posições com os valores informados no começo, normalizado."""
    vector = np.zeros(DIM, dtype="float32")
    vector[: len(values)] = values
    return l2_normalize(vector)


def make_face(embedding, size=120, is_real=None, x=10, y=40) -> Face:
    return Face(x=x, y=y, w=size, h=size, embedding=np.asarray(embedding), is_real=is_real)


@pytest.fixture
def base() -> AuthorizedBase:
    """maria e joao, três fotos cada, em direções bem diferentes."""
    maria = [unit(1, 0.05), unit(1, -0.05), unit(1, 0, 0.05)]
    joao = [unit(0, 1, 0.05), unit(0.05, 1), unit(0, 1, -0.05)]
    return AuthorizedBase(names=["maria"] * 3 + ["joao"] * 3, embeddings=np.vstack(maria + joao))


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((240, 240, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Pastas e imagens
# ---------------------------------------------------------------------------


def touch(root, *names):
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()


def test_list_images_pega_so_imagens_inclusive_em_subpastas(tmp_path):
    touch(tmp_path, "b.jpg", "a.PNG", "notas.txt", "joao/01.jpeg")

    found = [p.relative_to(tmp_path).as_posix() for p in list_images(tmp_path)]

    assert found == ["a.PNG", "b.jpg", "joao/01.jpeg"]


def test_list_images_avisa_quando_a_pasta_nao_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "nao_existe")


def test_nome_vem_do_arquivo_ou_da_subpasta(tmp_path):
    assert person_name(tmp_path / "maria.jpg", tmp_path) == "maria"
    assert person_name(tmp_path / "joao" / "02.jpg", tmp_path) == "joao"


def test_list_authorized_agrupa_fotos_da_mesma_pessoa(tmp_path):
    touch(tmp_path, "maria.jpg", "joao/01.jpg", "joao/02.jpg")

    names = [name for name, _ in list_authorized(tmp_path)]

    assert sorted(names) == ["joao", "joao", "maria"]


def test_le_e_grava_imagem_em_caminho_com_acento(tmp_path, blank_image):
    # cv2.imread/imwrite falham com acentos no Windows; o projeto não pode.
    path = tmp_path / "fotos_ção" / "joão.png"
    blank_image[0, 0] = (1, 2, 3)

    write_image(blank_image, path)

    assert np.array_equal(read_image(path), blank_image)


def test_read_image_falha_com_arquivo_que_nao_e_imagem(tmp_path):
    path = tmp_path / "falsa.jpg"
    path.write_text("isso não é uma imagem")

    with pytest.raises(ValueError):
        read_image(path)


# ---------------------------------------------------------------------------
# Embeddings e detecção
# ---------------------------------------------------------------------------


def test_distancia_de_cosseno():
    matrix = np.vstack([unit(1), unit(0, 1), unit(-1)])

    assert cosine_distances(unit(1), matrix) == pytest.approx([0, 1, 2], abs=1e-6)


def test_detect_faces_devolve_lista_vazia_sem_rosto(monkeypatch, blank_image):
    def sem_rosto(**kwargs):
        raise FaceNotDetected("Face could not be detected.")

    monkeypatch.setattr(faces.DeepFace, "represent", sem_rosto)

    assert detect_faces(blank_image) == []


def test_detect_faces_nao_esconde_outros_erros(monkeypatch, blank_image):
    # Ex.: detector mal instalado. Tratar isso como "sem rosto" esconderia o problema.
    def detector_quebrado(**kwargs):
        raise ValueError("Confirm that opencv is installed on your environment!")

    monkeypatch.setattr(faces.DeepFace, "represent", detector_quebrado)

    with pytest.raises(ValueError, match="opencv"):
        detect_faces(blank_image)


def fake_represent(**kwargs):
    return [
        {"embedding": [3.0, 4.0], "facial_area": {"x": 1, "y": 2, "w": 90, "h": 100}},
        {"embedding": [1.0, 0.0], "facial_area": {"x": 100, "y": 2, "w": 90, "h": 100}},
    ]


def test_detect_faces_converte_e_normaliza_a_saida(monkeypatch, blank_image):
    monkeypatch.setattr(faces.DeepFace, "represent", fake_represent)

    first, _ = detect_faces(blank_image)

    assert (first.x, first.y, first.w, first.h, first.size) == (1, 2, 90, 100, 90)
    assert first.embedding == pytest.approx([0.6, 0.8])
    assert first.is_real is None  # anti-spoofing não rodou


def test_anti_spoofing_avalia_cada_rosto_separadamente(monkeypatch, blank_image):
    class FakeSpoofModel:
        def analyze(self, img, facial_area):
            # Só o rosto em x=100 é falso.
            return facial_area[0] != 100, 0.9

    monkeypatch.setattr(faces.DeepFace, "represent", fake_represent)
    monkeypatch.setattr(faces, "build_model", lambda **kwargs: FakeSpoofModel())

    real, fake = detect_faces(blank_image, anti_spoofing=True)

    assert real.is_real is True
    assert fake.is_real is False
    assert fake.spoof_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Base de autorizados
# ---------------------------------------------------------------------------


def test_template_e_a_media_normalizada_das_fotos(base):
    assert base.people == ["joao", "maria"]
    assert base.photo_count("maria") == 3
    assert np.linalg.norm(base.templates, axis=1) == pytest.approx([1, 1])
    # O template da maria aponta para a direção comum das três fotos.
    assert cosine_distances(unit(1), base.templates[[1]])[0] < 0.01


# ---------------------------------------------------------------------------
# Decisão de acesso
# ---------------------------------------------------------------------------


def test_rosto_do_autorizado_e_liberado(base):
    rec = decide(make_face(unit(1, 0.02)), base, threshold=0.3)

    assert rec.status is Status.GRANTED
    assert rec.name == "maria"
    assert rec.distance < 0.3


def test_rosto_desconhecido_e_negado(base):
    rec = decide(make_face(unit(0, 0, 1)), base, threshold=0.3)

    assert rec.status is Status.UNKNOWN
    assert rec.name is None
    # O mais parecido continua registrado, para o relatório.
    assert rec.closest in {"maria", "joao"}


def test_rosto_entre_dois_autorizados_e_negado_como_ambiguo(base):
    # Mesma distância para maria e joao: liberar para qualquer um seria chute.
    rec = decide(make_face(unit(1, 1)), base, threshold=0.5, min_margin=0.05)

    assert rec.distance <= 0.5
    assert rec.margin < 0.05
    assert rec.status is Status.AMBIGUOUS


def test_rosto_pequeno_e_negado_mesmo_parecido(base):
    rec = decide(make_face(unit(1), size=MIN_FACE_SIZE - 1), base, threshold=0.3)

    assert rec.closest == "maria"
    assert rec.status is Status.LOW_QUALITY


def test_rosto_falso_e_negado_mesmo_identico_ao_autorizado(base):
    rec = decide(make_face(unit(1), is_real=False), base, threshold=0.3)

    assert rec.closest == "maria"
    assert rec.status is Status.SPOOF


def test_rosto_real_confirmado_e_liberado(base):
    assert decide(make_face(unit(1), is_real=True), base).granted


def test_com_um_unico_autorizado_nao_ha_margem_a_checar():
    single = AuthorizedBase(names=["maria"], embeddings=unit(1)[None])

    rec = decide(make_face(unit(1)), single, threshold=0.3, min_margin=0.05)

    assert rec.granted
    assert math.isinf(rec.margin)


# ---------------------------------------------------------------------------
# Confirmação na webcam
# ---------------------------------------------------------------------------


def granted(name="maria", distance=0.1) -> Recognition:
    return Recognition(face=make_face(unit(1)), status=Status.GRANTED, closest=name, distance=distance)


def denied(status=Status.UNKNOWN, name="maria") -> Recognition:
    return Recognition(face=make_face(unit(1)), status=status, closest=name, distance=0.9)


def test_webcam_so_libera_apos_confirmacoes_seguidas():
    tracker = ConfirmationTracker(required=3)

    statuses = [tracker.update([granted()])[0].status for _ in range(4)]

    assert statuses == [Status.VERIFYING, Status.VERIFYING, Status.GRANTED, Status.GRANTED]


def test_confirmacao_e_anunciada_uma_vez():
    tracker = ConfirmationTracker(required=2)

    announced = []
    for _ in range(4):
        tracker.update([granted()])
        announced.append(tracker.newly_confirmed)

    assert announced == [[], ["maria"], [], []]


@pytest.mark.parametrize(
    "interrupcao",
    [[], [denied(Status.SPOOF)], [denied(Status.UNKNOWN)], [granted("joao")]],
    ids=["saiu_da_camera", "rosto_falso", "desconhecido", "outra_pessoa"],
)
def test_qualquer_interrupcao_zera_a_confirmacao(interrupcao):
    tracker = ConfirmationTracker(required=3)

    tracker.update([granted()])
    tracker.update([granted()])
    tracker.update(interrupcao)

    assert tracker.update([granted()])[0].status is Status.VERIFYING


def test_negados_passam_pela_confirmacao_sem_mudar():
    tracker = ConfirmationTracker(required=3)
    rec = denied(Status.SPOOF)

    assert tracker.update([rec]) == [rec]


# ---------------------------------------------------------------------------
# Cadastro
# ---------------------------------------------------------------------------


def stub_detect(monkeypatch, faces_by_call):
    """Substitui a detecção do cadastro por respostas prontas, na ordem das fotos."""
    calls = iter(faces_by_call)
    monkeypatch.setattr(enroll, "detect_faces", lambda *args, **kwargs: next(calls))


def test_foto_de_cadastro_com_duas_pessoas_e_recusada(monkeypatch, tmp_path, blank_image):
    write_image(blank_image, tmp_path / "maria.png")
    stub_detect(monkeypatch, [[make_face(unit(1)), make_face(unit(0, 1))]])

    assert read_face(tmp_path / "maria.png", "m", "d") is None


def test_rosto_minusculo_ao_fundo_nao_impede_o_cadastro(monkeypatch, tmp_path, blank_image):
    write_image(blank_image, tmp_path / "maria.png")
    stub_detect(monkeypatch, [[make_face(unit(1)), make_face(unit(0, 1), size=10)]])

    assert read_face(tmp_path / "maria.png", "m", "d") == pytest.approx(unit(1))


def test_foto_de_cadastro_com_rosto_pequeno_e_recusada(monkeypatch, tmp_path, blank_image):
    write_image(blank_image, tmp_path / "maria.png")
    stub_detect(monkeypatch, [[make_face(unit(1), size=MIN_FACE_SIZE - 1)]])

    assert read_face(tmp_path / "maria.png", "m", "d") is None


def photos(*embeddings):
    return [(Path(f"{i}.jpg"), embedding) for i, embedding in enumerate(embeddings)]


def test_foto_trocada_e_removida_do_cadastro():
    approved = consistent_photos(
        "maria", photos(unit(1, 0.05), unit(1, -0.05), unit(0, 0, 1)), threshold=0.3
    )

    assert [path.name for path, _ in approved] == ["0.jpg", "1.jpg"]


def test_duas_fotos_que_nao_batem_bloqueiam_o_cadastro():
    # Não há como saber qual das duas está certa.
    assert consistent_photos("maria", photos(unit(1), unit(0, 1)), threshold=0.3) == []


def test_maioria_das_fotos_discordando_bloqueia_o_cadastro():
    embeddings = photos(unit(1), unit(0, 1), unit(0, 0, 1), unit(0, 0, 0, 1))

    assert consistent_photos("maria", embeddings, threshold=0.3) == []


def test_build_base_so_cadastra_quem_passou_no_controle(monkeypatch, tmp_path, blank_image):
    for name in ["joao/1.png", "joao/2.png", "maria/1.png", "maria/2.png", "maria/3.png"]:
        write_image(blank_image, tmp_path / name)

    stub_detect(
        monkeypatch,
        [
            [make_face(unit(1))],  # joao/1
            [make_face(unit(0, 1))],  # joao/2: outra pessoa -> joao bloqueado
            [make_face(unit(0, 0, 1))],
            [make_face(unit(0, 0, 1, 0.05))],
            [make_face(unit(0, 0, 1, -0.05))],
        ],
    )

    built = enroll.build_base(tmp_path, threshold=0.3)

    assert built.people == ["maria"]
    assert built.photo_count("maria") == 3


def test_cadastro_vazio_explica_o_que_fazer(tmp_path):
    with pytest.raises(ValueError, match="Coloque fotos"):
        enroll.build_base(tmp_path)


def test_cache_so_recalcula_quando_as_fotos_mudam(monkeypatch, tmp_path, blank_image):
    authorized = tmp_path / "autorizados"
    cache = tmp_path / "cache.npz"
    write_image(blank_image, authorized / "maria.png")

    calls = []

    def fake_detect(*args, **kwargs):
        calls.append(1)
        return [make_face(unit(1))]

    monkeypatch.setattr(enroll, "detect_faces", fake_detect)

    assert load_base(authorized, cache).people == ["maria"]
    assert load_base(authorized, cache).people == ["maria"]
    assert len(calls) == 1  # a segunda veio do cache

    write_image(blank_image, authorized / "joao.png")
    assert load_base(authorized, cache).people == ["joao", "maria"]
    assert len(calls) == 3  # recalculou as duas fotos


def test_cache_nao_aceita_objetos_python(monkeypatch, tmp_path, blank_image):
    # Um cache adulterado com objetos (pickle) é descartado em vez de carregado.
    authorized = tmp_path / "autorizados"
    cache = tmp_path / "cache.npz"
    write_image(blank_image, authorized / "maria.png")
    np.savez(cache, signature=np.array([{"malicioso": True}], dtype=object))

    monkeypatch.setattr(enroll, "detect_faces", lambda *args, **kwargs: [make_face(unit(1))])

    assert load_base(authorized, cache).people == ["maria"]


def test_cache_corrompido_e_recalculado(monkeypatch, tmp_path, blank_image):
    authorized = tmp_path / "autorizados"
    cache = tmp_path / "cache.npz"
    write_image(blank_image, authorized / "maria.png")
    cache.write_bytes(b"lixo")

    monkeypatch.setattr(enroll, "detect_faces", lambda *args, **kwargs: [make_face(unit(1))])

    assert load_base(authorized, cache).people == ["maria"]


# ---------------------------------------------------------------------------
# Desenho e relatório
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rec, text, color",
    [
        (granted("joão"), "ACESSO LIBERADO: joão", COLOR_GRANTED),
        (denied(Status.VERIFYING), "VERIFICANDO...", COLOR_VERIFYING),
        (denied(Status.UNKNOWN), "ACESSO NEGADO", COLOR_DENIED),
        (denied(Status.SPOOF), "ACESSO NEGADO: rosto falso", COLOR_DENIED),
        (denied(Status.AMBIGUOUS), "ACESSO NEGADO: nao identificado", COLOR_DENIED),
        (denied(Status.LOW_QUALITY), "ACESSO NEGADO: aproxime-se", COLOR_DENIED),
    ],
    ids=["liberado", "verificando", "desconhecido", "falso", "ambiguo", "pequeno"],
)
def test_texto_e_cor_de_cada_decisao(rec, text, color):
    assert label_for(rec) == (text, color)


@pytest.mark.parametrize("status, color", [(Status.GRANTED, COLOR_GRANTED), (Status.UNKNOWN, COLOR_DENIED)])
def test_caixa_e_desenhada_na_cor_da_decisao(blank_image, status, color):
    rec = Recognition(face=make_face(unit(1), size=100, x=50, y=60), status=status, closest="maria")

    annotated = annotate(blank_image, [rec])

    # Pixel na borda esquerda da caixa, no meio da altura do rosto.
    assert tuple(annotated[110, 50]) == color
    # A imagem original não pode ser alterada.
    assert not blank_image.any()


def test_texto_sem_acento_para_o_opencv():
    assert ascii_text("João Conceição") == "Joao Conceicao"


def test_relatorio_registra_imagem_sem_rosto():
    assert report_rows("vazia.jpg", []) == [
        {"imagem": "vazia.jpg", "rosto": 0, "status": "sem_rosto"}
    ]


def test_relatorio_tem_uma_linha_por_rosto():
    rows = report_rows("grupo.jpg", [granted("joão"), denied(Status.SPOOF)])

    assert [row["status"] for row in rows] == ["liberado", "rosto_falso"]
    assert rows[0]["pessoa"] == "joão"
    assert rows[1]["pessoa"] == ""
    assert rows[1]["mais_parecido"] == "maria"


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_factory(monkeypatch, tmp_path, base):
    """Cria um RecognitionWorker cujo reconhecimento devolve respostas prontas."""

    def factory(responses, required=2):
        sequence = iter(responses)
        monkeypatch.setattr(webcam, "recognize", lambda *args, **kwargs: next(sequence))
        log = webcam.AccessLog(tmp_path / "registro.csv")
        return webcam.RecognitionWorker(base, 0.3, True, ConfirmationTracker(required), log)

    return factory


def read_log(worker) -> list[dict]:
    if not worker.access_log.path.exists():
        return []
    with worker.access_log.path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_worker_registra_acesso_so_quando_confirma(worker_factory, blank_image):
    worker = worker_factory([[granted()]] * 3, required=2)

    worker.analyze(blank_image)
    assert worker.results[0].status is Status.VERIFYING
    assert read_log(worker) == []

    worker.analyze(blank_image)
    worker.analyze(blank_image)
    assert worker.results[0].status is Status.GRANTED

    (entry,) = read_log(worker)
    assert (entry["evento"], entry["pessoa"]) == ("liberado", "maria")


def test_worker_registra_cada_tentativa_de_rosto_falso_uma_vez(worker_factory, blank_image):
    spoof = [denied(Status.SPOOF)]
    worker = worker_factory([spoof, spoof, [], spoof])

    for _ in range(4):
        worker.analyze(blank_image)

    events = [entry["evento"] for entry in read_log(worker)]
    assert events == ["rosto_falso", "rosto_falso"]


def test_foto_capturada_nunca_sobrescreve_outra(tmp_path):
    assert webcam.next_photo_path(tmp_path / "nova").name == "01.jpg"

    touch(tmp_path, "maria/01.jpg", "maria/03.jpg")
    assert webcam.next_photo_path(tmp_path / "maria").name == "02.jpg"


@pytest.mark.parametrize("nome", ["../fora", "a/b", "..", " "])
def test_capturar_recusa_nome_que_sairia_da_pasta(nome, tmp_path):
    with pytest.raises(typer.BadParameter):
        capturar(nome, autorizados=tmp_path)


def test_worker_em_thread_analisa_e_para(worker_factory, blank_image):
    worker = worker_factory([[granted()]] * 1000)
    worker.start()

    worker.submit(blank_image)
    deadline = time.monotonic() + 5
    while not worker.results and time.monotonic() < deadline:
        time.sleep(0.01)

    worker.stop()
    worker.join(timeout=5)

    assert worker.results
    assert not worker.is_alive()


def test_worker_repassa_erro_para_a_thread_principal(monkeypatch, tmp_path, base, blank_image):
    def broken(*args, **kwargs):
        raise RuntimeError("modelo quebrou")

    monkeypatch.setattr(webcam, "recognize", broken)
    worker = webcam.RecognitionWorker(
        base, 0.3, True, ConfirmationTracker(), webcam.AccessLog(tmp_path / "r.csv")
    )
    worker.start()
    worker.submit(blank_image)
    worker.join(timeout=5)

    assert isinstance(worker.error, RuntimeError)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------


def test_sorteio_da_avaliacao_separa_cadastro_tentativas_e_impostores(tmp_path):
    for person in ["a", "b", "c"]:
        touch(tmp_path, *[f"{person}/{i}.jpg" for i in range(5)])
    for person in ["x", "y", "z"]:
        touch(tmp_path, f"{person}/0.jpg")

    split = evaluate.split_people(tmp_path, n_enroll=3, n_people=2, n_impostors=3)

    enrolled = set(split.enroll)
    assert len(enrolled) == 2
    assert all(len(paths) == 3 for paths in split.enroll.values())
    # Foto de cadastro nunca é usada como tentativa.
    enroll_paths = {p for paths in split.enroll.values() for p in paths}
    assert not enroll_paths & {p for _, p in split.genuine}
    # Impostores nunca são pessoas cadastradas.
    assert not enrolled & {person for person, _ in split.impostor}


def test_avaliacao_conta_impostor_liberado_como_falso_aceite(tmp_path):
    split = evaluate.Split(
        enroll={"a": [tmp_path / "a1"]},
        genuine=[("a", tmp_path / "a2")],
        impostor=[("x", tmp_path / "x1"), ("y", tmp_path / "y1")],
    )
    detected = {
        str(tmp_path / "a1"): make_face(unit(1)),
        str(tmp_path / "a2"): make_face(unit(1, 0.1)),
        str(tmp_path / "x1"): make_face(unit(1, 0.2)),  # impostor parecido com "a"
        str(tmp_path / "y1"): None,  # sem rosto detectado
    }

    loose, strict = evaluate.evaluate(split, detected, np.array([0.05, 0.001]))

    assert loose["FRR"] == 0 and loose["impostores_liberados"] == 1
    assert loose["FAR"] == pytest.approx(1 / 3)
    assert strict["FAR"] == 0 and strict["FRR"] == 1
