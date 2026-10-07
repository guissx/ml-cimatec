"""Testes da base, da arquitetura e do fluxo de treino e predição."""

import json

import numpy as np
import pytest

from module_cifar10.config import CLASS_NAMES, INPUT_SHAPE, NUM_CLASSES
from module_cifar10.dataset import load_data, normalize_images, save_data
from module_cifar10.modeling.evaluate import accuracy_per_class
from module_cifar10.modeling.model import build_model
from module_cifar10.modeling.predict import predict_classes
from module_cifar10.modeling.train import save_model, train_model


@pytest.fixture
def raw_split():
    """Mini base sintética no mesmo formato cru que o Keras devolve."""
    rng = np.random.default_rng(0)
    n = 32

    images = rng.integers(0, 256, size=(n, *INPUT_SHAPE), dtype=np.uint8)
    labels = rng.integers(0, NUM_CLASSES, size=(n, 1), dtype=np.uint8)

    return images, labels


def test_classes_tem_dez_nomes_unicos():
    assert NUM_CLASSES == 10
    assert len(set(CLASS_NAMES)) == 10


def test_normalize_images_vai_para_zero_um():
    images = np.array([[0, 128, 255]], dtype=np.uint8)
    normalizado = normalize_images(images)

    assert normalizado.dtype == np.float32
    assert normalizado.min() == 0.0
    assert normalizado.max() == 1.0


def test_load_data_normaliza_e_achata_os_rotulos(raw_split, tmp_path):
    path = tmp_path / "cifar10.npz"
    save_data(raw_split, raw_split, path)

    (X_train, y_train), (X_test, y_test) = load_data(path)

    assert X_train.shape == (32, *INPUT_SHAPE)
    assert X_train.max() <= 1.0
    # De (n, 1) para (n,): é o formato que a loss esparsa espera.
    assert y_train.shape == (32,)
    assert y_test.shape == (32,)


def test_modelo_tem_a_arquitetura_do_notebook():
    model = build_model()

    assert model.input_shape == (None, *INPUT_SHAPE)
    assert model.output_shape == (None, NUM_CLASSES)
    # Mesmo total de parâmetros reportado no summary do notebook.
    assert model.count_params() == 552_874


def test_saida_do_modelo_e_uma_distribuicao_de_probabilidade():
    model = build_model()
    proba = model.predict(np.zeros((4, *INPUT_SHAPE), dtype="float32"), verbose=0)

    assert proba.shape == (4, NUM_CLASSES)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_treino_salva_modelo_e_historico(raw_split, tmp_path):
    images, labels = raw_split
    model, history = train_model(normalize_images(images), labels.flatten(), epochs=1)

    model_path = tmp_path / "cnn.keras"
    history_path = tmp_path / "history.json"
    save_model(model, history, model_path, history_path)

    assert model_path.exists()
    salvo = json.loads(history_path.read_text(encoding="utf-8"))
    assert {"accuracy", "val_accuracy", "loss", "val_loss"} <= set(salvo)

    pred = predict_classes(model, normalize_images(images[:5]))
    assert pred.shape == (5,)
    assert pred.min() >= 0 and pred.max() < NUM_CLASSES


def test_accuracy_per_class_conta_acertos_por_classe():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    resultado = accuracy_per_class(y_true, y_pred)

    assert resultado[CLASS_NAMES[0]] == pytest.approx(0.5)
    assert resultado[CLASS_NAMES[1]] == pytest.approx(1.0)
    # Classes sem nenhum exemplo não aparecem, em vez de virar NaN.
    assert CLASS_NAMES[2] not in resultado
