"""Executa a atividade Keras e verifica as previsões e o cálculo da perda."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def activity():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "atividade_tensorflow.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{path}:cell-{index}", "exec"), namespace)
    return namespace


def test_forward_pass_produces_one_probability_per_order(activity):
    probabilities = activity["y_pred"].numpy()
    assert probabilities.shape == activity["y"].shape == (6, 1)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(
        activity["y_class"].numpy(), (probabilities >= activity["threshold"]).astype(int)
    )


def test_binary_cross_entropy_matches_manual_calculation(activity):
    y = activity["y"].numpy()
    probabilities = np.clip(activity["y_pred"].numpy(), 1e-7, 1 - 1e-7)
    expected = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
    assert float(activity["loss"]) == pytest.approx(float(expected), rel=1e-5)


def test_loss_penalizes_confident_wrong_predictions(activity):
    tf = activity["tf"]
    losses = [
        float(activity["loss_fn"](tf.constant([[1.0]]), tf.constant([[p]])))
        for p in (0.95, 0.60, 0.05)
    ]
    assert losses[0] < losses[1] < losses[2]
