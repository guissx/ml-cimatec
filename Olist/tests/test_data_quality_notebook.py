"""Testes da aula 04; requerem o ambiente requirements-profiling.txt."""

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandera")
pytest.importorskip("ydata_profiling")

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks" / "01_data_quality.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture
def orders():
    frame = pd.DataFrame({
        "order_id": [f"order-{i}" for i in range(5)],
        "customer_id": [f"customer-{i}" for i in range(5)],
        "order_status": ["delivered"] * 5,
    })
    for column, day in (
        ("order_purchase_timestamp", "2018-01-01"),
        ("order_approved_at", "2018-01-02"),
        ("order_delivered_carrier_date", "2018-01-03"),
        ("order_delivered_customer_date", "2018-01-05"),
        ("order_estimated_delivery_date", "2018-01-10"),
    ):
        frame[column] = pd.Series(pd.Timestamp(day), index=frame.index, dtype="datetime64[ns]")
    return frame


@pytest.mark.parametrize("start", [PROJECT, PROJECT / "notebooks", PROJECT.parent])
def test_setup_finds_project_from_supported_directories(start, monkeypatch):
    monkeypatch.chdir(start)
    namespace = {}
    exec(code_cells()[0], namespace)
    assert namespace["ROOT"] == PROJECT


def test_notebook_generates_report_and_validates_orders(orders, tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT)
    namespace = {}
    cells = code_cells()
    exec(cells[0], namespace)
    namespace.update(DATA=tmp_path, REPORTS=tmp_path)
    orders.to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    for cell in cells[1:]:
        exec(cell, namespace)
    assert namespace["validated_orders"] is not None
    assert namespace["quality_failures"].empty
    assert len(namespace["validated_orders"]) == len(orders)
    assert (tmp_path / "profile_olist_orders_profile.html").stat().st_size > 0


def test_quality_violation_is_reported_without_using_invalid_data(orders, monkeypatch):
    monkeypatch.chdir(PROJECT)
    namespace = {}
    cells = code_cells()
    exec(cells[0], namespace)
    orders.loc[1, "order_id"] = orders.loc[0, "order_id"]
    namespace["orders"] = orders
    # Contrato e visualização final devem funcionar mesmo quando há violações.
    for cell in cells[-2:]:
        exec(cell, namespace)
    assert namespace["validated_orders"] is None
    failures = namespace["quality_failures"]
    assert not failures.empty
    assert "order_id" in failures["column"].values
