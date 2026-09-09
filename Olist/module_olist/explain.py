"""Interpretação do modelo de atraso com SHAP.

O modelo diz que um pedido vai atrasar; o SHAP diz por quê. Cada valor SHAP é a
contribuição de uma feature para afastar aquela previsão da média — positivo
empurra para "atrasa", negativo para "no prazo".

Isso importa aqui por dois motivos. O primeiro é de negócio: apontar um pedido
como problemático sem saber a causa não gera ação. O segundo é de diagnóstico —
foi testando pedidos à mão que apareceu o modelo reagindo ao valor do pedido no
sentido contrário ao dos dados, e é exatamente esse tipo de coisa que o gráfico
de importância expõe de uma vez, em vez de um caso por vez.
"""

from pathlib import Path

from loguru import logger
import matplotlib
import pandas as pd

from module_olist.config import FEATURES, FIGURES_DIR, INTERIM_DATA_DIR, TARGET
from module_olist.modeling.interpret import (
    calculate_shap_values,
    create_explainer,
    prepare_data_for_sharp,
)
from module_olist.modeling.predict import load_model

# Backend sem janela: o script roda no terminal e só grava arquivos. Precisa
# ser definido antes de importar o pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"

# O TreeExplainer é exato, mas o custo cresce com o número de linhas. Uma
# amostra desse tamanho já estabiliza o ranking de importância, e mantém o
# beeswarm legível — 96 mil pontos por feature viram uma mancha.
SAMPLE_SIZE = 2_000


def load_sample(path: Path = DATASET_PATH, size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Lê a base intermediária e devolve uma amostra estratificada.

    Args:
        path (Path): Base gerada por module_olist.main.
        size (int): Quantidade de linhas da amostra.

    Returns:
        pd.DataFrame: Amostra contendo apenas as colunas de FEATURES.

    """
    data = pd.read_csv(path, usecols=FEATURES + [TARGET])

    if len(data) <= size:
        return data[FEATURES]

    # Estratificado pelo alvo: com 8% de atrasos, uma amostra simples poderia
    # trazer poucos pedidos atrasados e o SHAP explicaria quase só a classe
    # majoritária.
    amostra = data.groupby(TARGET, group_keys=False).apply(
        lambda grupo: grupo.sample(
            n=max(1, round(size * len(grupo) / len(data))),
            random_state=42,
        ),
        include_groups=False,
    )

    logger.info(
        "Amostra de {} pedidos ({:.1%} de atrasos, como na base completa)",
        len(amostra),
        data[TARGET].mean(),
    )

    return amostra[FEATURES]


def plot_importance(shap_values, X_transformed: pd.DataFrame, output_dir: Path) -> Path:
    """Gera o gráfico de importância média das features.

    Args:
        shap_values: Valores SHAP calculados.
        X_transformed (pd.DataFrame): Features já transformadas pelo preprocessor.
        output_dir (Path): Pasta onde a figura é gravada.

    Returns:
        Path: Arquivo gerado.

    """
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    caminho = output_dir / "shap_importance.png"

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()

    return caminho


def plot_beeswarm(shap_values, X_transformed: pd.DataFrame, output_dir: Path) -> Path:
    """Gera o gráfico de dispersão dos valores SHAP.

    Diferente do gráfico de barras, este mostra o sentido do efeito: para cada
    feature, se valores altos empurram a previsão para atraso ou para o prazo.

    Args:
        shap_values: Valores SHAP calculados.
        X_transformed (pd.DataFrame): Features já transformadas pelo preprocessor.
        output_dir (Path): Pasta onde a figura é gravada.

    Returns:
        Path: Arquivo gerado.

    """
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    caminho = output_dir / "shap_beeswarm.png"

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()

    return caminho


def rank_features(shap_values, X_transformed: pd.DataFrame) -> pd.DataFrame:
    """Ordena as features pelo tamanho médio da contribuição.

    Args:
        shap_values: Valores SHAP calculados.
        X_transformed (pd.DataFrame): Features já transformadas pelo preprocessor.

    Returns:
        pd.DataFrame: Feature, importância média absoluta e efeito médio com sinal.

    """
    valores = pd.DataFrame(shap_values, columns=X_transformed.columns)

    ranking = pd.DataFrame(
        {
            "feature": valores.columns,
            # O módulo mede o tamanho do efeito, independente da direção.
            "importancia": valores.abs().mean().to_numpy(),
            # A média com sinal mostra para que lado a feature empurra na média.
            "efeito_medio": valores.mean().to_numpy(),
        }
    )

    return ranking.sort_values("importancia", ascending=False).reset_index(drop=True)


def main() -> None:
    """Calcula os valores SHAP do modelo salvo e grava os gráficos."""
    artefato = load_model()

    X = load_sample()

    logger.info("Transformando as features com o preprocessor do pipeline...")
    X_transformed = prepare_data_for_sharp(artefato["model"], X)

    logger.info("Calculando os valores SHAP de {} pedidos...", len(X_transformed))
    explainer = create_explainer(artefato["model"])
    shap_values = calculate_shap_values(explainer, X_transformed)

    ranking = rank_features(shap_values, X_transformed)

    logger.info(
        "Contribuição das features para o modelo '{}':\n{}",
        artefato["name"],
        ranking.head(15).to_string(index=False, float_format="{:.4f}".format),
    )

    importancia = plot_importance(shap_values, X_transformed, FIGURES_DIR)
    beeswarm = plot_beeswarm(shap_values, X_transformed, FIGURES_DIR)

    logger.success("Gráficos gravados em {} e {}", importancia, beeswarm)


if __name__ == "__main__":
    main()
