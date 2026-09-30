"""Active learning com Label Spreading para a previsão de atraso na entrega.

Adaptação do exemplo "Label Propagation digits: Active learning" do
scikit-learn para a base da Olist. A ideia é responder à pergunta: se rotular
um pedido custasse caro, quantos rótulos seriam necessários e quais pedidos
valeria a pena rotular primeiro?

O experimento é uma simulação. Todos os pedidos já têm rótulo, mas o modelo só
enxerga uma pequena parte deles. A cada rodada:

1. O LabelSpreading é treinado com poucos pedidos rotulados e muitos pedidos
   sem rótulo (marcados com -1). Ele propaga os rótulos conhecidos para os
   vizinhos por meio de um grafo de similaridade.
2. Para cada pedido sem rótulo, calcula-se a entropia da distribuição prevista.
   Entropia alta significa que o modelo está em dúvida entre "atrasa" e
   "no prazo".
3. Os pedidos de maior entropia são "enviados ao especialista", ou seja, seus
   rótulos verdadeiros são revelados e passam a fazer parte do treino.

A mesma rotina roda com seleção aleatória, que serve de linha de base: o active
learning só vale a pena se chegar a um desempenho melhor com o mesmo número de
rótulos.
"""

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from module_olist.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from module_olist.modeling.split import split_data

DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"
HISTORY_PATH = REPORTS_DIR / "active_learning.csv"
FIGURE_PATH = FIGURES_DIR / "active_learning.png"

# Tamanho do conjunto de pedidos (rotulados + não rotulados) usado no grafo.
# O LabelSpreading monta uma matriz de vizinhança entre todos os pontos, então
# usar os 77 mil pedidos do treino deixaria cada rodada lenta demais.
POOL_SIZE = 5_000

# Quantos pedidos começam rotulados. Com ~8% de atrasos, 100 pedidos trazem em
# média 8 positivos: o suficiente para o modelo ver as duas classes.
N_INITIAL_LABELED = 100

# Quantos pedidos o "especialista" rotula a cada rodada.
N_QUERIES_PER_ITERATION = 50

# Quantidade de rodadas de consulta. Ao final, o modelo terá visto
# N_INITIAL_LABELED + MAX_ITERATIONS * N_QUERIES_PER_ITERATION rótulos.
MAX_ITERATIONS = 20

# Valor que o scikit-learn usa para indicar "sem rótulo" no semi-supervisionado.
UNLABELED = -1

STRATEGIES = ("uncertainty", "random")


def create_label_spreading_pipeline() -> Pipeline:
    """Monta o pipeline de pré-processamento e LabelSpreading.

    Diferente dos modelos de árvore do pipeline.py, o LabelSpreading depende de
    distância entre pedidos. Sem padronização, total_price (centenas de reais)
    dominaria purchase_hour (0 a 23) no cálculo dos vizinhos.

    Returns:
        Pipeline: Pipeline pronto para receber fit com rótulos -1 nos pedidos
            não rotulados.

    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        # O LabelSpreading não aceita valores ausentes, ao
                        # contrário do XGBoost e do LightGBM.
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LabelSpreading(
                    # O kernel knn gera um grafo esparso. O rbf, usado no exemplo
                    # dos dígitos, cria uma matriz densa de POOL_SIZE x POOL_SIZE.
                    kernel="knn",
                    n_neighbors=10,
                    # Fração do rótulo original que pode ser substituída pela
                    # informação dos vizinhos. Valores baixos confiam mais nos
                    # rótulos dados pelo especialista.
                    alpha=0.2,
                    max_iter=50,
                ),
            ),
        ]
    )


def select_queries(
    model: Pipeline,
    unlabeled_indices: np.ndarray,
    n_queries: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Escolhe quais pedidos não rotulados serão enviados ao especialista.

    Args:
        model (Pipeline): Pipeline já treinado no conjunto atual.
        unlabeled_indices (np.ndarray): Posições dos pedidos ainda sem rótulo.
        n_queries (int): Quantidade de pedidos a rotular.
        strategy (str): "uncertainty" (maior entropia) ou "random".
        rng (np.random.Generator): Gerador usado na estratégia aleatória.

    Returns:
        np.ndarray: Posições dos pedidos escolhidos.

    """
    n_queries = min(n_queries, len(unlabeled_indices))

    if strategy == "random":
        return rng.choice(unlabeled_indices, size=n_queries, replace=False)

    if strategy != "uncertainty":
        raise ValueError(f"Estratégia desconhecida: {strategy}. Use uma de {STRATEGIES}.")

    # label_distributions_ guarda, para cada pedido do conjunto, a probabilidade
    # de cada classe depois da propagação. A entropia resume a dúvida do modelo:
    # 0 quando ele tem certeza, máxima quando as classes estão empatadas.
    distributions = model.named_steps["model"].label_distributions_[unlabeled_indices]
    entropies = stats.entropy(np.nan_to_num(distributions).T)

    most_uncertain = np.argsort(entropies)[::-1][:n_queries]

    return unlabeled_indices[most_uncertain]


def run_active_learning(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "uncertainty",
    n_initial_labeled: int = N_INITIAL_LABELED,
    n_queries: int = N_QUERIES_PER_ITERATION,
    max_iterations: int = MAX_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Executa o ciclo de active learning e registra o desempenho a cada rodada.

    Args:
        X_pool (pd.DataFrame): Pedidos disponíveis para o grafo.
        y_pool (pd.Series): Rótulos verdadeiros desses pedidos, revelados aos
            poucos conforme o especialista é consultado.
        X_test (pd.DataFrame): Features do conjunto de teste.
        y_test (pd.Series): Target do conjunto de teste.
        strategy (str): "uncertainty" ou "random".
        n_initial_labeled (int): Pedidos rotulados antes da primeira rodada.
        n_queries (int): Pedidos rotulados a cada rodada.
        max_iterations (int): Quantidade de rodadas de consulta.
        random_state (int): Semente usada na amostra inicial e na estratégia
            aleatória.

    Returns:
        pd.DataFrame: Uma linha por rodada, com a quantidade de rótulos usados e
            as métricas de teste.

    """
    X_pool = X_pool.reset_index(drop=True)
    y_pool = y_pool.to_numpy()
    rng = np.random.default_rng(random_state)

    # As duas estratégias partem exatamente dos mesmos pedidos rotulados, então
    # qualquer diferença entre as curvas vem apenas da forma de escolher as
    # consultas. A estratificação garante positivos já na primeira rodada.
    initial_indices, _ = train_test_split(
        np.arange(len(y_pool)),
        train_size=n_initial_labeled,
        stratify=y_pool,
        random_state=random_state,
    )
    labeled = np.zeros(len(y_pool), dtype=bool)
    labeled[initial_indices] = True

    history = []

    for iteration in range(max_iterations + 1):
        y_train = np.where(labeled, y_pool, UNLABELED)

        model = create_label_spreading_pipeline()
        model.fit(X_pool, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        history.append(
            {
                "strategy": strategy,
                "iteration": iteration,
                "n_labeled": int(labeled.sum()),
                "n_positive_labeled": int(y_pool[labeled].sum()),
                "pr_auc": float(average_precision_score(y_test, y_proba)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
        )

        logger.info(
            "{} | rodada {:>2} | rótulos: {:>5} ({} atrasos) | pr_auc: {:.4f} | roc_auc: {:.4f}",
            strategy,
            iteration,
            history[-1]["n_labeled"],
            history[-1]["n_positive_labeled"],
            history[-1]["pr_auc"],
            history[-1]["roc_auc"],
        )

        unlabeled_indices = np.flatnonzero(~labeled)
        if iteration == max_iterations or len(unlabeled_indices) == 0:
            break

        # Aqui o especialista entraria em ação. Na simulação, basta marcar os
        # pedidos como rotulados: o y_pool já contém a resposta verdadeira.
        queries = select_queries(model, unlabeled_indices, n_queries, strategy, rng)
        labeled[queries] = True

    return pd.DataFrame(history)


def plot_learning_curves(history: pd.DataFrame, baseline: float, path=FIGURE_PATH) -> None:
    """Compara a PR-AUC das estratégias conforme novos rótulos são adicionados.

    Args:
        history (pd.DataFrame): Resultado concatenado de run_active_learning.
        baseline (float): Proporção de atrasos no teste, que é a PR-AUC de um
            modelo que chuta ao acaso.
        path (Path): Arquivo de destino da figura.

    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, group in history.groupby("strategy"):
        ax.plot(group["n_labeled"], group["pr_auc"], marker="o", label=strategy)

    ax.axhline(baseline, color="gray", linestyle="--", label="acaso")
    ax.set_xlabel("Pedidos rotulados")
    ax.set_ylabel("PR-AUC no teste")
    ax.set_title("Active learning com LabelSpreading")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.success("Curvas salvas em {}", path)


def main() -> None:
    """Roda a simulação de active learning com as duas estratégias."""
    logger.info("Lendo a base intermediária de {}", DATASET_PATH)
    dataset = pd.read_csv(DATASET_PATH, usecols=FEATURES + [TARGET])

    # Mesmo split do train.py: o teste continua sendo o mesmo usado para avaliar
    # os modelos supervisionados, o que permite comparar os números.
    X_train, X_test, y_train, y_test = split_data(dataset)

    X_pool, _, y_pool, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(POOL_SIZE, len(X_train)),
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    logger.info("Pool: {} pedidos | Teste: {} pedidos", len(X_pool), len(X_test))

    history = pd.concat(
        [
            run_active_learning(X_pool, y_pool, X_test, y_test, strategy=strategy)
            for strategy in STRATEGIES
        ],
        ignore_index=True,
    )

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    logger.success("Histórico salvo em {}", HISTORY_PATH)

    plot_learning_curves(history, baseline=float(y_test.mean()))


if __name__ == "__main__":
    main()
