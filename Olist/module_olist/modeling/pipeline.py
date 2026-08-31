"""Pipelines de pré-processamento e modelagem para a previsão de atraso na entrega."""

from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from module_olist.modeling.split import FEATURES

RANDOM_STATE = 42

# customer_state é a única feature em formato de texto ("SP", "RJ", ...).
# Nenhum estimador do scikit-learn aceita texto diretamente, então ela precisa
# ser convertida em colunas numéricas antes de chegar ao modelo.
CATEGORICAL_FEATURES = ["customer_state"]

# As demais features já são numéricas e seguem direto para o modelo.
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]


def create_preprocessor() -> ColumnTransformer:
    """Cria o pré-processamento aplicado antes do modelo.

    Returns:
        ColumnTransformer: Transformador que codifica as colunas categóricas
            e repassa as numéricas sem alteração.

    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    # Estados que aparecerem apenas no teste (ou em produção) não
                    # quebram a predição: viram uma linha de zeros em vez de erro.
                    handle_unknown="ignore",

                    # Devolve um array denso. Com 27 estados o resultado é pequeno
                    # e alguns modelos lidam melhor com matrizes densas.
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),

            # Modelos baseados em árvores não precisam de padronização de escala,
            # porque separam os dados por limiares e não por distância.
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],

        # Mantém os nomes originais das colunas ("customer_state_SP" em vez de
        # "categorical__customer_state_SP"), facilitando a leitura das
        # importâncias de variáveis depois do treino.
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def compute_scale_pos_weight(y) -> float:
    """Calcula o peso da classe minoritária para bases desbalanceadas.

    Args:
        y: Série com a variável-alvo contendo 0 e 1.

    Returns:
        float: Razão entre a quantidade de negativos e de positivos.

    """
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        raise ValueError("A base informada não possui exemplos da classe positiva.")

    # Na base da Olist apenas ~8% dos pedidos atrasam. Sem esse ajuste o modelo
    # tende a prever "no prazo" para quase tudo, já que isso sozinho já acerta
    # 92% dos casos.
    return negatives / positives


def build_models(scale_pos_weight: float | None = None) -> dict:
    """Instancia os modelos candidatos.

    Args:
        scale_pos_weight (float | None): Peso da classe positiva, normalmente
            obtido com compute_scale_pos_weight aplicado ao conjunto de treino.

    Returns:
        dict: Mapeamento entre o nome do modelo e o estimador correspondente.

    """
    return {
        # Implementação de referência do próprio scikit-learn: busca exata do
        # melhor corte e execução em uma única thread. É a mais lenta das três,
        # mas serve como linha de base confiável.
        #
        # Atenção: GradientBoostingClassifier não possui os parâmetros
        # class_weight nem scale_pos_weight. Para tratar o desbalanceamento com
        # esse modelo é necessário passar sample_weight na chamada do fit.
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),

        # Crescimento por nível: a profundidade é limitada por max_depth e as
        # árvores ficam simétricas.
        "xgboost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,

            # Cada árvore vê 80% das linhas e 80% das colunas, o que reduz o
            # risco de o modelo decorar o conjunto de treino.
            subsample=0.8,
            colsample_bytree=0.8,

            # Algoritmo baseado em histogramas: bem mais rápido em bases deste porte.
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Crescimento por folha: sempre abre a folha de maior ganho, mesmo que a
        # árvore fique assimétrica. Por isso o controle de complexidade aqui é
        # feito por num_leaves, e não por max_depth.
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,

            # No LightGBM o subsample só entra em ação com subsample_freq >= 1.
            # Sem esta linha o parâmetro acima seria simplesmente ignorado.
            subsample_freq=1,

            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,

            # Silencia os avisos de treino da biblioteca.
            verbose=-1,
        ),
    }


def create_pipeline(model) -> Pipeline:
    """Encadeia o pré-processamento e o modelo em um único objeto.

    Manter as duas etapas dentro de um Pipeline garante que o OneHotEncoder seja
    ajustado somente com os dados de treino. Se ele fosse ajustado na base
    completa antes da divisão, informação do conjunto de teste vazaria para o
    treino e as métricas ficariam otimistas.

    Args:
        model: Estimador do scikit-learn a ser usado na etapa final.

    Returns:
        Pipeline: Pipeline pronto para receber fit e predict.

    """
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", model),
        ]
    )


def create_gradient_boosting_pipeline() -> Pipeline:
    """Monta o pipeline do GradientBoostingClassifier.

    Returns:
        Pipeline: Pipeline pronto para receber fit e predict.

    """
    return create_pipeline(build_models()["gradient_boosting"])


def create_xgb_pipeline(scale_pos_weight: float | None = None) -> Pipeline:
    """Monta o pipeline do XGBoost.

    Args:
        scale_pos_weight (float | None): Peso da classe positiva.

    Returns:
        Pipeline: Pipeline pronto para receber fit e predict.

    """
    return create_pipeline(build_models(scale_pos_weight)["xgboost"])


def create_lgbm_pipeline(scale_pos_weight: float | None = None) -> Pipeline:
    """Monta o pipeline do LightGBM.

    Args:
        scale_pos_weight (float | None): Peso da classe positiva.

    Returns:
        Pipeline: Pipeline pronto para receber fit e predict.

    """
    return create_pipeline(build_models(scale_pos_weight)["lightgbm"])


def build_pipelines(scale_pos_weight: float | None = None) -> dict[str, Pipeline]:
    """Monta um pipeline completo para cada modelo candidato.

    Args:
        scale_pos_weight (float | None): Peso da classe positiva aplicado aos
            modelos que suportam esse parâmetro.

    Returns:
        dict[str, Pipeline]: Mapeamento entre o nome do modelo e seu pipeline.

    """
    pipelines = {
        name: create_pipeline(model)
        for name, model in build_models(scale_pos_weight).items()
    }

    logger.info("Pipelines disponíveis: {}", list(pipelines))

    return pipelines
