import pandas as pd
from sklearn.model_selection import train_test_split

from module_olist.config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def split_data(data: pd.DataFrame):
    """Divide a base de dados em treino e teste.

    Args:
        data (pd.DataFrame): Base de dados a ser dividida.

    Returns:
        tuple: X_train, X_test, y_train, y_test.

    """
    X = data[FEATURES]
    y = data[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,

        # Mantém a mesma proporção de atrasos nos dois conjuntos. Sem isso, com
        # apenas 8% de positivos, o sorteio poderia desbalancear treino e teste
        # e a métrica passaria a medir a sorte da divisão.
        stratify=y,
    )
