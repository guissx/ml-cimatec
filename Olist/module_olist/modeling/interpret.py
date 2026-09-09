import pandas as pd
from scipy import sparse
import shap


def prepare_data_for_sharp(pipeline, X):
    """Prepara os dados para o SHAP.

    Args:
        pipeline: Pipeline de pré-processamento.
        X: DataFrame com as features.

    Returns:
        Dados transformados para o SHAP.

    """
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X)

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Recuperar os nomes das features após a transformação
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    return X_transformed_df

def create_explainer(pipeline):
    model = pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model)
    return explainer

def calculate_shap_values(explainer, X_transformed):
    shap_values = explainer.shap_values(X_transformed)
    return shap_values
