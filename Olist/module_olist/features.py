from loguru import logger
import pandas as pd

from module_olist.dataset import aggregate_data, create_target


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features derived from the orders dataset.

    Args:
        data (pd.DataFrame): DataFrame containing order timestamps.

    Returns:
        pd.DataFrame: DataFrame with the new feature columns added.
    """
    data = data.copy()

    # Calcula quantos dias a empresa prometeu para realizar a entrega,
    # considerando como início o momento da aprovação do pagamento.
    data["promised_days"] = (
        data["order_estimated_delivery_date"]  # Data prometida para a entrega.
        - data["order_approved_at"]            # Data de aprovação do pagamento.
    ).dt.total_seconds().div(86_400)          # Converte segundos para dias => 24 * 60 * 60 = 86.400 segundos

    # Extrai o número do mês em que a compra foi realizada.
    # Exemplo: janeiro = 1, fevereiro = 2, ..., dezembro = 12.
    data["purchase_month"] = (
        data["order_purchase_timestamp"].dt.month
    )

    # Extrai o dia do mês em que a compra foi realizada.
    # Os valores variam de 1 a 31.
    data["purchase_day"] = (
        data["order_purchase_timestamp"].dt.day
    )

    # Extrai o dia da semana em que a compra foi realizada.
    #
    # O Pandas representa os dias da seguinte forma:
    # 0 = segunda-feira
    # 1 = terça-feira
    # 2 = quarta-feira
    # 3 = quinta-feira
    # 4 = sexta-feira
    # 5 = sábado
    # 6 = domingo
    data["purchase_weekday"] = (
        data["order_purchase_timestamp"].dt.dayofweek
    )

    # Extrai a hora em que a compra foi realizada.
    # Os valores variam de 0 a 23.
    #
    # Exemplo:
    # 0  = meia-noite
    # 8  = 8 horas
    # 14 = 14 horas
    # 23 = 23 horas
    data["purchase_hour"] = (
        data["order_purchase_timestamp"].dt.hour
    )

    # Gera estatísticas descritivas para o prazo prometido:
    # quantidade, média, desvio-padrão, mínimo, quartis e máximo.
    logger.info(
        "Estatísticas de promised_days:\n{}",
        data["promised_days"].describe().to_frame().T
    )

    # Conta quantos pedidos apresentam prazo prometido menor ou igual a zero.
    #
    # Esses casos seriam suspeitos porque significariam que a data prometida
    # ocorreu antes ou exatamente no momento da aprovação do pagamento.
    logger.info(
        "Prazos não positivos: {}",
        data["promised_days"].le(0).sum()
    )

    return data


def create_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a dataset for predicting late deliveries.

    Args:
        orders (pd.DataFrame): DataFrame containing order information.
        items (pd.DataFrame): DataFrame containing order items.
        customers (pd.DataFrame): DataFrame containing customer information.

    Returns:
        pd.DataFrame: DataFrame containing only delivered orders with the target variable.
    """
    orders = create_target(orders)
    items_agg = aggregate_data(items)

    # Adiciona as informações agregadas dos itens: quantidade de itens,
    # vendedores, preço total e frete total.
    dataset = orders.merge(
        items_agg,
        on="order_id",          # Coluna usada para relacionar as tabelas.
        how="left",             # Mantém todos os pedidos de orders.
        validate="one_to_one",  # Verifica se order_id é único nas duas tabelas.
    )

    # Adiciona a localização do cliente.
    dataset = dataset.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",         # Relaciona cada pedido ao seu cliente.
        how="left",               # Mantém todos os pedidos da base anterior.
        validate="many_to_one",   # Vários pedidos podem apontar para um cliente.
    )

    assert dataset["order_id"].is_unique, "A base final deve manter uma linha por pedido."
    assert len(dataset) == len(orders), "A integração alterou o total de pedidos."

    return dataset
