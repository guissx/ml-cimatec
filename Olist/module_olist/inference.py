"""Previsão de atraso para pedidos avulsos, a partir dos campos crus.

A diferença para modeling/predict.py é a entrada. O predict recebe as features
já calculadas, no formato em que o modelo foi treinado. Este módulo recebe um
pedido como ele existe no mundo real — datas, valores e estado — e deriva as
features antes de chamar o predict.

A derivação é feita pelo create_features de features.py, o mesmo usado no
treino. Reimplementar as contas aqui seria mais direto de ler, mas criaria duas
fontes de verdade: bastaria alguém mudar a fórmula de promised_days de um lado
para o modelo passar a receber, em produção, uma feature diferente da que
aprendeu — sem erro, sem aviso, só com a previsão piorando em silêncio.
"""

from datetime import datetime
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_olist.config import FEATURES, MODEL_PATH
from module_olist.features import create_features
from module_olist.modeling.predict import load_model, predict

app = typer.Typer(add_completion=False)

# Campos crus que o chamador precisa informar. As features do modelo saem
# destes: as quatro de calendário e promised_days são derivadas das datas.
CAMPOS_OBRIGATORIOS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]

COLUNAS_DATA = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]


def build_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Deriva as features do modelo a partir dos campos crus do pedido.

    Args:
        orders (pd.DataFrame): Pedidos com as colunas de CAMPOS_OBRIGATORIOS.

    Returns:
        pd.DataFrame: As colunas de FEATURES, prontas para o modelo.

    """
    faltando = [campo for campo in CAMPOS_OBRIGATORIOS if campo not in orders.columns]

    if faltando:
        raise ValueError(f"Campos ausentes no pedido: {faltando}")

    orders = orders.copy()

    # create_features usa o acessor .dt, que só existe em coluna datetime.
    # Strings passariam batido até estourar lá dentro com uma mensagem obscura.
    for coluna in COLUNAS_DATA:
        orders[coluna] = pd.to_datetime(orders[coluna])

    return create_features(orders)[FEATURES]


def predict_orders(orders: list[dict] | pd.DataFrame, artefato: dict | None = None) -> pd.DataFrame:
    """Prevê o atraso de vários pedidos informados em formato cru.

    Args:
        orders (list[dict] | pd.DataFrame): Pedidos a prever.
        artefato (dict | None): Saída de load_model. Carregado do disco se omitido.

    Returns:
        pd.DataFrame: Probabilidade de atraso e decisão para cada pedido.

    """
    if isinstance(orders, list):
        orders = pd.DataFrame(orders)

    artefato = artefato or load_model()

    return predict(build_features(orders), artefato)


def predict_order(order: dict, artefato: dict | None = None) -> dict:
    """Prevê o atraso de um único pedido.

    Args:
        order (dict): Campos crus do pedido.
        artefato (dict | None): Saída de load_model. Carregado do disco se omitido.

    Returns:
        dict: Probabilidade de atraso, decisão e o ponto de corte usado.

    """
    artefato = artefato or load_model()
    resultado = predict_orders([order], artefato).iloc[0]

    return {
        "late_probability": float(resultado["late_probability"]),
        "is_late": int(resultado["is_late_predicted"]),
        "threshold": float(artefato["threshold"]),
    }


# Casos de teste calibrados pela propria base, e nao por intuicao.
#
# Cada cenario varia UMA coisa por vez. Mudar prazo, estado, quantidade e valor
# ao mesmo tempo torna impossivel saber a qual deles o modelo reagiu — foi
# exatamente assim que a primeira versao destes exemplos atribuiu ao estado um
# efeito que vinha do valor do pedido.
#
# Referencia de atraso na base, para comparar com a saida do modelo:
#   promised_days -> 28,3% ate 7 dias, ~8,5% na faixa mediana, 2,4% acima de 40.
#   customer_state -> AL 23,9%, MA 19,7%, PI 16,0% no topo;
#                     RO 2,9%, AC 3,8%, AM 4,1% no fim, todos abaixo de SP (5,9%).
#                     Distancia nao explica o ranking.
_PEDIDO_BASE = {
    "order_purchase_timestamp": "2018-03-13 10:15:00",
    "order_approved_at": "2018-03-13 10:30:00",
    "order_estimated_delivery_date": "2018-04-05 00:00:00",  # 23 dias
    "item_count": 1,
    "seller_count": 1,
    "total_price": 85.50,
    "total_freight": 15.10,
    "customer_state": "SP",
}


def _cenario(**alteracoes) -> dict:
    """Copia o pedido de referencia alterando apenas os campos informados."""
    return {**_PEDIDO_BASE, **alteracoes}


EXEMPLOS = {
    # Varia so o prazo prometido.
    "prazo 45d (base atrasa 2,4%)": _cenario(
        order_estimated_delivery_date="2018-04-27 00:00:00"
    ),
    "prazo 23d (base atrasa 8,5%)": _cenario(),
    "prazo 7d (base atrasa 28,3%)": _cenario(
        order_estimated_delivery_date="2018-03-20 00:00:00"
    ),

    # Varia so o estado, mantendo o prazo apertado de 7 dias.
    "prazo 7d + AM (base atrasa 4,1%)": _cenario(
        order_estimated_delivery_date="2018-03-20 00:00:00", customer_state="AM"
    ),
    "prazo 7d + AL (base atrasa 23,9%)": _cenario(
        order_estimated_delivery_date="2018-03-20 00:00:00", customer_state="AL"
    ),

    # Varia so o tamanho do pedido. Na base, pedido caro atrasa MAIS (18,5% no
    # quartil superior contra 14,7% no inferior); o modelo inverte esse sinal.
    "prazo 7d + pedido grande": _cenario(
        order_estimated_delivery_date="2018-03-20 00:00:00",
        item_count=3,
        seller_count=2,
        total_price=289.90,
        total_freight=78.40,
    ),
}


@app.command()
def exemplos(model_path: Path = MODEL_PATH) -> None:
    """Roda os pedidos de exemplo e mostra a reação do modelo a cada um."""
    artefato = load_model(model_path)

    resultados = predict_orders(list(EXEMPLOS.values()), artefato)
    resultados.insert(0, "cenario", list(EXEMPLOS))
    resultados["late_probability"] = resultados["late_probability"].round(4)

    logger.info(
        "Previsões (threshold {:.2f}):\n{}",
        artefato["threshold"],
        resultados.to_string(index=False),
    )


@app.command()
def pedido(
    estimated_delivery: str = typer.Option(..., help="Data prometida de entrega."),
    state: str = typer.Option("SP", help="UF do cliente."),
    price: float = typer.Option(85.50, help="Valor total dos produtos."),
    freight: float = typer.Option(15.10, help="Valor total do frete."),
    items: int = typer.Option(1, help="Quantidade de itens."),
    sellers: int = typer.Option(1, help="Quantidade de vendedores."),
    purchase: str = typer.Option("", help="Momento da compra. Padrão: agora."),
    model_path: Path = MODEL_PATH,
) -> None:
    """Prevê o atraso de um pedido informado pela linha de comando."""
    # Horario local de proposito: purchase_hour foi aprendida no fuso em que a
    # compra foi registrada na base, nao em UTC. Converter para UTC deslocaria
    # a hora em tres unidades e o modelo leria outro periodo do dia.
    momento = purchase or datetime.now().isoformat(sep=" ", timespec="seconds")  # noqa: DTZ005

    resultado = predict_order(
        {
            "order_purchase_timestamp": momento,
            # Sem informação melhor, assume-se aprovação imediata do pagamento.
            "order_approved_at": momento,
            "order_estimated_delivery_date": estimated_delivery,
            "item_count": items,
            "seller_count": sellers,
            "total_price": price,
            "total_freight": freight,
            "customer_state": state.upper(),
        },
        load_model(model_path),
    )

    veredito = "ATRASA" if resultado["is_late"] else "no prazo"

    logger.info(
        "{} | probabilidade de atraso: {:.2%} (corte em {:.0%})",
        veredito,
        resultado["late_probability"],
        resultado["threshold"],
    )


if __name__ == "__main__":
    app()
