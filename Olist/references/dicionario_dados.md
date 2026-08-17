# Dicionário de Dados — Olist Brazilian E-Commerce

Base: `data/raw/` — 9 arquivos CSV, pedidos de **set/2016 a out/2018**.
Todos os IDs são hashes de 32 caracteres (anonimizados).

## Modelo relacional

```
customers (customer_id PK)
    └── orders (order_id PK, customer_id FK)
            ├── order_items (order_id + order_item_id PK)
            │       ├── products (product_id PK) ── category_translation
            │       └── sellers  (seller_id PK)
            ├── order_payments (order_id + payment_sequential PK)
            └── order_reviews  (review_id, order_id FK)

geolocation (zip_code_prefix) ── liga a customers e sellers por prefixo de CEP (N:N)
```

---

## 1. `olist_orders_dataset.csv` — 99.441 linhas × 8 colunas
Tabela-fato central. Um registro por pedido.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `order_id` | str | 0% | **PK**. Identificador do pedido (99.441 únicos) |
| `customer_id` | str | 0% | **FK → customers**. Chave por pedido, não por pessoa (ver nota em customers) |
| `order_status` | str | 0% | 8 valores: `delivered` (97,0%), `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created`, `approved` |
| `order_purchase_timestamp` | datetime | 0% | Momento da compra |
| `order_approved_at` | datetime | 0,16% | Aprovação do pagamento |
| `order_delivered_carrier_date` | datetime | 1,79% | Postagem — entrega à transportadora |
| `order_delivered_customer_date` | datetime | 2,98% | Entrega efetiva ao cliente |
| `order_estimated_delivery_date` | datetime | 0% | Prazo prometido ao cliente (apenas data, sem hora) |

> Nulos nas datas de entrega são **estruturais**, não erro: pedido cancelado/em trânsito não tem data de entrega.

---

## 2. `olist_order_items_dataset.csv` — 112.650 linhas × 7 colunas
Um registro por **item** do pedido. Granularidade mais fina que `orders`.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `order_id` | str | 0% | **FK → orders** (98.666 pedidos distintos) |
| `order_item_id` | int | 0% | Sequencial do item dentro do pedido (1 a 21). **PK composta** com `order_id` |
| `product_id` | str | 0% | **FK → products** |
| `seller_id` | str | 0% | **FK → sellers**. Um pedido pode ter vários vendedores |
| `shipping_limit_date` | datetime | 0% | Prazo limite para o vendedor postar |
| `price` | float | 0% | Preço unitário do item (R$ 0,85 a 6.735,00 — média 120,65) |
| `freight_value` | float | 0% | Frete rateado por item (R$ 0 a 409,68 — média 19,99) |

> **Atenção:** não há coluna de quantidade. Quantidade = número de linhas repetidas do mesmo `product_id` no pedido.
> O valor total do pedido = `SUM(price + freight_value)` agrupado por `order_id`.

---

## 3. `olist_order_payments_dataset.csv` — 103.886 linhas × 5 colunas
Um pedido pode ter múltiplos pagamentos (2.961 casos).

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `order_id` | str | 0% | **FK → orders** |
| `payment_sequential` | int | 0% | Sequencial do pagamento (1 a 29). **PK composta** |
| `payment_type` | str | 0% | `credit_card` (73,9%), `boleto` (19,0%), `voucher` (5,6%), `debit_card` (1,5%), `not_defined` (3 registros) |
| `payment_installments` | int | 0% | Número de parcelas (0 a 24; mediana 1) |
| `payment_value` | float | 0% | Valor da transação |

---

## 4. `olist_order_reviews_dataset.csv` — 99.224 linhas × 7 colunas

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `review_id` | str | 0% | Id da avaliação — **não é único**: 814 duplicados |
| `order_id` | str | 0% | **FK → orders**. 547 pedidos têm mais de uma avaliação |
| `review_score` | int | 0% | Nota de 1 a 5 |
| `review_comment_title` | str | **88,3%** | Título livre do comentário |
| `review_comment_message` | str | **58,7%** | Texto livre em português — matéria-prima para NLP |
| `review_creation_date` | datetime | 0% | Envio da pesquisa ao cliente |
| `review_answer_timestamp` | datetime | 0% | Resposta do cliente |

---

## 5. `olist_products_dataset.csv` — 32.951 linhas × 9 colunas

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `product_id` | str | 0% | **PK** |
| `product_category_name` | str | 1,85% | Categoria em português (73 valores distintos) |
| `product_name_lenght` | float | 1,85% | Nº de caracteres do nome (grafia original com erro: "lenght") |
| `product_description_lenght` | float | 1,85% | Nº de caracteres da descrição |
| `product_photos_qty` | float | 1,85% | Quantidade de fotos publicadas |
| `product_weight_g` | float | 0,01% | Peso em gramas (4 produtos com 0) |
| `product_length_cm` | float | 0,01% | Comprimento |
| `product_height_cm` | float | 0,01% | Altura |
| `product_width_cm` | float | 0,01% | Largura |

> Os 610 nulos ocorrem sempre nas mesmas 5 colunas simultaneamente — cadastro incompleto.

---

## 6. `olist_customers_dataset.csv` — 99.441 linhas × 5 colunas

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `customer_id` | str | 0% | **PK**. Chave **por pedido** — 1:1 com `orders` |
| `customer_unique_id` | str | 0% | Identificador real da pessoa (96.096 únicos) |
| `customer_zip_code_prefix` | int | 0% | 5 primeiros dígitos do CEP (14.994 valores) |
| `customer_city` | str | 0% | Cidade (4.119 valores, minúsculas, sem acento padronizado) |
| `customer_state` | str | 0% | UF (27 valores) |

> **Armadilha clássica:** para análise de recorrência/LTV use `customer_unique_id`, nunca `customer_id`.
> A diferença (99.441 − 96.096 = 3.345) é o número de pedidos adicionais feitos por clientes recorrentes.

---

## 7. `olist_sellers_dataset.csv` — 3.095 linhas × 4 colunas

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `seller_id` | str | 0% | **PK** |
| `seller_zip_code_prefix` | int | 0% | Prefixo do CEP (2.246 valores) |
| `seller_city` | str | 0% | Cidade (611 valores, com variações de grafia) |
| `seller_state` | str | 0% | UF (23 valores) |

---

## 8. `olist_geolocation_dataset.csv` — 1.000.163 linhas × 5 colunas

| Coluna | Tipo | Nulos | Descrição |
|---|---|---|---|
| `geolocation_zip_code_prefix` | int | 0% | Prefixo de CEP (19.015 únicos) |
| `geolocation_lat` | float | 0% | Latitude |
| `geolocation_lng` | float | 0% | Longitude |
| `geolocation_city` | str | 0% | Cidade (8.011 valores — muito ruidoso) |
| `geolocation_state` | str | 0% | UF |

> **Não é uma tabela de lookup limpa:** 261.831 linhas totalmente duplicadas e ~52 coordenadas por prefixo em média.
> Antes de qualquer join, agregue por `zip_code_prefix` (mediana de lat/lng), senão o join explode em cardinalidade.

---

## 9. `product_category_name_translation.csv` — 71 linhas × 2 colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `product_category_name` | str | Categoria em português |
| `product_category_name_english` | str | Tradução para inglês |

> Cobre 71 das 73 categorias. Faltam: `pc_gamer` e `portateis_cozinha_e_preparadores_de_alimentos`.
