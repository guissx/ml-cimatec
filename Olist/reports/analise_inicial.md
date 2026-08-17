# Análise Inicial — Olist Brazilian E-Commerce

Base bruta em `data/raw/`. Dicionário completo em [`references/dicionario_dados.md`](../references/dicionario_dados.md).

## 1. Visão geral

| Métrica | Valor |
|---|---|
| Período | 04/09/2016 a 17/10/2018 (~25 meses) |
| Pedidos | 99.441 |
| Itens vendidos | 112.650 |
| Clientes únicos | 96.096 (de 99.441 registros de cliente) |
| Vendedores | 3.095 |
| Produtos | 32.951 em 73 categorias |
| GMV (itens) | R$ 13.591.643,70 |
| Frete | R$ 2.251.909,54 |
| Total transacionado (pagamentos) | R$ 16.008.872,12 |
| Ticket médio por pedido | R$ 137,75 |

## 2. Qualidade dos dados

**Integridade referencial: praticamente perfeita.** Zero chaves órfãs em `product_id`, `seller_id` e `customer_id`. As exceções são todas explicáveis:

| Achado | Volume | Diagnóstico |
|---|---|---|
| Pedidos sem itens | 775 | 603 são `unavailable` e 164 `canceled` — comportamento esperado, não erro |
| Pedidos sem pagamento | 1 | Registro isolado |
| Pedidos sem avaliação | 768 | Cliente não respondeu a pesquisa |
| `review_id` duplicado | 814 | Mesma avaliação vinculada a mais de um pedido |
| Pedidos com >1 avaliação | 547 | Rompe a suposição de 1:1 com `orders` |
| Linhas duplicadas em geolocation | 261.831 (26%) | Exige deduplicação/agregação antes do join |
| `delivered` sem data de entrega | 8 | Inconsistência real de status |
| Entregue antes de sair para transportadora | 23 | Inversão temporal — erro de registro |
| Categorias sem tradução | 2 | `pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos` |
| `payment_value` ≤ 0 | 9 | Provável estorno |
| Frete zerado | 383 itens | Frete grátis ou promoção |
| Peso zerado | 4 produtos | Cadastro inválido |

**Nulos relevantes:** concentrados em texto de avaliação (`review_comment_title` 88,3%, `review_comment_message` 58,7%) e em atributos de produto (1,85% em bloco). As datas de entrega têm nulos estruturais (2,98%) explicados pelo status do pedido.

**Truncamento nas pontas da série:** set/2016 tem 4 pedidos, nov/2016 tem 0, set/2018 tem 16 e out/2018 tem 4. Os meses de borda são artefatos — recorte a série em **jan/2017 a ago/2018** para qualquer análise temporal ou modelo com componente sazonal.

## 3. Comportamento do negócio

**Status:** 97,0% dos pedidos chegam a `delivered`. Cancelamentos são 0,6% e `unavailable` 0,6%. A base é fortemente desbalanceada em favor do fluxo feliz — relevante se o alvo do modelo for cancelamento.

**Pagamento:** cartão de crédito domina (73,9%), boleto responde por 19,0%. A mediana é de 1 parcela, mas o parcelamento vai até 24x. 2.961 pedidos usam mais de uma forma de pagamento, o que exige agregação antes de juntar a `orders`.

**Cesta:** média de 1,14 itens por pedido — o e-commerce é essencialmente de compra unitária. Preço mediano de R$ 74,99 contra média de R$ 120,65, com máximo de R$ 6.735: distribuição fortemente assimétrica à direita, candidata a transformação log em modelagem.

**Concentração geográfica:** São Paulo responde por 42,0% dos clientes e 59,7% dos vendedores. Somando SP, RJ e MG chega-se a 66,6% da demanda. O eixo Sudeste concentra oferta e demanda, o que torna distância cliente–vendedor uma feature promissora para prazo de entrega.

**Categorias:** as cinco maiores em número de produtos são cama/mesa/banho (3.029), esporte/lazer (2.867), móveis/decoração (2.657), beleza/saúde (2.444) e utilidades domésticas (2.335). Cauda longa acentuada nas 73 categorias.

## 4. Logística — o achado mais forte

Sobre os 96.470 pedidos entregues com data válida:

| Métrica | Valor |
|---|---|
| Lead time médio (compra → entrega) | 12,56 dias |
| Mediana | 10,22 dias |
| Percentil 90 | 23,10 dias |
| Percentil 99 | 46,05 dias |
| Máximo | 209,63 dias |
| Entregas fora do prazo prometido | **8,11%** |
| Atraso médio quando ocorre | 9,55 dias |

A cauda é longa: 1% dos pedidos leva mais de 46 dias. O prazo prometido é cumprido em 92% dos casos, mas quando falha, falha feio.

**Atraso destrói a nota.** A relação é direta e monotônica:

| Situação | Nota média | Pedidos |
|---|---|---|
| Entregue no prazo | 4,29 | 88.661 |
| Entregue com atraso | **2,57** | 7.700 |

E o lead time cresce de forma consistente conforme a nota cai:

| Nota | Lead time médio |
|---|---|
| 5 | 10,7 dias |
| 4 | 12,3 dias |
| 3 | 14,3 dias |
| 2 | 16,7 dias |
| 1 | 21,3 dias |

Um pedido que leva 21 dias tende a virar nota 1 mesmo quando entregue. **Tempo de entrega é o principal driver de satisfação nesta base** — e é o eixo mais defensável para modelagem.

**Distribuição das notas:** 57,8% são nota 5 e 11,5% são nota 1 — formato em "J". Se o alvo for satisfação, trate como binário (1-2 = insatisfeito, 4-5 = satisfeito) em vez de regressão sobre 1-5.

## 5. Riscos ao construir a tabela analítica

1. **Granularidade.** `orders` é 1 linha/pedido, `order_items` é 1 linha/item, `payments` é 1 linha/transação. Joins diretos duplicam receita. Agregue itens e pagamentos por `order_id` **antes** de juntar.
2. **`customer_id` ≠ cliente.** Use `customer_unique_id` para recorrência, LTV e coorte.
3. **Geolocation explode joins.** Agregue por `zip_code_prefix` (mediana de lat/lng) antes de usar.
4. **Reviews não são 1:1 com pedidos.** 547 pedidos têm múltiplas avaliações — decida a regra (primeira, última ou média) explicitamente.
5. **Vazamento temporal.** `order_delivered_customer_date`, `review_score` e datas de entrega só existem depois do fato. Para prever atraso ou satisfação no momento da compra, essas colunas não podem entrar como features.
6. **Bordas da série.** Descarte set-dez/2016 e set-out/2018.

## 6. Caminhos de modelagem sugeridos

| Problema | Alvo | Observação |
|---|---|---|
| Previsão de atraso na entrega | `entregue > prazo prometido` (8,1% positivo) | Melhor caso de uso. Features disponíveis no ato da compra: UF cliente/vendedor, distância, peso, dimensões, categoria, preço, frete, mês |
| Previsão de lead time | dias até entrega (regressão) | Alvo contínuo, cauda longa — considerar log |
| Previsão de insatisfação | `review_score ≤ 2` | Fortemente mediada por atraso; cuidado com vazamento |
| NLP em comentários | texto de 40.977 avaliações | Sentimento / clusterização de motivos de queixa |
| Segmentação RFM | `customer_unique_id` | Limitado: no máximo ~3,5% dos clientes recompram na janela |

**Recomendação:** começar por previsão de atraso na entrega. É o problema com sinal mais claro nos dados, sem vazamento se as features forem restritas ao momento da compra, e com valor de negócio direto — atraso corta a nota média de 4,29 para 2,57.
