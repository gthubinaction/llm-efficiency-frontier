# Base empírica — LLM scaling (N, D, C, loss)

## Arquivos
- `llm_scaling_dataset.csv` — 245 pontos reais reconstruídos (nuvem completa N×D).
- `llm_scaling_frontier.csv` — 15 pontos da fronteira compute-óptima (envelope de menor loss por faixa de N); base recomendada para o modelo L(N).

## Procedência (o que responde DIRETO ao Revisor 1)
Os pontos vêm da reconstrução pública da Figura 4 de Hoffmann et al. (2022),
feita por Besiroglu, Erdil, Barnett & You (2024), "Chinchilla Scaling: A
replication attempt" (arXiv:2404.10102), com código e dados abertos no
repositório Epoch AI `epoch-research/analyzing-chinchilla`
(arquivo `data/svg_extracted_data.csv`).

Isso substitui a extração manual via WebPlotDigitizer por uma fonte:
- citável (preprint + repositório versionado),
- reproduzível por terceiros (não depende de digitalização manual sua),
- de escala muito maior (245 pontos vs. os 17 originais).

## Dicionário de dados
| coluna | significado | unidade |
|---|---|---|
| N | parâmetros não-embedding do modelo | contagem |
| D | tokens de treino, derivado de D = C / (6N) | tokens |
| C_flop | compute de treino (da figura) | FLOP |
| loss | cross-entropy de validação reconstruída | nats/token |
| tokens_per_param | D/N (sanidade: mediana ≈ 22,5 ≈ regra dos ~20 do Chinchilla) | — |
| source | rótulo de procedência | — |

## Faixas reais (importante para claims de extrapolação)
- N: 5,7×10⁷ a 1,6×10¹⁰  → **~2,5 ordens de grandeza**, NÃO 6 nem 10.
- D: 2,4×10⁸ a 3,2×10¹¹
- C: 1,4×10¹⁸ a 1,3×10²² FLOP
- loss: 2,08 a 5,01 (escala consistente: um único tokenizer/dataset)

O threshold de eficiência ~10⁹ fica DENTRO desse range → é interpolação,
não extrapolação. Mais defensável que a versão original.

## Caveats honestos (para a metodologia)
1. Todos os pontos são da família Chinchilla (um tokenizer/dataset). Bom: a
   loss está numa escala única, sem o problema apples-to-oranges de misturar
   Kaplan + Hoffmann. Limite: não cobre o regime N<5×10⁷.
2. O ajuste L vs N na NUVEM completa dá R²≈0,26 (a dispersão vem da dimensão
   D). Isso é evidência empírica de que o framing só-N precisa da redução
   pela fronteira compute-óptima — não de uma asserção. Use a fronteira.
3. Se quiser estender para N baixo (regime "early"), a opção é sobrepor a
   forma fechada de Kaplan L(N)=(Nc/N)^αN (Nc≈8,8×10¹³, αN≈0,076) rotulada
   explicitamente como "forma funcional publicada", NUNCA como dado bruto,
   e ciente de que a escala de loss difere de Hoffmann.

## Citações a incluir
- Hoffmann, J. et al. (2022). Training compute-optimal large language models. arXiv:2203.15556.
- Besiroglu, T., Erdil, E., Barnett, M., & You, J. (2024). Chinchilla Scaling: A replication attempt. arXiv:2404.10102.
- Kaplan, J. et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
