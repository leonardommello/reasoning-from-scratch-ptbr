# Capítulo 5: Inference-time scaling via self-refinement


&nbsp;
## Materiais complementares

- [self_refinement_math500.py](self_refinement_math500.py): script autônomo para avaliar modelos com self-refinement no dataset MATH-500

O script importa funcionalidades do pacote [`reasoning_from_scratch`](../../reasoning_from_scratch) para evitar duplicação de código. (Veja as [instruções de configuração do capítulo 2](../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.)



<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---



&nbsp;

## Self-refinement

O script [`self_refinement_math500.py`](self_refinement_math500.py) implementa o método de self-refinement do capítulo 5.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch05/CH05_F21_raschka.webp" width=600>

&nbsp;



| #  | Método          | Scorer    | Iterações  | Modelo    | Acurácia | Tempo     |
|----|-----------------|-----------|------------|-----------|----------|-----------|
| 1  | Baseline (ch03) | -         | -          | Base      | 15,2%    | 10,1 min  |
| 2  | Self-refinement | Nenhum    | 1          | Base      | 25,0%    | 84,8 min  |
| 3  | Self-refinement | Nenhum    | 2          | Base      | 22,0%    | 165,4 min |
|    |                 |           |            |           |          |           |
| 4  | Self-refinement | Heurística | 1          | Base      | 21,6%    | 84,7 min  |
| 5  | Self-refinement | Heurística | 2          | Base      | 20,8%    | 151,4 min |
|    |                 |           |            |           |          |           |
| 6  | Self-refinement | Logprob   | 1          | Base      | 21,4%    | 85,3 min  |
| 7  | Self-refinement | Logprob   | 2          | Base      | 22,0%    | 165,3 min |
|    |                 |           |            |           |          |           |
| 8  | Self-refinement | Logp-ex   | 1          | Base      | 20,4%    | 85,0 min  |
| 9  | Self-refinement | Logp-ex   | 2          | Base      | 21,2%    | 160,2 min |
|    |                 |           |            |           |          |           |
| 10 | Baseline (ch03) | -         | -          | Reasoning | 48,2%    | 182,1 min |
| 11 | Self-refinement | Nenhum    | 1          | Reasoning | 56,6%    | 498,8 min |
| 12 | Self-refinement | Nenhum    | 2          | Reasoning | 56,6%    | 713,9 min |
|    |                 |           |            |           |          |           |
| 13 | Self-refinement | Heurística | 1          | Reasoning | 57,8%    | 498,6 min |
| 14 | Self-refinement | Heurística | 2          | Reasoning | 57,8%    | 713,9 min |
|    |                 |           |            |           |          |           |
| 15 | Self-refinement | Logprob   | 1          | Reasoning | 48,4%    | 499,7 min |
| 16 | Self-refinement | Logprob   | 2          | Reasoning | 48,6%    | 753,0 min |

Os valores de acurácia e os tempos de execução mostrados na tabela foram calculados sobre todas as 500 amostras do conjunto de teste MATH-500, usando uma GPU "cuda" (DGX Spark).

Os códigos a seguir dão instruções de como rodar os experimentos de self-consistency das linhas 4 a 12 (troque `uv run` por `python` se você não usa `uv`).

**Linha 2:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**Linha 3:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**Linha 4:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**Linha 5:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**Linha 6:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**Linha 7:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```

**Linha 8:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob_extract"
```

**Linha 9:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob_extract"
```

**Linha 11:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**Linha 12:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**Linha 13:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**Linha 14:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**Linha 15:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**Linha 16:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```




&nbsp;

## Self-consistency com desempate baseado em scorer

O [`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) estende a self-consistency com desempate baseado nos scorers implementados no capítulo 5.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/majority-vote.webp" width=600>

&nbsp;



|   | Método                                     | Modelo | Acurácia | Tempo     |
|---|--------------------------------------------|-------|----------|-----------|
| 1 | Baseline do capítulo 4 com prompting de CoT | Base  | 33,4%    | 129,2 min |
| 2 | Self-consistency (n=3) + voto majoritário  | Base  | 43,2%    | 328,2 min |
| 3 | Self-consistency (n=3) + heurística        | Base  | 43,4%    | 326,5 min |
| 4 | Self-consistency (n=3) + logprob médio     | Base  | 44,8%    | 327,7 min |


Os valores de acurácia e os tempos de execução mostrados na tabela foram calculados sobre todas as 500 amostras do conjunto de teste MATH-500, usando uma GPU "cuda" (DGX Spark).

Os códigos a seguir dão instruções de como rodar os experimentos de self-consistency das linhas 2 a 4 (troque `uv run` por `python` se você não usa `uv`).

**Linha 2:**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "none"
```

**Linha 3:**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "heuristic"
```

**Linha 4:**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "logprob"
```

&nbsp;

## Best-of-N

O [`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) implementa a abordagem de inference-scaling best-of-N.

O best-of-N é parecido com a self-consistency no sentido de que geramos várias respostas. No entanto, em vez de escolher a resposta final por voto majoritário, pontuamos todas as respostas geradas usando uma função de score.

O [`best_of_n_math500.py`](best_of_n_math500.py) estende a self-consistency com desempate baseado nos scorers implementados no capítulo 5.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/best-of-n.webp" width=600>

&nbsp;

|   | Método                                    | Modelo | Acurácia | Tempo     |
|---|-------------------------------------------|-------|----------|-----------|
| 1 | Baseline com prompting de chain-of-thought | Base  | 33,4%    | 129,2 min |
| 2 | Best-of-N (n=3) + heurística              | Base  | 40,6%    | 327,7 min |
| 3 | Best-of-N (n=3) + logprob médio           | Base  | 43,2%    | 330,2 min |


Os valores de acurácia e os tempos de execução mostrados na tabela foram calculados sobre todas as 500 amostras do conjunto de teste MATH-500, usando uma GPU "cuda" (DGX Spark).

Os códigos a seguir dão instruções de como rodar os experimentos de self-consistency das linhas 2 e 3 (troque `uv run` por `python` se você não usa `uv`).

**Linha 2:**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "heuristic"
)
```

**Linha 3:**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "logprob"
)
```
s
