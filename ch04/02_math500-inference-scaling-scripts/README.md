# Capítulo 4: Melhorando o raciocínio com inference-time scaling


&nbsp;
## Materiais complementares

- [cot_prompting_math500.py](cot_prompting_math500.py): script autônomo para avaliar modelos com prompting de chain-of-thought no dataset MATH-500
- [self_consistency_math500.py](self_consistency_math500.py): script autônomo para avaliar modelos com amostragem por self-consistency no dataset MATH-500
- [run_all_experiments_math500.sh](run_all_experiments_math500.sh): um script bash de conveniência que roda todos os experimentos (linhas 4 a 12) listados neste README abaixo

Ambos os scripts de avaliação importam funcionalidades do pacote [`reasoning_from_scratch`](../../reasoning_from_scratch) para evitar duplicação de código. (Veja as [instruções de configuração do capítulo 2](../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.)



<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---



&nbsp;

## Prompting com chain-of-thought

O script [`cot_prompting_math500.py`](self_consistency_math500.py) implementa o método de prompting com chain-of-thought do capítulo 4.

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F04_raschka.webp" width=600>

&nbsp;

A tabela abaixo compara esta abordagem (linha 3) com os baselines do capítulo 3:

|    | Método                                       | Modelo    | Acurácia | Tempo      |
|----|----------------------------------------------|-----------|----------|------------|
| 1  | Baseline (capítulo 3), greedy decoding       | Base      | 15,2%    | 10,1 min   |
| 2  | Baseline (capítulo 3), greedy decoding       | Reasoning | 48,2%    | 182,1 min  |
| 3  | Prompting com chain-of-thought ("CoT")       | Base      | 40,6%    | 84,5 min   |

Os valores de acurácia e os tempos de execução mostrados na tabela foram calculados sobre todas as 500 amostras do conjunto de teste MATH-500, usando uma GPU "cuda" (DGX Spark).

Para rodar o experimento da primeira linha, use:

```bash
python cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

Ou, com `uv:`


```bash
uv run cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

Para opções adicionais, use a flag `--help`.



&nbsp;
## Amostragem por self-consistency

O script [`self_consistency_math500.py`](self_consistency_math500.py) implementa o método de amostragem do capítulo 4.

(Opcionalmente, há a variante [`self_consistency_math500_batched.py`](self_consistency_math500_batched.py), que executa todas as `--num_samples` como um batch, para processamento mais rápido. Note, porém, que isso exige mais memória de computação.)

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F17_raschka.webp" width=600>

&nbsp;

A tabela abaixo compara esta abordagem (linhas 4 a 12) com os baselines do capítulo 3 (linhas 1 e 2):

|      | Método                                    | Modelo    | Acurácia | Tempo     |
| ---- | ----------------------------------------- | --------- | -------- | --------- |
| 1    | Baseline (capítulo 3), greedy decoding    | Base      | 15,2%    | 10,1 min  |
| 2    | Baseline (capítulo 3), greedy decoding    | Reasoning | 48,2%    | 182,1 min |
| 3    | Prompting com chain-of-thought ("CoT")    | Base      | 40,6%    | 84,5 min  |
| 4    | Temperature e top-p ("Top-p")             | Base      | 17,8%    | 30,7 min  |
| 5    | "Top-p" + self-consistency (n=3)          | Base      | 29,6%    | 97,6 min  |
| 6    | "Top-p" + self-consistency (n=5)          | Base      | 27,8%    | 116,8 min |
| 7    | "Top-p" + self-consistency (n=10)         | Base      | 31,6%    | 300,4 min |
| 8    | "Top-p" + "CoT"                           | Base      | 33,4%    | 129,2 min |
| 9    | Self-consistency (n=3) + "Top-p" + "CoT"  | Base      | 42,2%    | 211,6 min |
| 10   | Self-consistency (n=5) + "Top-p" + "CoT"  | Base      | 48,0%    | 452,9 min |
| 11   | Self-consistency (n=10) + "Top-p" + "CoT" | Base      | 52,0%    | 862,6 min |
| 12   | Self-consistency (n=3) + "Top-p" + "CoT"  | Reasoning | 55,2%    | 544,4 min |

Os valores de acurácia e os tempos de execução mostrados na tabela foram calculados sobre todas as 500 amostras do conjunto de teste MATH-500, usando uma GPU "cuda" (DGX Spark).

Os códigos a seguir dão instruções de como rodar os experimentos de self-consistency das linhas 4 a 12 (troque `uv run` por `python` se você não usa `uv`).

**Linha 4:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500
```

**Linha 5:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500
```

**Linha 6:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500
```

**Linha 7:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500
```

**Linha 8:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Linha 9:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Linha 10:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Linha 11:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Linha 12:**

```bash
uv run self_consistency_math500.py \
    --which_model "reasoning" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```


Para opções adicionais, use a flag `--help`.

