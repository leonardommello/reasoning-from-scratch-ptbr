
# Benchmarking com MMLU

Este material complementar implementa três métodos diferentes para avaliar modelos no MMLU.
- O método 1 serve como uma introdução intuitiva
- O método 2 é o mais usado na prática
- O método 3 é um método mais robusto, mais adequado para modelos de raciocínio

- Note que o código carrega o [dataset MMLU](https://huggingface.co/datasets/cais/mmlu) do model hub do Hugging Face. Assim, você precisa instalar a biblioteca Python `datasets` antes de rodar o código:

```python
pip install datasets
```

ou

```python
uv add datasets
```

- Nas seções a seguir, aplicamos os métodos de avaliação do MMLU ao (`"high_school_mathematics"`)

- Note que há muitos outros subconjuntos interessantes; este foi escolhido por simplicidade e eficiência; você pode usar, por exemplo

  - Use `--subsets list` para listar outros subconjuntos disponíveis

  - Use, por exemplo, `--subsets "astronomy,high_school_mathematics"` para selecionar vários subconjuntos

  - Use `--subsets "all"` para avaliar em todos os subconjuntos

(Note que, por simplicidade e legibilidade do código, focamos em um cenário zero-shot, e não 5-shot.)

<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---

&nbsp;

## Método 1: correspondência de letra no MMLU

- Deixamos o modelo gerar a resposta
- Extraímos a primeira letra A/B/C/D gerada e a comparamos com a resposta correta
- Este é o método mais intuitivo, mas a desvantagem é que o modelo pode não responder com uma letra A/B/C/D

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_1.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model base     
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.200 [high_school_mathematics]
MMLU 150 acc=0.193 [high_school_mathematics]
MMLU 200 acc=0.235 [high_school_mathematics]
MMLU 250 acc=0.224 [high_school_mathematics]

MMLU letter accuracy: 58/270 = 21.48% in 69.1s
{'accuracy': 0.21481481481481482, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
qwen3-0.6B-reasoning.pth: 100% (1433 MiB / 1433 MiB)
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy: 57/270 = 21.11% in 43.6s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## Método 2: pontuação por log-probabilidade

- Passamos o prompt pelo modelo e obtemos as log-probabilidades (log-probs) do próximo token (veja o capítulo 4 para a discussão sobre log-probs)
- Para cada opção de letra, calculamos então qual ID de token apareceria primeiro se acrescentássemos aquela letra
- Em seguida, comparamos essas quatro log-probs e escolhemos a maior (max)

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_2.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.420 [high_school_mathematics]
MMLU 150 acc=0.400 [high_school_mathematics]
MMLU 200 acc=0.370 [high_school_mathematics]
MMLU 250 acc=0.344 [high_school_mathematics]

MMLU letter accuracy (log-prob): 93/270 = 34.44% in 22.5s
{'accuracy': 0.34444444444444444, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy (log-prob): 57/270 = 21.11% in 22.4s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## Método 3: teacher forcing

- Em vez de consultar a log-prob de cada uma das letras A/B/C/D, uma pontuação mais robusta (especialmente para modelos de raciocínio) é alimentar a letra junto com a string completa da resposta
- No nosso exemplo, as strings de resposta são "A. 7", "B. 11", "C. 16", "D. 8"
- Este método é conhecido pelo termo infeliz "teacher forcing"
- Este método é o mais confiável, mas a ressalva é que leva 4x mais tempo que a abordagem por log-probabilidade do método 2 (já que alimentamos o modelo com todas as 4 variantes de resposta)

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_3.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.310 [high_school_mathematics]
MMLU 150 acc=0.307 [high_school_mathematics]
MMLU 200 acc=0.315 [high_school_mathematics]
MMLU 250 acc=0.312 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 86/270 = 31.85% in 67.9s
{'accuracy': 0.31851851851851853, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.250 [high_school_mathematics]
MMLU 150 acc=0.267 [high_school_mathematics]
MMLU 200 acc=0.255 [high_school_mathematics]
MMLU 250 acc=0.280 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 78/270 = 28.89% in 68.8s
{'accuracy': 0.28888888888888886, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



## Baseline de chute aleatório

- Este baseline de chute aleatório serve apenas para colocar os números acima em perspectiva

- Espera-se que um modelo que chuta aleatoriamente, com probabilidade uniforme (igual) entre todas as respostas, alcance $25\%$ de acurácia

- No entanto, para quem chuta aleatoriamente, podemos esperar desvios em relação aos $25\%$ (dependendo do tamanho da amostra)

- Por exemplo, podemos modelar uma rodada de avaliação como uma binomial com $K$ acertos em $n$ questões:

  - $K \sim \mathrm{Binomial}(n,p)$ com $p=\tfrac14$ e $n=$ número de questões.
  - Acurácia $A = K/n$.

- Vamos percorrer isso para o subconjunto *high_school_mathematics*, com $n=270$

- Em geral, as propriedades da binomial são:

  - Média: $\mathbb{E}[K] = np$
  - Desvio padrão: $\sigma_K = \sqrt{np(1-p)}$

- Para a acurácia $A=K/n$:

  - Média: $\mathbb{E}[A] = p = 0.25$
  - Desvio padrão: $\sigma_A = \sqrt{\tfrac{p(1-p)}{n}}$

- Substituindo $n=270$:

  - $\mathbb{E}[A] = 25\%$
  - $\sigma_A = \sqrt{\tfrac{0.25\cdot 0.75}{270}} \approx 2.64\%$

- Convertendo os limites de acurácia de um desvio padrão ($\pm 1\sigma$) em contagens:

  - Inferior: $K \le \lfloor 270\,(0.25-0.02636)\rfloor = 60$
  - Superior: $K \ge \lceil 270\,(0.25+0.02636)\rceil = 75$
  - (Dentro da faixa está $K=61,\dots,74$; equivalentemente $A\in[22.36\%,\,27.64\%]$)

- Assim, a probabilidade de cair fora desse limite é:

  $$
  z = \pm\,\frac{75-67.5}{\sqrt{270\cdot 0.25\cdot 0.75}} \approx \pm 1.054, \qquad
  \Pr(|A-0.25|>0.02636) \approx 2\bigl(1-\Phi(1.054)\bigr) \approx 0.292.
  $$

  Ou seja, cerca de 29,2% das rodadas de chute aleatório ficam abaixo de 22,36% ou acima de 27,64%

- Isso significa que, em cerca de $29,2\%$ dos casos em que o modelo está chutando aleatoriamente (assumindo distribuição uniforme), obtemos uma acurácia abaixo de $22,36\%$ ou acima de $27,64\%$
- Abaixo, uma verificação empírica:


```bash
➜  02_mmlu git:(main) ✗ uv run 0_random_guessing_baseline.py --subset "high_school_mathematics"
Subset: high_school_mathematics | split: test | n=270
Gold distribution provided in the dataset:
  A: 57 (21.11%)
  B: 71 (26.30%)
  C: 71 (26.30%)
  D: 71 (26.30%)

Random guessing over 10,000 trials (uniform A/B/C/D, seed=42):
  Mean accuracy: 24.98%
  Std dev across trials: 2.65%

Selected quantiles of accuracy:
  1% quantile: 18.889%
  5% quantile: 20.741%
  25% quantile: 23.333%
  50% quantile: 24.815%
  75% quantile: 26.667%
  95% quantile: 29.259%
  99% quantile: 31.111%

Full frequency table of accuracies (rounded):
  0.160: 1 times (0.01%)
  0.170: 11 times (0.11%)
  0.180: 38 times (0.38%)
  0.190: 124 times (1.24%)
  0.200: 302 times (3.02%)
  0.210: 562 times (5.62%)
  0.220: 612 times (6.12%)
  0.230: 1254 times (12.54%)
  0.240: 1619 times (16.19%)
  0.250: 1096 times (10.96%)
  0.260: 1525 times (15.25%)
  0.270: 1248 times (12.48%)
  0.280: 572 times (5.72%)
  0.290: 565 times (5.65%)
  0.300: 281 times (2.81%)
  0.310: 132 times (1.32%)
  0.320: 28 times (0.28%)
  0.330: 24 times (0.24%)
  0.340: 5 times (0.05%)
  0.360: 1 times (0.01%)
```
