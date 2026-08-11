# Capítulo 6: Treinando modelos de raciocínio com aprendizado por reforço

&nbsp;

&nbsp;
## Materiais complementares

- [rlvr_grpo_original_no_kl.py](rlvr_grpo_original_no_kl.py): script que implementa o algoritmo GRPO original para treinar um modelo de raciocínio usando aprendizado por reforço com rewards verificáveis (RLVR). O algoritmo foi usado pelo [DeepSeek R1](https://arxiv.org/abs/2501.12948) e originalmente proposto no artigo do [DeepSeekMath](https://arxiv.org/abs/2402.03300). No entanto, este script omite o termo de divergência de KL (como recomendado por [DAPO](https://arxiv.org/abs/2503.14476), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [Olmo 3](https://arxiv.org/abs/2512.13961) e outros)
  - O termo de divergência de KL garante que o modelo treinado não se desvie demais do modelo original, mas pode prejudicar o desempenho (especialmente em tarefas de matemática)
  - Este script implementa conceitualmente o mesmo código do capítulo 6; no entanto, há dois pequenos ajustes de desempenho:
    1. Remoção do cast `.cpu()` no sampler `torch.multinomial`, para melhorar o throughput em 20%; veja o [PR #178](https://github.com/rasbt/reasoning-from-scratch/pull/178) para mais informações sobre como isso foi implementado
    2. Pular a atualização do modelo quando todos os rewards são iguais, ao usar a flag `--skip-zero-advantage-updates`, o que acelera ainda mais o treinamento e pode reduzir os requisitos de memória (porque as sequências longas, que excedem o `--max_new_tokens`, são as mais caras e frequentemente resultam em rewards zero, já que a resposta correta não é incluída por atingir o limite de tokens antes de gerá-la); veja o [PR #186](https://github.com/rasbt/reasoning-from-scratch/pull/186) para mais informações sobre como isso foi implementado
    - Caso você queira ver o script sem essas duas melhorias mencionadas acima, pode visualizar o código original [aqui](https://github.com/rasbt/reasoning-from-scratch/blob/da009e41aacb17a433968cf84a4a6cf2a0fa4655/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py)
- [rlvr_grpo_original_no_kl_batched.py](rlvr_grpo_original_no_kl_batched.py): igual ao anterior, mas com suporte a treinamento em batches. Note, porém, que isso aumenta os requisitos de memória e pode, portanto, exigir a redução do número de rollouts e dos comprimentos de rollout. O uso é o mesmo do script acima, exceto pela adição de `--num_batches`.
  - Note que, diferente do código [evaluate_math500_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py) do capítulo 3, este código não precisa importar o `Qwen3Model` de [qwen3_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3_batched.py), como explicado em mais detalhe no [PR #179](https://github.com/rasbt/reasoning-from-scratch/pull/179)

- [rlvr_grpo_original_no_kl_batched_fsdp.py](rlvr_grpo_original_no_kl_batched_fsdp.py): igual ao anterior, mas com suporte a treinamento em várias GPUs usando o FSDP do PyTorch. Este é o script recomendado para treinar se você tem acesso a várias GPUs. O uso é o mesmo do script acima, exceto pela adição de `--num_gpus`.

Os scripts importam algumas funcionalidades do pacote [`reasoning_from_scratch`](../../reasoning_from_scratch) para evitar duplicação de código. (Veja as [instruções de configuração do capítulo 2](../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.) No entanto, neste caso, o código também reimplementa as funções centrais do próprio capítulo, para permitir inspeção e modificação mais fáceis.



<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---


&nbsp;

|      | Método                                 | Step | Máx. de tokens | Nº de rollouts | Acurácia MATH-500 | Média de tokens |
| ---- | -------------------------------------- | ---- | ---------- | ------------ | ------------ | --------------- |
| 1    | Base (capítulo 3)                      | -    |            |              | 15,2%        | 78,85           |
| 2    | Reasoning (capítulo 3)                 | -    |            |              | 48,2%        | 1369,79         |
| 3    | GRPO original (capítulo 7)             | 50   | 512        | 8            | 33,4%        | 910,33          |
| 4    | GRPO original (capítulo 7)             | 100  | 512        | 8            | 0,4%         | 1168,05         |
| 5    | GRPO original, mas sem KL (este capítulo) | 50   | 512        | 8            | 47,4%        | 586,11          |
| 6    | GRPO original, mas sem KL (este capítulo) | 100  | 512        | 8            | 44,0%        | 555,95          |
| 7    | GRPO com mod. do Olmo 3 (capítulo 7)   | 50   | 512        | 8            | 46,4%        | 601,61          |
| 8    | GRPO com mod. do Olmo 3 (capítulo 7)   | 100  | 512        | 8            | 45,4%        | 589,51          |
| 9    | GRPO com mod. do DeepSeek V3.2 (capítulo 7) | 50   | 512        | 8            | 44,2%        | 618,49          |
| 10   | GRPO com mod. do DeepSeek V3.2 (capítulo 7) | 100  | 512        | 8            | 45,2%        | 676,96          |

Os checkpoints são salvos a cada 50 steps. Se você interromper um script com KeyboardInterrupt, ele também salvará o último step como checkpoint.

Note que o treinamento permite apenas até 512 tokens gerados (máx. de tokens na tabela acima), para torná-lo mais acessível em termos de memória de computação necessária.

No entanto, o script de avaliação (mesmo método do capítulo 3) permite até 2048 tokens gerados, e a coluna "Média de tokens" da tabela acima mede quantos tokens são usados em média sobre o dataset de teste MATH-500. (O treinamento é feito sobre os 12.000 exemplos do dataset MATH que não têm sobreposição com o conjunto de teste MATH-500. Veja [https://github.com/rasbt/math_full_minus_math500](https://github.com/rasbt/math_full_minus_math500) para mais detalhes.)

**Linha 1**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

- Dica: você pode adicionar `--show_eta` ao comando de execução acima para exibir uma estimativa do tempo total de execução do script na sua máquina

**Linha 2**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

**Linhas 3 e 4**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_advanced/rlvr_grpo_original.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

Depois, para avaliar o modelo, rode o script `evaluate_math500.py` sobre o checkpoint gerado. Por exemplo:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path checkpoints/rlvr_grpo_original/qwen3-0.6B-rlvr-grpo-step00050.pth
```

**Linhas 5 e 6**

```bash
uv run rlvr_grpo_original_no_kl.py \
--num_rollouts 8 \
--steps 100 \
--max_new_tokens 512
```

**Linhas 7 e 8**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_olmo3.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**Linhas 9 e 10**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_deepseek_v32.py \
--num_rollouts 8 \
--max_new_tokens 512 
```


<br>

Se você está com pouca RAM, considere reduzir o número de rollouts (`--num_rollouts`) ou o comprimento das respostas (`--max_new_tokens`). A tabela abaixo lista alguns requisitos de recursos, como referência.



| num_rollouts | max_new_tokens | RAM necessária (GB) |
| ------------ | -------------- | ----------------- |
| 8            | 1024           | 30,50 GB          |
| 8            | 512            | 20,31 GB          |
| 8            | 256            | 15,60 GB          |
| 4            | 1024           | 12,80 GB          |
| 4            | 512            | 14,60 GB          |
| 4            | 256            | 10,59 GB          |


Note que reduzir o número de tokens ou de rollouts provavelmente afetará o desempenho de forma negativa. Se você está usando um número baixo de rollouts, pode melhorar um pouco a estabilidade do treinamento aumentando o `--accum_steps` de 1 para 2 ou 4 (acumulação de gradiente); no entanto, isso exigirá mais tempo de computação.

Note que o método GRPO original ("vanilla") com essas configurações não é muito estável por mais de 50 steps, e você pode considerar as versões melhoradas do capítulo 7 se quiser treinar por mais de 50 steps.


<br>

Note que o algoritmo GRPO original pode ser melhorado de várias formas para estabilizar e melhorar o treinamento, que é o tema do [próximo capítulo](../../ch07).



&nbsp;
## Plotando execuções de treinamento

O [plot_metrics.py](plot_metrics.py) pode ser usado para plotar as execuções de treinamento em formato CSV. Um exemplo de execução com 200 steps está incluído na pasta `logs` (o arquivo de log foi criado com as configurações padrão, exceto pelo aumento de `--max_new_tokens 2048`):

```bash
uv run plot_metrics.py \
--csv logs/rlvr_grpo_original_no_kl_metrics.csv \
--moving_average 20
```

(A configuração `--moving_average 20` faz a média sobre 20% dos steps anteriores, para uma linha de tendência mais suave.)

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch06/other/plot.webp?1" width="600px">
