# Material complementar do capítulo 8: treinar com destilação

Esta pasta contém um script simples de destilação para treinar o modelo Qwen3 0.6B com traços de raciocínio gerados por um professor, como abordado no capítulo 8.

&nbsp;
## Arquivos

- [distill.py](distill.py): treina o Qwen3 0.6B com dados de destilação em formato JSON (mais sobre o formato na próxima seção).
  - Por padrão, treina o modelo base com o tokenizer base
  - Se você passar `--use_think_tokens`, ele usa o tokenizer de raciocínio e envolve o traço de raciocínio como `<think>...</think>` antes da resposta final, de forma parecida com o que é feito no capítulo 8
  - Após cada epoch, salva um checkpoint em `checkpoints/distill/` e acrescenta métricas de treinamento em `logs/distill_metrics.csv`
  - Se você inicializar a partir de `--checkpoint_path` (opcional), em vez do modelo base, pode continuar um checkpoint já existente
- [distill_batched.py](distill_batched.py): versão em batch do script acima.
  - Usa a implementação do Qwen3 em batch com suporte a padding, para que exemplos de comprimentos diferentes possam ser treinados juntos
  - Acrescenta um argumento `--batch_size`, para processar vários exemplos por step de otimização
  - Salva checkpoints em `checkpoints/distill_batched/` e acrescenta métricas em `logs/distill_batched_metrics.csv`
  - Naturalmente, note que a variante em batch usa bem mais memória de GPU (dependendo do batch size)

O script importa funcionalidades compartilhadas do pacote [`reasoning_from_scratch`](../../reasoning_from_scratch) para evitar duplicar o código de carregamento do modelo e de formatação de prompt. (Veja as [instruções de configuração do capítulo 2](../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.)


<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---


&nbsp;
## Formato dos dados de entrada

A entrada é a saída JSON produzida por [`../02_generate_distillation_data`](../02_generate_distillation_data). Cada linha deve ser assim:

```json
{
  "problem": "Compute 1/2 + 1/6.",
  "gtruth_answer": "2/3",
  "message_thinking": "I will rewrite the fractions with a common denominator.",
  "message_content": "The final answer is \\boxed{\\tfrac{2}{3}}."
}
```

Para o treinamento, apenas os seguintes campos são usados:

- `problem`: inserido no mesmo template de prompt de matemática usado no capítulo 3
- `message_content`: obrigatório; usado como resposta-alvo supervisionada
- `message_thinking`: opcional; se presente, é colocado antes de `message_content`

Linhas com campos ausentes ou malformados são puladas automaticamente, e exemplos mais longos que `--max_seq_len` são filtrados antes da divisão entre treinamento e validação.


&nbsp;
## Exemplo de execução

Para uma verificação rápida de sanidade, você pode treinar com uma pequena amostra gerada na pasta anterior:

```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1
```

Isso vai:

- carregar os pesos base do Qwen3 0.6B
- tokenizar os pares de prompt/resposta
- reservar 1 exemplo para validação
- salvar um checkpoint após cada epoch em `checkpoints/distill/`
- escrever métricas em CSV em `logs/distill_metrics.csv`

Se você quiser treinar com tags de raciocínio explícitas e o tokenizer de raciocínio, adicione `--use_think_tokens`:

```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1 \
  --use_think_tokens
```

Se você quiser treinar em batches, rode:

```bash
uv run distill_batched.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --batch_size 2 \
  --log_every 1
```


&nbsp;
## Opções úteis

```bash
uv run distill.py --help
```

Argumentos importantes:

- `--data_path`: caminho para o arquivo JSON de destilação
- `--dataset_size`: trunca o dataset antes da divisão (`0` usa todas as linhas)
- `--validation_size`: número absoluto de exemplos de validação
- `--epochs`: número de passagens sobre a partição de treinamento
- `--batch_size`: número de exemplos por step de otimização no `distill_batched.py`
- `--lr`: learning rate do AdamW
- `--max_seq_len`: descarta exemplos cuja sequência de prompt + resposta seja maior que este limite
- `--checkpoint_path`: inicializa a partir de um checkpoint de destilação anterior
- `--grad_clip_norm`: clipping de gradiente opcional
- `--use_think_tokens`: troca para o tokenizer de raciocínio e a formatação `<think>...</think>`

Veja a seção "Experimentos" abaixo para exemplos práticos.

&nbsp;

## Avaliando um checkpoint destilado

Depois do treinamento, você pode avaliar um checkpoint no MATH-500 usando o script de avaliação do capítulo 3.

Se você treinou sem `--use_think_tokens`, avalie-o como um modelo `base`:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model base \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```

**Importante:** se você treinou com `--use_think_tokens`, avalie-o como um modelo `reasoning`, para que o tokenizer de raciocínio seja usado:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model reasoning \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```


&nbsp;
## Experimentos

Os datasets de destilação usados no capítulo 8 estão disponíveis no meu repositório do Hugging Face, em [rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill). No capítulo 8, eles são carregados por uma função auxiliar que baixa as partições, por exemplo:

````python
from reasoning_from_scratch.ch08 import load_distill_data

_ = load_distill_data(
    partition="deepseek-r1-math-train.json",
    local_path="deepseek-r1-math-train.json"
)
_ = load_distill_data(
    partition="qwen3-235b-a22b-math-train.json",
    local_path="qwen3-235b-a22b-math-train.json"
)
````



Para os experimentos abaixo, usei os arquivos `deepseek-r1-math-train.json` e `qwen3-235b-a22b-math-train.json` dessa coleção de datasets.


&nbsp;

|      | Dados do professor                   | Epoch | Acurácia MATH-500 | Val loss final |
| ---- | ------------------------------------ | ----- | ------------ | -------------- |
| 1    | Base (capítulo 3)                    | -     | 15,2%        | -              |
| 2    | Reasoning (capítulo 3)               | -     | 48,2%        | -              |
| 3    | Dados de destilação do DeepSeek R1   | 1     | 30,6%        | 0,5436         |
| 4    | Dados de destilação do DeepSeek R1   | 2     | 32,4%        | 0,5349         |
| 5    | Dados de destilação do DeepSeek R1   | 3     | 33,6%        | 0,5343         |
| 6    | Dados de destilação do Qwen3 235B A22B | 1   | 45,0%        | 0,4043         |
| 7    | Dados de destilação do Qwen3 235B A22B | 2   | 43,8%        | 0,3963         |
| 8    | Dados de destilação do Qwen3 235B A22B | 3   | 44,2%        | 0,3948         |

O treinamento leva cerca de 30 min em uma H100 e cerca de 3 horas em um DGX Spark, e usa até 15 GB de RAM.

Abaixo estão os trechos de código para reproduzir os resultados reportados na tabela.

&nbsp;
**Linha 1**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

&nbsp;
**Linha 2**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

&nbsp;
**Linhas 3, 4 e 5**

```bash
uv run distill.py \
--data_path deepseek-r1-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

Depois, para avaliar os checkpoints de cada epoch, rode:

&nbsp;
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run-1/checkpoints/distill/qwen3-0.6B-distill-step06682-epoch1.pth
```

Para as linhas 4 e 5, troque o caminho do checkpoint por `...step13364-epoch2.pth` e `...step20046-epoch3.pth`, respectivamente.

&nbsp;
**Linhas 6, 7 e 8**

```bash
uv run distill.py \
--data_path qwen3-235b-a22b-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

Depois, para avaliar os checkpoints de cada epoch, rode:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run_11/checkpoints/distill/qwen3-0.6B-distill-step05746-epoch1.pth
```

Para as linhas 7 e 8, troque o caminho do checkpoint por `...step11492-epoch2.pth` e `...step17238-epoch3.pth`, respectivamente.
