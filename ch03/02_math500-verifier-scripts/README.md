# Capítulo 3: Avaliando modelos de raciocínio

&nbsp;


&nbsp;
## Materiais complementares

- [evaluate_math500.py](evaluate_math500.py): script autônomo para avaliar modelos no dataset MATH-500
- [evaluate_math500_batched.py](evaluate_math500_batched.py): igual ao anterior, mas processa vários exemplos em paralelo durante a geração (para maior throughput)
- [evaluate_json.py](evaluate_json.py): avalia arquivos JSON/JSONL de registros salvos e reporta a acurácia

Ambos os scripts de avaliação importam funcionalidades do pacote [`reasoning_from_scratch`](../../reasoning_from_scratch) para evitar duplicação de código. (Veja as [instruções de configuração do capítulo 2](../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.)



<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---



&nbsp;

## Uso do `evaluate_math500.py`

Rode com:

```bash
python evaluate_math500.py
```

Ou, com `uv:`


```bash
uv run evaluate_math500.py
```

Opções:

```bash
uv run evaluate_math500.py --help

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to use: "auto" (default) or any torch device string
                        (e.g., "cpu", "cuda", "cuda:0", "mps").
  --which_model {base,reasoning}
                        Model variant to load (default: "base").
  --dataset_size DATASET_SIZE
                        Number of MATH-500 examples to evaluate (default: 10).
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens to generate (default: 2048).
  --compile             Enable torch.compile.
  --verbose             Print per-sample correctness while evaluating.
```

&nbsp;
## Uso do `evaluate_math500_batch.py`

Esta versão estende o batching para a própria geração, permitindo decodificação em paralelo:

```bash
uv run evaluate_math500_batched.py --help
```

Opções extras:

```bash
  --batch_size BATCH_SIZE
                        Number of examples to generate in parallel (default: 4).
  --disable_efficient_mode
                        Use a simpler batched inference method. Slower and more
                        memory-intensive, but easier to debug.
```


&nbsp;


**Nota de implementação:**
Por padrão, a geração em batch interrompe as sequências que emitem um stop token. Com `--disable_efficient_mode`, todas as sequências continuam até a mais longa terminar. Isso afeta apenas a eficiência computacional, não os resultados qualitativos, já que os tokens depois do stop token são descartados.

&nbsp;

**Dica (dispositivos MPS):**
Rode com:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run evaluate_math500_batched.py
```

Algumas operações do PyTorch usadas na inferência em batch eficiente ainda não são suportadas no MPS. Como alternativa, você também pode usar `--disable_efficient_mode`.



&nbsp;

- `evaluate_math500.py --dataset_size 500`


| Dispositivo / tamanho do dataset            | Modelo base | Modelo de raciocínio |
| ------------------------------------------- | ---------- | --------------- |
| **Mac Mini M4 CPU** (500 exemplos, sequencial | 43,6 min | Não rodou (esquentou demais)           |
| **Mac Mini M4 GPU** (500 exemplos, sequencial) | 37,5 min | Não rodou (esquentou demais) |
| **DGX Spark** (500 exemplos, sequencial) | 10,0 min  | 182,2 min      |
| **H100 GPU** (500 exemplos, sequencial) | 13,3 min  | 185,4 min      |

<br>
<br>

- `evaluate_math500_batched.py --dataset_size 500 --batch_size 128`

| Dispositivo / tamanho do dataset                             | Modelo base | Modelo de raciocínio |
| ------------------------------------------------------------ | ---------- | --------------- |
| **Mac Mini M4 CPU** (500 exemplos, em batch, `--batch_size 128`) | 167,2 min | Não rodou (esquentou demais)           |
| **Mac Mini M4 GPU** (500 exemplos, em batch, `--batch_size 128`) | Erro*     | Erro           |
| **DGX Spark** (500 exemplos, em batch, `--batch_size 128`)    | 16,3 min  | 119,3 min      |
| **H100 GPU** (500 exemplos, em batch, `--batch_size 128`)     | 3,3 min   | 14,6 min       |



- A acurácia do modelo base é 15,6% (78/500); a acurácia do modelo de raciocínio é 50,8% (254/500).


&nbsp;
## Uso do `evaluate_json.py`

Use isto se você já tem registros salvos e quer apenas (re)calcular a acurácia:

```bash
uv run evaluate_json.py --json_path math500_base-mps-evaluate-script.jsonl
# Accuracy 15.6% (78/500)

Chaves opcionais:

```bash
uv run evaluate_json.py \
  --json_path my_records.json \
  --gtruth_answer "gtruth_answer" \
  --generated_text "generated_text"
```

