# Baixando e usando checkpoints de treinamento

Esta pasta explica como baixar e usar os checkpoints de destilação do capítulo 8 do model hub do Hugging Face, em [https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints).

Os checkpoints são arquivos `state_dict` puros do PyTorch, para o pacote `reasoning_from_scratch`. Não são checkpoints do Hugging Face Transformers.

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---

&nbsp;
## Pastas de checkpoint disponíveis

- `ch08_distill_deepseek_r1`: os 3 checkpoints de destilação do DeepSeek-R1 usados nas linhas 3 a 5 do [`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb)
- `ch08_distill_qwen3_235b_a22b`: os 3 checkpoints de destilação do Qwen3 235B A22B usados nas linhas 6 a 8 do [`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb)

Os checkpoints estão hospedados em:

- [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints)

&nbsp;
## Baixando um checkpoint

Use `download_qwen3_distill_checkpoints(...)` de [`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py):

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir="qwen3",
)
```

&nbsp;
## Qual tokenizer usar

Use o tokenizer de raciocínio para:

- `ch08_distill_deepseek_r1`
- `ch08_distill_qwen3_235b_a22b`

&nbsp;

## Exemplo de uso

O exemplo abaixo baixa um checkpoint, baixa o tokenizer correspondente, carrega o modelo e gera texto com o `generate_text_basic_stream_cache` do capítulo 2:

```python
from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_distill_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir=local_dir,
)
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
model = Qwen3Model(QWEN_CONFIG_06_B)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

prompt = render_prompt("Solve: If x + 7 = 19, what is x?")
input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
):
    token_id = token.squeeze(0).item()
    print(tokenizer.decode([token_id]), end="", flush=True)
```

&nbsp;
## Exemplo com Qwen3

Para `ch08_distill_qwen3_235b_a22b`, use a mesma função auxiliar com o outro `distill_type`:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

download_qwen3_distill_checkpoints(
    distill_type="qwen3_235b_a22b",
    step="05746",
    out_dir="qwen3",
)
```

&nbsp;
## Steps disponíveis

Steps salvos disponíveis para `deepseek_r1`:

- `06682`
- `13364`
- `20046`

Steps salvos disponíveis para `qwen3_235b_a22b`:

- `05746`
- `11492`
- `17238`
