# Baixando e usando checkpoints de treinamento

Esta pasta explica como baixar e usar os checkpoints de treinamento do capítulo 7 do Hugging Face, em [https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints).

Os checkpoints são arquivos `state_dict` puros do PyTorch, para o pacote `reasoning_from_scratch`. Não são checkpoints do Hugging Face Transformers.

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---

&nbsp;
## Pastas de checkpoint disponíveis

- `7_3_plus_tracking`: checkpoints de GRPO com acompanhamento adicional de métricas
- `7_4_plus_clip_ratio`: checkpoints de GRPO com policy ratios clipados
- `7_5_plus_kl`: checkpoints de GRPO com um termo de KL
- `7_6_plus_format_reward`: checkpoints de GRPO com um format reward explícito para tags `<think>`

Os checkpoints estão hospedados em:

- [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints)

&nbsp;
## Baixando um checkpoint

Use `download_qwen3_grpo_checkpoints(...)` de [`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py):

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir="qwen3",
)
```

&nbsp;
## Qual tokenizer usar

Use o tokenizer base para:

- `7_3_plus_tracking`
- `7_4_plus_clip_ratio`
- `7_5_plus_kl`

Use o tokenizer de raciocínio para:

- `7_6_plus_format_reward`

O motivo é que o `7_6_plus_format_reward` foi treinado a partir do modelo de raciocínio e espera a formatação de chat do modelo de raciocínio.

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
    download_qwen3_grpo_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir=local_dir,
)
download_qwen3_small(kind="base", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(tokenizer_file_path=local_dir / "tokenizer-base.json")
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
## Exemplo com format reward

Para o `7_6_plus_format_reward`, troque para o tokenizer de raciocínio:

```python
from pathlib import Path

from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
)

local_dir = Path("qwen3")
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
```

&nbsp;
## Exemplo do capítulo 6

A mesma função auxiliar também suporta o checkpoint original do capítulo 6, sem KL:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050", out_dir="qwen3")
```

&nbsp;
## Checkpoints disponíveis

Correspondência com as seções:

- `no_kl`: baseline do capítulo 6, da configuração original de GRPO sem KL
- `tracking`: seção 7.3 do capítulo principal
- `clip_ratio`: seção 7.4 do capítulo principal
- `kl`: seção 7.5 do capítulo principal
- `format_reward`: seção 7.6 do capítulo principal

Steps salvos disponíveis:

- `no_kl`: `00050`, `00100`, `00500`, `01000`, `01500`, `03000`, `05000`, `09000`
- `tracking`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `clip_ratio`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `kl`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `format_reward`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
