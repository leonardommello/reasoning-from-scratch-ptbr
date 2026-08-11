# Qwen3 otimizado

A implementação do Qwen3 feita do zero, usada neste livro, equilibra ser eficiente (tanto em CPU quanto em GPU) e enxuta, permanecendo fácil de ler por uma pessoa.

Como alternativa, você pode usar o `Qwen3Model` opcional, que é um substituto direto e um pouco mais eficiente em GPU. A versão otimizada, em [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) (discutida em mais detalhe no apêndice C), difere da implementação de referência em [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) de duas formas principais:

- Implementa a attention usando o `torch.nn.functional.scaled_dot_product`, embutido no PyTorch, em vez de uma implementação própria.
- Introduz um `KVCache` modificado, que pré-aloca os tensores de key/value. Isso aumenta o uso de memória, mas evita alocar novo espaço repetidamente durante a execução.


Para explorar as diferenças, recomendo abrir [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) e [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) lado a lado e/ou olhar um diff dos arquivos:

<br>

![](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/optimized-LLM/vscode.webp)

<br>

&nbsp;
## Como usar

O código otimizado pode ser usado como substituto direto do código usado nos capítulos principais, como mostrado abaixo.

**Antes:**

```python
from reasoning_from_scratch.qwen3 import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
```


**Depois:**

```python
from reasoning_from_scratch.qwen3_optimized import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
```

&nbsp;
## Como rodar comparações

Para avaliar o desempenho no seu sistema, você pode usar a função [`compare_inference.py`](compare_inference.py), contida nesta pasta:

```python
python compare_inference.py
```

ou

```python
uv run compare_inference.py
```

Depois, adicione as seguintes flags:

- `--device`: seleciona o dispositivo, por exemplo, `cpu`, `mps` ou `cuda`
- `--cache`: habilita o KV cache
- `--compile`: usa o `torch.compile`
- `--reasoning`: usa a variante de raciocínio do Qwen3 em vez do modelo base. O modelo base gera aproximadamente 50 tokens em resposta ao prompt fornecido. A variante de raciocínio gera cerca de 2000 tokens.
- `--optimize`: usa o modelo otimizado de `qwen3_optimized.py` em vez do modelo padrão de `qwen3.py`.

<br>

&nbsp;
### Modelo padrão



| Modelo   | Modo              | Comando                         | Hardware        | Tokens/sec    | Memória de GPU (VRAM) |
| -------- | ----------------- | ------------------------------- | --------------- | ------------- | ----------------- |
| qwen3.py | Normal            | --device cpu                    | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | Normal compilado  | --device cpu --compile          | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | KV cache          | --device cpu --cache            | Mac Mini M4 CPU | 28            | -                 |
| qwen3.py | KV cache compilado | --device cpu --compile --cache  | Mac Mini M4 CPU | 68            | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Normal            | --device mps                    | Mac Mini M4 GPU | 17            | -                 |
| qwen3.py | Normal compilado  | --device mps --compile          | Mac Mini M4 GPU | InductorError | -                 |
| qwen3.py | KV cache          | --device mps --cache            | Mac Mini M4 GPU | 18            | -                 |
| qwen3.py | KV cache compilado | --device mps --compile --cache  | Mac Mini M4 GPU | InductorError | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Normal            | --device cuda                   | NVIDIA H100 GPU | 51            | 1,55 GB           |
| qwen3.py | Normal compilado  | --device cuda --compile         | NVIDIA H100 GPU | 164           | 1,81 GB           |
| qwen3.py | KV cache          | --device cuda --cache           | NVIDIA H100 GPU | 48            | 1,52 GB           |
| qwen3.py | KV cache compilado | --device cuda --compile --cache | NVIDIA H100 GPU | 141           | 1,81 GB           |

<br>

&nbsp;
### Modelo otimizado


| Modelo             | Modo              | Comando                                     | Hardware        | Tokens/sec | Memória de GPU (VRAM) |
| ------------------ | ----------------- | ------------------------------------------- | --------------- | ---------- | ----------------- |
| qwen3_optimized.py | Normal            | --optimized --device cpu                    | Mac Mini M4 CPU | 5          | -                 |
| qwen3_optimized.py | Normal compilado  | --optimized --device cpu --compile          | Mac Mini M4 CPU | 7          | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device cpu --cache            | Mac Mini M4 CPU | 49         | -                 |
| qwen3_optimized.py | KV cache compilado | --optimized --device cpu --compile --cache  | Mac Mini M4 CPU | 51         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Normal            | --optimized --device mps                    | Mac Mini M4 GPU | 21         | -                 |
| qwen3_optimized.py | Normal compilado  | --optimized --device mps --compile          | Mac Mini M4 GPU | NameError  | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device mps --cache            | Mac Mini M4 GPU | 29         | -                 |
| qwen3_optimized.py | KV cache compilado | --optimized --device mps --compile --cache  | Mac Mini M4 GPU | 38         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Normal            | --optimized --device cuda                   | NVIDIA H100 GPU | 55         | 1,50 GB           |
| qwen3_optimized.py | Normal compilado  | --optimized --device cuda --compile         | NVIDIA H100 GPU | 173        | 1,81 GB           |
| qwen3_optimized.py | KV cache          | --optimized --device cuda --cache           | NVIDIA H100 GPU | 56         | 5,85 GB           |
| qwen3_optimized.py | KV cache compilado | --optimized --device cuda --compile --cache | NVIDIA H100 GPU | 177        | 5,85 GB           |

<br>

Comparando as 2 tabelas acima, podemos ver que a variante otimizada é claramente mais rápida em tokens/segundo na maioria dos casos.

No entanto, note que a versão não otimizada é mais rápida (68 tok/sec) que a versão otimizada (51 tok/sec) ao usar a versão compilada com KV cache.

A versão otimizada também usa mais RAM base (5,85 GB com KV cache) que a versão não otimizada (1,5 GB). Isso ocorre porque ela pré-aloca os tensores que guardam os valores de KV para o comprimento máximo de contexto suportado. (Assim, ao rodar a versão não otimizada com um prompt de 41k de contexto, o uso de RAM seria aproximadamente parecido.)

**Talvez a melhor recomendação seja usar a versão não otimizada (com `--cache` e `--compile`) ao usar CPU. Ao usar GPU, use a versão otimizada (com `--cache` e `--compile`).**
