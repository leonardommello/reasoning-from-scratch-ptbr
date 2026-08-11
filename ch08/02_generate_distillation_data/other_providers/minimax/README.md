# Provedor de destilação MiniMax

Esta pasta contém o script de geração hospedada específico do MiniMax, para a geração de dados de destilação do capítulo 8.

Rode os comandos abaixo a partir de `ch08/02_generate_distillation_data/`, para que caminhos relativos como `math_train_sample.json` continuem funcionando como estão escritos.

Os formatos JSON de entrada e saída são os mesmos documentados no [README principal](../../README.md#input-data-format).

&nbsp;
## Arquivos

- [generate_with_minimax.py](generate_with_minimax.py): usa a API na nuvem do MiniMax para gerar as respostas do modelo para destilação. O MiniMax oferece modelos como o MiniMax-M3 (janela de contexto de 512K) por uma API compatível com a da OpenAI.

&nbsp;
## Configuração do MiniMax

1. Crie uma conta na [plataforma MiniMax](https://platform.minimaxi.com/)
2. Gere uma API key nas configurações da sua conta
3. Guarde a API key em um local seguro (por exemplo, um gerenciador de senhas)

Modelos disponíveis:
- `MiniMax-M3` — modelo mais recente, com janela de contexto de 512K e saída máxima de 128K (padrão)
- `MiniMax-M2.7` — geração anterior, com janela de contexto de 1M
- `MiniMax-M2.7-highspeed` — variante mais rápida do M2.7, otimizada para throughput

&nbsp;
## Geração de dados com o MiniMax

O script do MiniMax funciona de forma parecida com o script do OpenRouter:

```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model MiniMax-M3 \
  --num_processes 1 \
  --out_file sample_minimax_outputs.json
```

Se você não usa `uv`, troque `uv run` por `python`.

O arquivo de saída tem a mesma estrutura dos gerados pelos scripts do Ollama e do OpenRouter.

**Nota:** o MiniMax exige que o parâmetro de temperature esteja no intervalo (0.0, 1.0]. O script limita automaticamente valores fora desse intervalo.

&nbsp;
## Gerando um dataset de destilação do MATH-500

Para gerar as respostas do professor para o conjunto de 500 exemplos do MATH-500, você pode omitir o `--math_json`; o script do MiniMax carrega automaticamente o `math500_test.json` (e salva uma cópia local no primeiro uso).

```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --dataset_size 500 \
  --model MiniMax-M3 \
  --num_processes 1 \
  --out_file math500_minimax_distill.json
```

&nbsp;
## Gerando um dataset de destilação com 12.000 amostras do MATH

Isso usa o mesmo conjunto de treinamento de 12.000 amostras sem sobreposição dos capítulos 6, 7 e 8. Se você ainda não o tem, baixe-o primeiro:

```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model MiniMax-M3 \
  --num_processes 50 \
  --resume \
  --out_file math12000_minimax_distill.json
```

Para execuções grandes, reduza ou aumente o `--num_processes` dependendo dos limites da sua conta e do throughput desejado.
