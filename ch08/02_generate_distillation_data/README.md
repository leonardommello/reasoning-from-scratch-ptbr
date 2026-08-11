# Material complementar do capítulo 8: gerar dados de destilação

Esta pasta contém scripts para gerar saídas de um modelo professor para problemas de matemática, que podem ser usadas como dados de destilação no treinamento de um modelo de raciocínio menor, como abordado no capítulo 8.

&nbsp;
**Sumário:**

- [Arquivos](#arquivos)
- [Formato dos dados de entrada](#formato-dos-dados-de-entrada)
- [Formato de saída](#formato-de-saída)
- [1. Geração local com Ollama](#1-geração-local-com-ollama)
  - [1.1 Configuração do Ollama](#11-configuração-do-ollama)
  - [1.2 Geração local de dados com Ollama](#12-geração-local-de-dados-com-ollama)
  - [1.3 Solução de problemas do Ollama](#13-solução-de-problemas-do-ollama)
    - [1.3.1 Ollama não está rodando](#131-ollama-não-está-rodando)
    - [1.3.2 Modelo do Ollama não baixado](#132-modelo-do-ollama-não-baixado)
- [2. Geração hospedada com OpenRouter](#2-geração-hospedada-com-openrouter)
  - [2.1 Configuração do OpenRouter](#21-configuração-do-openrouter)
  - [2.2 Geração de dados com OpenRouter](#22-geração-de-dados-com-openrouter)
- [Datasets para destilação](#datasets-para-destilação)
- [Estatísticas do dataset](#estatísticas-do-dataset)
- [Acurácia do professor](#acurácia-do-professor)
- [Gerando um dataset de destilação do MATH-500](#gerando-um-dataset-de-destilação-do-math-500)
- [Gerando um dataset de destilação com 12.000 amostras do MATH](#gerando-um-dataset-de-destilação-com-12000-amostras-do-math)


&nbsp;
## Arquivos

- [average_field_lengths_json.py](average_field_lengths_json.py): script utilitário para imprimir estatísticas básicas dos datasets gerados.
- [generate_with_ollama.py](generate_with_ollama.py): usa o Ollama para gerar as respostas do modelo para destilação. Este script é recomendado se você quer destilar de modelos menores, que consegue rodar localmente, por exemplo, Qwen3 4B, gpt-oss 20B, DeepSeek R1 32B etc.
- [generate_with_openrouter.py](generate_with_openrouter.py): usa modelos pela API do OpenRouter para gerar as respostas do modelo para destilação. Recomendado ao usar modelos maiores, como DeepSeek R1 (671B) ou Kimi K2.5 (1T), grandes demais para rodar localmente.
- [math_train_sample.json](math_train_sample.json): pequeno dataset de amostra, para verificações rápidas de sanidade.

&nbsp;
## Formato dos dados de entrada

Ambos os scripts esperam um arquivo JSON via `--math_json`. No mínimo, cada objeto deve ter:

- `problem` (string): a questão de matemática.
- `answer` (string): a resposta de ground truth.

Chaves extras, como `level`, `type` e `unique_id`, são ignoradas. Você pode olhar o arquivo [math_train_sample.json](math_train_sample.json) para um exemplo de estrutura, baseado no [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json) que usamos nos capítulos 6, 7 e 8.

Para aplicá-lo às 12.000 amostras completas, basta baixar o [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json) e passá-lo aos scripts via `--math_json math_full_minus_math500.json`. Note que isso levará bastante tempo, então recomendo truncar o arquivo para algumas centenas ou mil exemplos.


&nbsp;
## Formato de saída

Ambos os scripts escrevem um array JSON em que cada linha é assim:

```
{
  "problem": "...",             # The original "problem"
  "gtruth_answer": "...",       # The original "answer"
  "message_thinking": "...",    # The model's thinking stream
  "message_content": "..."      # The model's final answer
}
```

Notas:

- O `"answer"` original do arquivo JSON de entrada foi renomeado para `"gtruth_answer"`, para evitar ambiguidade (porque "answer" é um termo genérico que também poderia se referir à resposta do modelo).
- Os arquivos são escritos incrementalmente após cada amostra, então é possível trabalhar com arquivos intermediários ou interromper a execução.
- Os scripts têm uma opção `--resume`, para continuar uma execução interrompida.

&nbsp;
## 1. Geração local com Ollama


- O Ollama é uma aplicação de código aberto para rodar LLMs de forma eficiente.
- É um wrapper em torno do llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)), que implementa LLMs em C/C++ puro para maximizar a eficiência.
- Note que é uma ferramenta para usar LLMs para gerar texto (inferência), não para treinar ou fazer fine-tuning de LLMs.

&nbsp;
### 1.1 Configuração do Ollama


- Antes de rodar o código abaixo, instale o ollama visitando [https://ollama.com](https://ollama.com) e seguindo as instruções (por exemplo, clicando no botão "Download" e baixando a aplicação ollama para o seu sistema operacional).
- Usuários de macOS e Windows: clique na aplicação ollama que você baixou; se ela perguntar se deseja instalar o uso por linha de comando, responda "yes".
- Usuários de Linux podem usar o comando de instalação fornecido no site do ollama.
- Há 3 formas de rodar o ollama no nosso computador:


&nbsp;
**1. `ollama serve`**

- Isso roda o backend do ollama como um servidor, normalmente em `http://localhost:11434`. Ele não carrega um modelo até que o chamemos pela API. É isso que queremos se formos usar o ollama pelo Python.

&nbsp;
**2. `ollama run deepseek-r1:8b`**

- Este é um wrapper de conveniência. Se o servidor ainda não estiver rodando, ele o inicia, então baixa o modelo (na primeira vez) e nos deixa em um terminal interativo onde podemos conversar com o modelo. Nos bastidores, ele usa a mesma API do servidor.
- O modelo `deepseek-r1:8b` exigirá aproximadamente 30 GB de RAM com a configuração `--max_new_tokens 8192`.
  - Se você tem mais RAM, recomendo testar modelos maiores, para respostas de melhor qualidade, por exemplo, `deepseek-r1:32b` (exige aproximadamente 60 GB)
  - Se você tem menos RAM, tente selecionar um modelo menor; você encontra uma lista de modelos R1 menores [aqui](https://ollama.com/library/deepseek-r1). Além disso, em vez de usar um modelo DeepSeek, sinta-se à vontade para usar o campo "Search model" no [site do Ollama](https://ollama.com/) para selecionar outros modelos que achar interessantes.
  - Como alternativa, você também pode reduzir `--max_new_tokens 8192` para `--max_new_tokens 2048`, para diminuir o uso de RAM, mas isso pode truncar algumas respostas prematuramente.

&nbsp;
**3. Aplicativo desktop do Ollama**

- Isso roda o mesmo backend automaticamente e fornece uma interface gráfica por cima dele (como mostrado na figura acima).
  Ele também aplica valores padrão (system prompt, temperature, stop sequences), o que pode explicar por que as respostas parecem diferentes do uso direto da API.

&nbsp;
### 1.2 Geração local de dados com Ollama

```bash
uv run generate_with_ollama.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file sample_ollama_outputs.json
```

Se você não usa `uv`, troque `uv run` por `python`.

A saída esperada é a seguinte:

```
Loading model: deepseek-r1:8b
Using CUDA:0
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 3.2 min

Wrote 5 rows to: /home/rasbt/reasoning-from-scratch-codedev/ch08/sample_ollama_outputs.json
```

As entradas do arquivo [sample_ollama_outputs.json](sample_ollama_outputs.json) resultante são assim:

```json
  {
    "problem": "A rectangular band formation...",
    "gtruth_answer": "98",
    "message_thinking": "I need to find the largest number of...",
    "message_content": "The function is continuous..."
  },
```

O campo `"message_thinking"` contém a explicação em chain-of-thought, e `"message_content"` contém a resposta final. Por exemplo, eles poderiam ser conectados assim:

```python
complete_answer = f"<think>{data['message_thinking']}</think>\n\n{data['message_content']}"
```

Ou seja:

```
"<think>I need to find the largest number of...</think>

The function is continuous..."
```

&nbsp;
### 1.3 Solução de problemas do Ollama

Abaixo estão alguns problemas comuns ao rodar o script de geração de dados do Ollama.

&nbsp;
#### 1.3.1 Ollama não está rodando

Se você vir um erro como

```
Loading model: deepseek-r1:32b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt//reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: <urlopen error [Errno 111] Connection refused>
```

certifique-se de que o `ollama serve` está rodando (em outra aba do terminal).

&nbsp;
#### 1.3.2 Modelo do Ollama não baixado

Se você vir o seguinte erro:

```
Loading model: deepseek-r1:8b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: HTTP 404 from Ollama at http://localhost:11434/api/chat: {"error":"model 'deepseek-r1:8b' not found"}
```

isso significa que o modelo ainda não foi baixado. Nesse caso, rode `ollama run deepseek-r1:8b` em um terminal separado, o que baixará o modelo e iniciará um chat. Você pode testar o modelo no chat e depois sair via `\bye`.


&nbsp;
## 2. Geração hospedada com OpenRouter

O Ollama é conveniente se você quer rodar modelos localmente. No entanto, há vários modelos grandes (como o DeepSeek R1, de 671B parâmetros) grandes demais para rodar localmente no nosso hardware. Para esses casos, recomendo o [OpenRouter](https://openrouter.ai), que permite usar uma grande variedade de LLMs, tanto de pesos abertos quanto proprietários, hospedados na nuvem, por uma API parecida com a do ChatGPT.

Até o momento em que isto foi escrito, o [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) custa \$0,70 por 1 milhão de tokens de entrada e \$2,50 por 1 milhão de tokens de saída. Note que há muitos modelos mais baratos (e mais rápidos) no OpenRouter; até o modelo mais novo, [DeepSeek V3.2](https://openrouter.ai/deepseek/deepseek-v3.2), custa apenas $0,40 por 1 milhão de tokens de saída.

Dito isso, vamos fazer um cálculo simples de custo. Dado um comprimento médio de prompt de entrada de 11 tokens e um comprimento médio de resposta de 1524 tokens, custa cerca de $3,82 gerar as respostas para 1000 questões do MATH.

Veja o detalhamento:

- Total de tokens de entrada: 11 × 1000 = 11.000
- Total de tokens de saída: 1524 × 1000 = 1.524.000
- Custo de entrada: `(11,000 / 1,000,000) × $0.70 = $0.0077`
- Custo de saída: `(1,524,000 / 1,000,000) × $2.50 = $3.81`
- Custo total: `$3.81 + $0.0077 ≈ $3.82`

&nbsp;
### 2.1 Configuração do OpenRouter

A configuração é bem simples. Tudo o que você precisa fazer é criar uma conta no [OpenRouter](https://openrouter.ai/), gerar uma API key em [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) e guardar a API key em um local seguro (por exemplo, um gerenciador de senhas).


&nbsp;
### 2.2 Geração de dados com OpenRouter

O script do OpenRouter funciona de forma parecida com o do Ollama, exceto que prefixamos a API key como variável de ambiente:

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file sample_openrouter_outputs.json
```

Se você não usa `uv`, troque `uv run` por `python`.

A saída fica assim:

```
Loading model: deepseek/deepseek-r1
Using OpenRouter API: https://openrouter.ai/api/v1/chat/completions
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 2.2 min

Wrote 5 rows to: /Users/sebastian/Developer/reasoning-from-scratch/ch08/02_generate_distillation_data/sample_openrouter_outputs.json
```

O arquivo de saída [sample_openrouter_outputs.json](sample_openrouter_outputs.json) tem a mesma estrutura do gerado pelo script do Ollama.

**Dica:** se você está gerando muitos dados, rodar esse processo sequencial de destilação pode ser bem lento (por exemplo, ~100 horas para 12.000 respostas com o DeepSeek R1). Nesse caso, recomendo rodar várias threads paralelas de geração de dados, via `--num_processes`. Por exemplo, usar `--num_processes 50` com os modelos DeepSeek R1 reduz o tempo de execução de 100 horas para aproximadamente 2 horas.


&nbsp;
## Datasets para destilação

Uma coleção de datasets gerados pela abordagem com OpenRouter descrita acima pode ser encontrada aqui: [https://huggingface.co/datasets/rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill).

&nbsp;
## Estatísticas do dataset

Para checar as estatísticas dos datasets, use o script [average_field_lengths_json.py]:

```bash
uv run average_field_lengths_json.py \
--json_path sample_openrouter_outputs.json
```

```
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
Records: 5
Tokenizer: reasoning
Field             AvgTokens  MinTokens  MaxToken  Count
gtruth_answer          9.40          9        10      5
message_content      196.00        166       259      5
message_thinking     933.20        449      1676      5
problem               77.80         30       121      5
```

&nbsp;
## Acurácia do professor

Para calcular a acurácia do modelo que gerou o dataset (ou seja, o professor), use o script [../../ch03/02_math500-verifier-scripts/evaluate_json.py](../../ch03/02_math500-verifier-scripts/evaluate_json.py):

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_json.py \
--json_path sample_openrouter_outputs.json \
--gtruth_answer gtruth_answer \
--generated_text message_content
```

```
Accuracy: 100.0% (5/5)
```

&nbsp;
## Gerando um dataset de destilação do MATH-500

Para gerar as respostas do professor para o conjunto de 500 exemplos do MATH-500, você pode omitir o `--math_json`; ambos os scripts carregam automaticamente o `math500_test.json` (e salvam uma cópia local no primeiro uso).

**Ollama**

```bash
uv run generate_with_ollama.py \
  --dataset_size 500 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file math500_ollama_distill.json
```

**OpenRouter**

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --dataset_size 500 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file math500_openrouter_distill.json
```

&nbsp;
## Gerando um dataset de destilação com 12.000 amostras do MATH

Isso usa o mesmo conjunto de treinamento de 12.000 amostras sem sobreposição dos capítulos 6, 7 e 8. Se você ainda não o tem, baixe-o primeiro:

```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

**Ollama**

```bash
uv run generate_with_ollama.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --resume \
  --out_file math12000_ollama_distill.json
```

**OpenRouter**

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek/deepseek-r1 \
  --num_processes 50 \
  --resume \
  --out_file math12000_openrouter_distill.json
```

Para execuções grandes no OpenRouter, reduza ou aumente o `--num_processes` dependendo dos limites da sua conta e do throughput desejado.
