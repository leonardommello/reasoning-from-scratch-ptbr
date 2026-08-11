
# LLM-as-a-judge

Este material complementar implementa uma abordagem de LLM-as-a-judge, em que o gpt-oss:20b (via a biblioteca de código aberto Ollama) avalia as variantes base e de raciocínio do Qwen3 0.6B no MATH-500.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-f/Appendix_F_F06_raschka.webp" width="500px">




- O ollama é uma aplicação de código aberto para rodar LLMs de forma eficiente
- É um wrapper em torno do llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)), que implementa LLMs em C/C++ puro para maximizar a eficiência
- Note que é uma ferramenta para usar LLMs para gerar texto (inferência), não para treinar ou fazer fine-tuning de LLMs
- Antes de rodar o código abaixo, instale o ollama visitando [https://ollama.com](https://ollama.com) e seguindo as instruções (por exemplo, clicando no botão "Download" e baixando a aplicação ollama para o seu sistema operacional)
- Usuários de macOS e Windows: clique na aplicação ollama que você baixou; se ela perguntar se deseja instalar o uso por linha de comando, responda "yes"
- Usuários de Linux podem usar o comando de instalação fornecido no site do ollama
- Há 3 formas de rodar o ollama no nosso computador:



**1. `ollama serve`**

- Isso roda o backend do ollama como um servidor, normalmente em `http://localhost:11434`. Ele não carrega um modelo até que o chamemos pela API. É isso que queremos se formos usar o ollama pelo Python.

**2. `ollama run gpt-oss:20b`**

- Este é um wrapper de conveniência. Se o servidor ainda não estiver rodando, ele o inicia, então baixa o modelo (na primeira vez) e nos deixa em um terminal interativo onde podemos conversar com o modelo. Nos bastidores, ele usa a mesma API do servidor.

**3. Aplicativo desktop do Ollama**

- Isso roda o mesmo backend automaticamente e fornece uma interface gráfica por cima dele (como mostrado na figura acima).
Ele também aplica valores padrão (system prompt, temperature, stop sequences), o que pode explicar por que as respostas parecem diferentes do uso direto da API.



## Uso



As opções e os valores padrão são mostrados abaixo.

<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---



```bash
uv run ollama-judge.py --help
usage: ollama-judge.py [-h] [--device DEVICE]
                       [--which_model {base,reasoning}]
                       [--dataset_size DATASET_SIZE]
                       [--max_new_tokens MAX_NEW_TOKENS]
                       [--url URL]
                       [--judge_model JUDGE_MODEL]

options:
  -h, --help            show this help message and
                        exit
  --device DEVICE       Device e.g., "cpu",
                        "cuda", "cuda:0", "mps".
  --which_model {base,reasoning}
                        Candidate variant to use.
                        Defaults to "base".
  --dataset_size DATASET_SIZE
                        Number of MATH-500
                        examples to evaluate.
                        Default: 10
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens for
                        candidate generation.
                        Default: 2048
  --url URL             Ollama chat endpoint for
                        the judge. Default: "http:
                        //localhost:11434/api/chat
                        "
  --judge_model JUDGE_MODEL
                        Judge model name (Ollama).
                        Used only for scoring.
                        Default: "gpt-oss:20b"
```



**Modelo base**

```bash
➜  uv run ollama-judge.py
Using Apple Silicon GPU (MPS)
Model: base
Device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=1
[3/10] score=5
[4/10] score=5
[5/10] score=3
[6/10] score=5
[7/10] score=5
[8/10] score=3
[9/10] score=5
[10/10] score=1

Summary
-------
Average score: 3.800 over 10 example(s)
Counts: 1:2 2:0 3:2 4:0 5:6
```

**Modelo de raciocínio**

```bash
➜  uv run ollama-judge.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Model: reasoning
Device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=5
[3/10] score=5
[4/10] score=5
[5/10] score=4
[6/10] score=5
[7/10] score=5
[8/10] score=1
[9/10] score=5
[10/10] score=3

Summary
-------
Average score: 4.300 over 10 example(s)
Counts: 1:1 2:0 3:1 4:1 5:7
```

