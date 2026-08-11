
# Rankings de leaderboard

Este material complementar implementa duas formas diferentes de construir leaderboards no estilo do LM Arena (antigo Chatbot Arena) a partir de comparações par a par.

Ambas as implementações recebem uma lista de preferências par a par (esquerda: vencedor, direita: perdedor) de um arquivo json, pelo argumento `--path`. Aqui está um trecho do arquivo [votes.json](votes.json) fornecido:

```json
[
  ["GPT-5", "Claude-3"],
  ["GPT-5", "Llama-4"],
  ["Claude-3", "Llama-3"],
  ["Llama-4", "Llama-3"],
  ...
]
```



<br>

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---

&nbsp;
## Método 1: ratings Elo

- Implementa o popular método de rating Elo (inspirado nos rankings de xadrez), originalmente usado pelo LM Arena
- Veja o [notebook principal](../01_main-chapter-code/chF_main.ipynb) para detalhes

```bash
➜  03_leaderboards git:(main) ✗ uv run 1_elo_leaderboard.py --path votes.json

Leaderboard (Elo) 
-----------------------
 1. GPT-5       1095.9
 2. Claude-3    1058.7
 3. Llama-4      958.2
 4. Llama-3      887.2
```






&nbsp;
## Método 2: modelo de Bradley-Terry

- Implementa um [modelo de Bradley-Terry](https://en.wikipedia.org/wiki/Bradley–Terry_model), parecido com o novo leaderboard do LM Arena, conforme descrito no artigo oficial ([Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132))
- Assim como no leaderboard do LM Arena, os scores são reescalados para ficarem parecidos com os scores Elo originais
- O código aqui usa o otimizador Adam do PyTorch para ajustar o modelo (por familiaridade e legibilidade)



```bash
➜  03_leaderboards git:(main) ✗ uv run 2_bradley_terry_leaderboard.py --path votes.json 

Leaderboard (Bradley-Terry)
-----------------------------
 1. GPT-5       1140.6
 2. Claude-3    1058.7
 3. Llama-4      950.3
 4. Llama-3      850.4
```

