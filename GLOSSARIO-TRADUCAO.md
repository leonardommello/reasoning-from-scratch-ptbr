# Guia de tradução — pt-BR

Documento normativo da tradução deste fork. Toda tradução deve seguir estas regras.

## Escopo

| Alvo | Traduzir? |
|---|---|
| Células markdown dos `.ipynb` | Sim |
| Arquivos `.md` (READMEs, troubleshooting) | Sim |
| Comentários e docstrings `.py` | Sim |
| Strings de `print()` / logs | Sim, exceto as congeladas (abaixo) |
| Código, nomes de variáveis, funções, classes | **Não** |
| Nomes de arquivo, caminhos, URLs, chaves de dict | **Não** |
| Arquivos em `tests/` | **Não** (a suíte permanece em inglês) |
| Prompts enviados ao modelo | **Não** (mudam o comportamento do LLM) |
| Citação bibliográfica, título do livro, ISBN | **Não** |

## Strings congeladas

Estas aparecem em `print()` mas são verificadas por testes automatizados. **Não traduzir:**

| String | Origem | Teste |
|---|---|---|
| `PASS` / `FAIL` | `reasoning_from_scratch/ch03.py:389,409` | `tests/test_ch03.py:322` |
| `Time:` | ch02 | `tests/test_ch02.py:153` |
| `tokens/sec` | ch02 | `tests/test_ch02.py:154` |
| `Accuracy: {x}% ({n}/{m})` | scripts MATH-500 | `tests/test_math500_scripts.py:72,105` |
| `Average: {n} tokens` | ch08 | `tests/test_ch08.py:239,243` |
| `Shortest: {n} tokens (index {i})` | ch08 | `tests/test_ch08.py:240,244` |
| `Longest: {n} tokens (index {i})` | ch08 | `tests/test_ch08.py:241,245` |
| Textos de `argparse` (`usage`, help) | scripts CLI | `tests/test_ch07_scripts.py:37` |

Rodar `pytest tests/` após cada lote de tradução em `.py`.

## Jargão mantido em inglês

Sem itálico, sem tradução, sem parênteses explicativos após a primeira ocorrência:

`token`, `tokenizer`, `prompt`, `embedding`, `checkpoint`, `reward`, `rollout`,
`fine-tuning`, `pretraining`, `baseline`, `batch`, `batch size`, `learning rate`,
`loss`, `logits`, `dataset`, `benchmark`, `leaderboard`, `inference-time scaling`,
`chain-of-thought` (CoT), `self-consistency`, `best-of-N`, `self-refinement`,
`policy`, `on-policy`, `off-policy`, `clipping`, `KL divergence`, `advantage`,
`greedy decoding`, `sampling`, `temperature`, `top-k`, `top-p`, `beam search`,
`attention`, `transformer`, `layer`, `head`, `hidden state`, `KV cache`,
`throughput`, `latency`, `overhead`, `pipeline`, `wrapper`, `parser`,
`LLM`, `GRPO`, `RLVR`, `RL`, `SFT`, `MMLU`, `LLM-as-a-judge`, `GPU`, `CPU`.

Siglas nunca são expandidas em português.

## Termos traduzidos

| Inglês | pt-BR |
|---|---|
| reasoning model | modelo de raciocínio |
| reasoning | raciocínio |
| training | treinamento |
| to train | treinar |
| weights | pesos |
| layer (contexto matemático) | camada |
| distillation | destilação |
| teacher / student model | modelo professor / modelo aluno |
| reinforcement learning | aprendizado por reforço |
| verifier | verificador |
| accuracy | acurácia |
| evaluation | avaliação |
| exercise / solution | exercício / solução |
| chapter / appendix | capítulo / apêndice |
| setup | configuração (ambiente) / instalação |
| troubleshooting | solução de problemas |
| bonus material | material complementar |
| hardware requirements | requisitos de hardware |
| consumer hardware | hardware de consumo |
| step by step | passo a passo |
| from scratch | do zero |

## Estilo

- Português brasileiro, ortografia completa com acentos e cedilha.
- Tratamento: **você** (nunca "tu", nunca "o leitor").
- Voz ativa; presente do indicativo. "Carregamos o modelo", não "o modelo será carregado".
- Primeira pessoa do plural onde o original usa "we": "vamos ver", "usamos".
- Preservar exatamente: níveis de heading, links markdown, blocos de código,
  HTML embutido (`<br>`, `<img>`, `&nbsp;`), tabelas, badges, emojis.
- Não reordenar, não resumir, não adicionar conteúdo. Uma frase entra, uma sai.
- Comentários `.py`: manter a coluna/indentação original.

## Rastreabilidade

- Branch: `traducao-pt-br`
- Um commit por capítulo: `traduz: chNN — <alvo>`
- Progresso: `PROGRESSO-TRADUCAO.md`

## Conformidade com a Apache-2.0

O repositório original é `Copyright 2025-2026 Sebastian Raschka`, licenciado sob
Apache License 2.0. A Seção 2 concede licença perpétua e irrevogável para
reproduzir, preparar obras derivadas, exibir publicamente e distribuir a obra e
suas obras derivadas. Uma tradução é obra derivada, então esta tradução e sua
publicação estão cobertas pela licença.

A Seção 4 impõe obrigações que este fork cumpre:

| Obrigação | Como é cumprida |
|---|---|
| (a) entregar cópia da licença | `LICENSE` preservado sem alterações |
| (b) marcar arquivos modificados de forma proeminente | aviso de tradução no `README.md`, na célula de crédito de cada notebook e no rodapé do site |
| (c) preservar avisos de copyright, patente, marca e atribuição | tabela de crédito ao autor mantida no topo de cada notebook; links para o repositório e para o livro preservados |
| (d) preservar `NOTICE` | o repositório original não possui arquivo `NOTICE` |

Limites que esta tradução respeita:

- Traduz somente o que está no repositório. O texto do livro impresso publicado
  pela Manning não faz parte do repositório e não é reproduzido nem reconstruído aqui.
- Não se apresenta como edição oficial, como produto da Manning, nem como
  substituto do livro. O material publicado aponta para a compra do livro.
- Não remove nem obscurece a autoria original.
