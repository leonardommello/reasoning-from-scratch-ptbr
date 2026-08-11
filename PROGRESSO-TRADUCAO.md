# Progresso da tradução pt-BR

Branch: `traducao-pt-br` · Convenções: [GLOSSARIO-TRADUCAO.md](./GLOSSARIO-TRADUCAO.md)

## Etapa 1 — notebooks (19/19) ✅

Todos validados: JSON íntegro e aviso de modificação Apache-2.0 §4(b) presente.
Células de código, saídas de execução e blocos de código dentro do markdown
permanecem inalterados.

| Notebook | Células md | Status |
|---|---|---|
| `ch02_main.ipynb` | 77 | ✅ |
| `ch02_exercise-solutions.ipynb` | 8 | ✅ |
| `ch03_main.ipynb` | 42 | ✅ |
| `ch03_exercise-solutions.ipynb` | 25 | ✅ |
| `compare_with_current_parser.ipynb` | 14 | ✅ |
| `ch04_main.ipynb` | 49 | ✅ |
| `ch04_exercise-solutions.ipynb` | 24 | ✅ |
| `ch05_main.ipynb` | 50 | ✅ |
| `ch05_exercise-solutions.ipynb` | 26 | ✅ |
| `ch06_main.ipynb` | 85 | ✅ |
| `ch06_exercise-solutions.ipynb` | 13 | ✅ |
| `ch07_main.ipynb` | 123 | ✅ |
| `ch07_exercise-solutions.ipynb` | 12 | ✅ |
| `ch08_main.ipynb` | 85 | ✅ |
| `ch08_exercise-solutions.ipynb` | 12 | ✅ |
| `chC_main.ipynb` | 21 | ✅ |
| `chD_main.ipynb` | 25 | ✅ |
| `chE_main.ipynb` | 50 | ✅ |
| `chF_main.ipynb` | 34 | ✅ |

## Etapa 2 — documentação principal ✅

`README.md` · `troubleshooting.md` · `tests/README.md` · template de issue ·
READMEs dos 8 capítulos e dos apêndices C–G · READMEs de `01_main-chapter-code` ·
`02_setup-tips/README.md` · `gpu-instructions.md`

## Etapa 3 — site GitHub Pages ✅

`docs/index.html` — índice navegável, atribuição ao autor original, aviso de
tradução não oficial, nota de licença e link para compra do livro.

**Falta ativar:** em *Settings → Pages*, definir source = branch `traducao-pt-br`,
pasta `/docs`. O site fica em
`https://leonardommello.github.io/reasoning-from-scratch-ptbr/`.

## Etapa 4 — READMEs de material complementar ⬜

Pendentes (~20). São documentos de bônus, não bloqueiam o uso do repositório:

`ch02/03_optimized-LLM` · `ch02/04_torch-compile-windows` · `ch02/05_use_model` ·
`ch02/02_setup-tips/python-instructions.md` · `ch03/02_math500-verifier-scripts` ·
`ch03/03_advanced-parser` · `ch04/02_math500-inference-scaling-scripts` ·
`ch05/02_math500-more-inference-scaling-scripts` · `ch06/02_rlvr_grpo_scripts_intro` ·
`ch07/03_rlvr_grpo_scripts_advanced` · `ch07/04_download_trainining_checkpoints` ·
`ch08/02_generate_distillation_data` (+ `other_providers/minimax`) ·
`ch08/04_train_with_distillation` · `ch08/05_download_training_checkpoints` ·
`ch08/06_use_via_huggingface` (+ `export_approach`, `wrapper_approach`) ·
`chF/02_mmlu` · `chF/03_leaderboards` · `chF/04_llm-judge` · `chG/01_main-chapter-code`

## Etapa 5 — comentários e docstrings `.py` ⬜

91 arquivos, exceto `tests/`. Respeitar as strings congeladas do glossário.

## Verificação

- ✅ Todos os `.ipynb` são JSON válido
- ✅ Aviso Apache-2.0 §4(b) em todos os notebooks traduzidos
- ⬜ `pytest tests/` — **não executado**: `pytest` não está instalado neste
  ambiente. Nenhum arquivo `.py` foi modificado até aqui, então a suíte não foi
  afetada. Rodar antes de mexer na etapa 5:
  `SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests`

## Achado para reportar upstream

`ch08/01_main-chapter-code/ch08_main.ipynb`, célula de crédito: aponta para o
livro e o repositório errados (*Build a Large Language Model From Scratch* /
`rasbt/LLMs-from-scratch`) em vez do livro de raciocínio. Preservado fielmente na
tradução — corrigir atribuição por conta própria não seria adequado. Vale abrir
issue no repositório original.
