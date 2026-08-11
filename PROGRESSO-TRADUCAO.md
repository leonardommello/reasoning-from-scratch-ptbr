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

## Etapa 4 — READMEs de material complementar (6/19)

Concluídos ✅

`ch02/04_torch-compile-windows` · `ch03/03_advanced-parser` ·
`ch07/03_rlvr_grpo_scripts_advanced` · `ch08/06_use_via_huggingface` ·
`chF/03_leaderboards` · `chG/01_main-chapter-code`

Pendentes ⬜ (~101k chars). São documentos de bônus — descrevem como rodar
scripts e baixar checkpoints. Não bloqueiam o uso do repositório:

| Arquivo | Bytes |
|---|---|
| `ch08/02_generate_distillation_data/README.md` | 15.345 |
| `ch05/02_math500-more-inference-scaling-scripts/README.md` | 10.211 |
| `chF/02_mmlu/README.md` | 9.800 |
| `ch06/02_rlvr_grpo_scripts_intro/README.md` | 9.175 |
| `ch08/04_train_with_distillation/README.md` | 8.905 |
| `ch02/05_use_model/README.md` | 8.611 |
| `ch02/03_optimized-LLM/README.md` | 7.591 |
| `ch02/02_setup-tips/python-instructions.md` | 6.696 |
| `ch04/02_math500-inference-scaling-scripts/README.md` | 6.465 |
| `ch07/04_download_trainining_checkpoints/README.md` | 4.971 |
| `chF/04_llm-judge/README.md` | 4.603 |
| `ch03/02_math500-verifier-scripts/README.md` | 4.597 |
| `ch08/05_download_training_checkpoints/README.md` | 3.955 |
| `ch08/06_use_via_huggingface/{export,wrapper}_approach/README.md` | 12.385 |
| `ch08/02_generate_distillation_data/other_providers/minimax/README.md` | 3.067 |

## Etapa 5 — comentários `.py` (parcial)

Ferramenta: `tools_traducao/pycomments.py`, que substitui comentários por número
de linha preservando indentação e código, e **recusa** alterar os avisos de
atribuição exigidos pela Apache-2.0 §4(c).

| Alvo | Traduzido |
|---|---|
| `reasoning_from_scratch/` (pacote core) | 166/521 (31%) |
| scripts bônus em `ch0*/` | 5/330 (1%) |

Concluídos ✅ `ch02.py` · `ch03.py` · `ch04.py` · `ch05.py` · `ch06.py` ·
`ch07.py` · `ch08.py` · `utils.py` · `appendix_f.py` · `ch02_ex.py`

Pendentes ⬜ `qwen3.py` (123) · `qwen3_batched.py` (90) ·
`qwen3_optimized.py` (75) · `appendix_c.py` (31) — são os arquivos de
arquitetura do modelo. Mais os scripts bônus.

Uma linha ficou deliberadamente sem tradução: `ch03.py:236`, que contém um
caractere Unicode de sobrescrito. Traduzi-la sem conseguir inspecionar o byte
exato arriscaria corromper o arquivo.

## Verificação

- ✅ Todos os `.ipynb` são JSON válido
- ✅ Aviso Apache-2.0 §4(b) em todos os notebooks traduzidos
- ✅ Todo o pacote `reasoning_from_scratch/` passa em `ast.parse`
- ✅ Strings congeladas intactas (`PASS`/`FAIL`, `Accuracy:`, `Time:`,
  `tokens/sec`, `Average/Shortest/Longest: N tokens`)
- ✅ **`pytest tests/`**: 89 passaram, 14 falharam, 31 puladas — conjunto de
  falhas **idêntico** ao baseline capturado antes de qualquer alteração em `.py`.
  Zero regressões introduzidas.

As 14 falhas são pré-existentes e vêm de dependências opcionais ausentes neste
ambiente (`transformers`, e afins), não da tradução. Para reproduzir:

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 \
  pytest tests/ --ignore=tests/test_ch08_huggingface.py
```

## Achado para reportar upstream

`ch08/01_main-chapter-code/ch08_main.ipynb`, célula de crédito: aponta para o
livro e o repositório errados (*Build a Large Language Model From Scratch* /
`rasbt/LLMs-from-scratch`) em vez do livro de raciocínio. Preservado fielmente na
tradução — corrigir atribuição por conta própria não seria adequado. Vale abrir
issue no repositório original.
