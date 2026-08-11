# Capítulo 3: Parser avançado (material complementar)

Esta pasta contém o experimento de parser da [issue #133](https://github.com/rasbt/reasoning-from-scratch/issues/133), na qual foi proposto um parser híbrido de LaTeX para lidar com casos extremos que o parser atual do capítulo pode não cobrir.



&nbsp;

## Arquivos

- [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb): notebook com exemplos de uso
- [math500_gpt_answers.json](math500_gpt_answers.json): exemplos do MATH-500 com respostas de LLM, usados em uma seção do notebook acima
- [gen_llm_answers.py](gen_llm_answers.py): script de conveniência para obter respostas em box do modelo Qwen3, em formato json
- [evaluate_math500_advanced.py](evaluate_math500_advanced.py): igual ao script de avaliação de LLM do capítulo 3, [evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py), mas suporta `--hybrid_parser` como argumento adicional, para usar o parser híbrido alternativo, por exemplo:

```python
uv run evaluate_math500_advanced.py --dataset_size 500 --hybrid_parser
```



&nbsp;
## Como isso difere do parser do capítulo 3
evaluate_math500_advanced.py
O parser do capítulo, em [reasoning_from_scratch/ch03.py](../../reasoning_from_scratch/ch03.py), foi projetado para se manter compacto e didático:

- Foca em normalização leve, mais checagens de equivalência simbólica
- Trata as respostas principalmente como expressões aritméticas/simbólicas

O parser híbrido desta pasta (`latex_normalizer_hybrid.py`) parte de padrões e é mais abrangente:

- Reconhece formatos de resposta antes de recorrer ao parsing de fallback.
- Adiciona suporte a intervalos, uniões, equações, matrizes, notação de conjuntos, pertencimento (`\\in`) e `\\pm`
- Preserva melhor casos extremos importantes, como respostas com subscrito de base (`52_8`) e caixa de texto (`\\text{Evelyn}`)

Exemplos em que o comportamento difere:

- `52_8` -> o caminho do capítulo costuma resolver para `528`; o híbrido mantém `52_8`
- `11,\\! 111,\\! 111,\\! 100` -> o caminho do capítulo pode virar uma tupla; o híbrido normaliza para `11111111100`
- `(0,9) \\cup (9,36)` -> o caminho do capítulo normalmente permanece como texto; o híbrido retorna uma união simbólica

Trade-offs:

- Parser do capítulo: mais simples, mais rápido e mais fácil de interpretar
- Parser híbrido: melhor cobertura de casos extremos de LaTeX, mas com mais regras e complexidade; também acrescenta dependências do backend LaTeX do SymPy

&nbsp;
## Uso

Você pode importar o parser híbrido diretamente do pacote:

```python
from reasoning_from_scratch.bonus.parser import normalize_text_hybrid, sympy_parser_hybrid
```

Veja o [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb) para exemplos de uso mais detalhados.
