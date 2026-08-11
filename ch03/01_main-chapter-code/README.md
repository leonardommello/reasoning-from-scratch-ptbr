# Capítulo 3: Avaliando modelos de raciocínio

&nbsp;
## Código principal do capítulo

- [ch03_main.ipynb](ch03_main.ipynb): código principal do capítulo
- [ch03_exercise-solutions.ipynb](ch03_exercise-solutions.ipynb): soluções dos exercícios


&nbsp;
## Materiais complementares

- [../02_math500-verifier-scripts/evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py): script autônomo para avaliar modelos no dataset MATH-500
- [../02_math500-verifier-scripts/evaluate_math500_batched.py](../02_math500-verifier-scripts/evaluate_math500_batched.py): igual ao anterior, mas processa vários exemplos em paralelo durante a geração (para maior throughput)

Ambos os scripts de avaliação importam funcionalidades do pacote [`reasoning_from_scratch`](../../reasoning_from_scratch) para evitar duplicação de código. (Veja as [instruções de configuração do capítulo 2](../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.)
