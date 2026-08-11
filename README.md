# Build A Reasoning Model (From Scratch)

> **Tradução para português do Brasil.** Este é um fork traduzido do [repositório oficial](https://github.com/rasbt/reasoning-from-scratch) de Sebastian Raschka, licenciado sob Apache-2.0. Apenas a documentação, os comentários de código e as células de texto dos notebooks foram traduzidos — o código permanece idêntico ao original. O texto do livro impresso não faz parte deste repositório. Consulte o [guia de tradução](./GLOSSARIO-TRADUCAO.md) para as convenções adotadas.

Este repositório contém o código para desenvolver um modelo de raciocínio baseado em LLM e é o repositório de código oficial do livro [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B).


<br>
<br>

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="250px"></a>

(Impresso em cores.)

<br>

Em [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B), você vai aprender e entender como funciona um large language model (LLM) de raciocínio.

Raciocínio é um dos avanços recentes mais empolgantes e importantes na melhoria de LLMs, mas também é um dos mais fáceis de entender errado quando você só ouve o termo raciocínio e lê sobre ele na teoria. É por isso que este livro adota uma abordagem prática. Vamos começar com um LLM base pré-treinado e então adicionar capacidades de raciocínio nós mesmos, passo a passo em código, para que você veja exatamente como funciona.

Os métodos descritos neste livro conduzem você pelo processo de desenvolver seu próprio modelo de raciocínio — pequeno, porém funcional — com propósito educacional. Ele espelha as abordagens usadas na criação de modelos de raciocínio de larga escala como DeepSeek R1, GPT-5 Thinking e outros. Além disso, este livro inclui código para carregar os pesos de modelos pré-treinados existentes.

- Link para o [repositório de código-fonte oficial](https://github.com/rasbt/reasoning-from-scratch)
- Link para o [livro na Manning](https://mng.bz/lZ5B) (site da editora)
- Link para a página do livro na Amazon.com (a definir)
- ISBN 9781633434677



<br>
<br>

Para baixar uma cópia deste repositório, clique no botão [Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip) ou execute o seguinte comando no seu terminal:

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>


> **Dica:**
> O capítulo 2 traz dicas adicionais sobre instalar Python, gerenciar pacotes Python e configurar seu ambiente de programação.

<br>
<br>

## Sumário (em andamento)

[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

- [Guia de solução de problemas](./troubleshooting.md)

| Título do capítulo                                          | Código principal                                             |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Cap. 1: Entendendo modelos de raciocínio                    | Sem código                                                   |
| Cap. 2: Gerando texto com um LLM pré-treinado               | - [ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| Cap. 3: Avaliando modelos de raciocínio                     | - [ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br/>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| Cap. 4: Melhorando o raciocínio com inference-time scaling  | - [ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br/>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| Cap. 5: Inference-time scaling via self-refinement          | - [ch05_main.ipynb](ch05/01_main-chapter-code/ch05_main.ipynb)<br/>- [ch05_exercise-solutions.ipynb](ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) |
| Cap. 6: Treinando modelos de raciocínio com aprendizado por reforço | - [ch06_main.ipynb](ch06/01_main-chapter-code/ch06_main.ipynb)<br/>- [ch06_exercise-solutions.ipynb](ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb) |
| Cap. 7: Melhorando GRPO para aprendizado por reforço        | - [ch07_main.ipynb](ch07/01_main-chapter-code/ch07_main.ipynb)<br/>- [ch07_exercise-solutions.ipynb](ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb) |
| Cap. 8: Destilando modelos de raciocínio para raciocínio eficiente | - [ch08_main.ipynb](ch08/01_main-chapter-code/ch08_main.ipynb)<br/>- [ch08_exercise-solutions.ipynb](ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb) |
| Apêndice A: Referências e leitura complementar              | Sem código                                                   |
| Apêndice B: Soluções dos exercícios                         | O código e as soluções estão na subpasta de cada capítulo    |
| Apêndice C: Código-fonte do LLM Qwen3                       | - [chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb)  |
| Apêndice D: Usando LLMs maiores                             | - [chD_main.ipynb](chD/chD_main.ipynb)                       |
| Apêndice E: Batching e execução orientada a throughput      | - [chE_main.ipynb](chE/chE_main.ipynb)                       |
| Apêndice F: Abordagens comuns para avaliação de LLMs        | - [chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb)  |
| Apêndice G: Construindo uma interface de chat               | - [chG](chG)                                                 |

<br>
&nbsp;

O modelo mental abaixo resume as principais técnicas abordadas neste livro.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">



<br>



&nbsp;
## Livro complementar

Note que *Build A Reasoning Model (From Scratch)* é um livro autônomo, focado em métodos para melhorar o raciocínio de LLMs.

Neste livro, trabalhamos com um LLM base pré-treinado de código aberto (Qwen3) sobre o qual programamos e aplicamos métodos de raciocínio do zero. Isso inclui inference-time scaling, aprendizado por reforço e destilação.

No entanto, se você tem interesse em entender como um LLM base convencional é implementado, talvez goste do meu livro anterior, [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D).

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="120px"></a>

- [Link da Amazon](https://amzn.to/4fqvn0D)
- [Link da Manning](http://mng.bz/orYv)
- [Repositório no GitHub](https://github.com/rasbt/LLMs-from-scratch)


<br>
&nbsp;

## Requisitos de hardware

O código dos capítulos principais deste livro foi projetado para rodar majoritariamente em hardware de consumo, num tempo razoável, e não exige hardware de servidor especializado. Essa abordagem garante que um público amplo consiga acompanhar o material. Além disso, o código utiliza GPUs automaticamente quando disponíveis. Dito isso, os capítulos 2 a 4 funcionam bem em CPUs e GPUs. Para os capítulos 5 e 6, recomenda-se usar uma GPU caso você queira reproduzir os resultados do capítulo.


(Consulte o documento [setup_tips](ch02/02_setup-tips/python-instructions.md) para recomendações adicionais.)

&nbsp;
## Exercícios

Cada capítulo do livro inclui vários exercícios. As soluções estão resumidas no Apêndice B, e os notebooks de código correspondentes estão disponíveis nas pastas de código principal de cada capítulo neste repositório (por exemplo, [`ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb`](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb)).


&nbsp;
## Material complementar

Várias pastas contêm materiais opcionais, como bônus para leitores interessados:

- **Capítulo 2: Gerando texto com um LLM pré-treinado**
  - [Configuração opcional de Python e recomendações de GPU na nuvem](ch02/02_setup-tips)
  - [Usando uma versão do LLM otimizada para GPU](ch02/03_optimized-LLM)
  - [Usando `torch.compile()` no Windows](ch02/04_torch-compile-windows)
  - [Rodar inferência e conversar com o modelo](ch02/05_use_model)
- **Capítulo 3: Avaliando LLMs**
  - [Scripts verificadores do MATH-500](ch03/02_math500-verifier-scripts)
  - [Parser avançado](ch03/03_advanced-parser) (parser híbrido de LaTeX)
- **Capítulo 4: Melhorando o raciocínio com inference-time scaling**
  - [Inference scaling no MATH-500](ch04/02_math500-inference-scaling-scripts) (prompting com chain-of-thought, self-consistency)
- **Capítulo 5: Inference-time scaling via self-refinement**
  - [Mais inference scaling no MATH-500](ch05/02_math500-more-inference-scaling-scripts) (best-of-N, self-refinement)
- **Capítulo 6: Treinando modelos de raciocínio com aprendizado por reforço**
  - [Scripts de GRPO](ch06/02_rlvr_grpo_scripts_intro) com um modo em batch
- **Capítulo 7: Melhorando GRPO para aprendizado por reforço**
  - [Scripts avançados de GRPO](ch07/03_rlvr_grpo_scripts_advanced) (incluindo treinamento nos estilos DeepSeek-V3.2, Olmo3 e GDPO)
  - [Baixar checkpoints de treinamento](ch07/04_download_trainining_checkpoints) (como baixar e usar os checkpoints de GRPO dos capítulos 6 e 7)
- **Capítulo 8: Destilando modelos de raciocínio para raciocínio eficiente**
  - [Gerar dados de destilação](ch08/02_generate_distillation_data) (geração de saídas do modelo professor via Ollama ou OpenRouter)
  - [Treinar com destilação](ch08/04_train_with_distillation) (incluindo scripts de destilação com exemplo único e em batch)
  - [Baixar checkpoints de treinamento](ch08/05_download_training_checkpoints) (como baixar e usar os checkpoints de destilação do capítulo 8)
  - [Usar o Qwen3 com Hugging Face](ch08/06_use_via_huggingface) (como usar o modelo base e os checkpoints dos capítulos 6 a 8 com `transformers`)
- **Apêndice F: Abordagens comuns para avaliação de LLMs**
  - [Métodos de avaliação com MMLU](chF/02_mmlu)
  - [Leaderboards de LLMs](chF/03_leaderboards)
  - [LLM-as-a-judge](chF/04_llm-judge)
- **Apêndice G: Construindo uma interface de chat**
  - [Código da interface de chat](chG/01_main-chapter-code)


&nbsp;
## Dúvidas, feedback e contribuições para este repositório

Para problemas comuns, consulte o [guia de solução de problemas](./troubleshooting.md).

Todo tipo de feedback é bem-vindo, e o melhor canal é o [fórum de discussão da Manning](https://livebook.manning.com/forum?product=raschka2&page=1) ou o [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions). Da mesma forma, se você tiver dúvidas ou apenas quiser trocar ideias com outras pessoas, não hesite em postar no fórum também.

Note que, como este repositório contém o código correspondente a um livro impresso, no momento não posso aceitar contribuições que estendam o conteúdo do código dos capítulos principais, pois isso introduziria desvios em relação ao livro físico. Manter a consistência ajuda a garantir uma boa experiência para todos.

&nbsp;
## Citação

Se você achar este livro ou este código útil para sua pesquisa, considere citá-lo.

Citação no estilo Chicago:

> Raschka, Sebastian. *Build A Reasoning Model (From Scratch)*. Manning, 2025. ISBN: 9781633434677.

Entrada BibTeX:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Reasoning Model (From Scratch)},
  publisher    = {Manning},
  year         = {2025},
  isbn         = {9781633434677},
  url          = {https://mng.bz/lZ5B},
  github       = {https://github.com/rasbt/reasoning-from-scratch}
}
```
