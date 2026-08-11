# Material complementar do capítulo 8: usar o Qwen3 com Hugging Face

Esta pasta contém duas formas de usar o [`Qwen3Model`](../../reasoning_from_scratch/qwen3.py) feito do zero e os checkpoints `.pth` compatíveis deste repositório com o `transformers` do Hugging Face.

Ambas as abordagens permitem inferência e treinamento no estilo Hugging Face. A diferença está em querer um diretório de modelo Hugging Face reutilizável ou um wrapper local mais leve em torno do modelo PyTorch existente.

&nbsp;
## Abordagens


&nbsp;
### 1) `wrapper_approach`

A [./wrapper_approach](./wrapper_approach) mantém o modelo como um arquivo `.pth` local e encapsula o `Qwen3Model` em um `PreTrainedModel` local e fino, para que funcione com partes da API do Hugging Face.

Use esta abordagem se você quer:

- a menor quantidade de código extra
- experimentação local dentro deste repositório
- `model.generate(...)` e `transformers.Trainer` sem uma etapa de exportação
- carregar o modelo base ou os checkpoints dos capítulos 6 a 8 diretamente do `.pth`


&nbsp;
### 2) `export_approach`

A [./export_approach](./export_approach) converte os pesos do Qwen3 feito do zero, ou um checkpoint compatível, em uma pasta de modelo compatível com o Hugging Face.

Use esta abordagem se você quer:

- um diretório de modelo salvo, com `config.json`, arquivos de tokenizer e pesos
- `AutoConfig`, `AutoTokenizer` e `AutoModelForCausalLM`
- um fluxo de trabalho mais próximo de como os modelos Hugging Face costumam ser empacotados



&nbsp;
## Qual usar?

- Escolha a [wrapper_approach](wrapper_approach) para fins de aprendizado e se o objetivo for uma integração local mais leve com o `transformers`.
- Escolha a [export_approach](export_approach) se o objetivo for criar um pacote de modelo Hugging Face e otimizar o desempenho computacional.
