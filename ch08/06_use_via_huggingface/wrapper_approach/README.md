# Material complementar do capítulo 8: usar o Qwen3 por um wrapper local do Hugging Face

Esta pasta mostra como usar o [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py) feito do zero com a biblioteca `transformers` do Hugging Face, encapsulando-o em uma classe `PreTrainedModel` local e fina, para compatibilidade.

Isso permite que você use:

- `model.generate(...)`
- `transformers.Trainer`

diretamente com arquivos de modelo `.pth` locais deste repositório, incluindo os pesos base do Qwen3 e os checkpoints compatíveis dos capítulos 6 a 8.

&nbsp;
## Arquivos

- [hf_wrapper.py](hf_wrapper.py): wrapper `PreTrainedModel` local em torno do `Qwen3Model` feito do zero, que usamos ao longo do livro
- [hf_inference.py](hf_inference.py): geração de texto usando o wrapper e o tokenizer do repositório
- [hf_trainer.py](hf_trainer.py): exemplo com `Trainer`, usando o wrapper e o formato JSON de destilação do capítulo 8

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---

&nbsp;
## O que este wrapper faz

O wrapper mantém o modelo local a este repositório e o adapta à API do Hugging Face.

Concretamente, ele:

- carrega um arquivo de modelo `.pth` local diretamente no `Qwen3Model`
- encapsula esse modelo em uma interface `PreTrainedModel`
- expõe um método `forward(...)` compatível com o `Trainer`
- habilita o `model.generate(...)`
- continua usando o `Qwen3Tokenizer` do repositório

Por quê? Havia leitores curiosos para explorar mais os modelos no `transformers`, que tem mais recursos do que o código do zero deste repositório.

&nbsp;
## Limitações

Este é um wrapper pequeno e local em torno do modelo feito do zero.

Implicações importantes:

- é destinado a ambientes onde o `reasoning_from_scratch` está instalado
- não fornece um fluxo de trabalho com `AutoTokenizer.from_pretrained(...)`
- não cria um diretório de modelo reutilizável, com `config.json` e arquivos de tokenizer
- a geração é mantida intencionalmente simples, então ela recalcula o prefixo inteiro em vez de adaptar o KV cache do código do zero às classes de cache do Hugging Face; se você quiser suporte completo, precisará mudar para a [../export_approach](../export_approach)

Note que essas restrições mantêm o código curto e focado no uso local dentro deste repositório.

&nbsp;
## Passo 1: instalar as dependências

Este guia usa o Hugging Face Transformers, além das dependências do repositório.

```bash
pip install transformers accelerate
```

Ou, se você usa `uv`:

```bash
uv add --dev transformers accelerate
```

&nbsp;
## Passo 2: rodar inferência local encapsulada

Para rodar o modelo base pelo wrapper, use:

```bash
  uv run hf_inference.py \
    --tokenizer_kind base \
    --prompt "If x + 7 = 19, what is x?"
```

Para rodar a variante de raciocínio, use:

```bash
  uv run hf_inference.py \
    --tokenizer_kind reasoning \
    --prompt "If x + 7 = 19, what is x?"
```

Para rodar um checkpoint local:

```bash
uv run hf_inference.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --prompt "If x + 7 = 19, what is x?"
```

Se `--model_path` for omitido, o script baixa o modelo base ou de raciocínio padrão para o `--tokenizer_kind` selecionado. Se `--model_path` for fornecido, ele pode apontar para o arquivo `.pth` base do Qwen3 ou para qualquer checkpoint compatível produzido nos capítulos 6 a 8.

Internamente, o script de inferência:

1. constrói o modelo wrapper local
2. carrega o arquivo de modelo `.pth` selecionado no `Qwen3Model` encapsulado
3. tokeniza o prompt com o tokenizer do repositório
4. chama `model.generate(...)`

&nbsp;
## Passo 3: continuar o treinamento com o `Trainer`

O mesmo wrapper também pode ser usado com o `transformers.Trainer`:

```bash
uv run hf_trainer.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1
```

Assim como na inferência, `--model_path` pode apontar para os pesos base do Qwen3 ou para um checkpoint compatível dos capítulos 6 a 8.

O trainer mantém o mesmo objetivo restrito à resposta, usado no restante do capítulo 8:

- os tokens do prompt são mascarados
- apenas os tokens da resposta contribuem para a loss
- o modo de raciocínio envolve os traços do professor como `<think>...</think>`

O formato JSON de entrada corresponde aos dados de destilação gerados em [../../02_generate_distillation_data](../../02_generate_distillation_data).

&nbsp;
