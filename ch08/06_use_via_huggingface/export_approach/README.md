# Material complementar do capítulo 8: usar o código Qwen3 feito do zero via Hugging Face Transformers

Esta pasta mostra como converter o [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py) feito do zero, e qualquer checkpoint `.pth` compatível criado nos capítulos 6 a 8, em uma pasta compatível com o Hugging Face Transformers, e como rodá-lo com as funções de inferência do Hugging Face e com o `Trainer`.

A exportação é implementada como uma arquitetura `transformers` customizada, então funciona com as APIs padrão do Hugging Face, como `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `model.generate(...)` e `Trainer`. Mas, por ser código customizado, carregue-o com `trust_remote_code=True`.

&nbsp;
## Arquivos

- [hf_export.py](hf_export.py): converte os pesos do Qwen3 feito do zero, ou um checkpoint `.pth` salvo, em uma pasta de modelo do Hugging Face
- [hf_inference.py](hf_inference.py): roda a geração de texto com `AutoModelForCausalLM`
- [hf_trainer.py](hf_trainer.py): continua o treinamento de um modelo exportado com `transformers.Trainer`, no formato JSON de destilação do capítulo 8
- [hf_qwen3.py](hf_qwen3.py): implementação customizada de `PretrainedConfig` e `PreTrainedModel` do Hugging Face para a arquitetura Qwen3 exportada

Os scripts de exportação mantêm o código de modelo específico do Hugging Face localmente nesta pasta e importam utilitários compartilhados do pacote [`reasoning_from_scratch`](../../../reasoning_from_scratch) para o template de prompt do capítulo 3, as funções auxiliares de RoPE e as funções de download do Qwen3. (Veja as [instruções de configuração do capítulo 2](../../../ch02/02_setup-tips/python-instructions.md) para detalhes de instalação.)

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---

&nbsp;
## Passo 1: instalar as dependências

Este guia usa o Hugging Face Transformers, além das dependências do repositório. Para o `transformers.Trainer`, você também precisa do `accelerate`.

```bash
pip install transformers accelerate
```

Ou, se você usa `uv`:

```bash
uv add --dev transformers accelerate
```

&nbsp;
## Passo 2: exportar o modelo Qwen3 original

Para exportar o modelo base original como uma pasta do Hugging Face, rode:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind "base"  # or use "reasoning"
```

Se você já tem o modelo `.pth` bruto e o tokenizer localmente, pode evitar um download:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind base \
  --model_path ../../../ch02/01_main-chapter-code/qwen3/qwen3-0.6B-base.pth \
  --tokenizer_path ../../../ch02/01_main-chapter-code/qwen3/tokenizer-base.json
```

O mesmo também funciona com os arquivos `.pth` de checkpoint dos capítulos 6 a 8.

A pasta exportada conterá:

- `config.json`
- `generation_config.json`
- arquivos de tokenizer
- pesos do modelo (por padrão, como `model.safetensors`)
- uma cópia do módulo Python customizado exigido por `trust_remote_code=True`

&nbsp;
### O que o código de exportação faz

O exportador não traduz o modelo para a implementação oficial do Qwen no Hugging Face, e não modifica os pesos aprendidos. O que ele faz é o seguinte:

1. constrói um `PretrainedConfig` e um `PreTrainedModel` customizados do Hugging Face, que reproduzem a arquitetura do `Qwen3Model` feito do zero
2. carrega o `state_dict` original do `.pth` diretamente nesse modelo customizado do Hugging Face, sem renomear nem remodelar os parâmetros treináveis
3. salva o resultado usando o formato de pasta padrão do Hugging Face, para que `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `generate(...)` e `Trainer` consigam carregá-lo

As principais coisas adicionadas ou encapsuladas são:

- um arquivo de config do Hugging Face (`config.json`)
- uma classe de modelo do Hugging Face com assinatura de `forward(...)` compatível com o `transformers`
- arquivos de tokenizer do Hugging Face
- metadados de geração do Hugging Face (`generation_config.json`)
- um arquivo-fonte Python customizado que o `trust_remote_code=True` carrega

Há um pequeno detalhe extra durante a exportação. Ou seja, os checkpoints feitos do zero salvam apenas os pesos treináveis, enquanto a exportação para o Hugging Face também embute os buffers `cos` e `sin` de RoPE pré-computados, para que recarregar o modelo exportado seja numericamente consistente.

Para `--tokenizer_kind reasoning`, o exportador também anexa o chat template de raciocínio ao tokenizer, para que os scripts de inferência possam envolver os prompts automaticamente no formato de chat esperado.

Como esse módulo customizado do Hugging Face importa o pacote [`reasoning_from_scratch`](../../../reasoning_from_scratch) instalado, a pasta exportada é compatível desde que esse pacote esteja instalado no ambiente Python.

&nbsp;
## Passo 3: exportar um checkpoint salvo

O mesmo exportador também funciona para checkpoints de destilação do capítulo 8 ou qualquer outro arquivo `.pth` compatível produzido por este repositório.

Por exemplo, se você treinou um checkpoint do capítulo 8 com o tokenizer de raciocínio:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-distill \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --tokenizer_kind reasoning
```

Notas importantes:

- Use `--tokenizer_kind reasoning` para checkpoints de destilação do capítulo 8 e outros checkpoints treinados com o tokenizer de raciocínio.
- Use `--tokenizer_kind base` para checkpoints treinados com o tokenizer base.
- Se o JSON do tokenizer correspondente já estiver em disco, você pode passá-lo via `--tokenizer_path` para evitar um download.

&nbsp;
## Passo 4: rodar inferência com o Hugging Face

Depois da exportação, rode a inferência com `AutoTokenizer` e `AutoModelForCausalLM`:

```bash
uv run hf_inference.py \
  --model_dir hf-qwen3-base \
  --prompt "If x + 7 = 19, what is x?"
```

Internamente, o script:

1. carrega o modelo exportado com `trust_remote_code=True`
2. formata o prompt com o mesmo template de prompt de matemática do capítulo 3
3. aplica automaticamente o wrapper de chat de raciocínio quando o modelo exportado usa o tokenizer de raciocínio
4. chama `model.generate(...)`

Se você prefere a API bruta do Hugging Face diretamente, o padrão equivalente é:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-qwen3-base")
config = AutoConfig.from_pretrained("hf-qwen3-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "hf-qwen3-base",
    trust_remote_code=True,
)
```

&nbsp;
## Passo 5: continuar o treinamento com o `Trainer`

Você pode continuar o treinamento de um checkpoint exportado com o `Trainer` do Hugging Face, no mesmo formato JSON usado em [`../../04_train_with_distillation`](../../04_train_with_distillation).

Exemplo:

```bash
uv run hf_trainer.py \
  --model_dir hf-qwen3-base \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1 \
  --save_steps 10
```

O script mantém o mesmo objetivo de treinamento restrito à resposta, usado no código de destilação feito do zero:

- os tokens do prompt são mascarados e ficam fora da loss
- apenas os tokens da resposta destilada contribuem para a cross-entropy loss
- para exportações de raciocínio, o script envolve os traços do professor como `<think>...</think>` antes da resposta final



&nbsp;
## Carregando a exportação em outro lugar

Uma vez exportada, você pode copiar a pasta para outra máquina ou subi-la para o Hugging Face Hub e carregá-la lá também, desde que o `reasoning_from_scratch` esteja instalado naquele ambiente:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
```
