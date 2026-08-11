# Apêndice G: Construindo uma interface de chat



Esta pasta contém o código para rodar uma interface de usuário no estilo do ChatGPT, para interagir com os LLMs usados e/ou desenvolvidos neste livro, como mostrado abaixo.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



Para implementar essa interface de usuário, usamos o pacote Python de código aberto [Chainlit](https://github.com/Chainlit/chainlit).

&nbsp;
## Passo 1: instalar as dependências

Primeiro, instalamos o pacote `chainlit` e sua dependência:

```bash
pip install chainlit
```

Ou, se você usa `uv`:

```bash
uv add chainlit
```



&nbsp;

## Passo 2: rodar o código do `app`

Esta pasta contém 2 arquivos:

1. [`qwen3_chat_interface.py`](qwen3_chat_interface.py): este arquivo carrega e usa o modelo Qwen3 0.6B em modo de thinking.
2. [`qwen3_chat_interface_multiturn.py`](qwen3_chat_interface_multiturn.py): igual ao anterior, mas configurado para lembrar o histórico de mensagens.

(Abra e inspecione esses arquivos para saber mais.)

Rode um dos comandos a seguir no terminal para iniciar o servidor da interface:

```bash
chainlit run qwen3_chat_interface.py
```

ou, se você usa `uv`:

```bash
uv run chainlit run qwen3_chat_interface.py
```

Rodar um dos comandos acima deve abrir uma nova aba do navegador, onde você pode interagir com o modelo. Se a aba não abrir automaticamente, inspecione o comando no terminal e copie o endereço local para a barra de endereços do navegador (normalmente, o endereço é `http://localhost:8000`).

## Usando um checkpoint personalizado

Como o `chainlit run ...` é quem controla os argumentos de linha de comando, estes scripts leem o caminho de um checkpoint personalizado da variável de ambiente `CHECKPOINT_PATH`, em vez de usar `argparse`.

Exemplo no terminal:

```bash
CHECKPOINT_PATH=/absolute/path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run qwen3_chat_interface.py
```

Notas:

- Mantenha o `WHICH_MODEL` do script alinhado com o tokenizer que o checkpoint espera.
- Os checkpoints do capítulo 8, de [`ch08/05_download_training_checkpoints`](../../ch08/05_download_training_checkpoints), usam o tokenizer de raciocínio, então use `WHICH_MODEL = "reasoning"`.
- Quando `CHECKPOINT_PATH` está definido, o script baixa apenas o tokenizer para o `LOCAL_DIR`; ele não baixa novamente os pesos padrão do modelo.
