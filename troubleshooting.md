# Guia de solução de problemas

Esta página reúne problemas comuns e dicas de configuração encontrados ao longo do livro.

&nbsp;
## Bug de rolagem do JupyterLab

Se você está visualizando o código dos notebooks no JupyterLab em vez do VSCode, note que o JupyterLab (na configuração padrão) apresentou bugs de rolagem em versões recentes. Minha recomendação é ir em Settings -> Settings Editor e mudar o "Windowing mode" para "none" (como ilustrado abaixo), o que parece resolver o problema.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)


&nbsp;
## Capítulo 2

&nbsp;
### Problemas no download de arquivos

Use [esta página de discussão](https://github.com/rasbt/reasoning-from-scratch/discussions/145) se tiver qualquer problema com downloads de arquivos.

O código baixa dos seguintes endereços do Hugging Face, que você também pode abrir manualmente no navegador para checar se sua máquina ou rede está bloqueando:

- Arquivos do modelo e do tokenizer do capítulo 2: [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch/tree/main)
- Arquivo do modelo base: [qwen3-0.6B-base.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-base.pth)
- Arquivo do tokenizer base: [tokenizer-base.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-base.json)
- Arquivo do modelo de raciocínio: [qwen3-0.6B-reasoning.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-reasoning.pth)
- Arquivo do tokenizer de raciocínio: [tokenizer-reasoning.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-reasoning.json)
- Checkpoints de GRPO do capítulo 7: [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main)
- Checkpoints de destilação do capítulo 8: [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints/tree/main)

&nbsp;
#### Erros de SSL / proxy / certificado

Se o download de um modelo falhar com erros mencionando `SSL`, `CERTIFICATE_VERIFY_FAILED` ou `ProxyError`, o problema costuma ser do ambiente, e não um arquivo ausente.

Isso é incomum no geral, mas pode acontecer em máquinas de trabalho ou de instituições de ensino onde uma VPN, proxy, firewall ou antivírus intercepta o tráfego HTTPS. Nesse caso, tente o seguinte:

- Verifique se a URL correspondente do Hugging Face listada acima abre no seu navegador.
- Se o tokenizer baixa mas o arquivo de modelo `.pth` não, o proxy pode estar bloqueando arquivos maiores ou a extensão `.pth`.
- Peça à sua equipe de TI para liberar o download ou para tornar o certificado do proxy confiável para o Python.
- Em algumas máquinas gerenciadas, leitores relataram sucesso com `pip install pip-system-certs`, que faz o Python usar o repositório de certificados do sistema operacional.

&nbsp;
### `InductorError: CppCompileError`
Se você usa Linux e vê um `InductorError: CppCompileError: C++ compile error` ao executar `torch.compile`, contendo as seguintes linhas:

```python
Python.h: No such file or directory
81 | #include <Python.h>
| ^~~~~~~~~~
compilation terminated.
```

isso indica que seu runtime do Python pode estar sem alguns arquivos de cabeçalho C++ necessários para compilar o modelo para uso em CPU.

Você pode, por exemplo, checar se o arquivo existe: `ls -l /usr/include/python3.12/Python.h`.

Se ele não existir, você pode tentar instalar um runtime diferente do Python com

```bash
sudo apt-get install -y python3.12-dev build-essential
```

Ou pode desabilitar os requisitos de C++ no PyTorch antes de chamar `torch.compile`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = False

compiled_model = torch.compile(model)
```

Veja também a [#192](https://github.com/rasbt/reasoning-from-scratch/issues/192) para mais contexto.


&nbsp;
### CPU no Windows: `fatal error C1083` com `algorithm` ou `omp.h`

Se você está no Windows e o `torch.compile()` falha com

```text
fatal error C1083: Cannot open include file: 'algorithm': No such file or directory
```

ou

```text
fatal error C1083: Cannot open include file: 'omp.h': No such file or directory
```

o problema costuma estar na configuração local do compilador do Windows / OpenMP usada pelo TorchInductor (e não no código deste livro / repositório).

Um leitor relatou as seguintes dicas em um sistema Intel somente com CPU no [fórum](https://livebook.manning.com/forum?product=raschka2&comment=583365):

- Atualizar o PyTorch resolveu a ausência do cabeçalho `algorithm`.
- Mas a ausência do cabeçalho `omp.h` permaneceu.
- Usar um backend alternativo como `"eager"` ou `"aot_eager"` permitiu que o código rodasse.

Por exemplo:

```python
compiled_model = torch.compile(model, backend="eager")

# ou

compiled_model = torch.compile(model, backend="aot_eager")
```

Note que isso é uma solução de contorno, não uma correção completa. Pode ajudar, mas não usa o caminho completo de compilação do TorchInductor, então os ganhos de velocidade podem ser menores do que com um `torch.compile()` plenamente funcional.

**Mas tenha em mente também que o torch.compile não é essencial para este livro e você pode pular a seção inteira à vontade.**

De qualquer forma, se você está tentando fazer funcionar, antes de gastar muito tempo depurando pode ser útil rodar primeiro uma verificação mínima de sanidade:

```python
import torch

device = "cpu"  # ou "xpu"

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo = torch.compile(foo)
out = opt_foo(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device))
print(out.shape)
```

Se esse exemplo pequeno já falha, o problema provavelmente está na sua configuração de PyTorch / compilador, e não no código de modelo deste livro / repositório.

Para dicas adicionais de configuração, veja também [Usando `torch.compile()` no Windows](ch02/04_torch-compile-windows/README.md) e o [guia de CPU/XPU no Windows](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) do PyTorch. E, como mencionado antes, se o `torch.compile()` continuar instável no seu sistema, tudo bem pulá-lo nos exemplos do livro.



&nbsp;
## Capítulo 6

&nbsp;
### Checkpoints corrompidos

Em `train_rlvr_grpo` (capítulo 6), um `Ctrl+C` aciona o handler de `KeyboardInterrupt` para salvar um checkpoint `-interrupt`. Se você apertar `Ctrl+C` uma segunda vez antes de o salvamento terminar, isso pode interromper o `torch.save` no meio da escrita e deixar um arquivo `.pth` truncado. Espere pela mensagem do checkpoint `-interrupt` antes de sair.

Checkpoints de modelo corrompidos costumam levantar erros de carregamento ou falhar durante a avaliação; outro sinal revelador é que eles ficam bem menores do que os ~1,5 GB esperados.

&nbsp;
## Outros problemas

Para outros problemas, sinta-se à vontade para abrir uma nova [Issue](https://github.com/rasbt/reasoning-from-scratch/issues) no GitHub.
