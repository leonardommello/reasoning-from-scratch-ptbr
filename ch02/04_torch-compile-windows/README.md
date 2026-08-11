# Usando `torch.compile()` no Windows

O `torch.compile()` depende do *TorchInductor*, que compila kernels em JIT e exige um toolchain de compilador C/C++ funcional.

Assim, no Windows, a configuração necessária para fazer o `torch.compile` funcionar pode ser um pouco mais trabalhosa do que no Linux ou no macOS, que normalmente não exigem passos extras além de instalar o PyTorch.

Se você usa Windows e usar o `torch.compile` parece complicado ou trabalhoso demais, não se preocupe: todos os exemplos de código deste repositório funcionam bem sem compilação.

Abaixo estão algumas dicas que reuni com base nas recomendações de [Daniel Kleine](https://github.com/d-kleine) e no seguinte [guia do PyTorch](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html).

&nbsp;
## 1 Configuração básica (CPU ou CUDA)

&nbsp;
### 1.1 Instalar o Visual Studio 2022

- Selecione o workload **"Desktop development with C++"**.
- Certifique-se de incluir o **pacote de idioma inglês** (sem ele, você pode encontrar erros de codificação UTF-8).

&nbsp;
### 1.2 Abrir o prompt de comando correto


Inicie o Python pelo

**"x64 Native Tools Command Prompt for VS 2022"**

ou pelo

**"Visual Studio 2022 Developer Command Prompt"**.

Como alternativa, você pode inicializar o ambiente manualmente rodando:

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

&nbsp;
### 1.3 Verificar se o compilador funciona

Rode

   ```bash
   cl.exe
   ```

Se você vir as informações de versão impressas, o compilador está pronto.

&nbsp;
## 2 Solucionando erros comuns

&nbsp;
### 2.1 Erro: `cl not found`

Instale o **Visual Studio Build Tools** com o workload "C++ build tools" e rode o Python por um developer command prompt. (Veja este [guia](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) da Microsoft para detalhes)

&nbsp;
### 2.2 Erro: `triton not found` (ao usar CUDA)

Instale manualmente o build do Triton para Windows:

```bash
pip install "triton-windows<3.4"
```

ou, se você usa `uv`:

```bash
uv pip install "triton-windows<3.4"
```

(Como mencionado antes, o triton é exigido pelo TorchInductor para a compilação de kernels CUDA.)



&nbsp;
## 3 Notas adicionais

No Windows, o compilador `cl.exe` só é acessível de dentro do ambiente Visual Studio Developer. Isso significa que usar o `torch.compile()` em notebooks como o Jupyter pode não funcionar, a menos que o notebook tenha sido iniciado a partir de um Developer Command Prompt.

Como mencionado no início deste artigo, há também um [guia do PyTorch](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) que alguns usuários acharam útil para colocar o `torch.compile()` para rodar em builds de CPU no Windows. No entanto, note que ele se refere ao branch instável do PyTorch, então use-o apenas como referência.

**Se a compilação continuar causando problemas, sinta-se à vontade para pulá-la. É um bônus interessante, mas não é importante para acompanhar o livro.**

