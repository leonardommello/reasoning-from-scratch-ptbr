# Recomendações de configuração do Python

O código deste livro é em grande parte autocontido, e me esforcei para minimizar as dependências externas. No entanto, para manter o livro acessível, legível e bem abaixo de 2000 páginas, alguns pacotes Python são necessários.

Esta seção apresenta dois métodos amigáveis para iniciantes instalarem os pacotes necessários, para que você consiga rodar os exemplos de código.

Existem, claro, muitas outras formas de instalar e gerenciar pacotes Python. Se você é uma pessoa experiente em Python e já tem sua própria configuração ou preferências, sinta-se à vontade para pular esta seção.

Se nenhuma das duas opções abaixo funcionar para você, não hesite em entrar em contato, por exemplo, abrindo uma [Discussion](https://github.com/rasbt/reasoning-from-scratch/discussions).

&nbsp;
## Opção 1: usar `pip` (embutido, funciona em qualquer lugar)

Se você já usa uma versão recente do Python, pode instalar pacotes usando o instalador `pip`, que já vem embutido.

Usei Python 3.12 para este livro. No entanto, versões mais novas, como Python 3.13 e 3.14, assim como versões mais antigas, como 3.11 e 3.10, também funcionam bem, desde que sejam suportadas pelo PyTorch. Você pode checar sua versão do Python rodando:

```bash
python --version
```

Se você usa Python 3.9 ou mais antigo, considere instalar a versão mais recente pelo [python.org](https://www.python.org/downloads/) ou usar uma ferramenta como o [`pyenv`](https://github.com/pyenv/pyenv) para gerenciar versões. No entanto, se você estiver instalando uma nova versão do Python, certifique-se de que ela seja suportada pelo PyTorch, checando a recomendação no [site oficial do PyTorch](https://pytorch.org/get-started/locally/). O PyTorch normalmente fica alguns meses atrás do lançamento mais recente do Python, então versões do Python recém-lançadas não são suportadas nem recomendadas de imediato.

Para instalar novos pacotes, conforme necessário (por exemplo, PyTorch e Jupyter Lab), rode:

```bash
pip install torch jupyterlab
```

Como alternativa, você pode instalar de uma vez todos os pacotes Python necessários usados neste livro, pelo arquivo [`requirements.txt`](https://github.com/rasbt/reasoning-from-scratch/blob/main/requirements.txt):

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```


&nbsp;
## Opção 2: usar `uv` (mais rápido e amplamente recomendado)

Embora o `pip` continue sendo a forma clássica e oficial de instalar pacotes Python, o [`uv`](https://github.com/astral-sh/uv) é um gerenciador de pacotes Python moderno e amplamente recomendado, que automaticamente:

- Cria e gerencia um ambiente virtual
- Instala pacotes rapidamente
- Mantém um lockfile para instalações reproduzíveis
- Suporta comandos parecidos com os do `pip`

&nbsp;
### Instalando o `uv` e os pacotes Python

Para instalar o `uv`, você pode usar os comandos abaixo (veja também a página oficial de [instalação](https://docs.astral.sh/uv/getting-started/installation/) para as recomendações mais recentes).

&nbsp;
**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

&nbsp;
**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Uma vez instalado, você pode instalar novos pacotes Python de forma parecida com o que faria pelo `pip`, como descrito na seção anterior, exceto que você troca `pip` por `uv pip`. Por exemplo:

```bash
uv pip install torch jupyterlab
```

No entanto, se você usa `uv`, o que eu recomendo e uso, é ainda melhor usar a sintaxe nativa do `uv` em vez do `uv pip`, como descrito abaixo.

&nbsp;
### Fluxo de trabalho recomendado com `uv`

Em vez de usar `uv pip`, recomendo e uso o fluxo de trabalho nativo do `uv`.

Primeiro, clone o repositório do GitHub para a sua máquina local:



```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
```

Em seguida, navegue até essa pasta, por exemplo, no Linux e no macOS:

```bash
cd reasoning-from-scratch
```

Então, como essa pasta contém um arquivo `pyproject.toml` e um arquivo `.python-version`, você já está pronto: o `uv` criará automaticamente uma pasta de ambiente virtual (invisível por padrão, `.venv`) para este projeto `reasoning-from-scratch`, na qual instala todas as dependências na primeira vez que você rodar um script ou abrir o Jupyter Lab.

O arquivo `.python-version` atualmente fixa o Python 3.13 para o ambiente local do `uv`. Isso evita selecionar acidentalmente uma versão do Python mais nova que as versões do PyTorch testadas com este projeto. Se o `uv` usar uma versão diferente do Python, você pode redefinir a fixação local rodando:

```bash
uv python pin 3.13
uv sync
```

Você provavelmente não vai precisar, mas, de forma geral, é possível instalar pacotes adicionais que ainda não fazem parte dos requisitos listados no `pyproject.toml`, via `uv add`:


```bash
uv add llms_from_scratch
```

O comando acima então adiciona o pacote ao ambiente virtual e ao arquivo `pyproject.toml`.

&nbsp;
### Rodando código pelo `uv`

Esta seção descreve os comandos do `uv` para rodar o Jupyter Lab e scripts Python.

Para abrir o Jupyter Lab, execute:

```bash
uv run jupyter lab
```

Scripts Python podem ser rodados via:

```bash
uv run python script.py
```




> **Uso avançado:** esta seção descreve uma forma simples de usar o `uv`, familiar para quem usa `pip`. Se você tem interesse em um uso mais avançado, veja [este documento](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences) para instruções mais explícitas sobre gerenciar ambientes virtuais no `uv`.
> Se você usa macOS ou Linux e prefere os comandos nativos do uv, consulte [este tutorial](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md). Também recomendo consultar a [documentação oficial do uv](https://docs.astral.sh/uv/) para informações adicionais.



&nbsp;
### Dicas de JupyterLab

Se você está visualizando o código dos notebooks no JupyterLab em vez do VSCode, note que o JupyterLab (na configuração padrão) apresentou bugs de rolagem em versões recentes. Minha recomendação é ir em Settings -> Settings Editor e mudar o "Windowing mode" para "none" (como ilustrado abaixo), o que parece resolver o problema.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

&nbsp;
## Dúvidas?

Se você tiver qualquer dúvida, não hesite em entrar em contato pelo fórum de [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) neste repositório do GitHub.
