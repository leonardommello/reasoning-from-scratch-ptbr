# Testes

Este diretório contém a suíte de testes Python do repositório.

## Execuções locais

Instale primeiro o ambiente de desenvolvimento:

```bash
uv sync --group dev
```

### 1. Suíte normal, ignorando os testes caros (recomendado)

Recomendado para testes rápidos e desenvolvimento de novas funcionalidades.

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

Rodar um único arquivo de teste:

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests/test_ch03.py
```


Este é o equivalente local mais próximo da matriz de testes padrão do GitHub.

### 2. Suíte normal mais os testes caros

Alguns códigos são ignorados por padrão porque são relativamente caros de rodar. Recomendo rodar esses testes quando você terminar a depuração básica.

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

Note que isso roda os testes protegidos por `SKIP_EXPENSIVE` nos arquivos de teste, mas ainda exclui os testes de integração de rede/download reais, que baixam checkpoints grandes de modelo.

### 3. Somente os testes de download

Alguns testes verificam se os arquivos de checkpoint dos modelos estão disponíveis para download e se os servidores (ainda) funcionam. Não é necessário rodar esses testes localmente ou com regularidade. Servem mais para testes ocasionais.

Para rodar esses testes de download, use:

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests -k real_download
```

Como isso funciona:

- `pytest tests` coleta os testes do diretório `tests/`
- `-k real_download` mantém apenas os testes cujos nomes contêm `real_download`

Para um exemplo mais direcionado — por exemplo, para rodar diretamente o teste de snapshot real do apêndice D —, use:

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests/test_appendix_d.py -k real_download_1_7b
```

Os testes opcionais de download real cobrem atualmente:

- `tests/test_ch03.py`: download real do `math500_test.json` e downloads de tokenizer
- `tests/test_ch06.py`: download real do conjunto de treinamento de matemática
- `tests/test_ch07.py`: download real de arquivo raw do GitHub
- `tests/test_ch08.py`: downloads reais do dataset de destilação e do tokenizer
- `tests/test_appendix_d.py`: download real do snapshot `Qwen/Qwen3-1.7B-Base`
- `tests/test_qwen3.py`: comparação real do tokenizer `Qwen/Qwen3-0.6B`


### 4. Tudo (não recomendado)

Isso roda tudo na suíte de testes. Note que inclui tanto os testes computacionalmente caros (seção 3) quanto os caros testes de download (seção 4).

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests
```

Não é recomendado para testes de rotina ao fazer alterações no código, porque os downloads de arquivos são muito custosos e desnecessários de rodar com regularidade.


## O que roda no GitHub CI

A matriz de testes padrão do GitHub roda a suíte normal e omite os testes mais pesados:

- `.github/workflows/tests-linux.yml`
- `.github/workflows/tests-macos.yml`
- `.github/workflows/tests-windows.yml`
- `.github/workflows/basic-tests-pip.yml`

Esses workflows definem `SKIP_EXPENSIVE=1`, então os testes caros são pulados ali. O motivo é que o GitHub CI não tem os recursos computacionais necessários (como uma GPU) para rodar os testes caros.

Os testes de integração de rede/download reais rodam em um workflow separado:

- `.github/workflows/real-download-tests.yml`

Esse workflow define `RUN_REAL_DOWNLOAD_TESTS=1` e roda apenas os testes selecionados por `-k real_download`.
Ele não faz parte da matriz padrão de PR/push. Roda em uma agenda semanal e também pode ser iniciado manualmente via `workflow_dispatch`.
