
# Recursos de GPU na nuvem

Esta seção descreve alternativas na nuvem para rodar o código apresentado neste livro.

Embora o código rode em notebooks e desktops convencionais sem GPU dedicada, plataformas de nuvem com GPUs NVIDIA podem melhorar substancialmente o tempo de execução do código, especialmente nos capítulos 5 a 7.

&nbsp;

## Usando o Lightning Studio

Para uma experiência de desenvolvimento tranquila na nuvem, recomendo a plataforma [Lightning AI Studio](https://lightning.ai/), que permite configurar um ambiente persistente e usar tanto o VSCode quanto o Jupyter Lab em CPUs e GPUs na nuvem.

Assim que você iniciar um novo Studio, pode abrir o terminal e executar os seguintes passos de configuração para clonar o repositório e instalar as dependências:

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
pip install -r requirements.txt
```

(Diferente do Google Colab, isso só precisa ser executado uma vez, já que os ambientes do Lightning AI Studio são persistentes, mesmo que você alterne entre máquinas de CPU e GPU.)

Depois, navegue até o script Python ou o Jupyter Notebook que você quer rodar. Opcionalmente, você também pode conectar facilmente uma GPU para acelerar o tempo de execução do código — por exemplo, ao fazer o pretraining do LLM no capítulo 5 ou o fine-tuning nos capítulos 6 e 7.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## Usando o Google Colab

Para usar um ambiente do Google Colab na nuvem, acesse [https://colab.research.google.com/](https://colab.research.google.com/) e abra o notebook do capítulo desejado pelo menu do GitHub ou arrastando o notebook para o campo *Upload*, como mostrado na figura abaixo.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


Certifique-se também de enviar os arquivos relevantes (arquivos de dataset e arquivos .py dos quais o notebook importa) para o ambiente do Colab, conforme mostrado abaixo.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


Opcionalmente, você pode rodar o código em uma GPU alterando o *Runtime*, como ilustrado na figura abaixo.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;
## Dúvidas?

Se você tiver qualquer dúvida, não hesite em entrar em contato pelo fórum de [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) neste repositório do GitHub.
