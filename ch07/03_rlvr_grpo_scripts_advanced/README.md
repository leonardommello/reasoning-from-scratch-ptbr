# Capítulo 7: Melhorando a otimização de policy em aprendizado por reforço

Esta seção contém scripts avançados de GRPO que estendem a implementação do capítulo 6 com acompanhamento adicional, estabilização e variantes de modelagem de reward.


&nbsp;
## Visão geral dos scripts

&nbsp;
### Scripts principais

- `7_3_plus_tracking.py` (*7.3 Acompanhando métricas de desempenho mais avançadas do GRPO*): acompanha métricas adicionais de desempenho (estatísticas de advantage e entropia)
- `7_4_plus_clip_ratio.py` (*7.4 Estabilizando o GRPO em nível de sequência com policy ratios clipados*): como o anterior, mas calcula a loss de policy gradient com policy ratios clipados
- `7_5_plus_kl.py` (*7.5 Controlando o quanto o modelo muda com um termo de KL*): como o anterior, mas adiciona um termo de KL loss
- `7_6_plus_format_reward.py` (*7.6 Adicionando um format reward explícito*): como o anterior, mas adiciona um format reward para tokens `<think>` (uma diferença importante em relação aos outros scripts é que este é aplicado ao modelo de raciocínio, e não ao modelo base, já que ele já é familiarizado com esses tokens, como discutido no capítulo principal)

<br>

&nbsp;
### Scripts bônus de dicas e truques de GRPO

Desde que o GRPO foi publicado pela primeira vez, em abril de 2024 ([DeepSeekMath](https://arxiv.org/abs/2402.03300)), e se popularizou em janeiro de 2025 ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)), muitas melhorias foram sugeridas na literatura. Algumas das mais notáveis estão listadas abaixo:

1. Filtragem de sinal de gradiente zero ([DAPO, de Yu et al., 2025](https://arxiv.org/abs/2503.14476))
2. Amostragem ativa (DAPO)
3. Loss em nível de token (DAPO)
4. Sem KL loss (DAPO e [Dr. GRPO, de Liu et al., 2025](https://arxiv.org/abs/2503.20783))
5. Clip higher (DAPO)
6. Importance sampling truncado ([Yao et al., 2025](https://fengyao.notion.site/off-policy-rl))
7. Sem normalização por desvio padrão (Dr. GRPO)
8. Ajuste de KL com forças específicas por domínio; zero para matemática ([DeepSeek V3.2](https://arxiv.org/abs/2512.02556)
9. KL reponderado (DeepSeek V3.2)
10. Máscara de sequência off-policy (DeepSeek V3.2)
11. Manter a máscara de amostragem para top-p / top-k (DeepSeek V3.2)
12. Manter a normalização de advantage original do GRPO (DeepSeek V3.2)
13. Normalização por grupo, por reward, antes da agregação ([GDPO, de Liu et al., 2026](https://arxiv.org/abs/2601.05242))
14. Importance sampling e clipping em nível de sequência ([GSPO, de Zheng et al., 2025](https://arxiv.org/abs/2507.18071))
15. Clipar os pesos de importance sampling em vez das atualizações de token ([CISPO, de MiniMax et al., 2025](https://arxiv.org/abs/2506.13585))

(Pretendo escrever um texto mais detalhado sobre isso um dia, depois de terminar o conteúdo principal.)

<br>

Os scripts a seguir implementam algumas dessas melhorias:

- `7_7_improvements/olmo3_style.py`: implementa as melhorias 1 a 7, de forma parecida com o [Olmo 3](https://arxiv.org/abs/2512.13961), em cima do [7_5_plus_kl.py](7_5_plus_kl.py)

- `7_7_improvements/deepseek_v32_style.py`: implementa as melhorias 8 a 12, de forma parecida com o [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556), em cima do [7_5_plus_kl.py](7_5_plus_kl.py)

- `7_7_improvements/gdpo.py`: implementa o [GDPO](https://arxiv.org/abs/2601.05242) em cima do [7_6_plus_format_reward.py](7_6_plus_format_reward.py) (já que o GDPO é um ajuste para múltiplos rewards)

---

**Nota**: se você não usa `uv`, troque `uv run ...py` por `python ...py` nos exemplos abaixo.

---


&nbsp;
