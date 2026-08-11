# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from .ch02 import generate_text_basic_stream_cache
from .ch03 import extract_final_candidate
from .qwen3 import KVCache

from collections import Counter
import torch


def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
    generate_func=None,  # Novo
    **generate_kwargs  # Novo
):

    if generate_func is None:  # Novo
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)

    generated_ids = []
    for token in generate_func(  # Novo
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,  # Novo
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


def plot_scores_bar(
    next_token_logits, start=19_800, end=19_900,
    arrow=True, ylabel="Logit value"
):

    import matplotlib.pyplot as plt

    # Seleciona um trecho do vocabulário
    x = torch.arange(start, end)

    # .cpu() é um atalho para to(torch.device("cpu"))
    logits_section = next_token_logits[0, start:end].float().cpu()

    # Plota os logits
    plt.bar(x, logits_section)
    plt.xlabel("Vocabulary index")
    plt.ylabel(ylabel)

    # Destaca o logit máximo
    if arrow:
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 1.5
            },
            fontsize=10,
        )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def plot_logits_with_temperature(
    next_token_logits, start=19_800, end=19_900,
    temps=(0.5, 5.0),
):

    import matplotlib.pyplot as plt

    x = torch.arange(start, end)
    logits_orig = next_token_logits[0, start:end].float().cpu()

    # Aplica temperature scaling
    logits_scaled = [
        scale_logits_by_temperature(logits_orig, T) for T in temps
    ]
    # Plota os logits
    plt.plot(x, logits_orig, label="Original logits", lw=2)
    plt.plot(
        x, logits_scaled[0],
        label=f"T={temps[0]} (sharper)", ls="--", lw=1
    )
    plt.plot(
        x, logits_scaled[1],
        label=f"T={temps[1]} (flatter)", ls=":", lw=3
    )

    # Destaca o logit máximo
    max_idx = torch.argmax(logits_orig)
    plt.annotate(
        "Berlin",
        xy=(x[max_idx], logits_orig[max_idx]),
        xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
        fontsize=12,
    )

    plt.xlabel("Vocabulary index")
    plt.ylabel("Logit value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def count_samples(probas, num_samples=1000, threshold=1, tokenizer=None):
    # Sorteia amostras conforme as probabilidades
    samples = torch.multinomial(
        probas.cpu(), num_samples=num_samples, replacement=True
    )

    # Conta quantas vezes cada índice foi selecionado
    counts = torch.bincount(samples.squeeze(0), minlength=1)

    # Imprime os resultados
    for i, c in enumerate(counts):
        if c > threshold:
            if tokenizer is None:
                print(f"Vocab index {i}: {c.item()}x")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")


@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Passo 3.1: obtém os logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        ########################################
        # NOVO:
        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # Passo 3.2: aplica temperature scaling nos logits
            logits = scale_logits_by_temperature(out, temperature)

            # Passo 3.3: converte em probabilidades
            probas = torch.softmax(logits, dim=-1)

            # Passo 3.4: amostra o token conforme as probabilidades
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        #########################################
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    # Passo 4.1: ordena por probabilidade decrescente
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)

    # Passo 4.2: soma cumulativa
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # Passo 4.3.1: mantém os tokens em que a massa cumulativa anterior (antes do token) é < top_ps
    # Exemplo: [0.5, 0.41, 0.09] com top_p=0.9 deve manter os dois primeiros tokens
    prefix = cumprobas - sorted_probas   # massa cumulativa antes de cada token
    keep = prefix < top_p
    # Sempre mantém pelo menos um token (fallback para top_p muito pequeno ou não positivo)
    keep[:, 0] = True

    # Passo 4.3.2: zera tudo além do corte
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    # Passo 4.3.3: mapeia de volta para a ordem original
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    # Passo 4.4: renormaliza para somar 1
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    # keepdim=True tecnicamente não é necessário, mas faz o código funcionar em casos com batch
    return filtered / denom


@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Passo 3.1: obtém os logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # Passo 3.2: aplica temperature scaling nos logits
            logits = scale_logits_by_temperature(out, temperature)

            # Passo 3.3: converte em probabilidades
            probas = torch.softmax(logits, dim=-1)

            # (Novo) Passo 4: aplica o filtro top-p nas probabilidades
            probas = top_p_filter(probas, top_p)

            # Passo 3.4: amostra o token conforme as probabilidades
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []

    # 1) Amostra várias respostas
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )

        # 2) Extrai a resposta final (curta) de cada resposta
        short = extract_final_candidate(
            answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")

    # 3) Escolhe a resposta final mais frequente (voto de self-consistency)
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)

    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
