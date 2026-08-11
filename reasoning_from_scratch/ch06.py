# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from .qwen3 import KVCache
from .ch03 import (
    extract_final_candidate, grade_answer, render_prompt
)
from .ch04 import top_p_filter

import json
import time
from pathlib import Path

import requests
import torch


def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        "MATH/math_full_minus_math500.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            print("Using backup URL.")
            r = requests.get(backup_url, timeout=30)
            r.raise_for_status()

        data = r.json()

        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
        )

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)

        token_id = next_token.item()
        generated.append(token_id)

        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # Exige \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        # Estágio 1: gera os rollouts
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Estágio 2: calcula os rewards
        reward = reward_rlvr(text, example["answer"])

        # Estágio 4: calcula os logprobs
        logp = sequence_logprob(model, token_ids, prompt_len)

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    # Estágio 2: coleta todos os rewards
    rewards = torch.tensor(roll_rewards, device=device)

    # Estágio 3: calcula os advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # Estágio 4: coleta todos os logprobs
    logps = torch.stack(roll_logps)

    # Estágio 5: calcula a loss de policy gradient
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss  # No próximo capítulo, adicionamos um termo de KL aqui

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def append_csv_metrics(
    csv_log_path,
    step_idx,
    total_steps,
    loss,
    reward_avg,
    avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,

):
    if steps is None:
        steps = len(math_data)

    # Estágio 1: inicializa o otimizador
    # (o modelo já foi inicializado fora da função)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)
    try:
        # Estágio 2: itera sobre os steps de treinamento
        for step in range(steps):

            # Estágio 3: zera o gradiente da loss
            # (é boa prática fazer isso no início de cada step)
            optimizer.zero_grad()

            current_step = step + 1
            example = math_data[step % len(math_data)]

            # Estágio 4: calcula a loss do GRPO
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Estágio 5: backward pass, para calcular os gradientes da loss
            stats["loss_tensor"].backward()

            # Limita gradientes grandes, para melhorar a estabilidade do treinamento
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Estágio 6: atualiza os pesos do modelo usando os gradientes da loss
            optimizer.step()

            # Estágio 7: coleta rewards, losses e comprimentos de resposta
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path,
                current_step,
                steps,
                stats["loss"],
                reward_avg,
                avg_response_len,
            )

            # Imprime as métricas do step
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )

            # Amostra saídas (a cada 10 steps) para checar se o modelo
            # gera texto coerente
            if current_step % 10 == 0:
                print(f"[Step {current_step}] sample outputs")
                for i, sample in enumerate(stats["samples"][:3]):
                    text = sample["text"].replace("\n", "\\n")
                    print(
                        f"  {i+1}) reward={sample['reward']:.3f} "
                        f"len={sample['gen_len']}: {text}"
                    )
                print()

            # Estágio 8: salva o checkpoint do modelo
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

    # Salva um checkpoint do modelo caso interrompamos o treinamento cedo
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model

    return model
