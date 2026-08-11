# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path
import json
import re
import time

import requests
from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError
import torch

from .qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from .ch02 import (
    generate_text_basic_stream_cache
)

RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

LATEX_FIXES = [  # Formatação LaTeX a ser substituída
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # remove tokens especiais de chat, como <|assistant|>
SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
}


def load_model_and_tokenizer(which_model, device, use_compile, local_dir="qwen3"):
    if which_model == "base":

        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":

        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


def load_tokenizer_only(which_model, local_dir="qwen3"):
    if which_model == "base":
        download_qwen3_small(
            kind="base", tokenizer_only=True, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":
        download_qwen3_small(
            kind="reasoning", tokenizer_only=True, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    return tokenizer


def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)

    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
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


def get_last_boxed(text):
    # Encontra a última ocorrência de "\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # Pega a posição após "\boxed"
    current_idx = boxed_start_idx + len(r"\boxed")

    # Pula qualquer espaço em branco após "\boxed"
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # Espera uma chave de abertura "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Faz o parsing das chaves com aninhamento
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # Considera chaves desbalanceadas
    if brace_depth != 0:
        return None

    # Extrai o conteúdo dentro das chaves mais externas
    return text[content_start_idx:current_idx-1]


def extract_final_candidate(text, fallback="number_then_full"):
    # Valor de retorno padrão, se nada corresponder
    result = ""

    if text:
        # Prefere a última expressão em box, se houver
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        # Se não houver expressão em box, tenta o fallback
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                # Usa o último número
                result = m[-1]
            elif fallback == "number_then_full":
                # Caso contrário, retorna o texto inteiro se nenhum número for encontrado
                result = text
    return result


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    # Remove rótulos de múltipla escolha no início
    # Por exemplo, "c. 3" -> 3, ou "b: 2" -> 2
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    # Remove marcadores de grau
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)   # ^{\circ}
    text = re.sub(r"\^\s*\\circ", "", text)             # ^\circ
    text = text.replace("°", "")                        # Grau em Unicode

    # desfaz \text{...} se a string inteira estiver envolvida
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # remove delimitadores de matemática inline/display \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # canonicalização leve de LaTeX
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # convert unicode superscripts into exponent form (e.g., 2² -> 2**2)
    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    # números/raízes
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # frações
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # expoente e números mistos
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # 1,234 -> 1234
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr):
    # Para evitar travar em respostas longas e sem sentido
    # que alguns modelos mal treinados (capítulo 6) podem emitir
    if expr is None or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # Transformações padrão, como o tratamento de parênteses
                *spp.standard_transformations,

                # Permite símbolos de multiplicação omitidos (por exemplo, "2x" -> 2*x")
                spp.implicit_multiplication_application,
            ),

            # Avalia durante o parsing, para que constantes simples simplifiquem (por exemplo, 2+3 -> 5)
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None


def equality_check(expr_gtruth, expr_pred):
    # Primeiro, checa se as duas expressões são exatamente a mesma string
    if expr_gtruth == expr_pred:
        return True

    # Faz o parsing das duas expressões em objetos SymPy (retorna None se o parsing falhar)
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # Se ambas as expressões foram parseadas com sucesso, tenta a comparação simbólica
    if gtruth is not None and pred is not None:
        try:
            # Se a diferença é 0, elas são equivalentes
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False


def split_into_parts(text):
    result = [text]

    if text:
        # Checa se o texto parece uma tupla ou lista, por exemplo, "(a, b)" ou "[a, b]"
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # Divide nas vírgulas dentro dos colchetes e remove espaços em branco
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # Se o texto está vazio, retorna uma lista vazia
        result = []

    return result


def grade_answer(pred_text, gt_text):
    result = False  # Resultado padrão, caso as checagens falhem

    # Só continua se ambas as entradas forem strings não vazias
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )  # Quebra o ground truth em partes comparáveis

        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )  # Quebra a predição em partes comparáveis

        # Garante que ambos os lados tenham o mesmo número de partes válidas
        if (gt_parts and pred_parts
           and len(gt_parts) == len(pred_parts)):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )  # Checa cada parte quanto à equivalência matemática

    return result  # True apenas se todas as checagens passaram


def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)  # Roda a checagem de igualdade
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows

    # Calcula a largura máxima de cada coluna, para alinhar bem a tabela
    col_widths = [
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]

    # Imprime a tabela linha por linha
    for row in data:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)

    # Imprime o resumo dos testes que passaram
    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\nPassed {passed}/{len(rows)}")


def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template


def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy:  # Salva uma cópia local
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


def mini_eval_demo(model, tokenizer, device):
    ex = {  # Exemplo de teste com os campos "problem" e "answer"
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     # 1. Aplica o template de prompt
    gen_text = generate_text_stream_concat(   # 2. Gera a resposta
        model, tokenizer, prompt, device,
        max_new_tokens=64,
    )
    pred_answer = extract_final_candidate(gen_text)  # 3. Extrai e normaliza a resposta
    is_correct = grade_answer(                       # 4. Avalia a resposta
        pred_answer, ex["answer"]
    )
    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")


def eta_progress_message(
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)

    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"

    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")  # Torna o nome do arquivo compatível com o Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0  # Calcula o comprimento médio das respostas (veja o exercício 3.2)
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:  # Salva os resultados para inspeção
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])    # 1. Aplica o template de prompt
            gen_text = generate_text_stream_concat(   # 2. Gera a resposta
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            total_len += len(tokenizer.encode(gen_text))

            extracted = extract_final_candidate(  # 3. Extrai e normaliza a resposta
                gen_text
            )
            is_correct = grade_answer(            # 4. Avalia a resposta
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {  # Registro a ser salvo para inspeção
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:  # Imprime as respostas durante o processo de geração
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    # Imprime as informações de resumo
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    avg_len = total_len / num_examples
    print(f"Average response length: {avg_len:.2f} tokens")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc
