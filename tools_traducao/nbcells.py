"""Ferramentas de tradução: extrai e reinjeta células markdown de notebooks.

Uso:
    python tools_traducao/nbcells.py extract <notebook.ipynb> <saida.txt>
    python tools_traducao/nbcells.py inject  <notebook.ipynb> <traduzido.txt>
    python tools_traducao/nbcells.py check   <notebook.ipynb>

O formato intermediário usa um delimitador por célula:

    @@@CELL 0
    linha 1
    linha 2
    @@@CELL 1
    ...

Apenas células markdown são exportadas, na ordem em que aparecem. O índice é o
índice da célula dentro de nb["cells"], então a reinjeção não depende da ordem
do arquivo de texto. Células de código e outputs nunca são tocados.
"""

import json
import sys

DELIM = "@@@CELL "


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dump(nb, path):
    # indent=1 + newline final é o formato que o Jupyter grava, o que mantém
    # o diff do git legível e evita reformatação espúria do arquivo inteiro.
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def source_to_text(source):
    return "".join(source) if isinstance(source, list) else source


def text_to_source(text):
    """Converte texto em lista de linhas terminadas em \\n, como o Jupyter grava."""
    if not text:
        return []
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def extract(nb_path, out_path):
    nb = load(nb_path)
    chunks = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        chunks.append(f"{DELIM}{i}\n{source_to_text(cell['source'])}")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(chunks))
    print(f"extraídas {len(chunks)} células markdown de {nb_path}")


def parse_blocks(text):
    """Devolve {índice_da_célula: texto}. Erros de formato levantam exceção."""
    blocks = {}
    current = None
    buf = []
    for line in text.split("\n"):
        if line.startswith(DELIM):
            if current is not None:
                blocks[current] = "\n".join(buf)
            current = int(line[len(DELIM):].strip())
            buf = []
        else:
            buf.append(line)
    if current is not None:
        blocks[current] = "\n".join(buf)
    return blocks


def inject(nb_path, txt_path):
    nb = load(nb_path)
    with open(txt_path, encoding="utf-8") as fh:
        blocks = parse_blocks(fh.read())

    md_idx = {i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"}
    unknown = set(blocks) - md_idx
    if unknown:
        raise SystemExit(f"ERRO: índices inexistentes ou não-markdown: {sorted(unknown)}")

    for i, text in blocks.items():
        nb["cells"][i]["source"] = text_to_source(text)

    dump(nb, nb_path)
    faltando = sorted(md_idx - set(blocks))
    print(f"reinjetadas {len(blocks)}/{len(md_idx)} células em {nb_path}")
    if faltando:
        print(f"AINDA EM INGLÊS: {faltando}")


def check(nb_path):
    """Valida o JSON e relata células markdown que ainda parecem estar em inglês."""
    nb = load(nb_path)
    pistas = (" the ", " and ", " with ", " this ", " that ", "We ", "Note that")
    suspeitas = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        txt = source_to_text(cell["source"])
        if any(p in txt for p in pistas):
            suspeitas.append(i)
    total = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"{nb_path}: JSON válido, {total} células markdown, {len(suspeitas)} suspeitas")
    if suspeitas:
        print(f"  índices: {suspeitas}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        extract(sys.argv[2], sys.argv[3])
    elif cmd == "inject":
        inject(sys.argv[2], sys.argv[3])
    elif cmd == "check":
        check(sys.argv[2])
    else:
        raise SystemExit(__doc__)
