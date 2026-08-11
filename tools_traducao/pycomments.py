"""Traduz comentários de arquivos .py por número de linha, sem tocar no código.

Uso:
    python tools_traducao/pycomments.py list <arquivo.py>
    python tools_traducao/pycomments.py apply <arquivo.py> <traducoes.txt>

Formato do arquivo de traduções, um comentário por linha:

    12|# comentário traduzido
    47|    # comentário indentado traduzido

A indentação e a posição do `#` são preservadas exatamente como no original:
apenas o texto após o `#` é substituído. Comentários inline (após código na
mesma linha) também são suportados.

Nunca traduzir:
    - cabeçalhos de copyright, fonte e repositório (exigidos pela Apache-2.0 §4c)
    - comentários que são código comentado
    - strings congeladas listadas em GLOSSARIO-TRADUCAO.md
"""

import io
import re
import sys
import tokenize

# Padrões de comentários que NUNCA devem ser traduzidos, porque constituem
# avisos de atribuição exigidos pela seção 4(c) da Apache License 2.0.
ATRIBUICAO = re.compile(
    r"Copyright \(c\)|Source for |Code repository:|Apache License|LICENSE\.txt"
)


def comentarios(caminho):
    """Devolve [(linha, coluna, texto_do_comentario), ...] em ordem."""
    src = open(caminho, encoding="utf-8").read()
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out.append((tok.start[0], tok.start[1], tok.string))
    return out


def listar(caminho):
    for linha, col, texto in comentarios(caminho):
        marca = "  [ATRIBUICAO - NAO TRADUZIR]" if ATRIBUICAO.search(texto) else ""
        print(f"{linha}|{texto}{marca}")


def aplicar(caminho, traducoes_path):
    novos = {}
    with open(traducoes_path, encoding="utf-8") as fh:
        for bruta in fh:
            bruta = bruta.rstrip("\n")
            if not bruta.strip():
                continue
            num, _, texto = bruta.partition("|")
            novos[int(num)] = texto.strip()

    originais = {linha: (col, texto) for linha, col, texto in comentarios(caminho)}

    protegidos = [n for n in novos if n in originais
                  and ATRIBUICAO.search(originais[n][1])]
    if protegidos:
        raise SystemExit(
            f"ERRO: tentativa de alterar aviso de atribuicao nas linhas {protegidos}. "
            "A Apache-2.0 secao 4(c) exige preserva-los."
        )

    desconhecidos = sorted(set(novos) - set(originais))
    if desconhecidos:
        raise SystemExit(f"ERRO: linhas sem comentario no original: {desconhecidos}")

    linhas = open(caminho, encoding="utf-8").read().split("\n")
    for num, texto in novos.items():
        col, original = originais[num]
        linha = linhas[num - 1]
        # Preserva tudo antes do `#`, troca apenas o comentário.
        linhas[num - 1] = linha[:col] + texto

    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(linhas))
    print(f"{caminho}: {len(novos)} comentarios traduzidos")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        listar(sys.argv[2])
    elif cmd == "apply":
        aplicar(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
