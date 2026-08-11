"""Insere o aviso de modificação exigido pela Apache-2.0 §4(b) nos notebooks.

O aviso é acrescentado logo abaixo da tabela de crédito da primeira célula
markdown, de modo que a atribuição original (§4c) permaneça intacta e visível.

Uso:
    python tools_traducao/aviso_traducao.py <notebook.ipynb> [<notebook.ipynb> ...]

A operação é idempotente: rodar de novo não duplica o aviso.
"""

import sys

from nbcells import load, dump, source_to_text, text_to_source

MARCA = "<!-- aviso-traducao-ptbr -->"

AVISO = f"""{MARCA}
<sub>
<b>Tradução não oficial para português do Brasil.</b> Este arquivo é uma obra
derivada do repositório original de Sebastian Raschka
(<a href="https://github.com/rasbt/reasoning-from-scratch">rasbt/reasoning-from-scratch</a>),
licenciado sob Apache License 2.0. Apenas o texto foi traduzido; o código
permanece inalterado. Não é uma publicação oficial da Manning e não substitui o
livro. Detalhes das convenções em <code>GLOSSARIO-TRADUCAO.md</code>.
</sub>"""


def aplicar(nb_path):
    nb = load(nb_path)
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        texto = source_to_text(cell["source"])
        if MARCA in texto:
            print(f"{nb_path}: aviso já presente")
            return
        # A primeira célula markdown é a tabela de crédito ao autor original.
        cell["source"] = text_to_source(texto.rstrip() + "\n\n" + AVISO)
        dump(nb, nb_path)
        print(f"{nb_path}: aviso inserido")
        return
    print(f"{nb_path}: nenhuma célula markdown encontrada")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for caminho in sys.argv[1:]:
        aplicar(caminho)
