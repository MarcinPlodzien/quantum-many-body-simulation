#!/usr/bin/env python
"""
check_math.py -- lint the MARKDOWN of notebook sources for math that is not LaTeX or that renders badly on GitHub.

Rules (house style: EVERY formula is LaTeX; must render in Jupyter, Quarto AND the GitHub notebook viewer):
  U  unicode math characters in prose (Greek letters, bra-kets, sub/superscripts, operators) -> write $...$ LaTeX
  T  '|' inside $...$ within a markdown TABLE row (breaks the table everywhere)    -> use \\vert, \\lvert/\\rvert, \\langle/\\rangle
  E  \\begin{align|equation|gather|eqnarray} (not rendered on GitHub)               -> $$\\begin{aligned} ... \\end{aligned}$$
  M  macros unsupported by stock MathJax: \\ket \\bra \\braket \\bm \\mathds \\slashed \\qty \\dd   -> expand by hand  (\\tag is AMS: fine)
  D  display math '$$' sharing a line with prose, or not surrounded by blank lines (GitHub needs its own paragraph)
  Q  display math inside a blockquote ('> $$')                                     -> move it out of the blockquote
  L  display math inside a list item                                               -> move it out / un-indent

    python tools/check_math.py                # summary table for all sources
    python tools/check_math.py -v [pattern]   # every finding with line numbers (optionally only files matching pattern)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
UNI = re.compile(r"[\u0370-\u03FF\u2070-\u209F\u2100-\u214F\u2200-\u22FF\u27E8\u27E9\u2A00-\u2AFF\u00B2\u00B3\u00B9\u00D7\u00B1\u221A\u210F]")
BADENV = re.compile(r"\\begin\{(align\*?|equation\*?|gather\*?|eqnarray\*?|multline\*?)\}")
BADMAC = re.compile(r"\\(ket|bra|braket|ketbra|bm|mathds|slashed|qty|dd)\b")
INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def markdown_lines(path):
    """Yield (lineno, text) of markdown cells of a percent-format source, '# ' prefix stripped."""
    in_md = False
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("# %%"):
            in_md = "[markdown]" in line
            continue
        if in_md and line.startswith("#"):
            yield i, re.sub(r"^# ?", "", line)


def lint(path):
    out = []
    lines = list(markdown_lines(path))
    in_code = in_disp = False
    for k, (i, t) in enumerate(lines):
        if t.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        bare = re.sub(r"`[^`]*`", "", t)                       # ignore inline code
        prose = INLINE.sub("", re.sub(r"\$\$.*?\$\$", "", bare))
        if not in_disp and UNI.search(prose) and "$$" not in bare:
            out.append((i, "U", "unicode math in prose: " + " ".join(sorted(set(UNI.findall(prose))))))
        if bare.lstrip().startswith("|") and any("|" in m for m in INLINE.findall(bare)):
            out.append((i, "T", "'|' inside inline math in a table row"))
        if BADENV.search(bare):
            out.append((i, "E", "unsupported environment: " + BADENV.search(bare).group(0)))
        if BADMAC.search(bare):
            out.append((i, "M", "unsupported macro: " + BADMAC.search(bare).group(0)))
        n = bare.count("$$")
        if n:
            s = bare.strip()
            if s.startswith(">"):
                out.append((i, "Q", "display math inside a blockquote"))
            elif re.match(r"^\s*([-*+]|\d+\.)\s", bare) or (bare.startswith("   ") and not in_disp):
                out.append((i, "L", "display math inside a list item / indented"))
            opens_or_closes_alone = s in ("$$",) or (s.startswith("$$") and s.endswith("$$") and len(s) > 4)
            if not opens_or_closes_alone and not (in_disp and s.endswith("$$")) and not (s.startswith("$$") and n == 1):
                out.append((i, "D", "display math shares its line with prose"))
            if not in_disp:                                     # opening: previous line must be blank
                prev = lines[k - 1][1].strip() if k else ""
                if prev and not prev.startswith("$$"):
                    out.append((i, "D", "no blank line before display math"))
            closes = (n == 2) or in_disp
            if closes and k + 1 < len(lines) and lines[k + 1][1].strip() and n != 1 or (in_disp and n == 1 and k + 1 < len(lines) and lines[k + 1][1].strip()):
                out.append((i, "D", "no blank line after display math"))
            if n == 1:
                in_disp = not in_disp
    return out


def fix_blank_lines(md_lines):
    """Give every display-math block its own paragraph: blank line before the opening $$ and after the closing $$.
    Works on a list of markdown lines (no '# ' prefix); skips blockquotes, tables and fenced code."""
    out, in_disp, in_code = [], False, False
    for t in md_lines:
        s = t.strip()
        if s.startswith("```"):
            in_code = not in_code
        n = 0 if in_code or s.startswith(">") or s.startswith("|") else re.sub(r"`[^`]*`", "", t).count("$$")
        opening = n and not in_disp and s.startswith("$$")
        if opening and out and out[-1].strip():
            out.append("")
        out.append(t)
        if n == 1:
            in_disp = not in_disp
        closed = n and not in_disp and s.endswith("$$")
        if closed:
            out.append("\0")                      # marker: need a blank line unless the next one is blank
    res = []
    for j, t in enumerate(out):
        if t == "\0":
            if j + 1 < len(out) and out[j + 1].strip():
                res.append("")
            continue
        res.append(t)
    return res


def fix_source(path):
    """Apply fix_blank_lines to every markdown cell of a percent-format source AND its built notebook (no re-execution)."""
    import nbformat
    chunks = re.split(r"(?m)^(?=# %%)", path.read_text())
    new = []
    for ch in chunks:
        head, _, body = ch.partition("\n")
        if head.startswith("# %%") and "[markdown]" in head:
            lines = [re.sub(r"^# ?", "", l) for l in body.rstrip("\n").split("\n")]
            body = "\n".join(("# " + l) if l else "#" for l in fix_blank_lines(lines)) + "\n\n"
        new.append(head + "\n" + body if head.startswith("# %%") else ch)
    path.write_text("".join(new))
    nbp = (ROOT / path.relative_to(ROOT / "_src")).with_suffix(".ipynb")
    if nbp.exists():
        nb = nbformat.read(nbp, 4)
        for c in nb.cells[2:-1]:
            if c.cell_type == "markdown":
                c.source = "\n".join(fix_blank_lines(c.source.split("\n")))
        nbformat.write(nb, nbp)


if __name__ == "__main__":
    if "--fix-blank-lines" in sys.argv:
        for p in sorted((ROOT / "_src").rglob("*.py")):
            fix_source(p)
        print("blank lines around display math fixed in sources and notebooks")
    verbose = "-v" in sys.argv
    pat = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
    print(f"{'source':66s}    U    T    E    M    D    Q    L")
    for p in sorted((ROOT / "_src").rglob("*.py")):
        if pat not in str(p):
            continue
        f = lint(p)
        c = {k: sum(1 for x in f if x[1] == k) for k in "UTEMDQL"}
        print(f"{str(p.relative_to(ROOT / '_src'))[:66]:66s} " + " ".join(f"{c[k]:4d}" for k in "UTEMDQL"))
        if verbose:
            for i, k, msg in f:
                print(f"      line {i:5d} [{k}] {msg}")
