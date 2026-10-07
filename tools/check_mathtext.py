#!/usr/bin/env python
r"""Test-parse every math expression that a notebook hands to matplotlib.

matplotlib renders plot labels with its own mathtext parser, not LaTeX, and the two disagree:
`\frac34` and `\mathcal N` are valid LaTeX and fatal to mathtext, which needs `\frac{3}{4}` and
`\mathcal{N}`.  `\mathds` does not exist in mathtext at all.  The failure surfaces only when the
cell runs, which for a long notebook is minutes in, so it is worth catching statically.

Only STRING LITERALS are checked.  Markdown in `#` comments is rendered by the notebook front end,
which is real LaTeX, so those are skipped.  f-string placeholders are blanked before parsing.

    python tools/check_mathtext.py                 # every notebook source
    python tools/check_mathtext.py 48_quant        # only paths containing the pattern
"""
import ast, pathlib, re, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext

ROOT = pathlib.Path(__file__).resolve().parents[1]
pat = sys.argv[1] if len(sys.argv) > 1 else ""
parser = mathtext.MathTextParser("path")
PRINTF = re.compile(r"%[-+ #0-9.]*[difeEgGs]")
def literals(tree):
    """Every string constant, with f-string fields replaced by a harmless "0".

    The substitution is done from the AST, never by a regex over the text: a regex for {...}
    also eats the braces of \frac{3}{4} and turns it into \frac00, which is how the first
    version of this script reported 285 false failures.
    """
    inner = set()                              # Constant nodes that belong to an f-string
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                inner.add(id(v))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in inner:              # a fragment of an f-string, not a string of its own
                continue
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    parts.append("0")          # a formatted field: any number will do
            text = "".join(parts)
            if text:
                yield node.lineno, text

fail = 0
files = sorted(p for p in ROOT.glob("_src/*/*.py") if pat in str(p))
for f in files:
    src = f.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"{f.name}: cannot parse ({e})"); fail += 1; continue
    for lineno, s in literals(tree):
        if "$" not in s:
            continue
        # inline math is delimited by PAIRS of $, so split rather than regex-match: a regex for
        # $...$ mis-pairs the delimiters in a string that holds two separate spans.
        parts = s.split("$")
        if len(parts) % 2 == 0:
            # An odd number of $ means this literal is half of something: math assembled by
            # concatenation, r"$\mu=" + f"{mu:.3f}$", or a "$" used to strip markup. Neither is
            # resolvable without running the code, and flagging them produced only false alarms.
            continue
        for span in parts[1::2]:
            probe = PRINTF.sub("0", span)     # %.3f / %d are filled in at runtime
            try:
                parser.parse("$" + probe + "$")
            except Exception as e:
                msg = str(e).splitlines()[-1]
                print(f"{f.name}:{lineno}: mathtext cannot parse ${span}$\n    {msg[:100]}")
                fail += 1

if fail:
    print(f"\n{fail} unparseable expression(s). matplotlib needs braces: "
          r"\frac{3}{4}, \mathcal{N}; and \mathds does not exist in mathtext.")
    sys.exit(1)
print(f"all matplotlib math expressions parse ({len(files)} notebook sources)")
