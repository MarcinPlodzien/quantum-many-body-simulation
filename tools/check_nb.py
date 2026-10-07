#!/usr/bin/env python
"""
check_nb.py -- editorial audit of the built notebooks (no execution).

For every part*/**.ipynb reports: #cells, #figures, error outputs, stderr/warning outputs, forbidden words,
presence of the mandatory sections (learning goals, prerequisites, checkpoints/asserts, summary, exercises,
references), use of JAX transformations, hard-coded dtypes, and size on disk.

    python tools/check_nb.py            # table for all notebooks
    python tools/check_nb.py -v         # + details of every problem
"""
import pathlib
import re
import sys

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"smoq(?!\.jax)|julia|\bport(ed|ing)?\b|\.jl\b", re.I)
SECTIONS = {"learn": r"what you will learn", "prereq": r"prerequisite", "summary": r"summary|takeaway",
            "exerc": r"exercise", "refs": r"reference"}


def audit(path):
    nb = nbformat.read(path, 4)
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    user_code = "\n".join(c.source for c in code_cells if "ENGINE RECAP" not in c.source and "CONFIGURATION" not in c.source)
    problems, figs = [], 0
    for i, c in enumerate(code_cells):
        for o in c.get("outputs", []):
            if o.output_type == "error":
                problems.append(f"cell {i}: ERROR {o.ename}: {o.evalue[:80]}")
            if o.output_type == "stream" and o.name == "stderr":
                problems.append(f"cell {i}: stderr: {o.text.strip()[:100]}")
            if o.output_type == "stream" and re.search(r"warning", o.text, re.I) and o.name != "stderr":
                problems.append(f"cell {i}: warning text: {o.text.strip()[:100]}")
            if "data" in o and "image/png" in o["data"]:
                figs += 1
        if c.get("execution_count") is None and c.source.strip():
            problems.append(f"cell {i}: not executed")
    for m in FORBIDDEN.finditer(md + "\n" + user_code):
        problems.append(f"forbidden word: '{m.group(0)}'")
    missing = [k for k, pat in SECTIONS.items() if not re.search(pat, md, re.I)]
    hard = len(re.findall(r"complex128|complex64|float64|float32", user_code))
    jaxuse = {k: len(re.findall(p, user_code)) for k, p in
              {"jit": r"jax\.jit|@jit|\bjit\(", "vmap": r"vmap", "scan": r"lax\.scan|\bscan\(", "grad": r"grad\("}.items()}
    return dict(cells=len(nb.cells), md_words=len(md.split()), figs=figs, asserts=len(re.findall(r"\bassert\b", user_code)),
                missing=missing, hard=hard, jax=jaxuse, problems=problems, kb=path.stat().st_size // 1024)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    print(f"{'notebook':58s} cells words figs asrt  jit vmap scan grad  hard   KB  missing / #problems")
    for p in sorted(list(ROOT.glob("part*/*.ipynb")) + list(ROOT.glob("ch*/*.ipynb"))):
        a = audit(p)
        j = a["jax"]
        print(f"{str(p.relative_to(ROOT))[:58]:58s} {a['cells']:5d} {a['md_words']:5d} {a['figs']:4d} {a['asserts']:4d} "
              f"{j['jit']:4d} {j['vmap']:4d} {j['scan']:4d} {j['grad']:4d} {a['hard']:5d} {a['kb']:5d}  "
              f"{','.join(a['missing']) or '-'} / {len(a['problems'])}")
        if verbose:
            for pr in a["problems"]:
                print("      !", pr)
