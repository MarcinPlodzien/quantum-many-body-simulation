#!/usr/bin/env python
"""
build_nb.py -- turn a percent-format source file (_src/<part>/<name>.py) into an executed,
self-contained Jupyter notebook (<part>/<name>.ipynb).

Source format
-------------
    #@title: Time evolution I -- TEBD
    #@part: Part 2 -- Quantum dynamics
    #@description: one sentence shown on the website listing

    # %% [markdown]
    # ## 1. Motivation
    # Markdown text, every line prefixed with "# " (LaTeX with $...$ / $$...$$ is fine).

    # %%
    #@engine: apply_gate, rdm, tebd_gates
    (this cell is REPLACED by the source of those engine objects + all their dependencies,
     extracted from quantum_engine.py -> notebooks are self-contained, all notebooks share the same engine code)

    # %%
    ordinary python code cell

Usage
-----
    python tools/build_nb.py _src/part2_quantum_dynamics/02_tebd.py            # build + execute
    python tools/build_nb.py --no-exec _src/...py                               # build only
    python tools/build_nb.py --all                                              # everything
Environment: set JAX_PLATFORMS=cpu to execute on CPU.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
import time

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_raw_cell

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "quantum_engine.py"
SRC = ROOT / "_src"

AUTHOR_BLOCK = (
    "**Marcin Płodzień** — Institute of Theoretical Physics, Jagiellonian University  \n"
    "[ORCID 0000-0002-0835-1644](https://orcid.org/0000-0002-0835-1644) · "
    "[github.com/MarcinPlodzien](https://github.com/MarcinPlodzien)\n\n"
    "*Quantum Many-Body Simulation — {part}*"
)

CONFIG_MD = (
    "## Configuration\n\n"
    "Every notebook of this course starts with the same cell. Choose where to run (`DEVICE`) and the floating-point "
    "precision (`PRECISION`): `\"double\"` = `float64/complex128` (default, recommended for physics), "
    "`\"single\"` = `float32/complex64` (half the memory, faster on consumer GPUs, but watch the round-off). "
    "All code below derives its dtypes from `RDTYPE`/`CDTYPE` and its check tolerances from `TOL`, so nothing else "
    "needs to change.")

CONFIG_HEAD = (
    "# ==============================================================================\n"
    "# CONFIGURATION  -- adapt to your hardware, then run the notebook top to bottom\n"
    "# ==============================================================================\n"
    'DEVICE    = "auto"     # "auto": use a GPU if JAX finds one, else CPU | "cpu" | "gpu"\n'
    'PRECISION = "double"   # "double": float64/complex128 | "single": float32/complex64\n\n'
    "import os\n"
    'if DEVICE == "cpu":\n'
    '    os.environ["JAX_PLATFORMS"] = "cpu"          # must be set BEFORE jax is imported\n\n')

CONFIG_TAIL = (
    "\n\nimport time\n"
    "import matplotlib.pyplot as plt\n\n"
    'if DEVICE == "gpu" and jax.default_backend() == "cpu":\n'
    '    print("WARNING: DEVICE=\'gpu\' requested but JAX found no GPU -- running on CPU.")\n'
    'print(f"JAX {jax.__version__} | backend: {jax.default_backend()} | devices: {jax.devices()}")\n'
    'print(f"precision: {PRECISION} -> real {RDTYPE.__name__}, complex {CDTYPE.__name__}, check tolerance TOL={TOL:g}")\n')

FOOTER = (
    "---\n"
    "**About these lectures.** Written by Marcin Płodzień (Institute of Theoretical Physics, Jagiellonian University; "
    "[ORCID](https://orcid.org/0000-0002-0835-1644), [GitHub](https://github.com/MarcinPlodzien)) as self-study "
    "material for the course *Quantum Many-Body Simulation: from a single spin to quantum machine learning*. The notebook is self-contained: the *Engine "
    "recap* cell holds every function of the SmoQ.jax engine it uses. If you use this material in teaching or "
    "research, please credit the author:\n\n"
    "> M. Płodzień, *Quantum Many-Body Simulation: from a single spin to quantum machine learning*, hands-on lectures "
    "in JAX with the SmoQ.jax engine (2026), https://marcinplodzien.github.io/quantum-many-body-simulation/"
)


# ------------------------------------------------------------------------------
# Engine extraction: names -> source, with transitive dependencies
# ------------------------------------------------------------------------------
class Engine:
    def __init__(self, path: pathlib.Path):
        self.text = path.read_text()
        self.lines = self.text.splitlines()
        tree = ast.parse(self.text)
        end_pre = next(i for i, l in enumerate(self.lines) if "END PREAMBLE" in l)
        doc_end = tree.body[0].end_lineno  # module docstring
        self.preamble = "\n".join(l for l in self.lines[doc_end:end_pre]
                                  if not l.startswith("from __future__")).strip()
        self.nodes = {}   # name -> (node, order)
        for order, node in enumerate(tree.body):
            if node.lineno - 1 <= end_pre:
                continue
            for name in self._defined(node):
                self.nodes[name] = (node, order)
        # section banners:  "# ====" / "# N. Title" / "# ===="
        self.banners = [(i, self.lines[i + 1]) for i in range(len(self.lines) - 2)
                        if self.lines[i].startswith("# ====") and self.lines[i + 2].startswith("# ====")]

    @staticmethod
    def _defined(node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            out = []
            for t in node.targets:
                out += [e.id for e in ast.walk(t) if isinstance(e, ast.Name)]
            return out
        return []

    def _deps(self, node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in self.nodes}

    def extract(self, names, show=False):
        missing = [n for n in names if n not in self.nodes]
        if missing:
            raise SystemExit(f"[build_nb] unknown engine names: {missing}")
        todo, chosen = list(names), {}
        while todo:
            n = todo.pop()
            node, order = self.nodes[n]
            if order in chosen:
                continue
            chosen[order] = node
            todo += list(self._deps(node) - {n})
        out, last_banner = [], None
        for order in sorted(chosen):
            node = chosen[order]
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
            banner = max((b for b in self.banners if b[0] < start), default=None)
            if banner and banner != last_banner:
                out.append(f"\n# {'-' * 76}\n{banner[1]}\n# {'-' * 76}")
                last_banner = banner
            seg = "\n".join(self.lines[start:node.end_lineno])
            out.append(seg + ("\n" if isinstance(node, ast.FunctionDef) else ""))
        header = ('#| code-fold: true\n#| code-summary: "Engine recap — the simulator primitives used in this notebook '
                  '(click to expand)"\n'
                  "# ==============================================================================\n"
                  "# ENGINE RECAP  (copied from quantum_engine.py, the SmoQ.jax engine)\n"
                  "# Every function below is explained, with its mathematics and an independent check, in\n"
                  "# notebook 08b (Chapter 3), 'Building the quantum simulator engine'.\n"
                  "# States are rank-N tensors of shape (2,)*N; every operator acts through ONE einsum.\n"
                  "# ==============================================================================\n")
        if show:   # Part 0: the function was just derived in the notebook -> show its final engine form, unfolded
            header = ("# ==============================================================================\n"
                      "# FINAL ENGINE VERSION  (verbatim from quantum_engine.py -- this is what later notebooks reuse)\n"
                      "# ==============================================================================\n")
        return header + "\n".join(out).strip("\n") + "\n"

    def config_cell(self):
        return CONFIG_HEAD + self.preamble + CONFIG_TAIL


# ------------------------------------------------------------------------------
# Markdown normalisation for Quarto
# ------------------------------------------------------------------------------
_LIST_ITEM = re.compile(r"^\s{0,3}([*+-]|\d+[.)])\s+\S")


def pandoc_safe_lists(md: str) -> str:
    """Insert a blank line before a list that directly follows a paragraph line.

    Jupyter (CommonMark) lets a list interrupt a paragraph; Pandoc, which Quarto uses, does not, and renders
    "**Physics**\\n* item" as one run-on paragraph with literal asterisks. Code fences and $$ display math are left alone.
    """
    out, fence, disp = [], False, False
    for line in md.split("\n"):
        s = line.strip()
        if out and not fence and not disp and _LIST_ITEM.match(line):
            prev = out[-1]
            if prev.strip() and not _LIST_ITEM.match(prev) and not prev.startswith(("  ", "\t", "|", "#", ">")):
                out.append("")
        out.append(line)
        if s.startswith("```"):
            fence = not fence
        elif not fence and s.count("$$") % 2 == 1:
            disp = not disp
    return "\n".join(out)


# ------------------------------------------------------------------------------
# Percent-format parser
# ------------------------------------------------------------------------------
def parse_source(path: pathlib.Path, engine: Engine):
    text = path.read_text()
    meta = dict(re.findall(r"^#@(title|part|description):\s*(.+)$", text, flags=re.M))
    for k in ("title", "part"):
        if k not in meta:
            raise SystemExit(f"[build_nb] {path}: missing '#@{k}:' directive")
    chunks = re.split(r"^# %%", text, flags=re.M)[1:]
    # raw YAML front matter: read by Quarto (browser-tab title, listing description); ignored by Jupyter
    front = '---\npagetitle: "{}"\ndescription: "{}"\n---'.format(
        meta["title"].replace('"', "'"), meta.get("description", "").replace('"', "'"))
    cells = [new_raw_cell(front),
             new_markdown_cell(f"# {meta['title']}\n\n" + AUTHOR_BLOCK.format(part=meta["part"]))]
    config = [new_markdown_cell(CONFIG_MD), new_code_cell(engine.config_cell())]
    for i, ch in enumerate(chunks):
        if "#@config" in ch:                     # explicit placement of the configuration cell
            cells += config; config = []
            continue
        if i == 1 and config:                    # default: right after the first (introduction) cell
            cells += config; config = []
        head, _, body = ch.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in head:
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            cells.append(new_markdown_cell(pandoc_safe_lists(md)))
        else:
            lines = body.splitlines()
            directive = [l for l in lines if re.match(r"\s*#@engine(-show)?:", l)]
            if directive:  # one or more "#@engine: a, b, c" lines; any other lines are kept below the recap
                names = [n for l in directive for n in re.split(r"[,\s]+", l.split(":", 1)[1]) if n]
                rest = "\n".join(l for l in lines if l not in directive)
                show = any("#@engine-show:" in l for l in directive)
                body = engine.extract(names, show=show) + (("\n" + rest) if rest.strip() else "")
            if body.strip():
                cells.append(new_code_cell(body))
    cells.append(new_markdown_cell(FOOTER))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    if "description" in meta:
        nb.metadata["description"] = meta["description"]
    return nb


ENV_NOISE = ("Unable to import Axes3D",)      # warnings caused by the build machine, not by the notebook


def scrub_environment_noise(nb):
    """Drop stderr outputs that only reflect the local installation (they would also leak local paths)."""
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = [o for o in c.get("outputs", [])
                         if not (o.get("output_type") == "stream" and o.get("name") == "stderr"
                                 and any(m in o.get("text", "") for m in ENV_NOISE))]


def build(path: pathlib.Path, execute=True, timeout=1800):
    engine = Engine(ENGINE)
    path = path.resolve()
    rel = path.relative_to(SRC)
    out = (ROOT / rel).with_suffix(".ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = parse_source(path, engine)
    if execute:
        from nbclient import NotebookClient
        t0 = time.time()
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(out.parent)}}).execute()
        print(f"[build_nb] executed {rel} in {time.time() - t0:.1f}s")
        scrub_environment_noise(nb)
    nbformat.write(nb, out)
    print(f"[build_nb] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    srcs = sorted(SRC.rglob("*.py")) if a.all else [pathlib.Path(s) for s in a.sources]
    if not srcs:
        ap.error("give source files or --all")
    for s in srcs:
        build(s, execute=not a.no_exec, timeout=a.timeout)
