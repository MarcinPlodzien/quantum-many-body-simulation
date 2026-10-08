#!/usr/bin/env python
"""
build_tikz.py -- compile the TikZ figures of the notebooks into self-contained SVG files.

Sources       _src/<chapter>/figures/<name>.tex      (\\documentclass[tikz,border=4pt]{standalone}, quantikz allowed)
Output        <chapter>/figures/<name>.svg            (next to the notebooks; referenced as figures/<name>.svg)

Each source is compiled with pdflatex in a temporary directory and converted with `pdftocairo -svg`, which writes
every glyph as a vector path: the SVG needs no fonts.  A white background rectangle is inserted, so the figure stays
legible on the dark theme of the website.

Usage
-----
    python tools/build_tikz.py                                   # every figure whose .tex is newer than its .svg
    python tools/build_tikz.py --force                           # rebuild all
    python tools/build_tikz.py _src/ch11_*/figures/qae.tex       # selected sources (always rebuilt)
    python tools/build_tikz.py --png                             # also write <name>.png previews into the temp dir
Requires pdflatex (TeX Live, with quantikz for circuit figures) and pdftocairo (poppler-utils).
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "_src"


def target_of(tex):
    """_src/<chapter>/figures/<name>.tex -> <chapter>/figures/<name>.svg"""
    rel = tex.resolve().relative_to(SRC)
    return ROOT / rel.parent / (tex.stem + ".svg")


def add_white_background(svg_text):
    """Insert a white rectangle as the first drawn element (pdftocairo leaves the page transparent)."""
    m = re.search(r"<svg\b[^>]*>", svg_text)
    if m is None:
        raise ValueError("no <svg> element")
    rect = '\n<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>'
    return svg_text[:m.end()] + rect + svg_text[m.end():]


def build(tex, png_dir=None):
    out = target_of(tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tikz_") as tmp:
        tmp = pathlib.Path(tmp)
        shutil.copy2(tex, tmp / tex.name)
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                             cwd=tmp, capture_output=True, text=True)
        pdf = tmp / (tex.stem + ".pdf")
        if run.returncode != 0 or not pdf.exists():
            log = (tmp / (tex.stem + ".log")).read_text(errors="ignore") if (tmp / (tex.stem + ".log")).exists() else run.stdout
            errs = [l for l in log.splitlines() if l.startswith("!")] or log.splitlines()[-20:]
            raise SystemExit(f"[tikz] pdflatex failed for {tex}:\n  " + "\n  ".join(errs[:20]))
        svg = tmp / (tex.stem + ".svg")
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        out.write_text(add_white_background(svg.read_text()))
        if png_dir is not None:
            png_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["pdftocairo", "-png", "-r", "150", "-singlefile", str(pdf), str(png_dir / tex.stem)],
                           check=True)
    print(f"[tikz] {tex.relative_to(ROOT)} -> {out.relative_to(ROOT)}")


def main(argv):
    force = "--force" in argv
    png_dir = None
    if "--png" in argv:
        i = argv.index("--png")
        png_dir = pathlib.Path(argv[i + 1]) if i + 1 < len(argv) and not argv[i + 1].startswith("-") \
            and not argv[i + 1].endswith(".tex") else pathlib.Path(tempfile.gettempdir()) / "tikz_png"
    files = [pathlib.Path(a).resolve() for a in argv if a.endswith(".tex")]
    if not files:
        files = sorted(SRC.glob("*/figures/*.tex"))
        files = [f for f in files if force or not target_of(f).exists()
                 or target_of(f).stat().st_mtime < f.stat().st_mtime]
    for tex in files:
        build(tex, png_dir)
    if png_dir is not None:
        print(f"[tikz] PNG previews in {png_dir}")
    if not files:
        print("[tikz] all figures up to date")


if __name__ == "__main__":
    main(sys.argv[1:])
