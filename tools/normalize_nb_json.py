#!/usr/bin/env python
"""Write every built notebook in the canonical nbformat layout (cell sources and text outputs as lists of lines).

    python tools/normalize_nb_json.py [notebook.ipynb ...]      # default: all built notebooks

Both a string and a list of lines are valid nbformat, but Quarto 1.2 fails on string-valued cell sources
("cell.source.join is not a function").  Notebooks written by tools/build_nb.py are already canonical; this fixes
notebooks patched in place by other tools.  Content is unchanged: only the JSON layout is.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def lines(x):
    return x.splitlines(keepends=True) if isinstance(x, str) else x


def normalize(path):
    nb = json.loads(path.read_text())
    before = json.dumps(nb, sort_keys=True)
    for c in nb["cells"]:
        c["source"] = lines(c.get("source", []))
        for o in c.get("outputs", []):
            if "text" in o:
                o["text"] = lines(o["text"])
            for k, v in o.get("data", {}).items():
                if isinstance(v, str) and (k.startswith("text/") or k == "image/svg+xml"):
                    o["data"][k] = lines(v)
    if json.dumps(nb, sort_keys=True) != before:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        return True
    return False


def main():
    paths = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(ROOT.glob("ch*/*.ipynb")) + sorted(ROOT.glob("lecture_notes_notebooks/*.ipynb"))
    changed = [p for p in paths if normalize(p)]
    print(f"[normalize_nb_json] {len(changed)} of {len(paths)} notebooks rewritten")


if __name__ == "__main__":
    main()
