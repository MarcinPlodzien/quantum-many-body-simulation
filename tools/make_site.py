#!/usr/bin/env python
"""
make_site.py -- generate the Quarto website scaffolding from the notebook sources.

Reads the #@title / #@part / #@description directives of every _src/<part>/<NN_name>.py and writes
    _quarto.yml   (sidebar = parts -> notebooks, in order)
    index.qmd     (landing page: course description, author, table of contents with descriptions)
    engine.qmd    (the documented engine source, rendered as a page)

Then:   quarto preview          # local preview
        quarto render           # static site in _site/   (notebooks are NOT re-executed: stored outputs are used)
        quarto publish gh-pages # publish to GitHub Pages
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "_src"
sys.path.insert(0, str(ROOT / "tools"))

COURSE = "Numerical simulations of many-body quantum systems with JAX"
ENGINE_NB = "ch03_matrix_free_engine/08b_building_quantum_simulator_engine.ipynb"
PROJECT = "SmoQ.jax"                                   # the engine developed in the course
ENGINE_TAGLINE = "a matrix-free JAX engine for quantum many-body systems"
SITE_TITLE = "Quantum Many-Body Simulation"            # the course (navbar, notebook headers)
HERO_TITLE = "From a Single Spin to Quantum Machine Learning"
SITE_SUBTITLE = "Hands-on lectures on simulating quantum systems in JAX, from scratch"
AUTHOR = "Marcin Płodzień"
AFFIL = "Institute of Theoretical Physics, Jagiellonian University"
ORCID = "https://orcid.org/0000-0002-0835-1644"
GITHUB = "https://github.com/MarcinPlodzien"

REPO = "https://github.com/MarcinPlodzien"
HOMEPAGE = "http://chaos.if.uj.edu.pl/ZOA/marcinplodzien"
SITE_URL = "https://marcinplodzien.github.io/quantum-many-body-simulation/"
COURSE_REPO = "https://github.com/MarcinPlodzien/quantum-many-body-simulation"
FOOTER_LINKS = ("[Homepage](" + HOMEPAGE + ") · "
                "[Email](mailto:mplodzien@gmail.com) · [ORCID](https://orcid.org/0000-0002-0835-1644) · "
                "[Scholar](https://scholar.google.com/citations?user=eC9nCmgAAAAJ&hl=en) · [GitHub](https://github.com/MarcinPlodzien) · "
                "[Scopus](https://www.scopus.com/authid/detail.uri?authorId=55031869100) · "
                "[WoS](https://www.webofscience.com/wos/author/record/K-7326-2017)")


# Companion notebooks of the LaTeX lecture notes live in _src/lecture_notes_notebooks/ (the book links to them there);
# on the website they are listed inside the chapter whose topic they share.
SITE_CHAPTER = {
    "49_single_excitation_on_a_lattice": "ch01_computational_toolbox",
    "48_quantifying_entanglement": "ch09_entanglement_and_complexity",
}


def read_chapter_page(folder):
    """_src/<chapter>/_chapter.md: a 'goal:' line (one-line outcome for the landing-page table), '---', intro prose."""
    f = SRC / folder / "_chapter.md"
    if not f.exists():
        return "", ""
    head, _, body = f.read_text().partition("\n---\n")
    goal = re.search(r"^goal:\s*(.+)$", head, flags=re.M)
    return (goal.group(1).strip() if goal else ""), body.strip()


def ensure_front_matter(nb_path, meta):
    """Notebooks built before the builder learned about YAML front matter get it patched in (no re-execution)."""
    import nbformat
    from nbformat.v4 import new_raw_cell
    from build_nb import pandoc_safe_lists
    nb = nbformat.read(nb_path, 4)
    changed = False
    for c in nb.cells:                      # lists directly after a paragraph line: see build_nb.pandoc_safe_lists
        if c.cell_type == "markdown":
            fixed = pandoc_safe_lists(c.source)
            # header of notebooks built before the current course title (earlier titles of the course)
            fixed = re.sub(r"\*(Lecture notes “Numerical simulations of many-body quantum systems with JAX”|"
                           r"(SmoQ\.jax: )?Quantum System Simulators? (with|in) JAX|Quantum Many-Body Simulation in JAX) — [^*\n]*\*",
                           "*" + SITE_TITLE + " — " + meta.get("part", "") + "*", fixed)
            if fixed.startswith("---\n**About these"):                    # footer written under an earlier course title
                from build_nb import FOOTER
                fixed = FOOTER
            changed |= fixed != c.source
            c.source = fixed
    if nb.cells and nb.cells[0].cell_type == "raw":
        if changed:
            nbformat.write(nb, nb_path)
        return
    nb.cells.insert(0, new_raw_cell('---\npagetitle: "{}"\ndescription: "{}"\n---'.format(
        meta["title"].replace('"', "'"), meta.get("description", "").replace('"', "'"))))
    nbformat.write(nb, nb_path)


def collect():
    parts = {}
    srcs = [q for q in SRC.glob("*/*.py") if q.parent.name[:2] in ("pa", "ch") or q.stem in SITE_CHAPTER]
    for p in sorted(srcs, key=lambda q: (SITE_CHAPTER.get(q.stem, q.parent.name), q.stem)):
        text = p.read_text()
        meta = dict(re.findall(r"^#@(title|part|description):\s*(.+)$", text, flags=re.M))
        nb = (ROOT / p.relative_to(SRC)).with_suffix(".ipynb")
        if not nb.exists():
            print(f"[make_site] skipping {p.name}: notebook not built yet")
            continue
        ensure_front_matter(nb, meta)
        folder = SITE_CHAPTER.get(p.stem, p.parent.name)
        if folder != p.parent.name and folder not in parts:
            raise SystemExit(f"[make_site] {p.name}: website chapter {folder} has no notebooks of its own")
        parts.setdefault(folder, {"title": meta.get("part", folder), "items": []})
        parts[folder]["items"].append((nb.relative_to(ROOT).as_posix(), meta["title"], meta.get("description", "")))
    return parts


def chapter_label(title):
    """'Chapter 5 — Ground states and unitary dynamics' -> ('5', 'Ground states and unitary dynamics')."""
    m = re.match(r"Chapter\s+(\d+)\s*[—-]\s*(.+)", title)
    return (m.group(1), m.group(2).strip()) if m else ("", title)


def q(s):
    return '"' + s.replace('"', "'") + '"'


def main():
    parts = collect()
    (ROOT / "chapters").mkdir(exist_ok=True)
    chapters = []                                   # (folder, number, name, goal, page, items)
    for d, part in parts.items():
        num, name = chapter_label(part["title"])
        goal, intro = read_chapter_page(d)
        page = f"chapters/{d}.qmd"
        chapters.append((d, num, name, goal, page, part["items"]))
        # ------------------------------------------------------------ chapters/<folder>.qmd (concept page)
        thumb = f"assets/thumbs/{d[:4]}.png"
        pg = ["---", f"title: {q(f'{num} · {name}')}", "code-tools: false", "---", ""]
        if (ROOT / thumb).exists():
            pg += [f"![](../{thumb}){{width=60% fig-align=center fig-alt={q(name)}}}", ""]
        pg += [intro, "", "## Notebooks in this chapter", ""]
        for path, title, desc in part["items"]:
            pg += ["::: {.nb-item}", f"[{title}](../{path})", "", desc, ":::", ""]
        (ROOT / page).write_text("\n".join(pg) + "\n")
    # ---------------------------------------------------------------- _quarto.yml
    y = ["project:", "  type: website", "  output-dir: _site", "  render:", "    - index.qmd", "    - engine.qmd",
         '    - "chapters/*.qmd"', '    - "ch*/*.ipynb"', '    - "lecture_notes_notebooks/*.ipynb"', "",
         "execute:", "  enabled: false   # use the outputs stored in the notebooks; rebuild them with tools/build_nb.py", "",
         "website:", f"  title: {q(SITE_TITLE)}", f"  site-url: {q(SITE_URL)}", f"  repo-url: {q(COURSE_REPO)}",
         '  description: "Self-study lecture notes on simulating quantum many-body systems with a matrix-free JAX engine."',
         "  page-navigation: true", "  navbar:", "    background: primary", "    search: true", "    pinned: true",
         "    left:", '      - text: "Start here"', "        href: index.qmd", '      - text: "Chapters"', "        menu:"]
    for d, num, name, goal, page, items in chapters:
        y += [f"          - text: {q(f'{num} · {name}')}", f"            href: {page}"]
    y += ['      - text: "The engine"', "        menu:",
          '          - text: "Building the engine (notebook 08b)"', f"            href: {ENGINE_NB}",
          '          - text: "Engine source code"', "            href: engine.qmd",
          "    right:", "      - icon: github",
          f"        href: {q(GITHUB)}", "        aria-label: GitHub",
          "      - icon: house", f"        href: {q(HOMEPAGE)}", "        aria-label: Homepage",
          "  sidebar:", "    logo: assets/qusml_logo.svg", "    style: floating", "    search: true", "    collapse-level: 1", "    contents:",
          '      - text: "Start here"', "        href: index.qmd"]
    for d, num, name, goal, page, items in chapters:
        y += [f"      - section: {q(f'{num} · {name}')}", f"        href: {page}", "        contents:"]
        for path, title, _ in items:
            y += [f"          - href: {path}", f"            text: {q(title)}"]
    y += ['      - text: "Engine source code"', "        href: engine.qmd",
          "  page-footer:", "    border: true", f"    left: {q(f'© 2026 {AUTHOR} · {AFFIL}')}",
          "    right: >-", f"      {FOOTER_LINKS}", "",
          "format:", "  html:", "    theme:", "      light: [cosmo, assets/qusml.scss]", "      dark: [darkly, assets/qusml-dark.scss]", "    toc: true", "    toc-depth: 3",
          "    number-sections: false", "    code-fold: show", "    code-tools: true", "    code-copy: true",
          "    code-overflow: wrap", "    highlight-style: github", "    html-math-method: mathjax", "    grid:",
          "      sidebar-width: 340px", ""]
    (ROOT / "_quarto.yml").write_text("\n".join(y))
    # ---------------------------------------------------------------- index.qmd (landing page)
    n_nb = sum(len(c[5]) for c in chapters)
    first_nb = chapters[0][5][0][0]
    starter = (ROOT / "examples" / "starter.py").read_text().split("\n")
    starter = "\n".join(l for l in starter if not l.startswith(("# The simulator in five", "import sys, pathlib")))
    ix = ["---", f"pagetitle: {q(SITE_TITLE)}", "toc: false", "code-tools: false",
          "code-fold: false", "---", "",
          "::: {.hero}", "::: {.hero-text}", f'<h1 class="hero-title">{HERO_TITLE}</h1>', "", f'<p class="hero-tagline">{SITE_SUBTITLE}</p>', "",
          '<p class="hero-sub">Starting from an empty Python file, the lectures develop a matrix-free simulation engine '
          "and use it to study spin-1/2 systems, quantum circuits, unitary and dissipative dynamics, and quantum "
          "machine learning.</p>", "",
          f'<p class="hero-author"><a href="{HOMEPAGE}"><b>{AUTHOR}</b></a> · {AFFIL}</p>', "",
          f"[Start with notebook 00a]({first_nb}){{.btn .btn-primary}} [Browse the chapters](#chapters){{.btn .btn-outline-primary}} "
          f"[The engine]({ENGINE_NB}){{.btn .btn-outline-primary}}", ":::", "",
          "::: {.hero-image}", "![](assets/hero.png){fig-alt=\"Light cone after a local quench, domain-wall melting "
          "computed with MPS-TEBD, the magic of single-qubit states, and Husimi functions of one-axis-twisted cat states\"}",
          ":::", ":::", "",
          "## Preface", "",
          "Quantum many-body physics and quantum computing share one computational object: the state of many "
          "interacting two-level systems. Whether the two levels are the spin of an electron, two internal states of a "
          "trapped ion or the qubit of a superconducting processor, the state of $N$ of them is a vector of $2^N$ complex "
          "amplitudes, and every simulation is a sequence of operations on that vector. These lectures build, starting "
          f"from an empty Python file, a complete simulation engine for such systems, **{PROJECT}**, {ENGINE_TAGLINE}, "
          "and then use it to study their physics.", "",
          "The simulator is written in [JAX](https://docs.jax.dev) and grows chapter by chapter. It stores the state of "
          "$N$ spins-1/2 as a rank-$N$ tensor and applies every operator, whether a gate, a Hamiltonian term, a Kraus "
          "operator or a projector, through a single `einsum` contraction, so the $2^N\\times 2^N$ matrix of textbook "
          "treatments is never formed. Because the whole simulator is differentiable, the same code that evolves a "
          "spin chain also trains a quantum circuit, which makes it a natural laboratory for quantum machine learning. "
          "On this one primitive the notes construct a fully functional toolbox:", "",
          "::: {.feature-grid}"]
    FEATURES = [
        ("Spin-1/2 many-body physics", "Hamiltonians on arbitrary graphs, exact diagonalisation and matrix-free Lanczos, "
         "symmetries, entanglement, phase diagrams and quantum phase transitions.", ["ch02", "ch05", "ch13"]),
        ("Unitary dynamics", "Exact propagation, Runge–Kutta, Trotter–Suzuki, Chebyshev and Krylov integrators, compared "
         "on accuracy and cost, and applied to quantum quenches and light cones.", ["ch05"]),
        ("Dissipative dynamics", "The Lindblad master equation on the density tensor and its unravelling into quantum "
         "trajectories, with noise channels and decoherence of entangled states.", ["ch06", "ch03"]),
        ("Quantum circuits", "Gates as exponentials of Pauli operators, circuits as compiled lists of contractions, "
         "measurement, feed-forward, random circuits, and protocols such as teleportation.", ["ch04", "ch08"]),
        ("Quantum machine learning", "Parametrised circuits trained with gradients from the parameter-shift rule and "
         "from automatic differentiation, optimisers, the variational eigensolver, quantum autoencoders, training under "
         "shot noise and gate noise, and quantum reservoir computing, where many-body dynamics processes time series.",
         ["ch11", "ch12"]),
        ("Beyond the state vector", "Matrix product states with DMRG and TEBD for chains far longer than a state vector "
         "can hold, and diagnostics of entanglement, magic and metrological usefulness.", ["ch07", "ch09", "ch10"]),
    ]
    by_prefix = {c[0][:4]: c for c in chapters}
    for title, text, chs in FEATURES:
        links = " · ".join(f"[{by_prefix[k][1]} · {by_prefix[k][2]}]({by_prefix[k][4]})" for k in chs if k in by_prefix)
        ix += ["::: {.feature}", f"**{title}**", "", text, "", f"Chapters: {links}", ":::"]
    ix += [":::", "",
           "On first contact, these methods are usually met as black boxes inside large libraries. Writing the simulator "
           "oneself shows what such a library takes care of: how the ordering of the qubits fixes every index, why a time step is "
           "stable or unstable, what an algorithm costs and where its error comes from. Every derivation in the notes is "
           "carried out explicitly, every algorithm is validated against an exact or independent result, and the code is "
           "written to be read, with documented functions and with the JAX tools introduced where they are first needed: "
           "compiled loops (`jit`, `lax.scan`), batching (`vmap`) and automatic differentiation (`grad`). The notes "
           "therefore teach three things together: the physics, the numerical methods, and the practice of writing "
           "scientific code that is both fast and correct.", "",
           "## Quantum machine learning", "",
           "The differentiable simulator is used throughout Chapters 11 and 12 to build and train quantum "
           "machine-learning models, each derived, implemented and tested against exact results:", ""]
    QML = [("40", "parametrised quantum circuits and four ways to compute their gradients"),
           ("41", "optimisers for variational circuits, from gradient descent to the quantum natural gradient"),
           ("42", "the variational quantum eigensolver for ground states of spin chains"),
           ("43", "the measurement cost of a variational algorithm, with classical shadows in the training loop"),
           ("44", "training under gate noise: cost landscapes, noise-induced barren plateaus and error mitigation"),
           ("45", "the quantum autoencoder, compressing quantum data with a variational circuit"),
           ("46", "quantum reservoir computing, many-body dynamics as a machine for time-series prediction")]
    all_items = {pathlib.Path(path).name[:2]: (path, title) for c in chapters for path, title, _ in c[5]}
    for nn, what in QML:
        if nn in all_items:
            path, title = all_items[nn]
            ix.append(f"- [{what[0].upper() + what[1:]}]({path}) (notebook {nn})")
    ix += ["",
           "## Start here", "",
           f"The engine fits in one file, [`quantum_engine.py`](engine.qmd); notebook 08b, [Building the quantum simulator "
           f"engine]({ENGINE_NB}), explains every function in it with its mathematics and an independent check. Every "
           "capability above is a few lines "
           "of code on top of it. The snippet below prepares and samples a GHZ state, finds the ground state of a "
           "Heisenberg chain of sixteen spins, lets a Néel state melt under the same Hamiltonian, drives a decaying qubit "
           "with the Lindblad equation, and differentiates the energy of a variational circuit with respect to all its "
           "angles. It runs in about a minute or less on a laptop CPU (`python examples/starter.py`) and prints one labelled "
           "line per part; the notes explain every function it uses, starting from nothing.", "",
           "```python", starter.strip(), "```", "",
           "**Where to go next.**", "",
           f"- *First contact with numerics:* notebooks 00a, 00b, 49, 01 and 02 ([Chapter 1]({by_prefix['ch01'][4]})), then "
           f"[Chapter 2]({by_prefix['ch02'][4]}).",
           f"- *Comfortable with quantum mechanics and Python:* start at [Chapter 3]({by_prefix['ch03'][4]}), where the "
           "engine is built, and continue in order.",
           f"- *Interested in quantum information:* after Chapter 3, go to [Chapter 4]({by_prefix['ch04'][4]}) and "
           f"[Chapter 8]({by_prefix['ch08'][4]}).",
           f"- *Interested in quantum technologies:* [Chapter 10]({by_prefix['ch10'][4]}) on metrology and "
           f"[Chapter 11]({by_prefix['ch11'][4]}) on variational circuits build on Chapters 3–5.",
           f"- *Interested in quantum machine learning:* after Chapters 3–5, [Chapter 11]({by_prefix['ch11'][4]}) "
           f"(variational circuits, the variational eigensolver, quantum autoencoders) and "
           f"[Chapter 12]({by_prefix['ch12'][4]}) (quantum reservoir computing).", "",
           "## Chapters {#chapters}", "",
           f"{len(chapters)} chapters, each a short overview page followed by runnable notebooks ({n_nb} in total). The "
           "notebooks are numbered globally; each one carries the engine functions it uses in a folded *Engine recap* cell "
           "and ends with exercises, so any notebook also runs and can be studied on its own.", "",
           "::: {.chapter-grid}"]
    for d, num, name, goal, page, items in chapters:
        thumb = f"assets/thumbs/{d[:4]}.png"
        img = f"[![]({thumb}){{fig-alt={q(name)}}}]({page})" if (ROOT / thumb).exists() else ""
        ix += ["::: {.chapter-card}", img, "", f"[{num} · {name}]({page})", "", goal + ".", "",
               f'<span class="nbcount">{len(items)} notebook{"s" if len(items) != 1 else ""}</span>', ":::"]
    ix += [":::", "",
           "## Getting started", "", "```bash",
           "pip install -r requirements.txt          # jax, numpy, scipy, matplotlib, jupyter",
           "python examples/starter.py               # the snippet above",
           "jupyter lab                              # open any notebook in the ch*/ folders", "```", "",
           "The configuration cell at the top of every notebook selects the device (CPU or GPU) and the precision "
           "(single or double). Everything runs on a laptop CPU.", "",
           "## Who this is for", "",
           "Students who have taken one semester of quantum mechanics and a first Python course, and anyone who wants to "
           "learn how many-body quantum simulations are written in practice. Every derivation is carried out step by "
           "step, and every result is checked numerically.", "",
           "## How to cite", "",
           f"> M. Płodzień, *{SITE_TITLE}: {HERO_TITLE.lower()}*, "
           f"hands-on lectures in JAX with the {PROJECT} engine (2026), {SITE_URL}", ""]
    (ROOT / "index.qmd").write_text("\n".join(ix))
    # ---------------------------------------------------------------- engine.qmd
    eng = (ROOT / "quantum_engine.py").read_text()
    (ROOT / "engine.qmd").write_text(
        "---\ntitle: \"The SmoQ.jax engine\"\nsubtitle: \"quantum_engine.py: a matrix-free JAX engine for quantum many-body systems, fully documented\"\n"
        "code-fold: false\n---\n\n"
        "This page lists the complete source of the engine. To learn it, start with notebook 08b, "
        f"[Building the quantum simulator engine]({ENGINE_NB}), where every function is derived with its "
        "mathematics, its `einsum` string is read index by index, and an independent dense check confirms it. "
        "The *Engine recap* cell of each notebook is copied from this file, and `tests/test_core.py` validates it "
        "against dense linear algebra. The engine is also available as the Python package "
        "[SmoQ.jax](https://github.com/MarcinPlodzien/SmoQ.jax).\n\n"
        "```python\n" + eng + "\n```\n")
    print(f"[make_site] wrote _quarto.yml, index.qmd, engine.qmd, {len(chapters)} chapter pages  ({n_nb} notebooks)")


if __name__ == "__main__":
    main()
