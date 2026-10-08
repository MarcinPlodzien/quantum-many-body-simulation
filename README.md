# Quantum Many-Body Simulation

**From a single spin to quantum machine learning**: hands-on lectures on simulating quantum systems in JAX, from scratch, with **SmoQ.jax**, a matrix-free JAX engine for quantum many-body systems.

Website: **https://marcinplodzien.github.io/quantum-many-body-simulation/**

> The course is under active review: notebooks are being checked and improved, and their content may change.

**Marcin Płodzień** — Institute of Theoretical Physics, Jagiellonian University in Kraków ·
[ORCID 0000-0002-0835-1644](https://orcid.org/0000-0002-0835-1644) · [github.com/MarcinPlodzien](https://github.com/MarcinPlodzien)

Self-study lecture notes (Jupyter notebooks) that teach, side by side, the **physics** of quantum many-body systems,
the **numerical methods** used to simulate them, and **good implementation practice** in JAX.
Prerequisites: one semester of quantum mechanics and basic Python.

The common thread is a small **matrix-free simulator** built from scratch: a state of N spins is a rank-N tensor and
every operator (gate, Hamiltonian term, Kraus operator, projector) acts through a single `einsum` contraction —
no 2^N × 2^N matrix is ever built. `jit`, `vmap`, `lax.scan` and `grad` make it fast and differentiable, on CPU or GPU,
in single or double precision (configuration cell at the top of every notebook).

## Layout

| path | what |
|---|---|
| `ch01_computational_toolbox/` … `ch12_quantum_reservoir_computing/` | the executed notebooks (each one self-contained) |
| `quantum_engine.py` | the documented engine; the “Engine recap” cell of every notebook is copied from it |
| `tests/test_core.py` | validation of every engine primitive against dense linear algebra |
| `_src/<chapter>/` | notebook sources in percent format (edit these, never the `.ipynb`) |
| `tools/build_nb.py` | `_src/*.py` → executed `.ipynb` (adds title/author block, configuration cell, engine recap, footer) |
| `tools/make_site.py` | generates `_quarto.yml`, `index.qmd`, `engine.qmd` for the website |

## Quick start

```bash
pip install -r requirements.txt
jupyter lab ch01_computational_toolbox/01_jax_from_scratch.ipynb
```

The notebooks are numbered globally, `00a` to `46`, and are meant to be read in that order; the chapter folders
group them by subject. Every notebook opens with a configuration cell selecting CPU or GPU and single or double
precision, and carries the simulator primitives it uses in a folded *Engine recap* cell, so any one of them runs
on its own.

## Rebuilding notebooks and the website

```bash
JAX_PLATFORMS=cpu python tests/test_core.py                     # validate the engine
# rebuild + execute one notebook:
JAX_PLATFORMS=cpu python tools/build_nb.py _src/ch05_ground_states_and_unitary_dynamics/12_tebd_trotter_suzuki.py
python tools/build_nb.py --all                                  # rebuild everything
python tools/check_nb.py -v                                     # editorial audit of the built notebooks
python tools/check_math.py -v <name>                            # LaTeX / rendering audit of a source
python tools/make_site.py && quarto preview                     # website preview
quarto publish gh-pages                                         # publish to GitHub Pages
```

## How to cite

> M. Płodzień, *Quantum Many-Body Simulation: from a single spin to quantum machine learning*, hands-on lectures in JAX with the SmoQ.jax engine (2026), https://marcinplodzien.github.io/quantum-many-body-simulation/

## License

Code and lecture material (notebooks, text and figures) are released under the MIT License, see [LICENSE](LICENSE).
