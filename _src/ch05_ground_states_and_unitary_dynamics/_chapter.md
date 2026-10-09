goal: Find ground states with Lanczos and propagate states with Trotter, Chebyshev and Krylov methods
---
Each method of this chapter has its own notebook. Spin Hamiltonians on arbitrary graphs are written as lists of
local terms, and the Lanczos algorithm, derived from the power method, delivers their ground states without
building a matrix; symmetry sectors give the gap, and the Ising and XXZ chains, a $4\times4$ lattice and long-range
couplings serve as physics examples. Three propagators follow: Trotter–Suzuki splitting with two-site gates (orders
one, two and four), the Chebyshev expansion of the propagator, and Krylov-subspace exponentiation with an
a-posteriori error bound. Each is derived, implemented with `jit` and `lax.scan`, and validated against exact
evolution.

A head-to-head benchmark compares the integrators on accuracy and cost, uses the Loschmidt echo as a sensitive test,
and ends with a table for choosing an integrator. The final notebook uses them for the physics of quantum quenches in
spin chains: relaxation of the magnetisation, light cones, domain-wall melting, entanglement growth, integrable
versus non-integrable chains and the Loschmidt echo with its dynamical phase transitions.

Read the notebooks in the listed order. Notebook 14 builds on the Lanczos iteration of notebook 11 and compares
against notebooks 12 and 13; notebook 15 uses mainly the Trotter–Suzuki evolution of notebook 12 and needs the other
two integrators only as cross-checks, so a reader interested in the physics can go from 12 to 15 directly.

Prerequisites are notebook 01 (JAX: `jit`, `vmap`, `lax.scan`) of [Chapter 1](ch01_computational_toolbox.qmd), the
models, dense diagonalisation and textbook time evolution of notebooks 03 and 04 in
[Chapter 2](ch02_spin_systems_textbook_way.qmd), and [Chapter 3](ch03_matrix_free_engine.qmd): `apply_gate` and
`apply_hamiltonian` of notebook 05 and the entanglement entropy and Page value of notebook 06. The Trotter circuit of notebook 09 in
[Chapter 4](ch04_digital_quantum_circuits.qmd) is the gate sequence of notebook 12. Open systems follow in
[Chapter 6](ch06_open_quantum_systems.qmd), and the same Trotter gates act on matrix product states in
[Chapter 7](ch07_tensor_networks.qmd).
