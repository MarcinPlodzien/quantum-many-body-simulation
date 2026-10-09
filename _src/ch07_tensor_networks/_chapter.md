goal: Represent one-dimensional states as matrix product states and run DMRG and TEBD on them
---
Matrix product states store a weakly entangled state of a long chain with memory that grows linearly in its
length. The notebook constructs them from repeated singular value decompositions, explains canonical forms and
truncation, and builds two-site DMRG from its elements: the Hamiltonian as a matrix product operator, the
environments, the effective Hamiltonian applied matrix-free, Lanczos as the local solver, and the sweeps.

Time evolution follows with a compiled TEBD two-site update at fixed bond dimension, validated against
state-vector evolution and then applied to quenches of chains far longer than a state vector can hold.

The chapter is one long notebook. Sections 2–5 (why entanglement decides compressibility, the construction, canonical
forms, truncation, measurements) are needed for everything else; Section 6 (DMRG) and Sections 7–10 (TEBD, quenches of
40 and 60 spins and the entanglement barrier) are independent of each other and can be read in either order; Sections
11–12 give the cost and the limits of the method. Prerequisites are notebooks 01 and 02 (`jit`, `vmap`, `lax.scan`,
einsum and tensor-leg diagrams) of [Chapter 1](ch01_computational_toolbox.qmd); the Schmidt decomposition, the
entanglement entropy and the area law of notebook 06 in [Chapter 3](ch03_matrix_free_engine.qmd); and, in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd), the Lanczos algorithm of notebook 11 (for DMRG), the
Trotter–Suzuki gates of notebook 12 (for TEBD) and, helpful for the physics of the quenches, notebook 15. DMRG returns
for the critical Ising chain in [Chapter 13](ch13_quantum_phase_transitions.qmd).
