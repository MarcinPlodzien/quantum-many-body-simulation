goal: Build spin-chain Hamiltonians as dense matrices, diagonalise them and evolve states in time
---
A chain of $N$ spins-1/2 is put on the computer the way textbooks do it. Operators acting on one spin are lifted to
the full $2^N$-dimensional space with Kronecker products, model Hamiltonians (transverse-field Ising, Heisenberg)
are assembled as matrices, and their ground states and spectra follow from exact diagonalisation, which already shows
the finite-size precursor of the quantum phase transition of the Ising chain.

The second notebook solves the time-dependent Schrödinger equation with the same dense matrices: the exact
propagator, the explicit Euler and Runge–Kutta integrators and their stability, and Trotterization with its error
measured, applied to a quench of the Ising chain. Both notebooks end at the same place, the memory and time cost of a $2^N\times 2^N$
matrix, which is the reason for the matrix-free approach of [Chapter 3](ch03_matrix_free_engine.qmd). Prerequisites are
notebooks 01 (JAX) and 02 (einsum and `reshape`) of [Chapter 1](ch01_computational_toolbox.qmd).
