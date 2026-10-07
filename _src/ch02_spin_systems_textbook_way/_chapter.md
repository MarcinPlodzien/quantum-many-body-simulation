goal: Build spin-chain Hamiltonians as dense matrices, diagonalise them and evolve states in time
---
A chain of $N$ spins-1/2 is put on the computer the way textbooks do it. Operators acting on one spin are lifted to
the full $2^N$-dimensional space with Kronecker products, model Hamiltonians (transverse-field Ising, Heisenberg)
are assembled as matrices, and their ground states and spectra follow from exact diagonalisation.

The second notebook solves the time-dependent Schrödinger equation with the same dense matrices: the exact
propagator, the explicit Euler and Runge–Kutta integrators and their stability, and Trotterization derived from
scratch with its error measured. Both notebooks end at the same place, the memory and time cost of a $2^N\times 2^N$
matrix, which is the reason for the matrix-free approach of Chapter 3.
