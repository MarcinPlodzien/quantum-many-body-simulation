goal: Simulate a single quantum particle on a grid and write fast, batched, differentiable array code
---
The course opens with complete simulations of one particle in one dimension, where every number can be checked
against an exact solution. A free Gaussian wave packet on a ring is propagated with finite differences and with the
Fourier method; the harmonic oscillator is discretised, diagonalised, and propagated by spectral decomposition, by its
matrix exponential and with the Runge–Kutta and Crank–Nicolson integrators. In both, the numerical error is measured
against the analytic result and explained. Two further notebooks carry the same machinery into a
nonlinear problem, the bright solitons of the Gross–Pitaevskii equation: their ground state by imaginary-time
relaxation, their real-time motion with a split-step integrator, and their collisions.

A third single-particle problem removes the continuum altogether. Notebook 49 follows one excitation hopping on a
chain of sites, whose Hamiltonian is the three-point Laplacian of notebook 00a read as an exact model rather than as
an approximation. Its cosine band, its light cone and its closed-form propagator, a Bessel function, are solved on a
ring and on an open chain, and the same matrix returns in the spin-chain and tensor-network chapters as a reference
solution. It needs only notebook 00a and can be read directly after it. The two soliton notebooks are not needed
later in the course and can be left for a second reading.

The chapter closes with the two tools used everywhere afterwards. JAX turns NumPy-style code into compiled
(`jit`), batched (`vmap`), looped (`lax.scan`) and differentiable (`grad`) programs, and `einsum` expresses every
contraction of the course in index notation, from matrix products to partial traces.
