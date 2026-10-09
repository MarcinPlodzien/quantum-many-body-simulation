goal: Simulate open quantum systems with the Lindblad equation and with quantum trajectories
---
A quantum system coupled to an environment is described by a density matrix that obeys the
Gorini–Kossakowski–Sudarshan–Lindblad master equation. The first notebook introduces jump operators, applies the
Lindbladian matrix-free to the rank-$2N$ density tensor, compares Runge–Kutta and Trotter–Kraus time stepping, and
checks the results against analytic single-qubit solutions before moving to a dissipative spin chain. With
$\sigma^\pm=(X\pm iY)/2$ and $|0\rangle$ the $+1$ eigenstate of $Z$, decay of the excited level $|1\rangle$ to $|0\rangle$
has the jump operator $\sigma^+=|0\rangle\langle1|$; quantum-optics texts call the same operator $\sigma^-$.

The second notebook unravels the same dynamics into pure-state quantum trajectories with random quantum jumps,
proves that their average obeys the Lindblad equation, and uses `vmap` over trajectories to reach system sizes for
which the density tensor no longer fits in memory, up to dephasing-induced diffusion in a chain of 16 spins.

Read notebook 16 before notebook 17. The density tensor ($4^N$ numbers) gives averages without statistical error and is the method of
choice up to about ten spins; trajectories ($2^N$ numbers each, with a statistical error $\propto1/\sqrt M$ for $M$
trajectories) take over beyond, and they are the only route above about 13 spins, where the density tensor and its Runge–Kutta
copies no longer fit into the memory of a laptop. Prerequisites are notebook 01 (`jit`, `vmap`, `lax.scan`, random keys) of
[Chapter 1](ch01_computational_toolbox.qmd); `apply_gate`, partial traces, density matrices, Kraus channels and their
stochastic unravelling, notebooks 05 to 07 in [Chapter 3](ch03_matrix_free_engine.qmd); and the Trotter–Suzuki gates
of notebook 12 and, for the last section of notebook 17, the domain-wall quench of notebook 15 in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd). Long chains with little entanglement
follow in [Chapter 7](ch07_tensor_networks.qmd), with matrix product states.
