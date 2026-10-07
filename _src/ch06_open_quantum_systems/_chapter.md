goal: Simulate open quantum systems with the Lindblad equation and with quantum trajectories
---
A quantum system coupled to an environment is described by a density matrix that obeys the
Gorini–Kossakowski–Sudarshan–Lindblad master equation. The first notebook introduces jump operators, applies the
Lindbladian matrix-free to the rank-$2N$ density tensor, compares Runge–Kutta and Trotter–Kraus time stepping, and
checks the results against analytic single-qubit solutions before moving to a dissipative spin chain.

The second notebook unravels the same dynamics into pure-state quantum trajectories with random quantum jumps,
proves that their average obeys the Lindblad equation, and uses `vmap` over trajectories to reach system sizes for
which the density tensor no longer fits in memory.
