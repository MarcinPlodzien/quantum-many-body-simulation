goal: Act on a many-body state with einsum, and compute observables, entanglement, channels and measurements
---
This chapter builds the simulator used by the rest of the course. A state of $N$ spins is stored as a rank-$N$
tensor with one index of dimension two per spin, and an operator on one or two spins acts on it through a single
`einsum` contraction, so the $2^N\times 2^N$ matrix is never formed. The same idea gives the Hamiltonian applied to
a state, and with it the first matrix-free time evolution of a chain of twenty spins.

The following notebooks add what is needed to extract physics from such a state: expectation values, correlators
and Pauli strings; reduced density matrices and entanglement entropies; density tensors with unitary and Kraus
evolution for mixed states; and projective measurements with sampled bit strings and shot noise.
