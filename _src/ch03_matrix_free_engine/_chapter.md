goal: Act on a many-body state with einsum, and compute observables, entanglement, channels and measurements
---
This chapter builds the simulator used by the rest of the course. A state of $N$ spins is stored as a rank-$N$
tensor with one index of dimension two per spin, and an operator on one or two spins acts on it through a single
`einsum` contraction, so the $2^N\times 2^N$ matrix is never formed. The same idea gives the Hamiltonian applied to
a state, and with it the first matrix-free time evolution of a chain of twenty spins.

The following notebooks add what is needed to extract physics from such a state: expectation values, correlators
and Pauli strings; reduced density matrices, entanglement entropies and a first phase diagram (the XXZ chain in a
transverse field); density tensors with unitary and Kraus evolution for mixed states; and projective measurements
with sampled bit strings and shot noise. Read them in the listed order. The last notebook, "Building the quantum
simulator engine", is the reference for every engine function: its Sections 2–12 accompany this chapter,
Sections 13–18 summarise later chapters, and the function table of its Section 19.2 serves as an index.

Prerequisites are notebooks 01 (JAX) and 02 (einsum) of [Chapter 1](ch01_computational_toolbox.qmd) and the dense
Kronecker-product construction of [Chapter 2](ch02_spin_systems_textbook_way.qmd). The engine built here runs the
circuits of [Chapter 4](ch04_digital_quantum_circuits.qmd) and all later chapters.
