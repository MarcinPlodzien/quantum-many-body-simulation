goal: Write quantum circuits as lists of einsum contractions and sample random unitaries and circuits
---
Quantum gates are small unitaries on one or two qubits, which makes a quantum circuit a special case of the engine
of Chapter 3. The first notebook introduces rotations as exponentials of Pauli matrices, the standard gate set,
circuits as data that can be compiled with `jit`, circuit identities, universality and the cost of approximating a
rotation with the finite Clifford+$T$ gate set, a Trotter step written as a circuit, and circuits with noise after
every gate on the density tensor.

The second notebook turns to randomness: Haar-random states and unitaries, the Porter–Thomas distribution of output
probabilities, random brick-wall circuits and the growth of their entanglement, the single-qubit Clifford group as a
unitary 3-design, and cross-entropy benchmarking of random circuit sampling. Read the two notebooks in the listed
order; the second uses the gates, the circuit lists and the Clifford gates of the first.

Prerequisites are notebook 01 (JAX) of [Chapter 1](ch01_computational_toolbox.qmd), the Trotterization of notebook 04
in [Chapter 2](ch02_spin_systems_textbook_way.qmd), and [Chapter 3](ch03_matrix_free_engine.qmd): `apply_gate` of
notebook 05 above all, with entanglement entropy (06), Kraus channels and trajectories (07) and sampled bit strings
(08). The Trotter circuits continue as TEBD in [Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd), and the
parametrised circuits as variational algorithms in [Chapter 11](ch11_variational_quantum_circuits.qmd).
