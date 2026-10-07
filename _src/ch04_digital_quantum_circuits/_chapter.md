goal: Write quantum circuits as lists of einsum contractions and sample random unitaries and circuits
---
Quantum gates are small unitaries on one or two qubits, which makes a quantum circuit a special case of the engine
of Chapter 3. The first notebook introduces rotations as exponentials of Pauli matrices, the standard gate set,
circuits as data that can be compiled with `jit`, circuit identities, and a Trotter step written as a circuit.

The second notebook turns to randomness: Haar-random states and unitaries, the Porter–Thomas distribution of output
probabilities, random brick-wall circuits and the growth of their entanglement, the single-qubit Clifford group,
and cross-entropy benchmarking of random circuit sampling.
