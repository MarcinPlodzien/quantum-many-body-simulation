goal: Quantify mixed-state entanglement, multipartite correlations, magic and circuit complexity
---
This chapter collects diagnostics that characterise a quantum state beyond its energy and local observables.
Entanglement negativity, computed from the partial transpose of the density tensor, detects entanglement in mixed
states. Many-body Bell correlators extend the CHSH test to $N$ parties: a single number certifies correlations that
no local hidden-variable model reproduces and bounds how many parties share the entanglement, up to all $N$.

The stabilizer Rényi entropy measures non-stabilizerness ("magic"), the resource that separates classically
simulable Clifford circuits from universal ones, and the final notebook combines entanglement, magic and spectral
statistics to discuss how random circuits generate states that are hard to simulate.

Read notebook 25 first; it introduces the partial transpose and the PPT criterion. Notebook 48 covers the same criterion more briefly and adds the matrix-product-state route, for which it needs
notebook 18 of [Chapter 7](ch07_tensor_networks.qmd). Notebook 26 can follow notebook 25 directly. Notebook 27 does
not depend on notebooks 25 and 26, and notebook 28 builds on notebook 27. Prerequisites are notebooks 01 and 02 of
[Chapter 1](ch01_computational_toolbox.qmd); the reduced density matrices, density tensors and channels of notebooks
06 and 07 in [Chapter 3](ch03_matrix_free_engine.qmd); the gates, the Clifford group and the random circuits of
notebooks 09 and 10 in [Chapter 4](ch04_digital_quantum_circuits.qmd), for notebooks 27 and 28; Lanczos, TEBD and the
light cone after a quench, notebooks 11, 12 and 15 in [Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd); the
quantum trajectories of notebook 17 in [Chapter 6](ch06_open_quantum_systems.qmd), for the noisy Bell correlator; and
CHSH, Werner states and GHZ decoherence from notebooks 19, 21 and 22 in
[Chapter 8](ch08_quantum_information_protocols.qmd). Notebook 26 also uses material from later chapters: Adam and SPSA
from notebook 41 of [Chapter 11](ch11_variational_quantum_circuits.qmd) in its Section 7, and one-axis twisting from
notebooks 33 and 34 of [Chapter 10](ch10_quantum_metrology_protocols.qmd) in its Section 11. Magic returns in
notebook 37 of Chapter 10, as a property of the probe states used in metrology.
