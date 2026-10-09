goal: Train parametrised quantum circuits: gradients, optimisers, VQE, measurement cost and noise
---
A variational quantum algorithm adjusts the angles of a parametrised circuit to minimise a cost function. The
chapter derives four ways to obtain gradients (finite differences, the parameter-shift rule, SPSA and automatic
differentiation), compares optimisers from plain gradient descent to the quantum natural gradient, and applies
them in the variational quantum eigensolver to ground states of spin chains, checked against exact diagonalisation.

The remaining notebooks address what a real device adds: the number of measurements a cost evaluation needs,
including classical shadows inside the optimisation loop; gate noise and its effect on landscapes, training and
error mitigation. The same machinery then optimises probe states for quantum metrology, compresses quantum data in
the quantum autoencoder and uses its compression cost to locate phase boundaries without labels, and classifies data
with quantum kernels. The quantum approximate optimisation algorithm turns it to classical problems: MaxCut and number
partitioning become diagonal Ising Hamiltonians, and alternating cost and mixer layers are trained to sample their
optimal bit strings.

Read notebooks 40 and 41 first; every other notebook uses their gradients and optimisers. The eigensolver follows in
the order 42a, 42, 43: notebook 42a introduces the Hamiltonians, the ansätze and the training, notebook 42 the error
relations, symmetries, excited states and shot noise, and notebook 43 the measurement cost. Notebook 44 takes the
eigensolver as given, so read it after the eigensolver notebooks, although it is listed with the foundations. The
applications can then be read in any order: notebook 44b builds on notebook 44 and needs notebooks 29 to 34 and 37 of
[Chapter 10](ch10_quantum_metrology_protocols.qmd); notebook 45 uses the noise models of notebook 44 in its Section
12; notebook 45b builds on notebook 45, and its Sections 3.1 and 6.4 use the fidelity susceptibility and the
Kosterlitz–Thouless point of notebook 47 in [Chapter 13](ch13_quantum_phase_transitions.qmd); notebook 45c needs
notebooks 40 to 42; notebook 45d needs none of the chapter's earlier notebooks. Prerequisites are notebook 01 (JAX:
`jit`, `vmap`, `grad`, `lax.scan`) of [Chapter 1](ch01_computational_toolbox.qmd); the state tensors, density tensors,
channels and measurements of notebooks 05 to 08 in [Chapter 3](ch03_matrix_free_engine.qmd); the gates and random
circuits of notebooks 09 and 10 in [Chapter 4](ch04_digital_quantum_circuits.qmd); the Hamiltonians and Lanczos ground
states of notebook 11, and the Trotter splitting of notebook 12 for notebook 45c, in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd); and the classical shadows of notebook 24 in
[Chapter 8](ch08_quantum_information_protocols.qmd) for notebook 43.
