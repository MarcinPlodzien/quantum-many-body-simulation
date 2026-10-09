goal: Compute quantum Fisher information and simulate metrology protocols from preparation to estimation
---
Quantum metrology asks how precisely a parameter imprinted on a quantum state can be estimated. The chapter follows one
path through the subject.

**Foundations.** Notebook 29 builds estimation theory from a classical meter up to the quantum Fisher information, the
quantum Cramér–Rao bound, and the two limits it sets: the standard quantum limit for independent particles and the
Heisenberg limit for entangled ones. Notebook 30 evaluates the quantum Fisher information of mixed states through the
symmetric logarithmic derivative and runs the protocol prepare–encode–measure–estimate end to end.

**Protocols.** Ramsey interferometry at the standard quantum limit (31), GHZ interferometry at the Heisenberg limit (32),
spin squeezing by one-axis twisting (33), and the same twisting continued to a GHZ-like cat, with its readouts and its
fragility under noise (34).

**Practical limits.** What a scrambling unitary does to a probe (35), how much quantum Fisher information survives
particle loss and how encoded probes protect it (37), and how fast scramblers hide the phase information from subsystems
(38).

**Measuring the quantum Fisher information.** Notebook 36 estimates it for a prepared state from randomised
measurements (classical shadows) and counts the measurements an entanglement certificate needs.

**Many-body dynamics.** Notebook 39 follows how metrologically useful entanglement grows and spreads in quenched spin
chains.

**Variational optimisation.** Notebook 44b in [Chapter 11](ch11_variational_quantum_circuits.qmd) optimises probe
states and measurements under noise with automatic differentiation.

Read notebooks 29 to 34 in order; they form the main thread, and every later notebook uses notebooks 29 and 30.
Notebooks 35 to 39 are closer to current research and can be read selectively: notebook 35 builds on the probes of
notebooks 33 and 34, and notebook 38 builds on notebook 35; notebooks 36, 37 and 39 rest on notebooks 29 and 30 and
take the states of notebooks 33 to 35 only as examples. Prerequisites are notebooks 01 and 02 of
[Chapter 1](ch01_computational_toolbox.qmd); the state tensors, reduced density matrices, channels and measurements of
notebooks 05 to 08 in [Chapter 3](ch03_matrix_free_engine.qmd); the gates and Haar-random unitaries of notebooks 09
and 10 in [Chapter 4](ch04_digital_quantum_circuits.qmd); TEBD and quench dynamics, notebooks 12 and 15 in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd), for notebooks 38 and 39; GHZ states and classical shadows,
notebooks 22 and 24 in [Chapter 8](ch08_quantum_information_protocols.qmd), for notebooks 32 and 36; and the
stabilizer Rényi entropy of notebook 27 in [Chapter 9](ch09_entanglement_and_complexity.qmd), for notebook 37.
