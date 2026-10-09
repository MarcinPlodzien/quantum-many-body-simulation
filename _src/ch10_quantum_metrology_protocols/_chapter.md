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

**Variational optimisation.** Notebook 44b in Chapter 11 optimises probe states and measurements under noise with
automatic differentiation.
