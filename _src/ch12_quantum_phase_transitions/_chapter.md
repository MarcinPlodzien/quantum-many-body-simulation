goal: Locate a quantum phase transition numerically from finite chains and finite-size scaling
---
A quantum phase transition occurs in the ground state as a parameter of the Hamiltonian is varied. The notebook uses
the transverse-field Ising chain, which has an exact solution, as a benchmark and locates its transition with four
independent signatures: the order parameter, the energy gap, the entanglement entropy with its central charge, and
the fidelity susceptibility. Finite-size scaling and data collapse turn results on finite chains into statements
about the thermodynamic limit. DMRG carries the same analysis to 128 spins, and the XXZ chain adds a second
universality class and a Kosterlitz-Thouless transition that finite-size scaling cannot pin down.

Prerequisites are the restarted Lanczos solver and symmetry sectors of notebook 11 in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd), whose Section 11 takes a first look at the same transition,
and the Jordan-Wigner mapping to free fermions quoted in notebook 15 of the same chapter; entanglement entropy from
Schmidt values, notebook 06 in [Chapter 3](ch03_matrix_free_engine.qmd); matrix product states and DMRG, notebook 18
in [Chapter 7](ch07_tensor_networks.qmd); and `jax.jit` from notebook 01 in
[Chapter 1](ch01_computational_toolbox.qmd). One remark relates the fidelity susceptibility to the quantum Fisher
information of notebook 29 in [Chapter 10](ch10_quantum_metrology_protocols.qmd).
