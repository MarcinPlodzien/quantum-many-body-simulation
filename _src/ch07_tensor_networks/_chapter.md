goal: Represent one-dimensional states as matrix product states and run DMRG and TEBD on them
---
Matrix product states store a weakly entangled state of a long chain with memory that grows linearly in its
length. The notebook constructs them from repeated singular value decompositions, explains canonical forms and
truncation, and builds two-site DMRG element by element: the Hamiltonian as a matrix product operator, the
environments, the effective Hamiltonian applied matrix-free, Lanczos as the local solver, and the sweeps.

Time evolution follows with a compiled TEBD two-site update at fixed bond dimension, validated against
state-vector evolution and then applied to quenches of chains far longer than a state vector can hold.
