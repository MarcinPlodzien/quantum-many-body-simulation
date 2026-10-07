goal: Train parametrised quantum circuits: gradients, optimisers, VQE, measurement cost and noise
---
A variational quantum algorithm adjusts the angles of a parametrised circuit to minimise a cost function. The
chapter derives four ways to obtain gradients (finite differences, the parameter-shift rule, SPSA and automatic
differentiation), compares optimisers from plain gradient descent to the quantum natural gradient, and applies
them in the variational quantum eigensolver to ground states of spin chains, checked against Lanczos.

The remaining notebooks address what a real device adds: the number of measurements a cost evaluation needs,
including classical shadows inside the optimisation loop; gate noise and its effect on landscapes, training and
error mitigation; and the quantum autoencoder as a variational compression task.
