goal: Use driven many-body dynamics as a trainable machine for time-series prediction
---
In quantum reservoir computing the many-body dynamics itself is left untrained: an input sequence is written into
a spin system, the Hamiltonian evolution acts as a nonlinear feature map with fading memory, and only a linear
readout is fitted by ridge regression. The notebook measures the memory capacity of the reservoir, applies it to
standard benchmark tasks against classical baselines tuned on the same validation data, and compares density-tensor
and trajectory simulations under finite measurement statistics and decoherence. At equal feature count the
four-qubit reservoir has a larger error than the tuned classical echo-state network on every task.

The notebook first introduces classical reservoir computing (the echo-state network, the ridge-regression readout,
memory capacity and the benchmark tasks), so no machine-learning background is needed. Prerequisites are notebook 01
(`jit`, `vmap`, `lax.scan`, random keys) of [Chapter 1](ch01_computational_toolbox.qmd); density tensors, partial
traces, Kraus channels and measurement with reset, notebooks 07 and 08 in [Chapter 3](ch03_matrix_free_engine.qmd);
the Trotter–Suzuki gates of notebook 12 in [Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd); and the Lindblad
equation and quantum trajectories of notebooks 16 and 17 in [Chapter 6](ch06_open_quantum_systems.qmd), used in
Sections 4.5 and 10.
