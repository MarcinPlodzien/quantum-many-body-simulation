goal: Use driven many-body dynamics as a trainable machine for time-series prediction
---
In quantum reservoir computing the many-body dynamics itself is left untrained: an input sequence is written into
a spin system, the Hamiltonian evolution acts as a nonlinear feature map with fading memory, and only a linear
readout is fitted by ridge regression. The notebook measures the memory capacity of the reservoir, applies it to
standard benchmark tasks against classical baselines tuned on the same validation data, and compares density-tensor
and trajectory simulations under finite measurement statistics and decoherence. At equal feature count the
four-qubit reservoir has a larger error than the tuned classical echo-state network on every task.
