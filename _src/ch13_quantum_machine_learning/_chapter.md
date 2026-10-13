goal: Train quantum circuits and quantum dynamics as learning machines: gradients, optimisers, VQE, noise, kernels, reservoir computing, data re-uploading and tensor-network models
---
Quantum machine learning uses a quantum system as a trainable model. Data enter through the preparation of a state or
through the drive of a Hamiltonian, the output is read from expectation values or measurement statistics, and a
classical optimiser adjusts the free parameters. The models differ in where the trainable parameters sit. In a
variational quantum circuit they are the angles of the gates; in a quantum reservoir the quantum dynamics stays fixed
and only a classical linear readout is trained; in a data re-uploading classifier the input itself is written into the
circuit again and again, between trainable layers. Because the simulator is differentiable, all three are trained with
the same code that evolves a spin chain.

**Variational quantum circuits.** A variational quantum algorithm adjusts the angles of a parametrised circuit to
minimise a cost function. Notebooks 40 to 42 derive four ways to obtain gradients (finite differences, the
parameter-shift rule, SPSA and automatic differentiation), compare optimisers from plain gradient descent to the
quantum natural gradient, and apply them in the variational quantum eigensolver to ground states of spin chains,
checked against exact diagonalisation.

The remaining variational notebooks address what a real device adds: the number of measurements a cost evaluation
needs, including classical shadows inside the optimisation loop; gate noise and its effect on landscapes, training and
error mitigation. The same machinery then optimises probe states for quantum metrology, compresses quantum data in
the quantum autoencoder and uses its compression cost to locate phase boundaries without labels, and classifies data
with quantum kernels. The quantum approximate optimisation algorithm turns it to classical problems: MaxCut and number
partitioning become diagonal Ising Hamiltonians, and alternating cost and mixer layers are trained to sample their
optimal bit strings.

Read notebooks 40 and 41 first; every other variational notebook uses their gradients and optimisers. The eigensolver
follows in the order 42a, 42, 43: notebook 42a introduces the Hamiltonians, the ansätze and the training, notebook 42
the error relations, symmetries, excited states and shot noise, and notebook 43 the measurement cost. Notebook 44 takes
the eigensolver as given, so read it after the eigensolver notebooks, although it is listed with the foundations. The
applications can then be read in any order: notebook 44b builds on notebook 44 and needs notebooks 29 to 34 and 37 of
[Chapter 10](ch10_quantum_metrology_protocols.qmd); notebook 45 uses the noise models of notebook 44 in its Section
12; notebook 45b builds on notebook 45, and its Sections 3.1 and 6.4 use the fidelity susceptibility and the
Kosterlitz–Thouless point of notebook 47 in [Chapter 11](ch11_quantum_phase_transitions.qmd); notebook 45c needs
notebooks 40 to 42; notebook 45d needs none of the earlier notebooks of the chapter. Prerequisites are notebook 01
(JAX: `jit`, `vmap`, `grad`, `lax.scan`) of [Chapter 1](ch01_computational_toolbox.qmd); the state tensors, density
tensors, channels and measurements of notebooks 05 to 08 in [Chapter 3](ch03_matrix_free_engine.qmd); the gates and
random circuits of notebooks 09 and 10 in [Chapter 4](ch04_digital_quantum_circuits.qmd); the Hamiltonians and Lanczos
ground states of notebook 11, and the Trotter splitting of notebook 12 for notebook 45c, in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd); and the classical shadows of notebook 24 in
[Chapter 8](ch08_quantum_information_protocols.qmd) for notebook 43.

**Quantum reservoir computing.** In quantum reservoir computing (notebook 46) the many-body dynamics itself is left
untrained: an input sequence is written into a spin system, the Hamiltonian evolution acts as a nonlinear feature map
with fading memory, and only a linear readout is fitted by ridge regression. The notebook measures the memory capacity
of the reservoir, applies it to standard benchmark tasks against classical baselines tuned on the same validation
data, and compares density-tensor and trajectory simulations under finite measurement statistics and decoherence. At
equal feature count the four-qubit reservoir has a larger error than the tuned classical echo-state network on every
task.

Notebook 46 first introduces classical reservoir computing (the echo-state network, the ridge-regression readout,
memory capacity and the benchmark tasks), so no machine-learning background is needed, and it does not depend on the
variational notebooks. Its prerequisites are notebook 01 (`jit`, `vmap`, `lax.scan`, random keys) of
[Chapter 1](ch01_computational_toolbox.qmd); density tensors, partial traces, Kraus channels and measurement with
reset, notebooks 07 and 08 in [Chapter 3](ch03_matrix_free_engine.qmd); the Trotter–Suzuki gates of notebook 12 in
[Chapter 5](ch05_ground_states_and_unitary_dynamics.qmd); and the Lindblad equation and quantum trajectories of
notebooks 16 and 17 in [Chapter 6](ch06_open_quantum_systems.qmd), used in Sections 4.5 and 10.

**Data re-uploading classifier.** A classifier built from a single qubit becomes expressive when the classical input
is encoded into the circuit repeatedly, each encoding followed by a trainable rotation. Notebook 55
develops this model on one qubit and on a few qubits, with and without entanglement, and shows that $L$ encodings make
the output a Fourier series of degree $L$ in the data. It needs the Adam optimiser and the training loop of notebooks
40 and 41, notebook 01 (`jit`, `vmap`, `grad`, `lax.scan`) of [Chapter 1](ch01_computational_toolbox.qmd), the Bloch
vectors, fidelities and reduced density matrices of notebook 06 in [Chapter 3](ch03_matrix_free_engine.qmd), and the
rotations and the CZ gate of notebook 09 in [Chapter 4](ch04_digital_quantum_circuits.qmd).

**Physics-inspired machine-learning models.** The tools of many-body physics also build classical models. Notebook 56
maps every image to a product state of one qubit per pixel and stores the weight tensor of a linear classifier on these
$2^{196}$-dimensional states as a matrix product state with a label index. It trains the model on handwritten digits
by DMRG-like sweeps and by automatic differentiation of the whole contraction, and compares the entanglement of the
trained classifier with that of the superposition of its training images. It needs the matrix product states, the
singular value decomposition and the Schmidt values, the canonical forms and truncation, and the two-site DMRG sweeps of
notebook 18 in [Chapter 7](ch07_tensor_networks.qmd), gradient descent and the Adam optimiser of notebook 41, the tensor-leg diagrams of notebook 02 and notebook 01 (`jit`, `vmap`, `grad`, `lax.scan`) of
[Chapter 1](ch01_computational_toolbox.qmd).

Notebook 57 turns the same matrix product state into a generative model. The probability of a binary image is the
squared amplitude of the state, the Born rule, so the normalisation follows from the canonical form and new images are
drawn exactly, pixel by pixel, without a Markov chain. The notebook trains the model by two-site sweeps on the negative
log-likelihood, learns the bars-and-stripes data set exactly and relates the bond dimensions it needs to the Schmidt
ranks of the data, and on handwritten digits generates samples, completes images with missing pixels and measures the
gap between training and test likelihood. It builds on notebook 56 and needs the same prerequisites, together with the sampling of bit strings from a state
vector of notebook 08 in [Chapter 3](ch03_matrix_free_engine.qmd).
