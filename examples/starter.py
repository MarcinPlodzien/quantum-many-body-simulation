# The simulator in five snippets. Run from the repository root:  python examples/starter.py
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax, jax.numpy as jnp
from quantum_engine import *        # the simulator built in the notes

key = jax.random.PRNGKey(0)

# 1. Quantum circuit: a three-qubit GHZ state, sampled
psi = zero_state(3)                 # rank-3 tensor of shape (2, 2, 2)
psi = apply_gate(psi, H, (0,))
psi = apply_gate(psi, CNOT, (0, 1))
psi = apply_gate(psi, CNOT, (1, 2))
bits = sample_bitstrings(key, psi, shots=6)
print("GHZ samples:", bits.tolist())     # only 000 and 111

# 2. Many-body physics: Heisenberg chain of 16 spins, Lanczos
N = 16
H_xxx = heisenberg_terms(N)         # local terms, no 2^N x 2^N matrix
E0, gs = lanczos_ground_state(H_xxx, N)
S_half = entanglement_entropy(gs, list(range(N // 2)))
print(f"E0 = {E0:.6f},  half-chain entropy = {S_half:.4f} bits")

# 3. Unitary dynamics: a Neel state melts (2nd-order TEBD)
neel = product_state("01" * (N // 2))
psi_t, z0 = tebd_evolve(neel, H_xxx, dt=0.05, n_steps=40,
                        observe=lambda p: expect_local(p, Z, (0,)))
print("<Z_0>(t) every 10 steps:", [round(float(z), 3) for z in z0[::10]])

# 4. Dissipative dynamics: driven, decaying qubit (Lindblad)
rho = to_dm(zero_state(1))
drive = [((0,), 0.5 * X)]
decay = [((0,), SM, 0.2)]           # jump operator sigma^-, rate 0.2
for _ in range(200):
    rho = lindblad_rk4_step(rho, drive, decay, dt=0.05)
print(f"excited population at t = 10: {dm_matrix(rho)[1, 1].real:.4f}")

# 5. Variational circuit: energy gradient by autodiff
n, layers = 4, 2
H4 = heisenberg_terms(n)
cost = lambda th: energy(H4, hardware_efficient_ansatz(th, n, layers))
theta = 0.1 * jax.random.normal(key, (hea_num_params(n, layers),))
g = jax.grad(cost)(theta)
print(f"energy = {cost(theta):.4f}, gradient has {g.size} components")
