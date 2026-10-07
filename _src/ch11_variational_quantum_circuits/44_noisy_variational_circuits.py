#@title: Noisy variational circuits — landscapes, plateaus, training and error mitigation
#@part: Chapter 11 — Variational quantum circuits
#@description: Gate noise as a channel after every gate, the dictionary between measured error rates and channel parameters, two exact-in-expectation simulators (density tensor and quantum trajectories) shown to agree, the contraction of the cost landscape derived and measured, the exact invariance of the minimum under global depolarising noise and its shift under amplitude damping, noise-induced barren plateaus, training with density-tensor gradients against SPSA on trajectory costs, optimal-parameter resilience, the depth trade-off and zero-noise extrapolation.

# %% [markdown]
# ## 1. Introduction and motivation
#
# Every variational quantum algorithm written so far in this chapter assumed a perfect device. The circuit
# $U(\boldsymbol\theta)$ was a product of exact unitaries, the state stayed pure, and the cost
# $C(\boldsymbol\theta)=\langle\psi(\boldsymbol\theta)\vert\hat H\vert\psi(\boldsymbol\theta)\rangle$ was the energy of
# that pure state. Present-day hardware is not like that. A two-qubit gate on a superconducting processor fails a few
# times in a thousand; qubits dephase while they wait; excited states decay. The circuit that runs is not
# $U(\boldsymbol\theta)$ but a **quantum channel** $\mathcal E_{\boldsymbol\theta}$, and the state it produces is mixed.
#
# This changes the optimisation problem itself, not merely the accuracy of its answer. Three questions must be answered
# before a variational algorithm can be trusted on a noisy device.
#
# 1. **What does noise do to the landscape?** If it only rescales the cost, the minimum is where it always was and the
#    algorithm still finds the right state. If it deforms the cost, the algorithm optimises the wrong function.
# 2. **What does noise do to the gradients?** A landscape that flattens exponentially with circuit depth cannot be
#    trained at all, whatever the optimiser.
# 3. **Can the damage be undone afterwards?** Error *correction* needs many more qubits than exist today; error
#    *mitigation* trades extra circuit runs for a better estimate of the noiseless expectation value, and works now.
#
# **Road map.**
#
# * **Section 3** fixes the noise model — a channel after every gate — and derives the dictionary between the numbers a
#   laboratory reports (average gate fidelity, $T_1$, $T_2$) and the channel parameters our simulator takes.
# * **Section 4** builds the two simulators. The **density tensor** propagates $\rho$ exactly at cost $O(4^N)$; the
#   **quantum trajectory** unravelling keeps a pure state at cost $O(2^N)$ per sample and converges as $1/\sqrt M$. They
#   are checked against each other on the energy of a noisy variational state.
# * **Section 5** derives how a Pauli expectation value contracts under local depolarising noise — a factor
#   $(1-4p/3)$ per noisy qubit per layer — and measures the law where its assumptions hold exactly.
# * **Section 6** proves that a **global** depolarising channel commutes with every unitary, so that
#   $C_{\rm noisy}(\boldsymbol\theta)=(1-q)\,C(\boldsymbol\theta)+q\,\mathrm{Tr}\,\hat H/2^N$ and the minimiser is
#   **unchanged**. Section 7 shows by an exactly solvable one-qubit example that **non-unital** noise (amplitude damping)
#   does move the minimiser, and gives the shifted angle in closed form.
# * **Section 8** measures **noise-induced barren plateaus**: the variance of a gradient component against circuit depth,
#   with and without noise.
# * **Sections 9 to 11** train under noise — exact gradients through the Kraus einsums against SPSA on trajectory
#   estimates — test **optimal-parameter resilience** (train noisy, evaluate noiselessly) and measure the **depth
#   trade-off** between expressivity and accumulated noise.
# * **Section 12** implements **zero-noise extrapolation** and shows both a regime where it recovers four digits and one
#   where it fails.
# * **Section 13** measures the cost of both simulators against $N$ and locates the crossover.
#
# ### What you will learn
#
# *Physics*
# * how gate errors, dephasing and energy relaxation are represented as quantum channels, and how the channel parameters
#   are read off from $T_1$, $T_2$ and a randomised-benchmarking error rate;
# * why a global depolarising channel leaves the *position* of the variational minimum exactly where it was, while
#   amplitude damping shifts it;
# * why the useful circuit depth has an optimum that moves to smaller depth as the error rate grows;
# * what error mitigation can and cannot do.
#
# *Numerical methods*
# * two representations of the same open-system dynamics, one deterministic and $O(4^N)$, one stochastic and $O(2^N)$,
#   and how to validate either against the other with honest error bars;
# * differentiating a channel: reverse-mode automatic differentiation straight through a chain of Kraus einsums;
# * estimating a decay exponent from data and comparing it with a derived prediction;
# * Richardson extrapolation of a noisy expectation value in the noise strength.
#
# *Implementation practice*
# * `lax.scan` over circuit layers so that the compiled graph does not grow with depth;
# * `vmap` over noise strengths (the channel parameter is a traced number) and over random initialisations at the same
#   time, so a whole two-dimensional sweep is one compiled program;
# * keeping the *monitored* quantity exact while the *optimised* quantity is noisy.
#
# ### Prerequisites
#
# * [07 — density matrices and quantum channels](../ch03_matrix_free_engine/07_density_matrices_and_quantum_channels.ipynb):
#   the density tensor, the Kraus form, the channel zoo, the stochastic unravelling;
# * [09 — quantum gates and circuits](../ch04_digital_quantum_circuits/09_quantum_gates_and_circuits.ipynb):
#   circuits as gate lists, `noisy_circuit_dm` and `noisy_circuit_mcwf`;
# * [40 — parametrized gates and gradients](../ch11_variational_quantum_circuits/40_parametrized_gates_and_gradients.ipynb):
#   ansätze, cost functions, the parameter-shift rule, SPSA, barren plateaus in the noiseless case;
# * [41 — optimisers](../ch11_variational_quantum_circuits/41_optimizers.ipynb): Adam, SPSA with Spall gains, the
#   `lax.scan` training loop and `vmap` over random starts;
# * [42 — the variational quantum eigensolver](../ch11_variational_quantum_circuits/42_variational_quantum_eigensolver.ipynb):
#   the algorithm itself, which is taken as given here.
#
# **Conventions and sizes.** The test problem throughout is the ground state of an XXZ chain in a transverse field on
# $N=4$ qubits, with the hardware-efficient ansatz of notebook 40. Four qubits keep the density tensor at $4^4=256$
# complex numbers, so exact gradients through the channel are cheap and every experiment can be repeated over many
# random initialisations. The trajectory simulator is pushed to $N=12$ in the cost study of Section 13.

# %% [markdown]
# ## 2. Engine recap and notebook helpers
#
# From the engine we need the two ways of acting on a density tensor (`apply_gate_dm`, `apply_kraus_dm`), the stochastic
# unravelling (`apply_kraus_mcwf`), the three channels of the noise model, the ansatz and its parameter count, the
# Hamiltonian term list and the exact ground state from Lanczos.

# %%
#@engine: apply_gate, apply_gate_dm, apply_kraus_dm, apply_kraus_mcwf, kraus_depolarizing, kraus_dephasing, kraus_amplitude_damping, I2, X, Y, Z, H, CZ, P0, P1, ry, rz, zero_state, product_state, to_dm, dm_matrix, rdm, rdm_dm, expect_local, expect_local_dm, expect_pauli_string, expect_pauli_string_dm, fidelity_pure, haar_state, haar_unitary, heisenberg_terms, apply_hamiltonian, energy, dense_hamiltonian, lanczos_ground_state, hea_num_params, hardware_efficient_ansatz

# %%
# ==============================================================================
# PLOT STYLE + helpers reused from notebook 41 (training loop, statistics)
# ==============================================================================
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # colour-blind-friendly, fixed order
MARKERS = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=PALETTE), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "legend.frameon": False})


def max_abs(a):
    """Largest absolute entry of an array, as a Python float."""
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def bands(hist):
    """Median and interquartile band of a batch of histories, shape (n_runs, n_steps) -> three curves."""
    h = np.asarray(hist)
    return np.percentile(h, 25, axis=0), np.median(h, axis=0), np.percentile(h, 75, axis=0)


def mean_and_se(x):
    """Sample mean and standard error of the mean of a 1D array of independent samples."""
    x = np.asarray(x)
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size))


def opt_adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    """Adam as a pair (init, update) of pure functions -- the optimiser interface of notebook 41."""
    def init(theta):
        return (jnp.zeros_like(theta), jnp.zeros_like(theta))

    def update(theta, state, g, k):
        m, v = state
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        return theta - lr * (m / (1 - b1 ** k)) / (jnp.sqrt(v / (1 - b2 ** k)) + eps), (m, v)

    return init, update


def train(theta0, key, grad_rule, opt, monitor, n_steps):
    """One training run compiled as a single `lax.scan` (notebook 41, Step 8).

    grad_rule(theta, key, k) -> gradient estimate;  opt = (init, update);  monitor(theta) -> recorded diagnostic.
    """
    init_fn, update_fn = opt

    def body(carry, k):
        theta, state, key = carry
        key, sub = jax.random.split(key)
        theta, state = update_fn(theta, state, grad_rule(theta, sub, k), k)
        return (theta, state, key), monitor(theta)

    (theta, _, _), hist = lax.scan(body, (theta0, init_fn(theta0), key), jnp.arange(1, n_steps + 1))
    return theta, hist


def random_starts(n_runs, n_params, seed, scale=jnp.pi):
    """`n_runs` independent parameter vectors drawn uniformly from [-scale, scale]^n, plus one PRNG key each."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return (jax.random.uniform(k1, (n_runs, n_params), minval=-scale, maxval=scale),
            jax.random.split(k2, n_runs))


def timed(fn, *args, repeat=3):
    """Best of `repeat` wall-clock timings AFTER one warm-up call, so compilation is excluded."""
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts)

# %% [markdown]
# ## 3. The noise model and its parameters
#
# ### 3.1 A channel after every gate
#
# The standard phenomenological model of a gate-based device is: **after every gate, every qubit the gate touched passes
# through a single-qubit channel**. The channel is the vehicle for two physically distinct effects.
#
# * **Gate errors** — miscalibrated pulses, crosstalk, leakage — are modelled by a **depolarising** channel, because a
#   randomised gate error looks isotropic on the Bloch sphere once averaged over the randomising circuits used to
#   measure it. Two-qubit gates are an order of magnitude worse than single-qubit gates, so the model carries two error
#   probabilities, $p_1$ and $p_2>p_1$.
# * **Idling** — the qubit waiting while other gates are executed — is modelled by **dephasing** (loss of the relative
#   phase, time constant $T_2$) and **amplitude damping** (decay $\vert1\rangle\to\vert0\rangle$, time constant $T_1$).
#
# The three channels, with the Kraus operators the engine uses, are
#
# $$\begin{aligned}
# \mathcal D_p(\rho)&=(1-p)\,\rho+\frac p3\bigl(X\rho X+Y\rho Y+Z\rho Z\bigr), &&\text{depolarising},\\
# \mathcal Z_p(\rho)&=(1-p)\,\rho+p\,Z\rho Z, &&\text{dephasing},\\
# \mathcal A_\gamma(\rho)&=K_0\rho K_0^\dagger+K_1\rho K_1^\dagger,\quad
#  K_0=\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix},\ K_1=\begin{pmatrix}0&\sqrt\gamma\\0&0\end{pmatrix},
#  &&\text{amplitude damping}.
# \end{aligned}\tag{1}$$
#
# ### 3.2 Depolarising strength from an average gate fidelity
#
# A laboratory does not publish $p$; it publishes an *error per gate* $r$ measured by randomised benchmarking, which is
# $1-F_{\rm avg}$ with $F_{\rm avg}$ the gate fidelity averaged over input states. The dictionary follows from one
# algebraic identity. For a single qubit,
#
# $$\rho+X\rho X+Y\rho Y+Z\rho Z=2\,\mathrm{Tr}(\rho)\,\mathbb 1=2\cdot\mathbb 1,\tag{2}$$
#
# which is checked by writing $\rho=\tfrac12(\mathbb 1+\mathbf r\cdot\boldsymbol\sigma)$ and using
# $\sigma_a\sigma_b\sigma_a=-\sigma_b$ for $a\neq b$ and $\sigma_a\sigma_a\sigma_a=\sigma_a$: the three conjugations
# flip the sign of two of the three Bloch components each, so
# $\sum_a\sigma_a\rho\,\sigma_a=\tfrac12(3\cdot\mathbb 1-\mathbf r\cdot\boldsymbol\sigma)$, and adding $\rho$ gives
# $2\cdot\mathbb 1$. Substituting Eq. (2) into the first line of Eq. (1),
#
# $$\mathcal D_p(\rho)=\Bigl(1-\frac{4p}{3}\Bigr)\rho+\frac{4p}{3}\,\frac{\mathbb 1}{2}.\tag{3}$$
#
# So the depolarising channel shrinks the Bloch vector by $\lambda\equiv1-4p/3$ and mixes in a fraction $4p/3$ of the
# maximally mixed state. The fidelity of the output with the (pure) input is
# $\langle\psi\vert\mathcal D_p(\vert\psi\rangle\langle\psi\vert)\vert\psi\rangle=\lambda+\tfrac{4p}{3}\cdot\tfrac12
# =1-\tfrac{2p}{3}$, independently of $\vert\psi\rangle$. Hence
#
# $$F_{\rm avg}=1-\frac{2p}{3},\qquad r=1-F_{\rm avg}=\frac{2p}{3},\qquad \boxed{\;p=\tfrac32\,r\;}.\tag{4}$$
#
# A two-qubit gate with an error per gate $r_2=5\cdot10^{-3}$ — a good number on today's hardware — corresponds to
# $p_2=7.5\cdot10^{-3}$ in our parametrisation *if* the whole error is assigned to one qubit; our model applies
# $\mathcal D_{p_2}$ to **both** qubits of the gate, so $p_2=\tfrac34 r_2$ per qubit reproduces the same total error to
# first order. We keep the simpler convention $p_2=\tfrac32r_2$ per qubit and remember that the model is then slightly
# pessimistic.
#
# ### 3.3 Dephasing and damping from $T_1$ and $T_2$
#
# Amplitude damping removes population from $\vert1\rangle$: from Eq. (1),
# $\rho_{11}\to(1-\gamma)\rho_{11}$. Comparing with the exponential law $\rho_{11}(t)=e^{-t/T_1}\rho_{11}(0)$ for an
# idling time $t$,
#
# $$\gamma=1-e^{-t/T_1}.\tag{5}$$
#
# Dephasing multiplies the coherence by $1-2p$, because $Z\rho Z$ flips the sign of $\rho_{01}$:
# $\rho_{01}\to(1-p)\rho_{01}-p\rho_{01}$. Comparing with $\rho_{01}(t)=e^{-t/T_\varphi}\rho_{01}(0)$,
#
# $$p=\tfrac12\bigl(1-e^{-t/T_\varphi}\bigr).\tag{6}$$
#
# (The measured $T_2$ combines pure dephasing with the dephasing that damping itself produces,
# $1/T_2=1/T_\varphi+1/(2T_1)$; Eq. (6) uses the pure-dephasing part.) Both formulas are checked numerically below by
# applying the channel $n$ times and comparing with the exponential.

# %%
# ==============================================================================
# STEP 1: the noise-parameter dictionary, verified numerically
# ==============================================================================
# CHECK A: the depolarising channel contracts the Bloch vector by 1 - 4p/3 (Eq. 3)
p_chk = 0.07
rho_1q = 0.5 * (I2 + 0.6 * X + 0.4 * Y - 0.5 * Z)                     # an arbitrary mixed qubit
rho_out = apply_kraus_dm(rho_1q, kraus_depolarizing(p_chk), [0])
bloch_in = jnp.array([jnp.real(jnp.trace(rho_1q @ P)) for P in (X, Y, Z)])
bloch_out = jnp.array([jnp.real(jnp.trace(rho_out @ P)) for P in (X, Y, Z)])
lam_chk = 1 - 4 * p_chk / 3
print(f"depolarising p = {p_chk}")
print(f"  Bloch vector in  : {np.asarray(bloch_in)}")
print(f"  Bloch vector out : {np.asarray(bloch_out)}")
print(f"  measured shrink factor {float(bloch_out[0] / bloch_in[0]):.12f}  vs 1-4p/3 = {lam_chk:.12f}")
assert max_abs(bloch_out - lam_chk * bloch_in) < TOL

# CHECK B: the average gate fidelity of Eq. (4). The six +-X, +-Y, +-Z eigenstates form a 2-design, so
# averaging <psi|D_p(|psi><psi|)|psi> over them gives EXACTLY the Haar average.
six = [product_state(c) for c in ("0", "1", "+", "-", "r", "l")]
F_six = [float(jnp.real(jnp.vdot(psi, apply_kraus_dm(to_dm(psi), kraus_depolarizing(p_chk), [0]) @ psi)))
         for psi in six]                       # for N=1 the density TENSOR is already the 2x2 matrix
F_avg = float(np.mean(F_six))
print(f"  average gate fidelity over the 6-state 2-design: {F_avg:.12f}   vs 1-2p/3 = {1 - 2 * p_chk / 3:.12f}")
print(f"  error per gate r = {1 - F_avg:.6f}  ->  p = 3r/2 = {1.5 * (1 - F_avg):.6f}   (input was {p_chk})")
assert abs(F_avg - (1 - 2 * p_chk / 3)) < TOL

# CHECK C: T1 and T2 laws, Eqs. (5) and (6)
T1, T2phi, dt, n_steps_T = 20.0, 12.0, 0.5, 60
gam_step = 1 - np.exp(-dt / T1)
p_step = 0.5 * (1 - np.exp(-dt / T2phi))
rho_T = to_dm(product_state("+"))                                     # |+> has both population and coherence
pop, coh = [], []
for n in range(n_steps_T + 1):
    m = dm_matrix(rho_T)
    pop.append(float(jnp.real(m[1, 1])))
    coh.append(float(jnp.abs(m[0, 1])))
    rho_T = apply_kraus_dm(rho_T, kraus_amplitude_damping(gam_step), [0])
    rho_T = apply_kraus_dm(rho_T, kraus_dephasing(p_step), [0])
t_axis = dt * np.arange(n_steps_T + 1)
pop_pred = 0.5 * np.exp(-t_axis / T1)
print(f"\nT1 = {T1}, T_phi = {T2phi}, step dt = {dt}  ->  gamma = {gam_step:.6f}, p_dephase = {p_step:.6f}")
print(f"  population  : max |measured - 0.5 exp(-t/T1)| = {np.max(np.abs(np.array(pop) - pop_pred)):.2e}")

fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.semilogy(t_axis, pop, "o", ms=3.5, color=PALETTE[0], label=r"population $\rho_{11}$ (simulated)")
ax.semilogy(t_axis, pop_pred, "-", lw=1.4, color=PALETTE[0], label=r"$\frac{1}{2} e^{-t/T_1}$")
ax.semilogy(t_axis, coh, "s", ms=3.5, color=PALETTE[1], label=r"coherence $\vert\rho_{01}\vert$ (simulated)")
ax.set_xlabel("idle time $t$ (arbitrary units)")
ax.set_ylabel("population / coherence")
ax.set_title("Repeated channels reproduce the exponential decay laws")
ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The three identities of Section 3 hold to machine precision. The Bloch vector shrinks by exactly $1-4p/3$; the average
# fidelity over the six-state 2-design equals $1-2p/3$ to twelve digits, so a reported error per gate $r$ is converted
# to a channel parameter by $p=\tfrac32 r$ with no ambiguity; and repeating the two idle channels $n$ times reproduces
# $e^{-t/T_1}$ and the pure-dephasing decay of the coherence with a maximum deviation at the level of round-off.
#
# The figure shows the coherence falling faster than the population, which is the generic situation: it is damaged both
# by pure dephasing and by the damping itself, since a decay event destroys the phase relation along with the
# population.
#
# > **Physics insight.** The depolarising channel is not a claim that gate errors really are isotropic. It is the
# > *average* of an arbitrary error channel over the random single-qubit Cliffords used in randomised benchmarking — the
# > same twirling operation that makes the benchmarking curve a single exponential. Modelling gate errors as
# > depolarising is exactly as accurate as the number $r$ that the twirl produces, and no more.

# %% [markdown]
# ## 4. Two simulators for the same noisy circuit
#
# ### 4.1 The noisy hardware-efficient ansatz
#
# The circuit is the ansatz of notebook 40: $L$ repetitions of [a rotation block $R_y(\theta)R_z(\theta)$ on every
# qubit, then a chain of $CZ$ gates], followed by one final rotation block, so the parameter count is $n=2N(L+1)$. The
# noisy version inserts a channel after every gate: $\mathcal D_{p_1}$ on each qubit after its rotation pair, and
# $\mathcal D_{p_2}$ on **both** qubits of every $CZ$.
#
# Counting the noise locations will matter in Sections 5 and 8. Per layer, each qubit passes through
#
# * one channel of strength $p_1$ (after its rotations), and
# * one channel of strength $p_2$ per $CZ$ that touches it — two for a bulk qubit of an open chain, one at each end.
#
# For $N=4$ the chain has three bonds, so qubits $0$ and $3$ collect one $CZ$ channel per layer and qubits $1$ and $2$
# collect two.
#
# ### 4.2 Representation 1: the density tensor
#
# $\rho$ is a rank-$2N$ array (notebook 07): $N$ ket axes and $N$ bra axes. A unitary acts as $U$ on the ket axis and
# $U^*$ on the bra axis (`apply_gate_dm`); a channel is the single einsum of `apply_kraus_dm`, which sums over the Kraus
# index internally. The result is **exact**: no sampling, no statistical error. The price is $4^N$ complex numbers and
# $O(2^k4^N)$ work per $k$-qubit operation.
#
# The implementation writes the $L$ repeated layers as a `lax.scan` over the rows of $\boldsymbol\theta$. Without it the
# traced graph would contain one copy of every gate in the circuit, and the compilation time of the *gradient* would
# grow linearly with the depth — which matters in Section 8, where depths up to $L=12$ are differentiated.
#
# ### 4.3 Representation 2: quantum trajectories
#
# The unravelling of notebook 07 keeps a **pure** state and replaces each channel by a random choice of one Kraus
# operator, $\vert\psi\rangle\to K_m\vert\psi\rangle/\lVert K_m\vert\psi\rangle\rVert$ with probability
# $\lVert K_m\vert\psi\rangle\rVert^2$ (`apply_kraus_mcwf`). The average of $\vert\psi\rangle\langle\psi\vert$ over
# trajectories is exactly $\rho$, so any expectation value is estimated without bias:
#
# $$\mathbb E\bigl[\langle\psi_{\rm traj}\vert\hat O\vert\psi_{\rm traj}\rangle\bigr]=\mathrm{Tr}(\hat O\rho),
#   \qquad \text{statistical error}\ \propto\frac1{\sqrt M}.\tag{7}$$
#
# Memory is $O(2^N)$ per trajectory and trajectories are independent, so `vmap` over PRNG keys runs them all in one
# compiled program.

# %%
# ==============================================================================
# STEP 2: the noisy ansatz on a density tensor and as a quantum trajectory
# ==============================================================================
# PARAMETERS of the test problem: XXZ chain in a transverse field
N_Q = 4                                     # qubits
JXX, JYY, JZZ, HX = -1.0, -1.0, -0.5, -0.3  # H = Jxx sum XX + Jyy sum YY + Jzz sum ZZ + hx sum X
P2_DEF = 0.02                               # depolarising probability after a two-qubit gate (per touched qubit)
P1_RATIO = 0.1                              # single-qubit gates are 10x better:  p1 = P1_RATIO * p2
L_DEF = 2                                   # layers of the default ansatz

TERMS = heisenberg_terms(N_Q, Jxx=JXX, Jyy=JYY, Jzz=JZZ, hx=HX)
E0_EXACT, PSI_GS = lanczos_ground_state(TERMS, N_Q)
TRACE_H = float(jnp.real(jnp.trace(dense_hamiltonian(TERMS, N_Q))))


def rot_block_dm(rho, row, kraus1, N):
    """One rotation block Ry(theta)Rz(theta) per qubit on a density tensor, each followed by its channel."""
    for q in range(N):
        rho = apply_gate_dm(rho, ry(row[q, 0]), [q])
        rho = apply_gate_dm(rho, rz(row[q, 1]), [q])
        rho = apply_kraus_dm(rho, kraus1, [q])
    return rho


def ent_block_dm(rho, kraus2, N):
    """One CZ chain on a density tensor, with a channel on BOTH qubits of every gate."""
    for q in range(N - 1):
        rho = apply_gate_dm(rho, CZ, [q, q + 1])
        rho = apply_kraus_dm(rho, kraus2, [q])
        rho = apply_kraus_dm(rho, kraus2, [q + 1])
    return rho


def noisy_hea_dm(theta, N, layers, p1, p2, chan1=kraus_depolarizing, chan2=kraus_depolarizing):
    """Noisy hardware-efficient ansatz, density-tensor representation.

    MATH  rho = (final rotations) o [ ENT o ROT ]^layers  applied to |0..0><0..0|, with a single-qubit channel
          after every gate: chan1(p1) after each rotation pair, chan2(p2) on both qubits of every CZ.
    COST  O(4^N) memory, O(2^k 4^N) per operation.
    JAX   the repeated layers are a `lax.scan`, so the traced graph (and the gradient graph) has a size that does
          NOT grow with `layers`; p1, p2 are traced, so `vmap` over noise strengths compiles once.
    """
    K1, K2 = chan1(p1), chan2(p2)
    th = theta.reshape(layers + 1, N, 2)
    rho = to_dm(zero_state(N))
    rho, _ = lax.scan(lambda r, row: (ent_block_dm(rot_block_dm(r, row, K1, N), K2, N), None), rho, th[:layers])
    return rot_block_dm(rho, th[layers], K1, N)


def rot_block_mcwf(key, psi, row, kraus1, N):
    """The same rotation block on a PURE state: after each gate, ONE Kraus operator is sampled."""
    for q in range(N):
        psi = apply_gate(psi, ry(row[q, 0]), [q])
        psi = apply_gate(psi, rz(row[q, 1]), [q])
        key, sub = jax.random.split(key)                    # fresh randomness at every noise location
        psi = apply_kraus_mcwf(sub, psi, kraus1, [q])
    return key, psi


def ent_block_mcwf(key, psi, kraus2, N):
    """The same CZ chain on a pure state, sampling a Kraus operator on both qubits of every gate."""
    for q in range(N - 1):
        psi = apply_gate(psi, CZ, [q, q + 1])
        for t in (q, q + 1):
            key, sub = jax.random.split(key)
            psi = apply_kraus_mcwf(sub, psi, kraus2, [t])
    return key, psi


def noisy_hea_mcwf(key, theta, N, layers, p1, p2, chan1=kraus_depolarizing, chan2=kraus_depolarizing):
    """ONE quantum trajectory of the same noisy ansatz.  Memory O(2^N); `vmap` over keys for many trajectories."""
    K1, K2 = chan1(p1), chan2(p2)
    th = theta.reshape(layers + 1, N, 2)

    def body(carry, row):
        k, psi = carry
        k, psi = rot_block_mcwf(k, psi, row, K1, N)
        k, psi = ent_block_mcwf(k, psi, K2, N)
        return (k, psi), None

    (key, psi), _ = lax.scan(body, (key, zero_state(N)), th[:layers])
    _, psi = rot_block_mcwf(key, psi, th[layers], K1, N)
    return psi


def energy_dm(terms, rho):
    """<H> = sum_k Tr(rho h_k) for a density tensor and a Hamiltonian given as a list of local terms."""
    return sum(expect_local_dm(rho, h, q) for q, h in terms)


print(f"XXZ chain in a transverse field, N = {N_Q}")
print(f"  H = {JXX} sum XX + {JYY} sum YY + {JZZ} sum ZZ + {HX} sum X")
print(f"  exact ground energy (Lanczos)  E0 = {E0_EXACT:.10f}")
print(f"  Tr H = {TRACE_H:.3e}   (every term is a traceless Pauli string)")
print(f"  ansatz: L = {L_DEF} layers, n = {hea_num_params(N_Q, L_DEF)} parameters; "
      f"noise p2 = {P2_DEF}, p1 = {P1_RATIO * P2_DEF}")

# %%
# ==============================================================================
# CHECKPOINT 1: at p = 0 the noisy simulators must reproduce the noiseless circuit exactly
# ==============================================================================
theta_test = jax.random.uniform(jax.random.PRNGKey(0), (hea_num_params(N_Q, 3),), minval=-jnp.pi, maxval=jnp.pi)
psi_pure = hardware_efficient_ansatz(theta_test, N_Q, 3)
err_dm = max_abs(noisy_hea_dm(theta_test, N_Q, 3, 0.0, 0.0) - to_dm(psi_pure))
psi_traj0 = noisy_hea_mcwf(jax.random.PRNGKey(1), theta_test, N_Q, 3, 0.0, 0.0)
err_mc = max_abs(psi_traj0 - psi_pure)
print("p = 0 against the engine's `hardware_efficient_ansatz`:")
print(f"  density tensor     max |rho - |psi><psi||  = {err_dm:.2e}")
print(f"  single trajectory  max |psi_traj - psi|    = {err_mc:.2e}")
assert err_dm < TOL and err_mc < TOL

# ==============================================================================
# CHECKPOINT 2: with noise, the trajectory average must reproduce the density tensor
# ==============================================================================
M_CHK = 4000
E_ref = float(jax.jit(lambda t: energy_dm(TERMS, noisy_hea_dm(t, N_Q, L_DEF, P1_RATIO * P2_DEF, P2_DEF)))(theta_test[:hea_num_params(N_Q, L_DEF)]))
th_chk = theta_test[:hea_num_params(N_Q, L_DEF)]
traj_energy = jax.jit(jax.vmap(lambda k: energy(TERMS, noisy_hea_mcwf(k, th_chk, N_Q, L_DEF,
                                                                     P1_RATIO * P2_DEF, P2_DEF))))
samples = np.asarray(jax.block_until_ready(traj_energy(jax.random.split(jax.random.PRNGKey(7), M_CHK))))
E_mc, se_mc = mean_and_se(samples)
print(f"\nenergy of the noisy state at a random theta (p2 = {P2_DEF}):")
print(f"  density tensor (exact)      E = {E_ref:.6f}")
print(f"  {M_CHK} trajectories          E = {E_mc:.6f} +- {se_mc:.6f}")
print(f"  deviation = {abs(E_mc - E_ref) / se_mc:.2f} standard errors")
assert abs(E_mc - E_ref) < 5 * se_mc

# convergence of the trajectory estimate with M
Ms = np.array([25, 50, 100, 200, 400, 800, 1600, 3200])
errs = np.array([abs(np.mean(samples[:m]) - E_ref) for m in Ms])
ses = np.array([np.std(samples[:m], ddof=1) / np.sqrt(m) for m in Ms])
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
axes[0].hist(samples, bins=40, color=PALETTE[0], alpha=0.75)
axes[0].axvline(E_ref, color="k", ls="--", lw=1.4, label="density tensor (exact)")
axes[0].axvline(E_mc, color=PALETTE[1], ls="-", lw=1.4, label=f"mean of {M_CHK} trajectories")
axes[0].set_xlabel(r"$\langle H\rangle$ of one trajectory"); axes[0].set_ylabel("count")
axes[0].set_title("Trajectories scatter; their mean does not"); axes[0].legend(fontsize=8)
axes[1].loglog(Ms, np.maximum(errs, 1e-6), MARKERS[0] + "-", ms=6, color=PALETTE[0],
               label="measured $\\vert E_M - E_{\\mathrm{exact}}\\vert$")
axes[1].loglog(Ms, ses, MARKERS[1] + "--", ms=5, color=PALETTE[1], label="standard error of the mean")
axes[1].loglog(Ms, ses[0] * np.sqrt(Ms[0] / Ms), "k:", lw=1.4, label=r"$\propto 1/\sqrt{M}$")
axes[1].set_xlabel("number of trajectories $M$"); axes[1].set_ylabel("error in the energy")
axes[1].set_title("Statistical convergence of the unravelling"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# Two simulators built from completely different primitives — one sums over Kraus indices inside an einsum on a
# rank-$2N$ tensor, the other samples a single Kraus operator per noise location on a rank-$N$ tensor — agree on the
# energy of the same noisy state.
#
# The left panel shows *why* the agreement has to be stated with an error bar. A single trajectory is a pure state with
# its own energy: most land in a narrow peak near the mean, but a tail of unlucky jump sequences reaches below $-1$,
# and the spread over trajectories is $0.30$ against a mean of $0.488$. There is no sense in which one
# trajectory "is" the noisy state. Only the mean is meaningful, and the right panel confirms that its deviation from the
# exact value tracks the standard error of the mean and falls as $1/\sqrt M$.
#
# > **Numerical practice.** Quoting a trajectory result without its standard error is not a measurement. In the
# > comparisons below, every trajectory number is reported as mean $\pm$ standard error, and "agreement" always means
# > "within a few standard errors", never "the digits look similar".

# %% [markdown]
# ## 5. What noise does to the cost landscape
#
# ### 5.1 A Pauli operator under a depolarising channel
#
# The cost is a sum of Pauli expectation values, so it is enough to know what the channel does to one Pauli string. Work
# in the Heisenberg picture: $\mathrm{Tr}\bigl(P\,\mathcal D_p(\rho)\bigr)=\mathrm{Tr}\bigl(\mathcal D_p^\dagger(P)\,\rho\bigr)$,
# where the adjoint channel is obtained by moving the Kraus operators to the other side of the trace. Since all Kraus
# operators here are (multiples of) Hermitian Paulis, $\mathcal D_p^\dagger=\mathcal D_p$ as a map on operators. Take
# $P=Z$ on the qubit the channel acts on:
#
# $$\mathcal D_p^\dagger(Z)=(1-p)Z+\frac p3\bigl(XZX+YZY+ZZZ\bigr)=(1-p)Z+\frac p3(-Z-Z+Z)
#   =\Bigl(1-\frac{4p}{3}\Bigr)Z=\lambda Z,\tag{8}$$
#
# using $XZX=-Z$, $YZY=-Z$, $ZZZ=Z$. The same algebra gives $\lambda X$ and $\lambda Y$ — every non-identity Pauli is an
# eigenoperator of the depolarising channel with eigenvalue $\lambda=1-4p/3$ — while $\mathcal D_p^\dagger(\mathbb 1)=\mathbb 1$.
#
# A Pauli **string** of weight $w$ (non-identity on $w$ qubits) therefore picks up one factor $\lambda$ per noisy qubit
# in its support. If every qubit passes through $c$ channels per layer and there are $D$ layers, and **if the string is
# not changed by the intervening unitaries**, then
#
# $$\langle P\rangle_{\rm noisy}=\lambda^{\,w\,c\,D}\,\langle P\rangle_{\rm ideal},\qquad \lambda=1-\frac{4p}{3}.\tag{9}$$
#
# Equation (9) says that the whole landscape is squeezed towards zero, exponentially in the depth, and *faster for
# observables of higher weight*. Since $\mathrm{Tr}(\hat H)=0$ for a Hamiltonian built from non-identity Pauli strings,
# zero is also the value of the cost on the maximally mixed state. **The landscape flattens towards the maximally mixed
# value.**
#
# The italicised assumption is the weak point and must be stated clearly: in a real circuit the unitaries between the
# noise layers rotate $P$ into a *sum* of Pauli strings of generally higher weight, and higher weight means faster
# contraction. Equation (9) is therefore a *lower bound* on the amount of contraction, and we expect the measured decay
# to be somewhat faster. Section 5.2 verifies Eq. (9) exactly in a setting where the assumption holds, and Section 8
# measures the excess in a real circuit.

# %%
# ==============================================================================
# STEP 3: the contraction law of Eq. (9), where its assumptions hold exactly
# ==============================================================================
# Setting: a fixed Haar-random state, then D rounds of depolarising noise on EVERY qubit and nothing else
# (c = 1 channel per qubit per round, no unitaries in between) -> Eq. (9) should be exact.
psi_rand = haar_state(jax.random.PRNGKey(3), N_Q)
rho_rand = to_dm(psi_rand)
P_TEST = (("ZIII", 1), ("XXII", 2), ("XYZI", 3), ("XYZX", 4))
p_c = 0.05
lam_c = 1 - 4 * p_c / 3
print(f"D rounds of depolarising noise (p = {p_c}, lambda = 1 - 4p/3 = {lam_c:.6f}) on all {N_Q} qubits")
print(f"{'string':>8s} {'w':>2s} {'D':>2s} {'<P> measured':>14s} {'lambda^(wD) <P>_0':>19s} {'error':>10s}")
for ops, w in P_TEST:
    ideal = float(expect_pauli_string(psi_rand, ops))
    for D in (1, 3, 6):
        rho_d = rho_rand
        for _ in range(D):
            for q in range(N_Q):
                rho_d = apply_kraus_dm(rho_d, kraus_depolarizing(p_c), [q])
        got = float(expect_pauli_string_dm(rho_d, ops))
        pred = lam_c ** (w * D) * ideal
        print(f"{ops:>8s} {w:2d} {D:2d} {got:14.8f} {pred:19.8f} {abs(got - pred):10.1e}")
        assert abs(got - pred) < 1e-9

# %% [markdown]
# Equation (9) is exact to round-off in every case, and the pattern of the numbers is the physics: at $p=0.05$ one round
# of noise multiplies a weight-1 observable by $0.933$ and a weight-4 observable by $0.933^4=0.759$; after six rounds
# the weight-1 expectation value retains $66\%$ of its ideal size and the weight-4 one retains $19\%$. **Collective,
# high-weight observables are destroyed first** — which is the same statement as the fragility of GHZ coherences under
# local noise.
#
# ### 5.2 A slice through the noisy cost landscape
#
# We now put a real circuit between the noise layers and look at the cost itself. Fix a random $\boldsymbol\theta$, vary
# one angle across its full period, and plot $C(\boldsymbol\theta)$ for several noise strengths. The noise probability
# is a *traced* argument of `noisy_hea_dm`, so one `vmap` over the noise axis and one over the angle axis compile into a
# single program.

# %%
# ==============================================================================
# STEP 4: one-angle slices of the noisy cost, and the amplitude of the landscape vs p
# ==============================================================================
P_SLICE = jnp.asarray([0.0, 0.01, 0.03, 0.10])
n_par_def = hea_num_params(N_Q, L_DEF)
theta_slice = jax.random.uniform(jax.random.PRNGKey(12), (n_par_def,), minval=-jnp.pi, maxval=jnp.pi)
angles = jnp.linspace(-jnp.pi, jnp.pi, 121)


@jax.jit
def slice_cost(p, phi):
    """Cost as a function of angle 0 and of the two-qubit error probability -- both traced."""
    th = theta_slice.at[0].set(phi)
    return energy_dm(TERMS, noisy_hea_dm(th, N_Q, L_DEF, P1_RATIO * p, p))


curves = np.asarray(jax.vmap(lambda p: jax.vmap(lambda a: slice_cost(p, a))(angles))(P_SLICE))

P_AMP = jnp.asarray(np.linspace(0.0, 0.25, 26))
amp = np.asarray(jax.jit(jax.vmap(lambda p: jnp.max(jax.vmap(lambda a: slice_cost(p, a))(angles))
                                  - jnp.min(jax.vmap(lambda a: slice_cost(p, a))(angles))))(P_AMP))

# Effective exponent: fit  ln(amp/amp_0) = n_eff * ln(lambda)  with lambda = 1 - 4 p / 3.
lam_axis = 1 - 4 * np.asarray(P_AMP) / 3
n_eff = float(np.polyfit(np.log(lam_axis[1:]), np.log(amp[1:] / amp[0]), 1)[0])
# Counted exponents for a single Pauli string of weight w, L layers (Section 4.1):
#   CZ channels per layer: weight-1 term -> mean 1.5 per qubit; weight-2 term on a bond -> mean 10/3
#   rotation-block channels: w per layer+1, at strength p1 = P1_RATIO * p2
n_w1 = 1.5 * L_DEF + P1_RATIO * 1 * (L_DEF + 1)
n_w2 = (10 / 3) * L_DEF + P1_RATIO * 2 * (L_DEF + 1)
print(f"amplitude of the one-angle slice: p2=0 -> {amp[0]:.4f};  "
      f"p2=0.05 -> {amp[int(np.argmin(np.abs(np.asarray(P_AMP) - 0.05)))]:.4f};  "
      f"p2=0.10 -> {amp[int(np.argmin(np.abs(np.asarray(P_AMP) - 0.10)))]:.4f};  "
      f"p2=0.25 -> {amp[-1]:.4f}")
print(f"fit amplitude(p) / amplitude(0) = lambda^n_eff  ->  n_eff = {n_eff:.2f}")
print(f"  counted exponent for a weight-1 Pauli term over L={L_DEF} layers: {n_w1:.2f}")
print(f"  counted exponent for a weight-2 Pauli term over L={L_DEF} layers: {n_w2:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
for j, p in enumerate(np.asarray(P_SLICE)):
    axes[0].plot(np.asarray(angles), curves[j], "-", lw=1.8, color=PALETTE[j], label=f"$p_2={p:g}$")
axes[0].axhline(TRACE_H / 2 ** N_Q, color="k", ls=":", lw=1.2,
                label=r"maximally mixed value $\mathrm{Tr}\,H/2^N=0$")
axes[0].set_xlabel(r"angle $\theta_0$"); axes[0].set_ylabel(r"$\langle H\rangle$")
axes[0].set_title("The landscape is squeezed towards the mixed-state value"); axes[0].legend(fontsize=8)

axes[1].semilogy(np.asarray(P_AMP), amp, MARKERS[0] + "-", ms=5, color=PALETTE[0],
                 label="measured amplitude of the slice")
axes[1].semilogy(np.asarray(P_AMP), amp[0] * lam_axis ** n_eff, "k-", lw=1.2,
                 label=rf"fit $\lambda^{{{n_eff:.2f}}}$")
axes[1].semilogy(np.asarray(P_AMP), amp[0] * lam_axis ** n_w1, "k:", lw=1.3,
                 label=rf"counted, weight 1: $\lambda^{{{n_w1:.2f}}}$")
axes[1].semilogy(np.asarray(P_AMP), amp[0] * lam_axis ** n_w2, "k--", lw=1.3,
                 label=rf"counted, weight 2: $\lambda^{{{n_w2:.2f}}}$")
axes[1].set_xlabel(r"two-qubit error probability $p_2$")
axes[1].set_ylabel(r"$\max_\theta C-\min_\theta C$ along the slice")
axes[1].set_title("Amplitude of the landscape against noise"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The left panel shows the shape of the slice surviving while its amplitude shrinks: the maxima and minima stay at
# almost the same angles, and the whole curve is pulled towards $\mathrm{Tr}\,\hat H/2^N=0$, the value the cost takes on
# the maximally mixed state. That is exactly the behaviour Eq. (9) predicts, and it is the first hint of the result
# proved in the next section.
#
# The right panel turns the observation into a number. The measured amplitude follows a clean power law in $\lambda$
# over the whole range, $\mathrm{amplitude}\propto\lambda^{4.79}$ — the fitted line and the data are
# indistinguishable from $p_2=0$ to $p_2=0.25$, so Eq. (9) has the right *functional form* even for a full circuit.
# The exponent, however, is not the one a single Pauli weight predicts. Counting noise locations for $L=2$ layers as in
# Section 4.1 gives $3.30$ for a weight-1 term ($X_q$ in the field part of $\hat H$) and $7.27$ for a weight-2 term
# ($X_iX_{i+1}$ and its partners), and the measured $4.79$ sits between them.
#
# That is the expected outcome and it says something useful. The amplitude of a *one-angle* slice is not the size of
# one Pauli expectation value: it is the size of the part of $C$ that depends on $\theta_0$, and that part is a mixture
# of contributions of different weights sitting at different distances from the qubit whose angle is being scanned.
# Equation (9) therefore predicts the family of curves; which member of the family applies has to be measured.
#
# > **Common pitfall.** "The landscape flattens" is often said as if it implied "the minimum moves". The left panel says
# > otherwise for this noise model. Whether the minimum moves is a separate question, and it has a sharp answer.

# %% [markdown]
# ## 6. Global depolarising noise leaves the minimum exactly where it was
#
# ### 6.1 The channel and its two properties
#
# The **global** depolarising channel on $N$ qubits replaces the state by the maximally mixed one with probability $q$:
#
# $$\mathcal G_q(\rho)=(1-q)\,\rho+q\,\frac{\mathbb 1}{2^N}.\tag{10}$$
#
# **Property 1 — it commutes with every unitary.** For any unitary $V$,
#
# $$V\,\mathcal G_q(\rho)\,V^\dagger=(1-q)V\rho V^\dagger+q\,\frac{V\mathbb 1V^\dagger}{2^N}
#   =(1-q)V\rho V^\dagger+q\,\frac{\mathbb 1}{2^N}=\mathcal G_q\bigl(V\rho V^\dagger\bigr),\tag{11}$$
#
# because $V\mathbb 1V^\dagger=\mathbb 1$. **Property 2 — global depolarising channels compose into one.** Applying
# $\mathcal G_{q_1}$ then $\mathcal G_{q_2}$ gives $\mathcal G_q$ with $1-q=(1-q_1)(1-q_2)$, since the identity is a
# fixed point of every such channel.
#
# Together these have a strong consequence. A circuit in which a global depolarising channel of strength $q_l$ follows
# layer $l$ is **identical** to the noiseless circuit followed by one global depolarising channel of total strength
# $1-q=\prod_l(1-q_l)$: commute every channel through all the later unitaries with Eq. (11), then merge them with
# Property 2.
#
# ### 6.2 The cost, and why its minimiser does not move
#
# The cost of the noisy state is therefore
#
# $$C_{\rm noisy}(\boldsymbol\theta)=\mathrm{Tr}\bigl(\hat H\,\mathcal G_q(\vert\psi(\boldsymbol\theta)\rangle\langle\psi(\boldsymbol\theta)\vert)\bigr)
#   =(1-q)\,C(\boldsymbol\theta)+q\,\frac{\mathrm{Tr}\,\hat H}{2^N}.\tag{12}$$
#
# Both $q$ and $\mathrm{Tr}\hat H/2^N$ are independent of $\boldsymbol\theta$, and $1-q>0$ for $q<1$. An affine function
# of $C$ with a positive slope is minimised exactly where $C$ is:
#
# $$\arg\min_{\boldsymbol\theta}C_{\rm noisy}=\arg\min_{\boldsymbol\theta}C,\qquad
#   \nabla C_{\rm noisy}=(1-q)\,\nabla C.\tag{13}$$
#
# The *location* of every stationary point, the ordering of every pair of parameter values, and the direction of every
# gradient are untouched. What noise destroys is the *scale*: the gradient shrinks by $1-q$, which for a deep circuit is
# exponentially small — and that, not a deformation, is the mechanism of the barren plateau in Section 8. For our
# Hamiltonian $\mathrm{Tr}\hat H=0$, so Eq. (12) is a pure rescaling.

# %%
# ==============================================================================
# STEP 5: the global depolarising channel -- commutation, composition, and Eqs. (12)-(13)
# ==============================================================================
def apply_global_depolarizing(rho, q):
    """Global depolarising channel, Eq. (10):  rho -> (1-q) rho + q * 1/2^N.

    IMPLEMENTATION  the maximally mixed state is the identity matrix reshaped to a rank-2N tensor, divided by 2^N.
    """
    N = rho.ndim // 2
    eye = jnp.eye(2 ** N, dtype=rho.dtype).reshape((2,) * (2 * N))
    return (1 - q) * rho + q * eye / 2 ** N


q_g = 0.3
rho_g = to_dm(hardware_efficient_ansatz(theta_slice, N_Q, L_DEF))
V4 = haar_unitary(jax.random.PRNGKey(5), 4)                                  # an arbitrary 2-qubit unitary
lhs = apply_gate_dm(apply_global_depolarizing(rho_g, q_g), V4, [1, 2])
rhs = apply_global_depolarizing(apply_gate_dm(rho_g, V4, [1, 2]), q_g)
print(f"Property 1 (Eq. 11): max |V G_q(rho) V^dag - G_q(V rho V^dag)| = {max_abs(lhs - rhs):.2e}")
assert max_abs(lhs - rhs) < TOL

q1, q2 = 0.2, 0.35
comp = apply_global_depolarizing(apply_global_depolarizing(rho_g, q1), q2)
merged = apply_global_depolarizing(rho_g, 1 - (1 - q1) * (1 - q2))
print(f"Property 2 (composition): max |G_q2 o G_q1 - G_q| = {max_abs(comp - merged):.2e}")
assert max_abs(comp - merged) < TOL

C_ideal = float(energy(TERMS, hardware_efficient_ansatz(theta_slice, N_Q, L_DEF)))
C_noisy = float(energy_dm(TERMS, apply_global_depolarizing(rho_g, q_g)))
print(f"\nEq. (12) at a random theta, q = {q_g}:")
print(f"  C_ideal                       = {C_ideal:+.12f}")
print(f"  C_noisy (measured)            = {C_noisy:+.12f}")
print(f"  (1-q) C_ideal + q Tr H / 2^N  = {(1 - q_g) * C_ideal + q_g * TRACE_H / 2 ** N_Q:+.12f}")
assert abs(C_noisy - ((1 - q_g) * C_ideal + q_g * TRACE_H / 2 ** N_Q)) < TOL

# the gradient is rescaled, not rotated (Eq. 13)
g_ideal = jax.grad(lambda t: energy(TERMS, hardware_efficient_ansatz(t, N_Q, L_DEF)))(theta_slice)
g_noisy = jax.grad(lambda t: energy_dm(TERMS, apply_global_depolarizing(
    to_dm(hardware_efficient_ansatz(t, N_Q, L_DEF)), q_g)))(theta_slice)
cos_angle = float(jnp.dot(g_ideal, g_noisy) / (jnp.linalg.norm(g_ideal) * jnp.linalg.norm(g_noisy)))
print(f"  |grad C_noisy| / |grad C_ideal| = {float(jnp.linalg.norm(g_noisy) / jnp.linalg.norm(g_ideal)):.12f}"
      f"   vs 1-q = {1 - q_g}")
print(f"  cosine of the angle between the two gradients = {cos_angle:.12f}")
assert abs(cos_angle - 1.0) < TOL

# %%
# ==============================================================================
# STEP 6: the minimum of a two-angle slice, with and without global depolarising noise
# ==============================================================================
grid2 = jnp.linspace(-jnp.pi, jnp.pi, 81)


@jax.jit
def surface(q):
    def one(a, b):
        th = theta_slice.at[0].set(a).at[1].set(b)
        return energy_dm(TERMS, apply_global_depolarizing(to_dm(hardware_efficient_ansatz(th, N_Q, L_DEF)), q))
    return jax.vmap(lambda a: jax.vmap(lambda b: one(a, b))(grid2))(grid2)


Q_SURF = 0.8
S0, Sq = np.asarray(surface(0.0)), np.asarray(surface(Q_SURF))

# The strong statement: the two surfaces are the SAME function, up to the affine map of Eq. (12).
affine_err = float(np.max(np.abs(Sq - ((1 - Q_SURF) * S0 + Q_SURF * TRACE_H / 2 ** N_Q))))
print(f"max |C_noisy(theta) - [(1-q) C_ideal(theta) + q Tr H / 2^N]| over the {grid2.size}x{grid2.size} grid: "
      f"{affine_err:.2e}")
assert affine_err < TOL

# Consequence: the sets of grid minima coincide.  (This landscape has TWO exactly degenerate global minima,
# so `argmin`, which breaks ties arbitrarily, is not the right thing to compare -- the minimising SET is.)
tol_min = 1e-9
set0 = set(map(tuple, np.argwhere(S0 <= S0.min() + tol_min)))
setq = set(map(tuple, np.argwhere(Sq <= Sq.min() + tol_min * (1 - Q_SURF))))
print(f"grid points within {tol_min:g} of the minimum:")
print(f"  q = 0.0 : {len(set0)} point(s) at " + ", ".join(
    f"({float(grid2[i]):+.4f}, {float(grid2[j]):+.4f})" for i, j in sorted(set0)) + f"   C = {S0.min():+.6f}")
print(f"  q = {Q_SURF} : {len(setq)} point(s) at " + ", ".join(
    f"({float(grid2[i]):+.4f}, {float(grid2[j]):+.4f})" for i, j in sorted(setq)) + f"   C = {Sq.min():+.6f}")
print(f"  the two minimising sets are identical: {set0 == setq}")
print(f"  ratio of the two minima {Sq.min() / S0.min():.12f}  vs 1-q = {1 - Q_SURF}")
assert set0 == setq

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
for ax, S, ttl in zip(axes, (S0, Sq), ("noiseless, $q=0$", f"global depolarising, $q={Q_SURF}$")):
    im = ax.contourf(np.asarray(grid2), np.asarray(grid2), S.T, levels=24, cmap="viridis")
    for i, j in sorted(set0):
        ax.plot([float(grid2[i])], [float(grid2[j])], "w*", ms=15)
    ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")
    ax.set_title(ttl + " (stars: the two degenerate minima)")
    fig.colorbar(im, ax=ax, label=r"$\langle H\rangle$")
fig.tight_layout(); plt.show()

# %% [markdown]
# Both properties hold to machine precision, Eq. (12) reproduces the noisy cost to twelve digits, and the gradient is
# rescaled by exactly $1-q$ with the cosine of the angle between the two gradients equal to $1$ — the noisy gradient
# points in precisely the same direction as the noiseless one.
#
# The contour plots make the consequence visible. At $q=0.8$ only a fifth of the original contrast survives, so the
# colour scale spans a fifth of the range; the *pattern* of the level sets is identical, and the two surfaces are
# verified to be the same function up to the affine map of Eq. (12), grid point by grid point, to $10^{-16}$.
#
# This landscape happens to have **two exactly degenerate global minima** — visible as the two dark spots, related by
# $(\theta_0,\theta_1)\to(\theta_0+\pi,\theta_1+\pi)$ — which is why the comparison is made between the minimising
# *sets* rather than between two calls to `argmin`: a tie is broken by round-off, and at $q=0.8$ the round-off falls
# differently. The sets coincide, and the two minimum values are in the exact ratio $1-q$.
#
# A variational algorithm running on a device whose only error is global depolarising noise converges to the same angles
# as on a perfect device — it just has to see through a landscape whose features are $(1-q)$ times as large, which
# requires $(1-q)^{-2}$ times as many measurement shots.
#
# > **Physics insight.** This is the theoretical basis of *optimal-parameter resilience* (Sharma, Khatri, Cerezo and
# > Coles, 2020): for several classes of noise the global optimum of the noisy cost coincides with the noiseless one.
# > Section 10 tests how far the idea survives when the noise is local rather than global, which is the realistic case.

# %% [markdown]
# ## 7. Non-unital noise does move the minimum
#
# Equation (11) used only one property of the depolarising channel: it leaves $\mathbb 1$ invariant. A channel with
# $\mathcal E(\mathbb 1)=\mathbb 1$ is called **unital**. Amplitude damping is not unital — it drives every state towards
# $\vert0\rangle$, so
# $\mathcal A_\gamma(\mathbb 1/2)=\tfrac12\mathrm{diag}(1+\gamma,1-\gamma)\neq\mathbb 1/2$ — and the argument fails. Here
# is the smallest example in which the consequence can be computed in closed form.
#
# ### 7.1 One qubit, solved exactly
#
# Take one qubit, the cost $C(\theta)=-\langle X\rangle-h\langle Z\rangle$ with a fixed $h>0$, and the one-parameter
# ansatz $\vert\psi(\theta)\rangle=R_y(\theta)\vert0\rangle$, whose Bloch vector is
# $\mathbf r=(\sin\theta,\,0,\,\cos\theta)$. Apply a channel after the gate and read off the Bloch map.
#
# * **Noiseless:** $C(\theta)=-\sin\theta-h\cos\theta$. Setting $C'(\theta)=-\cos\theta+h\sin\theta=0$ gives
#
#   $$\tan\theta_\star=\frac1h.\tag{14}$$
#
# * **Depolarising**, which multiplies every Bloch component by $\lambda=1-4p/3$:
#   $C_\lambda(\theta)=-\lambda(\sin\theta+h\cos\theta)=\lambda\,C(\theta)$. The minimiser is unchanged — this is
#   Eq. (13) again, since a single-qubit depolarising channel *is* the global one for $N=1$.
#
# * **Amplitude damping**, whose Bloch map is $x\to\sqrt{1-\gamma}\,x$, $y\to\sqrt{1-\gamma}\,y$,
#   $z\to(1-\gamma)z+\gamma$ (the last term is the non-unital piece):
#
#   $$C_\gamma(\theta)=-\sqrt{1-\gamma}\,\sin\theta-h\bigl[(1-\gamma)\cos\theta+\gamma\bigr].$$
#
#   Differentiating, $-\sqrt{1-\gamma}\cos\theta+h(1-\gamma)\sin\theta=0$, so
#
#   $$\tan\theta_\star(\gamma)=\frac{\sqrt{1-\gamma}}{h(1-\gamma)}=\frac{1}{h\sqrt{1-\gamma}}.\tag{15}$$
#
# The constant $-h\gamma$ is an offset and drops out of the derivative; what survives is that the transverse components
# are damped by $\sqrt{1-\gamma}$ while the longitudinal one is damped by $1-\gamma$. **The two damping factors are
# different, so their ratio changes with $\gamma$, and the optimal angle rotates towards the equator.** Equation (15)
# reduces to Eq. (14) at $\gamma=0$ and diverges as $\gamma\to1$: with complete damping the state is $\vert0\rangle$
# whatever $\theta$ is, and the optimum becomes degenerate.

# %%
# ==============================================================================
# STEP 7: the one-qubit example, minimised on a fine grid and compared with Eqs. (14)-(15)
# ==============================================================================
H_FIELD = 0.5


def cost_1q(theta, chan, par):
    """C(theta) = -<X> - h<Z> for Ry(theta)|0> followed by the channel `chan(par)` -- density tensor, one qubit."""
    rho = apply_gate_dm(to_dm(zero_state(1)), ry(theta), [0])
    rho = apply_kraus_dm(rho, chan(par), [0])
    return -(expect_local_dm(rho, X, [0]) + H_FIELD * expect_local_dm(rho, Z, [0]))


fine = jnp.linspace(0.0, jnp.pi, 20001)
cases = [("noiseless", kraus_depolarizing, 0.0, np.arctan(1 / H_FIELD)),
         ("depolarising $p=0.2$", kraus_depolarizing, 0.2, np.arctan(1 / H_FIELD)),
         ("amplitude damping $\\gamma=0.3$", kraus_amplitude_damping, 0.3,
          np.arctan(1 / (H_FIELD * np.sqrt(1 - 0.3)))),
         ("amplitude damping $\\gamma=0.6$", kraus_amplitude_damping, 0.6,
          np.arctan(1 / (H_FIELD * np.sqrt(1 - 0.6))))]

print(f"one qubit, C(theta) = -<X> - {H_FIELD}<Z>;  grid spacing {float(fine[1] - fine[0]):.2e}")
print(f"{'case':>32s} {'theta* measured':>16s} {'theta* predicted':>17s} {'difference':>12s} {'C(theta*)':>11s}")
fig, ax = plt.subplots(figsize=(7.0, 4.3))
for j, (name, chan, par, pred) in enumerate(cases):
    vals = np.asarray(jax.jit(jax.vmap(lambda t: cost_1q(t, chan, par)))(fine))
    tmin = float(fine[int(np.argmin(vals))])
    print(f"{name.replace('$', '').replace(chr(92) + 'gamma', 'gamma'):>32s} {tmin:16.6f} {pred:17.6f} "
          f"{tmin - pred:+12.2e} {vals.min():11.6f}")
    assert abs(tmin - pred) < 2e-3
    ax.plot(np.asarray(fine), vals, "-", lw=1.8, color=PALETTE[j], label=name)
    ax.plot([tmin], [vals.min()], MARKERS[j], ms=8, color=PALETTE[j])
ax.axvline(np.arctan(1 / H_FIELD), color="k", ls="--", lw=1.1, label=r"noiseless optimum $\arctan(1/h)$")
ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$C(\theta)$")
ax.set_title("Unital noise rescales the cost; non-unital noise also moves its minimum")
ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The measured minimisers reproduce Eqs. (14) and (15) to the resolution of the grid — the differences in the table are
# $5\cdot10^{-5}$, a third of the grid spacing $1.57\cdot10^{-4}$. The depolarising curve is a scaled copy of the
# noiseless one and its minimum sits at $1.107097$, on the dashed line at $\arctan(1/h)=1.107149$; the two
# amplitude-damping curves have their minima at $1.174641$ and $1.264491$, against the predicted $1.174589$ and
# $1.264519$ of Eq. (15).
#
# The shift is not small: at $\gamma=0.6$ the optimal angle has moved by $0.157$ radians, about $9$ degrees. An
# algorithm that trains under damping and then reports its angles as "the" answer is reporting an answer that is biased,
# and the bias grows with the damping.
#
# > **Physics insight.** The distinction that matters is *unital* against *non-unital*, not "weak" against "strong".
# > Depolarising, dephasing and bit flips are unital and preserve the identity, so they contract the landscape about the
# > mixed state without moving its stationary points. Amplitude damping has a preferred state, and a preferred state is
# > a bias.

# %% [markdown]
# ## 8. Noise-induced barren plateaus
#
# Notebook 40 measured the barren plateau of a *noiseless* circuit: for a hardware-efficient ansatz the variance of a
# gradient component falls exponentially with the number of **qubits**. Wang and co-workers (2021) showed that local
# noise produces a second, independent mechanism: at fixed $N$, the variance falls exponentially with the circuit
# **depth**. The reasoning is Eq. (9). Each noisy layer multiplies every Pauli expectation value by $\lambda^{wc}$, so
# after $L$ layers the whole cost function — and with it every derivative — is compressed by $\lambda^{wcL}$, and a
# variance, being quadratic in the cost, by $\lambda^{2wcL}$:
#
# $$\mathrm{Var}\bigl[\partial_kC\bigr]_{\rm noisy}\ \approx\ \lambda^{2wcL}\;\mathrm{Var}\bigl[\partial_kC\bigr]_{\rm noiseless},
#   \qquad \ln\frac{\mathrm{Var}_{\rm noisy}}{\mathrm{Var}_{\rm noiseless}}\approx 2\,w\,c\,L\,\ln\lambda.\tag{16}$$
#
# The counting of $c$ for our circuit was done in Section 4.1. A two-body term $Z_iZ_{i+1}$ on bond $(i,i+1)$ sees, per
# layer, the $CZ$ channels of both its qubits: $1+2=3$ for the end bonds and $2+2=4$ for the middle one, so $10/3$ on
# average over the three bonds of an $N=4$ chain, plus $2\times0.1$ from the two rotation-block channels at
# $p_1=0.1p_2$. Hence $wc=10/3+0.2=3.53$ for the dominant terms of $\hat H$, and Eq. (16) predicts a slope
# $2\times3.53\times\ln\lambda$ in a plot of $\ln(\mathrm{Var}_{\rm noisy}/\mathrm{Var}_{\rm noiseless})$ against $L$.
#
# The measurement takes $R$ uniformly random parameter vectors at each depth, differentiates the noisy cost with respect
# to all of them by reverse-mode automatic differentiation **through the Kraus einsums**, and records the sample
# variance of the first component. Because $p$ is a traced argument, one `vmap` over noise strengths and one over random
# draws compile into a single program per depth.

# %%
# ==============================================================================
# STEP 8: gradient variance against circuit depth, with and without noise
# ==============================================================================
# PARAMETERS
R_BP = 24                                   # random parameter vectors per depth
DEPTHS_BP = (1, 2, 4, 6, 8, 12)
P_BP = jnp.asarray([0.0, 0.005, 0.02])      # two-qubit error probabilities

var_bp = {}
t_bp = time.perf_counter()
for L in DEPTHS_BP:
    n_par = hea_num_params(N_Q, L)
    th_bp = jax.random.uniform(jax.random.PRNGKey(10 + L), (R_BP, n_par), minval=-jnp.pi, maxval=jnp.pi)
    grads = jax.jit(jax.vmap(lambda p: jax.vmap(
        jax.grad(lambda t: energy_dm(TERMS, noisy_hea_dm(t, N_Q, L, P1_RATIO * p, p))))(th_bp)))
    G = np.asarray(jax.block_until_ready(grads(P_BP)))               # (n_p, R_BP, n_par)
    var_bp[L] = [float(np.var(G[j, :, 0])) for j in range(P_BP.size)]

print(f"variance of dC/dtheta_0 over {R_BP} uniformly random parameter vectors "
      f"({time.perf_counter() - t_bp:.1f} s)")
print(f"{'L':>3s} {'n':>5s} " + " ".join(f"{('p2=' + f'{float(p):g}'):>12s}" for p in P_BP))
for L in DEPTHS_BP:
    print(f"{L:3d} {hea_num_params(N_Q, L):5d} " + " ".join(f"{v:12.4e}" for v in var_bp[L]))

WC = 10 / 3 + 2 * P1_RATIO                  # noise channels per layer seen by an average two-body term
print(f"\nfitted exponential decay of the RATIO Var(p)/Var(0) against depth  (predicted slope 2*{WC:.2f}*ln(1-4p/3))")
fits = {}
for j, p in enumerate(np.asarray(P_BP)[1:], start=1):
    y = np.log([var_bp[L][j] / var_bp[L][0] for L in DEPTHS_BP])
    slope = float(np.polyfit(DEPTHS_BP, y, 1)[0])
    pred = 2 * WC * np.log(1 - 4 * float(p) / 3)
    fits[float(p)] = slope
    print(f"  p2 = {float(p):.4f}:  measured slope {slope:+.4f}   predicted {pred:+.4f}   "
          f"ratio measured/predicted {slope / pred:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
for j, p in enumerate(np.asarray(P_BP)):
    axes[0].semilogy(DEPTHS_BP, [var_bp[L][j] for L in DEPTHS_BP], MARKERS[j] + "-", ms=6, color=PALETTE[j],
                     label=f"$p_2={p:g}$")
axes[0].set_xlabel("number of layers $L$"); axes[0].set_ylabel(r"$\mathrm{Var}[\partial C/\partial\theta_0]$")
axes[0].set_title(f"Gradient variance vs depth, $N={N_Q}$"); axes[0].legend(fontsize=8)

for j, p in enumerate(np.asarray(P_BP)[1:], start=1):
    ratio = [var_bp[L][j] / var_bp[L][0] for L in DEPTHS_BP]
    axes[1].semilogy(DEPTHS_BP, ratio, MARKERS[j] + "-", ms=6, color=PALETTE[j], label=f"measured, $p_2={p:g}$")
    axes[1].semilogy(DEPTHS_BP, np.exp(2 * WC * np.log(1 - 4 * float(p) / 3) * np.array(DEPTHS_BP)), "--",
                     lw=1.3, color=PALETTE[j], label=f"Eq. (16), $p_2={p:g}$")
axes[1].set_xlabel("number of layers $L$")
axes[1].set_ylabel(r"$\mathrm{Var}_{\mathrm{noisy}}/\mathrm{Var}_{\mathrm{noiseless}}$")
axes[1].set_title("The noise-induced part of the decay"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# At $N=4$ the noiseless variance (left panel, blue) is essentially flat: four qubits are far too few for the
# *noiseless* barren plateau of notebook 40, which needs the Hilbert-space dimension to be large. The noisy curves fall,
# and they fall faster the larger $p_2$ is. That separation is the point: this plateau is caused by the noise alone, and
# it appears at a system size where the noiseless landscape is perfectly healthy.
#
# The right panel divides out the noiseless baseline, which removes the sampling scatter common to both, and compares
# with Eq. (16). The measured decay is steeper than predicted by $15\%$ at $p_2=0.005$ and by $8\%$ at $p_2=0.02$ — the
# direction anticipated in Section 5.1, since the estimate assumed a fixed Pauli weight $w=2$ while the circuit spreads
# each term over strings of higher weight, and higher weight contracts faster. The *rate* of the exponential is
# therefore predicted to within a fifth by counting noise locations on a diagram.
#
# The practical statement is the one that ends the argument for hardware: a gradient of size $\epsilon$ needs
# $O(\epsilon^{-2})$ measurement shots to resolve, so a variance falling like $e^{-\alpha L}$ makes the shot cost grow
# like $e^{\alpha L}$. **No optimiser can fix this** — the information is not in the data. Stilck França and
# García-Patrón (2021) turned the same mechanism into a sharper statement: above a fixed noise rate, the output of a
# noisy variational circuit is so close to the maximally mixed state that its energy can be matched by a trivial
# classical algorithm.

# %% [markdown]
# ## 9. Training under noise
#
# Two ways of driving the optimiser are compared, both minimising the **same** noisy cost.
#
# 1. **Exact gradients on the density tensor.** `noisy_hea_dm` is an ordinary JAX function — a chain of einsums — so
#    `jax.grad` differentiates straight through the Kraus operators and returns $\nabla C_{\rm noisy}$ exactly. This is
#    a *simulator privilege*: no device can do it, because no device has $\rho$.
# 2. **SPSA on a trajectory estimate.** The cost is estimated from $M$ trajectories, which is what a device with a
#    finite measurement budget delivers, and the gradient comes from the two-evaluation SPSA rule of notebook 41 with
#    Spall's probe schedule $c_k=c/k^{0.101}$.
#
# The diagnostic recorded at every iteration is the **exact** density-tensor cost for both methods, and, separately, the
# noiseless energy of the same parameters — the quantity Section 10 needs. Twelve random initialisations are run in one
# `vmap`.

# %%
# ==============================================================================
# STEP 9: exact density-tensor gradients vs SPSA on trajectory-estimated costs
# ==============================================================================
# PARAMETERS
R_TRAIN, N_STEPS, LR = 12, 150, 0.05
M_TRAJ = 32                                   # trajectories per cost evaluation of the stochastic method
C_SPSA, GAMMA_SPSA = 0.2, 0.101

P1_DEF = P1_RATIO * P2_DEF
cost_noisy = jax.jit(lambda t: energy_dm(TERMS, noisy_hea_dm(t, N_Q, L_DEF, P1_DEF, P2_DEF)))
cost_ideal = jax.jit(lambda t: energy(TERMS, hardware_efficient_ansatz(t, N_Q, L_DEF)))
monitor_both = lambda t: jnp.stack([cost_noisy(t), cost_ideal(t)])


def cost_traj(key, theta, M=M_TRAJ):
    """Trajectory estimate of the noisy energy: mean over M independent unravellings (unbiased, error ~ 1/sqrt(M))."""
    ks = jax.random.split(key, M)
    return jnp.mean(jax.vmap(lambda k: energy(TERMS, noisy_hea_mcwf(k, theta, N_Q, L_DEF, P1_DEF, P2_DEF)))(ks))


def grad_spsa_traj(c=C_SPSA, gamma=GAMMA_SPSA):
    """SPSA gradient rule (notebook 41, Eq. 12) evaluated on the trajectory cost -- 2 estimates per iteration."""
    def rule(theta, key, k):
        c_k = c / k ** gamma
        kd, kp, km = jax.random.split(key, 3)
        delta = jax.random.rademacher(kd, theta.shape).astype(theta.dtype)
        return (cost_traj(kp, theta + c_k * delta) - cost_traj(km, theta - c_k * delta)) / (2 * c_k) * delta
    return rule


th_tr, ks_tr = random_starts(R_TRAIN, n_par_def, seed=42)
METHODS = (("exact gradient, density tensor", lambda t, k, i: jax.grad(cost_noisy)(t)),
           (f"SPSA on M={M_TRAJ} trajectories", grad_spsa_traj()))
hist_train, theta_train = {}, {}
for name, rule in METHODS:
    t0 = time.perf_counter()
    thf, h = jax.jit(jax.vmap(lambda t, k: train(t, k, rule, opt_adam(LR), monitor_both, N_STEPS)))(th_tr, ks_tr)
    theta_train[name] = jax.block_until_ready(thf)
    hist_train[name] = np.asarray(h)                      # (runs, steps, 2): [noisy cost, noiseless cost]
    print(f"{name:34s} {time.perf_counter() - t0:6.1f} s   "
          f"final noisy E: median {np.median(hist_train[name][:, -1, 0]):+.5f}, "
          f"best {np.min(hist_train[name][:, -1, 0]):+.5f}")

print(f"\nreference values:  exact ground energy E0 = {E0_EXACT:+.5f}")
print(f"{'method':>34s} {'median E_noisy':>15s} {'median E_ideal(theta*)':>23s} {'best E_ideal(theta*)':>21s}")
for name, _ in METHODS:
    h = hist_train[name]
    print(f"{name:>34s} {np.median(h[:, -1, 0]):15.5f} {np.median(h[:, -1, 1]):23.5f} {np.min(h[:, -1, 1]):21.5f}")

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
it = np.arange(1, N_STEPS + 1)
for j, (name, _) in enumerate(METHODS):
    for col, ax, lab in ((0, axes[0], "noisy cost"), (1, axes[1], "noiseless energy of the same angles")):
        lo, med, hi = bands(hist_train[name][:, :, col])
        ax.fill_between(it, lo, hi, color=PALETTE[j], alpha=0.15)
        ax.plot(it, med, "-", lw=1.8, color=PALETTE[j], label=name)
axes[0].set_ylabel(r"$C_{\mathrm{noisy}}(\theta)$"); axes[0].set_title("What the optimiser minimises")
axes[1].set_ylabel(r"$\langle H\rangle$ of $\vert\psi(\theta)\rangle$")
axes[1].set_title("What the angles are worth on a perfect device")
for ax in axes:
    ax.axhline(E0_EXACT, color="k", ls=":", lw=1.2, label="exact ground energy $E_0$")
    ax.set_xlabel("iteration"); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %%
# ==============================================================================
# CHECKPOINT 3: the trained state, measured on the density tensor and with trajectories
# ==============================================================================
theta_star = theta_train["exact gradient, density tensor"][
    int(np.argmin(hist_train["exact gradient, density tensor"][:, -1, 0]))]
E_dm_star = float(cost_noisy(theta_star))
samples_star = np.asarray(jax.block_until_ready(jax.jit(jax.vmap(
    lambda k: energy(TERMS, noisy_hea_mcwf(k, theta_star, N_Q, L_DEF, P1_DEF, P2_DEF))))(
    jax.random.split(jax.random.PRNGKey(99), 4000))))
E_mc_star, se_star = mean_and_se(samples_star)
print("energy of the best trained noisy state:")
print(f"  density tensor          {E_dm_star:+.6f}")
print(f"  4000 trajectories       {E_mc_star:+.6f} +- {se_star:.6f}   "
      f"({abs(E_mc_star - E_dm_star) / se_star:.2f} standard errors)")
print(f"  noiseless at the same angles {float(cost_ideal(theta_star)):+.6f}")
print(f"  exact ground energy          {E0_EXACT:+.6f}")
assert abs(E_mc_star - E_dm_star) < 5 * se_star

# %% [markdown]
# The two panels separate two things that are easy to conflate.
#
# **Left: the optimiser works.** Both methods reduce the noisy cost. The exact-gradient run reaches a median of
# $-3.758$ and has converged by iteration 40; the SPSA run reaches $-3.327$ after 150 iterations with $M=32$
# trajectories per evaluation and is still improving. The stochastic method is slower per iteration, as it was in
# notebook 41, and for the same reason: two noisy scalars per step against $n=24$ exact ones. Its interquartile band is
# also much wider, because the noise of the estimate is inherited by the path the optimiser takes.
#
# **Right: the noisy cost is not the physics.** The same angles, evaluated on a perfect device, give a median energy of
# $-4.382$ for the exact-gradient run against the exact ground energy $E_0=-4.4548$ — a far better number than the
# $-3.758$ the noisy device reports. The gap between the two panels is the noise floor, and it does not shrink with
# training. Reporting the noisy cost as "the variational energy" would understate the quality of the state found by
# about $0.62$ in units where the answer is $-4.45$.
#
# The checkpoint confirms once more that the two simulators describe the same state: at the angles that minimised the
# noisy cost, the trajectory estimate $-3.7812\pm0.0209$ agrees with the exact density-tensor value $-3.7784$ within
# $0.13$ standard errors.
#
# > **Numerical practice.** The *variational principle does not protect the noisy cost*. $C(\boldsymbol\theta)\ge E_0$
# > holds for the noisy cost too, since $\mathrm{Tr}(\hat H\rho)\ge E_0$ for any state — but the bound is far from tight,
# > and a noisy energy above $E_0$ says nothing about how close the state is to the ground state.

# %% [markdown]
# ## 10. Optimal-parameter resilience
#
# Section 6 proved a clean statement for global depolarising noise: the minimiser does not move. Sharma, Khatri, Cerezo
# and Coles (2020) proved related statements for several noise models in variational *compiling*, and gave the effect
# its name — **optimal parameter resilience**. Real noise is local, not global, so the proof does not apply directly and
# the question becomes quantitative: how much does the minimiser move?
#
# The experiment is the one the name suggests. **Train under noise, then evaluate the parameters found on a noiseless
# simulator.** If the resilience holds, the noiseless energy of the noisy optimum should stay close to the noiseless
# optimum's own energy, however large the noise gets. Two channels are compared at five noise strengths each: the
# unital depolarising channel and the non-unital amplitude-damping channel, both applied after every gate. Because the
# noise strength is a traced argument, the whole $5\times8$ grid of (noise level, random start) is one compiled program
# per channel.

# %%
# ==============================================================================
# STEP 10: train noisy, evaluate noiseless -- for a unital and a non-unital channel
# ==============================================================================
# PARAMETERS
R_RES, N_STEPS_RES = 8, 150
P_RES = jnp.asarray([0.0, 0.01, 0.02, 0.04, 0.08])

th_res, ks_res = random_starts(R_RES, n_par_def, seed=1234)


def train_over_noise(chan):
    """Train at every noise level of P_RES from every start of th_res; record (noisy cost, noiseless energy)."""
    def one(p, t, k):
        cn = lambda th: energy_dm(TERMS, noisy_hea_dm(th, N_Q, L_DEF, P1_RATIO * p, p, chan, chan))
        mon = lambda th: jnp.stack([cn(th), cost_ideal(th)])
        return train(t, k, lambda th, kk, i: jax.grad(cn)(th), opt_adam(LR), mon, N_STEPS_RES)[1]
    return jax.jit(jax.vmap(lambda p: jax.vmap(lambda t, k: one(p, t, k))(th_res, ks_res)))(P_RES)


res = {}
for name, chan in (("depolarising (unital)", kraus_depolarizing),
                   ("amplitude damping (non-unital)", kraus_amplitude_damping)):
    t0 = time.perf_counter()
    res[name] = np.asarray(jax.block_until_ready(train_over_noise(chan)))   # (n_p, runs, steps, 2)
    print(f"{name:32s} trained at {P_RES.size} noise levels x {R_RES} starts in {time.perf_counter() - t0:.1f} s")

E_ref_ideal = float(np.min(res["depolarising (unital)"][0, :, -1, 1]))      # best noiseless optimum found
print(f"\nbest noiseless optimum of this ansatz: E = {E_ref_ideal:.6f}   (exact E0 = {E0_EXACT:.6f})")
print(f"{'channel':>32s} {'strength':>9s} {'E_noisy median':>15s} {'E_ideal(theta*) median':>23s} "
      f"{'best':>10s} {'penalty vs best':>16s}")
for name in res:
    for j, p in enumerate(np.asarray(P_RES)):
        H = res[name][j]
        print(f"{name if j == 0 else '':>32s} {p:9.3f} {np.median(H[:, -1, 0]):15.5f} "
              f"{np.median(H[:, -1, 1]):23.5f} {np.min(H[:, -1, 1]):10.5f} "
              f"{np.min(H[:, -1, 1]) - E_ref_ideal:+16.5f}")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for j, name in enumerate(res):
    best = [np.min(res[name][i, :, -1, 1]) - E_ref_ideal for i in range(P_RES.size)]
    med = [np.median(res[name][i, :, -1, 1]) - E_ref_ideal for i in range(P_RES.size)]
    ax.plot(np.asarray(P_RES), best, MARKERS[j] + "-", ms=7, color=PALETTE[j], label=f"{name}, best of {R_RES}")
    ax.plot(np.asarray(P_RES), med, MARKERS[j] + "--", ms=5, color=PALETTE[j], alpha=0.6, label=f"{name}, median")
ax.axhline(0.0, color="k", ls=":", lw=1.2, label="noiseless training")
ax.set_xlabel(r"channel strength after two-qubit gates ($p_2$ or $\gamma_2$)")
ax.set_ylabel(r"$E_{\mathrm{ideal}}(\theta^\star_{\mathrm{noisy}})-E_{\mathrm{ideal}}(\theta^\star_{\mathrm{ideal}})$")
ax.set_title("Optimal-parameter resilience: the price of having trained under noise")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); plt.show()

# %% [markdown]
# The measurement supports the resilience claim, and it supports it for **both** channels — including the non-unital
# one, for which Section 7 proved that the minimiser must move.
#
# Read the last column. Training at $p_2=0.08$, a catastrophic error rate at which the noisy device reports an energy of
# $-2.451$ instead of $-4.455$, still yields angles whose *noiseless* energy is $-4.3966$, only $0.042$ above the best
# noiseless optimum $-4.4384$ of the same ansatz. At the smallest non-zero noise, $p_2=0.01$, the penalty is
# $2.7\cdot10^{-3}$ — three orders of magnitude smaller than the $0.35$ by which the reported energy is wrong.
#
# The two channels are *not* identical, and the difference has the sign Section 7 demands. The non-unital channel costs
# more at **every** noise level: $5.2\cdot10^{-3}$ against $2.7\cdot10^{-3}$ at strength $0.01$, $0.052$ against
# $0.034$ at $0.04$, and $0.058$ against $0.042$ at $0.08$. The gap is a consistent factor of about $1.4$, small on the
# scale of the figure but visible in every row.
#
# The smallness of both penalties is not a contradiction of Section 7. That section showed the minimiser *moves*; it did
# not say how much the cost pays for the move. Near a minimum the cost is quadratic in the displacement, so a shift of
# the optimum by $\delta$ costs only $O(\delta^2)$ in energy. The honest summary of these numbers is: **the parameters
# are far more robust than the cost value is**, whether the noise is unital or not, and the useful output of a noisy
# variational run is the parameter vector, not the energy it reported.
#
# > **Common pitfall.** This conclusion is a measurement on one Hamiltonian, one ansatz and two channels, at
# > $N=4$. It is not a theorem. What *is* a theorem is Section 6: for global depolarising noise the optimum is exactly
# > unchanged. For everything else, resilience is an empirical property that has to be re-measured.

# %% [markdown]
# ## 11. The depth trade-off
#
# Notebook 41 measured that on a *noiseless* simulator more layers are better: extra parameters raise the success rate
# and lower the achievable floor. Section 8 measured that on a *noisy* device more layers cost exponentially in gradient
# size. The two effects pull in opposite directions, so there must be an optimum, and it must move to smaller depth as
# the error rate grows.
#
# The experiment trains the ansatz at depths $L=1,\dots,6$ and five noise strengths from $0$ to $0.02$, with eight
# random starts each, and records the best noisy energy reached. Depth is a static quantity (it is the length of the
# `lax.scan`), so one program is compiled per depth; the noise axis is vmapped inside it.

# %%
# ==============================================================================
# STEP 11: best achievable noisy energy against depth, for several error rates
# ==============================================================================
# PARAMETERS
DEPTHS_TO = (1, 2, 3, 4, 5, 6)
P_TO = jnp.asarray([0.0, 0.0005, 0.002, 0.008, 0.02])
R_TO, N_STEPS_TO = 8, 150

best_E = np.zeros((len(DEPTHS_TO), P_TO.size))
med_E = np.zeros_like(best_E)
ideal_floor = np.zeros(len(DEPTHS_TO))
t_to = time.perf_counter()
for iL, L in enumerate(DEPTHS_TO):
    n_par = hea_num_params(N_Q, L)
    th_to, ks_to = random_starts(R_TO, n_par, seed=77)

    def one(p, t, k, L=L):
        cn = lambda th: energy_dm(TERMS, noisy_hea_dm(th, N_Q, L, P1_RATIO * p, p))
        ci = lambda th: energy(TERMS, hardware_efficient_ansatz(th, N_Q, L))
        return train(t, k, lambda th, kk, i: jax.grad(cn)(th), opt_adam(LR),
                     lambda th: jnp.stack([cn(th), ci(th)]), N_STEPS_TO)[1]

    Hh = np.asarray(jax.block_until_ready(
        jax.jit(jax.vmap(lambda p: jax.vmap(lambda t, k: one(p, t, k))(th_to, ks_to)))(P_TO)))
    best_E[iL] = Hh[:, :, -1, 0].min(axis=1)
    med_E[iL] = np.median(Hh[:, :, -1, 0], axis=1)
    ideal_floor[iL] = Hh[0, :, -1, 1].min()

print(f"best noisy energy over {R_TO} random starts ({time.perf_counter() - t_to:.1f} s); exact E0 = {E0_EXACT:.5f}")
print(f"{'L':>3s} {'n':>5s} " + " ".join(f"{('p2=' + f'{float(p):g}'):>10s}" for p in P_TO))
for iL, L in enumerate(DEPTHS_TO):
    print(f"{L:3d} {hea_num_params(N_Q, L):5d} " + " ".join(f"{best_E[iL, j]:10.4f}" for j in range(P_TO.size)))
print("\noptimal depth for each error rate:")
for j, p in enumerate(np.asarray(P_TO)):
    L_star = DEPTHS_TO[int(np.argmin(best_E[:, j]))]
    print(f"  p2 = {p:.4f}  ->  L* = {L_star}   (best energy {best_E[:, j].min():+.4f})")
print("\nthe two competing terms:")
print(f"{'L':>3s} {'noiseless error E-E0':>21s} {'noise penalty at p2=' + f'{float(P_TO[-1]):g}':>26s}")
for iL, L in enumerate(DEPTHS_TO):
    print(f"{L:3d} {ideal_floor[iL] - E0_EXACT:21.3e} {best_E[iL, -1] - ideal_floor[iL]:26.3f}")
print(f"  noiseless error falls by a factor {(ideal_floor[0] - E0_EXACT) / (ideal_floor[-1] - E0_EXACT):.0f} "
      f"between L = {DEPTHS_TO[0]} and L = {DEPTHS_TO[-1]}")

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
for j, p in enumerate(np.asarray(P_TO)):
    axes[0].plot(DEPTHS_TO, best_E[:, j], MARKERS[j % len(MARKERS)] + "-", ms=6, color=PALETTE[j],
                 label=f"$p_2={p:g}$")
    k = int(np.argmin(best_E[:, j]))
    axes[0].plot([DEPTHS_TO[k]], [best_E[k, j]], "*", ms=15, color=PALETTE[j])
axes[0].axhline(E0_EXACT, color="k", ls=":", lw=1.2, label="exact $E_0$")
axes[0].set_xlabel("number of layers $L$"); axes[0].set_ylabel(r"best $\langle H\rangle$ reached")
axes[0].set_title("Expressivity against accumulated noise (stars: optimal depth)")
axes[0].legend(fontsize=8)

axes[1].plot(DEPTHS_TO, ideal_floor - E0_EXACT, MARKERS[0] + "-", ms=6, color=PALETTE[0],
             label="noiseless ansatz error $E-E_0$")
axes[1].plot(DEPTHS_TO, best_E[:, -1] - ideal_floor, MARKERS[1] + "-", ms=6, color=PALETTE[1],
             label=f"noise penalty at $p_2={float(P_TO[-1]):g}$")
axes[1].set_yscale("log")
axes[1].set_xlabel("number of layers $L$"); axes[1].set_ylabel("energy error")
axes[1].set_title("The two competing terms"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The optimum exists and it moves, exactly as the two mechanisms require.
#
# * On a **perfect** device ($p_2=0$) the energy falls monotonically with depth, from $-4.376$ at $L=1$ to $-4.454$ at
#   $L=6$, which is $E_0$ to three decimals: extra layers only ever help, and the deepest circuit tried is the best one.
# * At $p_2=5\cdot10^{-4}$ — a two-qubit error per gate of $r_2=3\cdot10^{-4}$, better than any device today — the
#   optimum is already at $L=2$.
# * At $p_2=2\cdot10^{-3}$ it is still $L=2$; at $p_2=8\cdot10^{-3}$ and $p_2=2\cdot10^{-2}$ it has dropped to $L=1$,
#   the shallowest circuit in the study.
#
# The right panel shows the two terms whose sum produces the optimum. The noiseless ansatz error falls steeply with
# depth — from $7.883\cdot10^{-2}$ at $L=1$ to $3.073\cdot10^{-4}$ at $L=6$, a factor of $257$ — while the noise
# penalty at $p_2=0.02$ grows from $0.347$ to $1.330$. The first curve keeps falling as the ansatz becomes able to
# represent the ground state; the second never stops rising. Wherever the falling curve stops falling faster than the
# rising one rises, the sum turns around.
#
# > **Physics insight.** This is the central design constraint of the NISQ era in one figure. A circuit that is deep
# > enough to be interesting is too deep to survive, unless the error rate falls below roughly (one over the number of
# > gates). Every entry in the table can be reproduced with a different ansatz, and the numbers will change; the shape
# > of the argument will not.

# %% [markdown]
# ## 12. Zero-noise extrapolation
#
# Error *correction* removes errors by encoding one logical qubit in many physical ones, and needs resources no current
# device has. Error *mitigation* leaves the errors in place and corrects the **estimate** afterwards, by running the
# circuit at several noise levels and extrapolating. The idea is due to Temme, Bravyi and Gambetta (2017) and,
# independently, Li and Benjamin (2017).
#
# ### 12.1 The method
#
# Let $E(\lambda)$ be the expectation value measured when every channel strength is multiplied by $\lambda\ge1$. Under
# mild assumptions $E(\lambda)$ is smooth and $E(0)$ is the noiseless value we want but cannot measure: on hardware
# $\lambda$ can be raised (by stretching pulses, or by replacing each gate $G$ by $GG^\dagger G$, which is the same
# unitary with three times the error) but not lowered. **Extrapolate to $\lambda=0$.** With $\lambda_1<\dots<\lambda_m$
# and the corresponding measurements, fit a polynomial of degree $m-1$ and evaluate it at zero. This is Richardson
# extrapolation, and the result is the linear combination
#
# $$\hat E(0)=\sum_{i=1}^{m}c_i\,E(\lambda_i),\qquad
#   c_i=\prod_{j\neq i}\frac{\lambda_j}{\lambda_j-\lambda_i},\qquad \sum_ic_i=1,\tag{17}$$
#
# which follows from writing the Lagrange interpolating polynomial through the points
# $(\lambda_i,E(\lambda_i))$ and evaluating it at $\lambda=0$. For $m=2$ with $\lambda=1,2$ this is
# $\hat E(0)=2E(1)-E(2)$, ordinary linear extrapolation.
#
# ### 12.2 Why it works, and when it stops
#
# Expand the noisy expectation value in the noise strength: $E(\lambda)=E_0+a_1\lambda+a_2\lambda^2+\dots$ The degree
# $m-1$ Richardson combination annihilates the terms $\lambda^1,\dots,\lambda^{m-1}$ exactly and leaves an error of
# order $\lambda^m$. So mitigation helps when the series converges quickly, that is when $\lambda\times$ (total error
# probability of the circuit) is small; it fails when the higher-order terms are not small, and it fails badly, because
# the coefficients $c_i$ alternate in sign and grow with $m$ — the combination $3E(1)-3E(2)+E(3)$ amplifies any
# statistical error by $\sqrt{9+9+1}=4.4$. **Mitigation trades bias for variance**, and both halves of that trade must
# be quoted.

# %%
# ==============================================================================
# STEP 12: Richardson extrapolation of the noisy energy in the noise scaling factor
# ==============================================================================
LAMBDAS = np.array([1.0, 2.0, 3.0])


def richardson_weights(lams):
    """Coefficients c_i of Eq. (17): the Lagrange polynomial through (lam_i, E_i), evaluated at lam = 0."""
    lams = np.asarray(lams, dtype=float)
    return np.array([np.prod([lams[j] / (lams[j] - lams[i]) for j in range(lams.size) if j != i])
                     for i in range(lams.size)])


print("Richardson coefficients of Eq. (17)")
for m in (2, 3, 4):
    c = richardson_weights(LAMBDAS[:m] if m <= 3 else np.arange(1.0, m + 1))
    print(f"  m = {m}: c = {np.round(c, 4)}   sum {c.sum():.6f}   noise amplification "
          f"sqrt(sum c^2) = {np.sqrt((c ** 2).sum()):.3f}")

E_exact_star = float(cost_ideal(theta_star))
print(f"\nzero-noise extrapolation at the trained angles; the target is the noiseless value {E_exact_star:+.6f}")
print(f"{'base p2':>9s} {'E(lam=1)':>11s} {'E(2)':>11s} {'E(3)':>11s} "
      f"{'linear (m=2)':>13s} {'Richardson (m=3)':>17s} {'|err| lin':>11s} {'|err| Rich':>11s}")
zne_rows = []
for base in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
    f_base = jax.jit(lambda t, s, b=base: energy_dm(TERMS, noisy_hea_dm(t, N_Q, L_DEF, P1_RATIO * b * s, b * s)))
    ys = np.array([float(f_base(theta_star, lam)) for lam in LAMBDAS])
    lin = float(richardson_weights(LAMBDAS[:2]) @ ys[:2])
    rich = float(richardson_weights(LAMBDAS) @ ys)
    zne_rows.append((base, ys, lin, rich))
    print(f"{base:9.4f} {ys[0]:11.5f} {ys[1]:11.5f} {ys[2]:11.5f} {lin:13.5f} {rich:17.5f} "
          f"{abs(lin - E_exact_star):11.2e} {abs(rich - E_exact_star):11.2e}")

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
lam_fine = np.linspace(0, 3.2, 100)
for j, (base, ys, lin, rich) in enumerate(zne_rows[::2]):
    c = PALETTE[j]
    axes[0].plot(LAMBDAS, ys, MARKERS[j] + "", ms=8, color=c, label=f"measured, $p_2={base:g}$")
    axes[0].plot(lam_fine, np.polyval(np.polyfit(LAMBDAS, ys, 2), lam_fine), "-", lw=1.4, color=c)
    axes[0].plot([0.0], [rich], "*", ms=14, color=c)
axes[0].axhline(E_exact_star, color="k", ls="--", lw=1.2, label="noiseless value")
axes[0].set_xlabel(r"noise scaling factor $\lambda$"); axes[0].set_ylabel(r"$\langle H\rangle$")
axes[0].set_title("Fit in the noise strength, extrapolate to zero"); axes[0].legend(fontsize=8)

bases = np.array([r[0] for r in zne_rows])
series = [("unmitigated ($m=1$)", [abs(r[1][0] - E_exact_star) for r in zne_rows], 1),
          ("linear, $m=2$", [abs(r[2] - E_exact_star) for r in zne_rows], 2),
          ("Richardson, $m=3$", [abs(r[3] - E_exact_star) for r in zne_rows], 3)]
for k, (lab, ys_, sl) in enumerate(series):
    axes[1].loglog(bases, ys_, MARKERS[k] + "-", ms=6, color=PALETTE[k], label=lab)
    axes[1].loglog(bases, ys_[0] * (bases / bases[0]) ** sl, "k:", lw=1.0,
                   label=r"$\propto p_2^m$" if k == 0 else None)
axes[1].set_xlabel(r"base error probability $p_2$"); axes[1].set_ylabel("absolute error of the estimate")
axes[1].set_title(r"Residual error of an order-$m$ extrapolation")
axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The two regimes announced in Section 12.2 are both visible.
#
# **Where it helps.** At $p_2=2.5\cdot10^{-3}$ the raw measurement is wrong by $7.7\cdot10^{-2}$; linear extrapolation
# from two points reduces the error to $1.3\cdot10^{-3}$, and the three-point Richardson combination to
# $2.2\cdot10^{-5}$ — three and a half orders of magnitude of bias removed for the price of two extra circuit families.
# The right panel shows why: the unmitigated error is linear in $p_2$, the two-point estimate is quadratic and the
# three-point estimate cubic, and each measured curve lies on the dotted reference line of its own order over more than
# a decade.
#
# **Where it fails.** At $p_2=0.08$ the raw error is $1.9$, the linear estimate is still wrong by $0.81$ and Richardson
# by $0.34$. Extrapolation is not magic: at $\lambda=3$ and $p_2=0.08$ the circuit has an aggregate error probability of
# order one, the state is close to maximally mixed — the measured $E(3)=-0.70$ against a noiseless $-4.36$ — and
# $E(\lambda)$ has flattened out. A function that has already saturated carries little information about its value at
# $\lambda=0$, and no polynomial through three of its points can invent it.
#
# The table of coefficients quantifies the other half of the trade. Going from $m=2$ to $m=4$ raises
# $\sqrt{\sum_ic_i^2}$ from $2.24$ to $8.31$: shot noise in the inputs is amplified by that factor, so the shot budget
# must grow by its square — a factor $14$ — to keep the statistical error where it was. A high-order extrapolation on a
# device is therefore a deliberate exchange of a bias one can bound for a variance one must pay for.
#
# > **Numerical practice.** Every number in this section came from *exact* density-tensor evaluations, so the
# > statistical half of the trade-off was switched off deliberately in order to expose the bias half cleanly. Exercise 6
# > puts the shot noise back.

# %% [markdown]
# ## 13. Cost: density tensor against trajectories
#
# The two representations scale differently, and the crossover decides which one to use.
#
# * The **density tensor** holds $4^N$ complex numbers and costs $O(2^k4^N)$ per $k$-qubit operation. It is exact, and
#   `jax.grad` runs through it, but $N=13$ already needs $4^{13}\times16$ bytes $\approx1$ TB.
# * **Trajectories** hold $2^N$ complex numbers each. $M$ of them cost $O(M\,2^N)$ and carry a statistical error
#   $\propto1/\sqrt M$, so *at fixed accuracy* $M$ is fixed and the total cost is $O(M2^N)$ — exponentially cheaper in
#   the exponent, at the price of never being exact.
#
# The crossover is where $M\,2^N\approx4^N$, i.e. $2^N\approx M$. The measurement below uses $M=200$, so the prediction
# is a crossover near $N=\log_2 200\approx7.6$.

# %%
# ==============================================================================
# STEP 13: measured cost of one circuit evaluation, density tensor vs M trajectories
# ==============================================================================
M_COST, L_COST = 200, 2
N_DM = (2, 3, 4, 5, 6, 7, 8)
N_MC = (2, 3, 4, 5, 6, 7, 8, 10, 12)


def make_dm_run(N):
    """A compiled noisy-circuit evaluation on the density tensor for N qubits."""
    th = jax.random.uniform(jax.random.PRNGKey(1), (2 * N * (L_COST + 1),), minval=-jnp.pi, maxval=jnp.pi)
    return jax.jit(lambda: noisy_hea_dm(th, N, L_COST, P1_DEF, P2_DEF))


def make_mc_run(N, M):
    """The same circuit as M quantum trajectories, vmapped over PRNG keys."""
    th = jax.random.uniform(jax.random.PRNGKey(1), (2 * N * (L_COST + 1),), minval=-jnp.pi, maxval=jnp.pi)
    keys = jax.random.split(jax.random.PRNGKey(0), M)
    return jax.jit(lambda: jax.vmap(lambda k: noisy_hea_mcwf(k, th, N, L_COST, P1_DEF, P2_DEF))(keys))


t_dm = {N: timed(make_dm_run(N), repeat=5) for N in N_DM}
t_mc = {N: timed(make_mc_run(N, M_COST), repeat=5) for N in N_MC}
print(f"one evaluation of the noisy ansatz, L = {L_COST} layers (compilation excluded)")
print(f"{'N':>3s} {'DM [ms]':>10s} {'DM memory':>12s} {'MCWF x' + str(M_COST) + ' [ms]':>18s} {'ratio DM/MCWF':>14s}")
for N in N_MC:
    dm = f"{t_dm[N] * 1e3:10.2f}" if N in t_dm else f"{'-':>10s}"
    mem = 16 * 4 ** N
    mem_s = f"{mem / 1024:.1f} kB" if mem < 1024 ** 2 else f"{mem / 1024 ** 2:.1f} MB"
    ratio = f"{t_dm[N] / t_mc[N]:14.2f}" if N in t_dm else f"{'-':>14s}"
    print(f"{N:3d} {dm} {mem_s:>12s} {t_mc[N] * 1e3:18.2f} {ratio}")

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.semilogy(N_DM, [t_dm[N] * 1e3 for N in N_DM], MARKERS[0] + "-", ms=6, color=PALETTE[0], label="density tensor")
ax.semilogy(N_MC, [t_mc[N] * 1e3 for N in N_MC], MARKERS[1] + "-", ms=6, color=PALETTE[1],
            label=f"{M_COST} trajectories")
ax.semilogy(N_DM, t_dm[N_DM[-1]] * 1e3 * 4.0 ** (np.array(N_DM) - N_DM[-1]), "k--", lw=1.1, label=r"$\propto4^N$")
ax.semilogy(N_MC, t_mc[N_MC[-1]] * 1e3 * 2.0 ** (np.array(N_MC) - N_MC[-1]), "k:", lw=1.4, label=r"$\propto2^N$")
ax.set_xlabel("number of qubits $N$"); ax.set_ylabel("wall time per evaluation [ms]")
ax.set_title("Cost of the two representations of the same noisy circuit")
ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# Each curve follows its predicted exponent over the range where the representation is usable: the density tensor
# steepens towards the $4^N$ reference line, the trajectory batch tracks $2^N$. The *timing* crossover, however, has not
# been reached by $N=8$ — the ratio in the last column is still below one, so the density tensor is the faster of the
# two everywhere it can be run at all. The naive estimate $2^N\approx M$, which put the crossover at $N\approx7.6$,
# counts only floating-point work; a trajectory also pays for one PRNG split and one categorical draw at every noise
# location, and there are $O(NL)$ of them, which is the constant the estimate ignores.
#
# What forces the switch is therefore **memory**, not speed. The density tensor is $1$ MB at $N=8$, $16$ MB at $N=10$
# and $256$ MB at $N=12$ — per intermediate array, in a graph that holds several of them and, under `jax.grad`, keeps
# many more. A trajectory is $64$ kB at $N=12$, and two hundred of them run in about a second.
#
# The decision rule is not only about resources. The density tensor gives an exact number and an exact gradient; the
# trajectories give an estimate with an error bar and, in practice, a gradient that has to come from SPSA or parameter
# shift. For the small systems of this notebook the density tensor is the right tool, which is why every *theoretical*
# statement above was checked on it and the trajectories were used to confirm that the statements survive the
# unravelling.

# %% [markdown]
# ## 14. Key takeaways
#
# * **The conversion from laboratory numbers to channel parameters is a two-line derivation, not a convention.** The
#   depolarising channel contracts the Bloch vector by $1-4p/3$ and has average gate fidelity $1-2p/3$, so a reported
#   error per gate $r$ becomes $p=\tfrac32r$; damping and dephasing follow from $\gamma=1-e^{-t/T_1}$ and
#   $p=\tfrac12(1-e^{-t/T_\varphi})$. All three were confirmed to twelve digits.
# * **Two independent simulators agree.** The density tensor and the trajectory unravelling gave the same energy of the
#   same noisy variational state within one standard error at $4000$ trajectories, and the trajectory error fell as
#   $1/\sqrt M$ over two decades.
# * **Local depolarising noise contracts every Pauli expectation value by $\lambda^{wcD}$** with $\lambda=1-4p/3$,
#   Eq. (9) — exact to round-off where its assumption holds. In a real circuit the functional form survives (the
#   amplitude of a landscape slice followed $\lambda^{4.79}$ over the whole range $0\le p_2\le0.25$) but the effective
#   exponent has to be measured: it fell between the weight-1 and weight-2 counts, $3.30$ and $7.27$.
# * **Global depolarising noise does not move the minimum.** It commutes with every unitary, so
#   $C_{\rm noisy}=(1-q)C+q\,\mathrm{Tr}\hat H/2^N$ — verified grid point by grid point to $10^{-16}$ on an
#   $81\times81$ slice. The gradient is rescaled by exactly $1-q$ without rotating (cosine $1.000000000000$), and the
#   set of minimising grid points is unchanged at $q=0.8$.
# * **Non-unital noise does move it.** For one qubit with $C=-\langle X\rangle-h\langle Z\rangle$ the optimal angle
#   obeys $\tan\theta_\star=1/(h\sqrt{1-\gamma})$ under amplitude damping, measured at $1.174641$ and $1.264491$ for
#   $\gamma=0.3$ and $0.6$ against a noiseless $1.107097$ — a shift of $9$ degrees at $\gamma=0.6$.
# * **Noise produces a barren plateau in the circuit depth.** At $N=4$, where the noiseless gradient variance is flat in
#   depth, the noisy variance fell exponentially with slopes $-0.054$ and $-0.207$ per layer at $p_2=0.005$ and
#   $p_2=0.02$, $15\%$ and $8\%$ steeper than the prediction of Eq. (16) obtained by counting noise locations.
# * **The parameters are more robust than the cost.** Training at an error rate so high that the device reports $-2.451$
#   instead of $-4.455$ still produced angles worth $-4.397$ on a noiseless simulator, $0.042$ from the best this ansatz
#   can do. Amplitude damping cost consistently more than depolarising — by a factor of about $1.4$ at every noise
#   level, the sign Section 7 requires — but both penalties stayed small, because the cost is quadratic in the
#   displacement of a minimiser.
# * **There is an optimal depth and it shrinks with the error rate.** Measured: $L^\star=6$ (the largest tried) at
#   $p_2=0$, $L^\star=2$ at $p_2=5\cdot10^{-4}$ and $2\cdot10^{-3}$, $L^\star=1$ at $8\cdot10^{-3}$ and
#   $2\cdot10^{-2}$.
# * **Zero-noise extrapolation removes bias order by order and costs variance.** Richardson extrapolation from three
#   noise levels reduced the error at $p_2=2.5\cdot10^{-3}$ from $7.7\cdot10^{-2}$ to $2.2\cdot10^{-5}$, with residuals
#   scaling as $p_2^m$ for $m=1,2,3$; at $p_2=0.08$ it left an error of $0.34$, because the observable had already
#   saturated. The amplification of statistical noise grows from $2.24$ at $m=2$ to $8.31$ at $m=4$.
# * **Memory, not speed, decides which representation to use.** The density tensor was still the faster of the two at
#   every $N$ it could be run at ($N\le8$), because a trajectory pays $O(NL)$ PRNG splits that the $M\,2^N$ estimate
#   ignores; but it needs $256$ MB per array at $N=12$, where a trajectory needs $64$ kB.
#
# ## 15. Exercises
#
# 1. ★ **Dephasing instead of depolarising.** Re-run the landscape slice of Section 5.2 with `kraus_dephasing` as both
#    channels. Dephasing is unital, so the minimum should not move much; is the contraction of the amplitude still
#    described by Eq. (9), and with what $\lambda$? (Derive the eigenvalue of $\mathcal Z_p^\dagger$ on $X$, $Y$ and $Z$
#    first — they are not all the same.)
# 2. ★ **The weight dependence, directly.** Fix a depth and a noise strength and measure
#    $\langle P\rangle_{\rm noisy}/\langle P\rangle_{\rm ideal}$ for Pauli strings of weight $1,2,3,4$ *through the real
#    noisy ansatz*, not the bare noise rounds of Step 3. Plot $\ln$ of the ratio against $w$ and compare its slope with
#    $cL\ln\lambda$.
# 3. ★★ **A global depolarising fit (extend the code).** Many experiments model an entire noisy circuit as one global
#    depolarising channel. Fit $q$ by least squares from the noisy and noiseless costs at 50 random
#    $\boldsymbol\theta$, then test the model by predicting the noisy cost at 50 fresh ones. How good is the
#    single-parameter description at $p_2=0.005$, and at $p_2=0.05$?
# 4. ★★ **Resilience of the state, not the energy (physics).** Section 10 measured the energy penalty. Measure instead
#    the fidelity $\lvert\langle\psi(\boldsymbol\theta^\star_{\rm noisy})\vert\psi(\boldsymbol\theta^\star_{\rm ideal})\rangle\rvert^2$
#    between the two *states* the two optima prepare. Does it degrade faster than the energy does, and why should it?
# 5. ★★ **Noise scaling by gate folding (extend the code).** Replace each $CZ$ by $CZ\,CZ^\dagger\,CZ$ in the gate list
#    and add its noise channels; this triples the error of that gate without changing the ideal unitary, which is how
#    $\lambda=3$ is realised on hardware. Compare the extrapolation obtained this way with the one of Section 12, where
#    $\lambda$ multiplied the channel parameter directly.
# 6. ★★ **The variance half of the trade (extend the code).** Estimate each $E(\lambda_i)$ from $M$ trajectories instead
#    of exactly, and plot the total error (bias plus statistics) of the $m=1,2,3$ estimates against $M$ at fixed
#    $p_2=0.01$. Below which shot budget does mitigation make the answer *worse*?
# 7. ★★★ **Where the plateau bites (physics).** Repeat Section 8 for $N=3,4,5,6$ and extract the decay rate per layer
#    for each. Combine it with the noiseless barren-plateau exponent measured in notebook 40 to estimate the largest
#    $(N,L)$ at which a gradient could still be resolved with $10^6$ shots per circuit at $p_2=10^{-3}$.
# 8. ★★★ **Mitigation inside the loop (extend the code).** Train with a *mitigated* cost: at each iteration evaluate
#    $E(1)$, $E(2)$, $E(3)$ and feed the Richardson combination to the optimiser. Compare the final noiseless energy of
#    the angles found with the unmitigated run of Section 9, and count the circuit evaluations each spent. Is mitigating
#    the *training* worth it, or is mitigating only the *final* measurement enough?
#
# ## References
#
# * M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2000) —
#   channels, the Kraus representation, the depolarising and amplitude-damping channels.
# * M. A. Nielsen, *A simple formula for the average gate fidelity of a quantum dynamical operation*,
#   Phys. Lett. A **303**, 249 (2002) — the relation between process fidelity and average gate fidelity used in Eq. (4).
# * E. Magesan, J. M. Gambetta and J. Emerson, *Characterizing quantum gates via randomized benchmarking*,
#   Phys. Rev. A **85**, 042311 (2012) — where the error per gate $r$ of Section 3.2 comes from, and why twirling makes
#   the depolarising model the right one.
# * J. Dalibard, Y. Castin and K. Mølmer, *Wave-function approach to dissipative processes in quantum optics*,
#   Phys. Rev. Lett. **68**, 580 (1992) — the trajectory unravelling of Section 4.3.
# * J. Preskill, *Quantum computing in the NISQ era and beyond*, Quantum **2**, 79 (2018) — the regime this notebook
#   simulates.
# * M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan,
#   L. Cincio and P. J. Coles, *Variational quantum algorithms*, Nat. Rev. Phys. **3**, 625 (2021) — the review,
#   including noise and trainability.
# * S. Wang, E. Fontana, M. Cerezo, K. Sharma, A. Sone, L. Cincio and P. J. Coles, *Noise-induced barren plateaus in
#   variational quantum algorithms*, Nat. Commun. **12**, 6961 (2021) — the exponential decay in depth measured in
#   Section 8.
# * K. Sharma, S. Khatri, M. Cerezo and P. J. Coles, *Noise resilience of variational quantum compiling*,
#   New J. Phys. **22**, 043006 (2020) — optimal-parameter resilience, tested in Section 10.
# * D. Stilck França and R. García-Patrón, *Limitations of optimization algorithms on noisy quantum devices*,
#   Nat. Phys. **17**, 1221 (2021) — how a noisy variational output approaches the maximally mixed state.
# * K. Temme, S. Bravyi and J. M. Gambetta, *Error mitigation for short-depth quantum circuits*,
#   Phys. Rev. Lett. **119**, 180509 (2017) — zero-noise extrapolation and probabilistic error cancellation.
# * Y. Li and S. C. Benjamin, *Efficient variational quantum simulator incorporating active error minimization*,
#   Phys. Rev. X **7**, 021050 (2017) — the independent proposal of extrapolation in the noise strength.
# * S. Endo, Z. Cai, S. C. Benjamin and X. Yuan, *Hybrid quantum-classical algorithms and quantum error mitigation*,
#   J. Phys. Soc. Jpn. **90**, 032001 (2021) — the review of mitigation techniques and their costs.
