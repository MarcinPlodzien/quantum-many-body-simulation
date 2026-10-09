#@title: Measurements — Born rule, collapse, sampling and shot noise
#@part: Chapter 3 — The matrix-free engine
#@description: Projective measurements of single spins in any Pauli basis, jit/vmap-compatible stochastic code with explicit PRNG keys, sampling bit strings, shot-noise statistics with error bars, reset and mid-circuit measurements.

# %% [markdown]
# ## 1. Introduction and motivation
#
# A simulator hands us the full wave function: $2^N$ complex amplitudes, to sixteen digits, which a laboratory never sees.
# What an experiment on $N$ spins (trapped ions, superconducting qubits, Rydberg atoms, cold atoms under a quantum-gas
# microscope) returns is a *click pattern*: one string of $N$ bits per run of the experiment — a "shot". Expectation
# values are then **estimated** by averaging over many shots and come with statistical error bars that shrink only like
# $1/\sqrt{\text{shots}}$. If we want our simulations to say something about experiments — how many shots are needed to see an
# effect, whether a variational algorithm survives its own measurement noise, how a protocol with feedback behaves —
# we have to simulate the measurement process itself.
#
# Measurement is also much more than "reading out the answer at the end":
#
# * it is the only **non-unitary, random** element of quantum mechanics — the state *changes* (collapses) when we look;
# * measuring *part* of an entangled system changes the state of the rest: this can **destroy** entanglement (a cat state dies
#   when one spin is observed) or **create** it between distant spins (measurement-based quantum computing, entanglement swapping);
# * **mid-circuit measurements with feedback** are the core of teleportation, quantum error correction, qubit reset and reuse,
#   quantum reservoir computing, and of measurement-induced phase transitions — all topics of later chapters of these notes.
#
# **What we will do.** We derive the Born rule and the collapse for one spin out of $N$ directly on the state *tensor*
# (Sections 3–4), turn them into a function `measure_qubit` that contains **no Python branching on random values** and can therefore
# be compiled with `jax.jit` and batched over thousands of shots with `jax.vmap` (Sections 5, 6 and 8), add the active `reset_qubit`
# (Section 7), study what a measurement does to the *other* spins (Section 9), sample complete bit strings
# (`sample_bitstrings`, Section 10), do **shot-noise statistics with error bars** (Section 11), look at the correlations of a
# Bell pair measured along different axes (Section 12), and finally put measurements *inside* compiled loops: the quantum
# Zeno effect and entanglement created by measurement (Section 13). Section 14 measures the cost.
#
# ### What you will learn
#
# *Physics*
# * the measurement postulate for a subsystem: Born probabilities from the reduced density matrix, collapse by projection;
# * measuring $X$, $Y$ or along any axis = rotate, measure $Z$, rotate back;
# * measurement back-action on entangled states: GHZ versus W; unread measurement = dephasing; reset as a quantum channel;
# * shot noise, standard errors, the central limit theorem at work; Bell correlations $E(\theta)=\cos\theta$; the quantum Zeno effect.
#
# *Numerical methods*
# * sampling from a discrete distribution three ways (threshold on a uniform number, Gumbel-max / `categorical`, inverse CDF) and their costs;
# * sequential (chain-rule) sampling versus joint sampling of bit strings;
# * estimators with error bars, $\chi^2$ and $z$-score sanity checks for stochastic code.
#
# *Implementation practice*
# * explicit PRNG keys: splitting, reproducibility, one key per random decision;
# * replacing `if random < p:` by `jnp.where` so that the code survives `jit`, `vmap` and `lax.scan`;
# * `vmap` over keys = many shots; `lax.scan` with keys as the scanned input = repeated measurements in time.
#
# ### Prerequisites
# * [01 — JAX](../ch01_computational_toolbox/01_jax_from_scratch.ipynb): `jit`, `vmap`, `lax.scan`, PRNG keys;
# * [05 — matrix-free operators](05_matrix_free_operators.ipynb): state tensors and `apply_gate`;
# * [06 — states, observables, entanglement](06_states_observables_entanglement.ipynb): reduced density matrices `rdm`, expectation values, entanglement entropy;
# * helpful but not required: [07 — density matrices and quantum channels](07_density_matrices_and_quantum_channels.ipynb).
#
# **Conventions**: spin/qubit $q$ = tensor axis $q$; $|0\rangle=(1,0)^T$ is the $+1$ eigenstate of $Z$ ("up"), $|1\rangle$ the $-1$ eigenstate;
# measurement **outcome $m\in\{0,1\}$ means eigenvalue $(-1)^m$**; bit strings are written $s_0s_1\dots s_{N-1}$ with spin 0 leftmost (most significant bit of the flat index).

# %% [markdown]
# ## 2. Engine recap and helpers
#
# From earlier notebooks we reuse the state constructors, `apply_gate` (one einsum that applies a small matrix to chosen axes of the
# state tensor), the reduced density matrix `rdm`, expectation values and the entanglement entropy. The measurement functions themselves
# are **not** in this recap — we build them from scratch below.

# %%
#@engine: apply_gate, rdm, I2, X, Y, Z, H, S, SDG, P0, P1, XX, ZZ, PAULI, rx, ry, zero_state, product_state, ghz_state, w_state, bell_state, cluster_state, haar_state, to_dm, dm_matrix, expect_local, expect_pauli_string, entanglement_entropy, heisenberg_terms, apply_hamiltonian, energy, lanczos_ground_state

# %%
# ==============================================================================
# PLOT STYLE + small helpers
# ==============================================================================
import itertools

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # colour-blind-friendly, fixed order
MARKERS = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=PALETTE), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "legend.frameon": False})


def kron_chain(mats):
    """Dense reference: mats[0] (x) mats[1] (x) ...  -- builds a 2^N x 2^N matrix, VALIDATION ONLY."""
    out = np.eye(1, dtype=complex)
    for m in mats:
        out = np.kron(out, np.asarray(m))
    return out


def max_abs(a):
    """Largest absolute entry, as a Python float."""
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def bits_to_str(bits):
    return "".join(str(int(b)) for b in np.asarray(bits))

# %% [markdown]
# ## 3. Theory: the measurement postulate for one spin out of $N$
#
# **The postulate** (as in your quantum mechanics course; Nielsen & Chuang, Sec. 2.2; Sakurai and Napolitano, Ch. 1). An observable is a Hermitian operator with spectral decomposition
# $O=\sum_m\lambda_m\Pi_m$, where $\Pi_m$ projects onto the eigenspace of the eigenvalue $\lambda_m$
# ($\Pi_m\Pi_{m'}=\delta_{mm'}\Pi_m$, $\sum_m\Pi_m=\mathbb 1$). Measuring $O$ on the state $|\psi\rangle$
#
# 1. gives the result $\lambda_m$ with probability $p(m)=\langle\psi|\Pi_m|\psi\rangle=\|\Pi_m|\psi\rangle\|^2$ (**Born rule**);
# 2. leaves the system in the state $|\psi_m\rangle=\Pi_m|\psi\rangle/\sqrt{p(m)}$ (**collapse**, projection postulate).
#
# **One spin out of many.** We measure $Z_q$ on spin $q$ of an $N$-spin register. The eigenvalues are $\pm1=(-1)^m$ and the projectors are
#
# $$\Pi_m=\mathbb 1\otimes\dots\otimes\underbrace{|m\rangle\langle m|}_{\text{spin }q}\otimes\dots\otimes\mathbb 1,\qquad P_0=|0\rangle\langle0|=\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad P_1=|1\rangle\langle1|=\begin{pmatrix}0&0\\0&1\end{pmatrix}.$$
#
# $\Pi_m$ is highly degenerate: it projects onto a $2^{N-1}$-dimensional subspace, so the post-measurement state is in general *still a superposition* (and may still be entangled).
# In tensor language $\Pi_m$ simply **keeps the slice $s_q=m$ of the state tensor and zeroes the other slice**:
#
# $$(\Pi_m\psi)[s_0,\dots,s_q,\dots,s_{N-1}]=\delta_{s_q,m}\;\psi[s_0,\dots,s_q,\dots,s_{N-1}].$$
#
# Hence the Born probability is the total weight of that slice,
#
# $$p(m)=\sum_{s\,:\,s_q=m}|\psi[s_0,\dots,s_{N-1}]|^2=\big(\rho_q\big)_{mm},\qquad(1)$$
#
# which we recognise as a diagonal element of the single-spin reduced density matrix $\rho_q[a,a']=\sum_{\rm rest}\psi[..a..]\psi^*[..a'..]$ of notebook 06.
# This is general: **the statistics of any measurement on a subsystem depend only on its reduced density matrix.**
#
# ### 3.1 General measurements, the Lüders rule and unread outcomes
#
# Three qualifications belong to the statement above; all three come back later in these notes.
#
# * **Projective measurements as a special case.** The postulate as written is the *projective* (von Neumann) case. The general measurement
#   postulate replaces the projectors by any family of operators $M_m$ with $\sum_mM_m^\dagger M_m=\mathbb 1$, giving
#   $p(m)=\langle\psi\vert M_m^\dagger M_m\vert\psi\rangle$ and $\vert\psi_m\rangle=M_m\vert\psi\rangle/\sqrt{p(m)}$;
#   projective measurements are the special case $M_m=\Pi_m$ with $\Pi_m^\dagger=\Pi_m=\Pi_m^2$. The $M_m$ are the Kraus
#   operators of [notebook 07](07_density_matrices_and_quantum_channels.ipynb). A photodetector that absorbs the photon it
#   counts is a measurement of this general kind and is *not* described by the projection rule above.
# * **The post-measurement state is a second assumption.** For a degenerate eigenvalue, $\vert\psi_m\rangle\propto\Pi_m\vert\psi\rangle$
#   is the *Lüders rule*: the state is disturbed as little as the outcome allows. It is what an ideal repeatable apparatus does —
#   measure twice in a row and get the same answer — and it is the rule we implement; a real apparatus can reproduce the Born
#   probabilities and still leave a different state behind.
# * **Selective versus non-selective.** The update is the state we assign *given* that the outcome $m$ has been read. If the
#   measurement happens but the result is discarded, the correct description is the average $\sum_mp(m)\vert\psi_m\rangle\langle\psi_m\vert$
#   — a density matrix. Section 9.2 computes it and identifies it with a noise channel. "Collapse" is the
#   name of this update rule; the formalism prescribes it without saying what brings it about, which is the measurement problem.
#
# ### 3.2 The algorithm
#
# The algorithm has three steps: (i) compute $p(0)$ from Eq. (1); (ii) draw the outcome $m$ at random with these probabilities; (iii) apply the $2\times2$ projector $P_m$ to axis $q$
# — this is just `apply_gate` with a non-unitary matrix — and renormalise. Cost: $O(2^N)$ per measurement, no large matrix anywhere.

# %% [markdown]
# ## 4. From formula to code, step 1: Born probabilities and collapse (deterministic parts)
#
# For $N=3$ and $q=1$ Eq. (1) reads $p(b)=\sum_{a,c}\psi[a,b,c]\psi^*[a,b,c]$: the letter $b$ is kept, $a$ and $c$ are summed. As an einsum: `"abc,abc->b"`.
# We compute the probabilities in three independent ways — slice weights (the einsum), the diagonal of the RDM, and the textbook expectation value of the
# $8\times8$ projector built with Kronecker products — and then the collapsed state, by hand with `"Bb,abc->aBc"` (the projector contracted with the middle axis), against the dense projector.

# %%
# ==============================================================================
# STEP 1: p(m) and the collapsed state for N = 3, q = 1 -- by hand, three ways
# ==============================================================================
psi3 = haar_state(jax.random.PRNGKey(3), 3)                     # a generic state: no symmetry hides a bug
v3 = np.asarray(psi3).reshape(-1)                               # the same state as a length-8 vector (dense reference)

p_slices = jnp.real(jnp.einsum("abc,abc->b", psi3, psi3.conj()))                # (i)   weight of the slices s_1 = 0, 1
p_rdm = jnp.real(jnp.diag(rdm(psi3, (1,))))                                     # (ii)  diagonal of the 1-spin RDM
Pi = [kron_chain([I2, P, I2]) for P in (P0, P1)]                                # (iii) dense 8x8 projectors 1 (x) P_m (x) 1
p_dense = np.array([np.real(v3.conj() @ Pi_m @ v3) for Pi_m in Pi])
print("p(m) from slices   :", np.asarray(p_slices))
print("p(m) from the RDM  :", np.asarray(p_rdm))
print("p(m) dense <Pi_m>  :", p_dense, "  sum =", p_dense.sum())
assert max_abs(p_slices - p_dense) < TOL and max_abs(p_rdm - p_dense) < TOL

for m, P in enumerate((P0, P1)):
    phi = jnp.einsum("Bb,abc->aBc", P, psi3)                    # projector on the middle axis (what apply_gate does)
    phi = phi / jnp.linalg.norm(phi)                            # renormalise: divide by sqrt(p(m))
    ref = Pi[m] @ v3 / np.sqrt(p_dense[m])
    err = max_abs(phi.reshape(-1) - ref)
    print(f"CHECKPOINT collapse on outcome {m}: max error vs dense projector = {err:.1e};  "
          f"weight left in the other slice = {float(jnp.sum(jnp.abs(phi[:, 1 - m, :]) ** 2)):.1e}")
    assert err < TOL

# %% [markdown]
# The three routes to $p(m)$ agree and sum to one; after the projection the "wrong" slice of the tensor is exactly empty and the state matches the dense textbook result.
# The norm of the projected tensor *before* renormalisation is $\sqrt{p(m)}$ — a useful fact: probabilities can always be read off from norms of projected states.

# %% [markdown]
# ## 5. Step 2: the random outcome — explicit keys and no Python `if`
#
# **Drawing the outcome.** A two-outcome random variable with $P(m{=}0)=p_0$ is obtained from one uniform random number $u\in[0,1)$: set $m=0$ if $u<p_0$ and $m=1$ otherwise.
#
# **Randomness in JAX** (recap of notebook 01; JAX documentation, *Pseudorandom numbers*). There is no hidden global random state. Every random function takes an explicit **key**; the same key always produces the same number.
# To get independent numbers we `split` a key into new ones and *never reuse a key*. This discipline is what makes a stochastic simulation reproducible, parallelisable
# (each shot gets its own key — no ordering issues) and compatible with `jit`.
#
# **The NumPy-style version** of a measurement translates the rule directly:
# ```python
# if u < p0:  project with P0   else:  project with P1
# ```
# It works in eager mode. But under `jax.jit` (or `vmap`) the function is *traced*: JAX runs it once with abstract placeholders ("tracers") instead of numbers in order to record the computation.
# A tracer has no value, so Python cannot decide the `if` — the cell below shows the resulting error.

# %%
# ==============================================================================
# STEP 2a: the naive version with a Python branch -- fine eagerly, breaks under jit
# ==============================================================================
def measure_z_naive(key, psi, q):
    """Z measurement of spin q with a Python `if` on the random number (DO NOT use: not jit/vmap-able)."""
    p0 = jnp.real(rdm(psi, [q])[0, 0])
    u = jax.random.uniform(key)
    if u < p0:                                   # <-- Python control flow on a random VALUE
        outcome, proj = 0, P0
    else:
        outcome, proj = 1, P1
    psi = apply_gate(psi, proj, [q])
    return outcome, psi / jnp.linalg.norm(psi)


key = jax.random.PRNGKey(42)
print("eager call works    : outcome =", measure_z_naive(key, psi3, 1)[0])
try:
    jax.jit(measure_z_naive, static_argnums=2)(key, psi3, 1)
except Exception as err:                         # jax.errors.TracerBoolConversionError
    print("under jit it fails  :", type(err).__name__)

# %% [markdown]
# **The cure: compute both options, select with arithmetic.** The outcome is the *array expression* `(u >= p0)` cast to an integer, and the projector is selected with
# `jnp.where(outcome == 0, P0, P1)` — a $2\times2$ array whose entries depend on the traced outcome. There is no branch any more: the compiled program is the same for both outcomes, only the numbers differ.
# (Alternatives are `lax.cond`/`lax.switch`; for selecting between two tiny matrices `jnp.where` is the simplest, and under `vmap` a `cond` is converted into a select anyway.)
#
# > **Common pitfall.** Renormalising never divides by zero: outcome 0 is chosen only if $u<p_0$, which requires $p_0>0$; outcome 1 only if $u\ge p_0$, which for $u<1$ requires $p_0<1$, i.e. $p_1>0$.
# > An outcome of probability zero is never selected, so the projected state never has zero norm. Both halves of the argument use the
# > half-open range $u\in[0,1)$ of `jax.random.uniform`, and both are exact-arithmetic statements: the outcome is decided with the
# > *computed* $p_0$, while the division is by the norm the projection actually produces. A probability that is zero on paper but comes
# > out of the contraction as $10^{-17}$ is therefore selected with probability $10^{-17}$ and would then be renormalised from a norm of
# > about $10^{-8}$ — still not a division by zero, but a reminder that probabilities below the working precision are not resolved.

# %%
# ==============================================================================
# STEP 2b: branch-free Z measurement -- jit- and vmap-compatible
# ==============================================================================
def my_measure_z(key, psi, q):
    """Projective Z measurement of spin q.  Returns (outcome, post-measurement state).

    MATH            p(0) = (rho_q)[0,0];  outcome m ~ {p(0), 1-p(0)};  psi -> P_m psi / ||P_m psi||
    IMPLEMENTATION  outcome = (u >= p0) as an int;  projector = jnp.where(outcome == 0, P0, P1)
    JAX             no Python branch on traced values -> works inside jit / vmap / scan. `q` is a static int.
    COST            O(2^N): one RDM einsum, one projector einsum, one norm.
    """
    p0 = jnp.real(rdm(psi, [q])[0, 0])                          # Born rule, Eq. (1)
    outcome = (jax.random.uniform(key) >= p0).astype(jnp.int32)  # 0 with probability p0, else 1
    proj = jnp.where(outcome == 0, P0, P1)                       # select the 2x2 projector WITHOUT branching
    psi = apply_gate(psi, proj, [q])                             # keep one slice of axis q, zero the other
    return outcome, psi / jnp.linalg.norm(psi)


measure_z_jit = jax.jit(my_measure_z, static_argnums=2)          # q defines the einsum string -> static argument
worst = 0.0
for seed in range(6):
    key = jax.random.PRNGKey(seed)
    m_naive, psi_naive = measure_z_naive(key, psi3, 1)
    m_jit, psi_jit = measure_z_jit(key, psi3, 1)
    ref = Pi[int(m_jit)] @ v3 / np.sqrt(p_dense[int(m_jit)])    # dense textbook collapse for this outcome
    worst = max(worst, max_abs(psi_jit.reshape(-1) - ref))
    print(f"key {seed}: naive outcome {m_naive}, jitted branch-free outcome {int(m_jit)}, "
          f"same state: {bool(max_abs(psi_naive - psi_jit) < TOL)}")
    assert int(m_jit) == m_naive
print(f"CHECKPOINT post-measurement state vs dense projector: worst error = {worst:.1e}")
assert worst < TOL

# %% [markdown]
# With the same key both versions make the same random decision and produce the same collapsed state, equal to the dense textbook projection — but only the branch-free one compiles.
# The signature `(key, psi, q) -> (outcome, new_psi)` is deliberate. The function is **pure**: it does not modify `psi` (JAX arrays are immutable anyway) and it does not advance any hidden random state.

# %% [markdown]
# ## 6. Measuring $X$, $Y$ (or any axis): rotate, measure $Z$, rotate back
#
# Suppose $P$ is an observable with eigenvalues $\pm1$ that is related to $Z$ by a unitary change of basis, $P=U^\dagger ZU$. Its projectors are $\Pi^P_m=U^\dagger P_mU$ (check: $\sum_m(-1)^mU^\dagger P_mU=U^\dagger ZU$). Then
#
# $$p(m)=\|U^\dagger P_mU|\psi\rangle\|^2=\|P_m\,(U|\psi\rangle)\|^2,\qquad|\psi_m\rangle\propto U^\dagger\,P_m\,(U|\psi\rangle).$$
#
# Read from right to left: **rotate with $U$, do an ordinary $Z$ measurement, rotate back with $U^\dagger$.** Most hardware measures this way, since most platforms can only read out in one fixed basis.
#
# Which eigenstate belongs to which outcome follows from the same line: $\Pi^P_m=U^\dagger P_mU$ projects onto $U^\dagger|m\rangle$, so
# **outcome $m$ means the eigenvector $U^\dagger|m\rangle$ of $P$, with eigenvalue $(-1)^m$.**
#
# * $X$: $U=H$ (Hadamard), because $HZH=X$ and $H^\dagger=H$. Outcome 0 $\leftrightarrow H|0\rangle=|{+}\rangle$, outcome 1 $\leftrightarrow H|1\rangle=|{-}\rangle$.
# * $Y$: $U=HS^\dagger$, because $(HS^\dagger)^\dagger Z(HS^\dagger)=S\,(HZH)\,S^\dagger=SXS^\dagger=Y$, using $S=\mathrm{diag}(1,i)$ and
#   $SXS^\dagger=\begin{pmatrix}0&-i\\i&0\end{pmatrix}$. Here $U=\tfrac1{\sqrt2}\begin{pmatrix}1&-i\\1&i\end{pmatrix}$, so outcome 0 belongs to
#   $U^\dagger|0\rangle=(|0\rangle+i|1\rangle)/\sqrt2$ (the $+1$ eigenstate of $Y$, written `"r"` in `product_state`) and outcome 1 to $(|0\rangle-i|1\rangle)/\sqrt2$ (`"l"`).
# * $Z$: $U=\mathbb 1$.

# %%
# ==============================================================================
# STEP 3: measurement in the X, Y or Z basis
# ==============================================================================
BASIS_ROT = {"Z": I2, "X": H, "Y": H @ SDG}          # U maps the eigenbasis of the Pauli onto |0>,|1>:  P = U^dag Z U
for name, U in BASIS_ROT.items():
    err = max_abs(U.conj().T @ Z @ U - PAULI[name])
    print(f"CHECKPOINT U^dag Z U = {name}: error = {err:.1e}")
    assert err < TOL


def my_measure_qubit(key, psi, q, basis="Z"):
    """Projective measurement of spin q in the eigenbasis of X, Y or Z. Returns (outcome, post-state);
    outcome 0 <-> eigenvalue +1, outcome 1 <-> eigenvalue -1."""
    U = BASIS_ROT[basis]                              # `basis` is a Python string -> static under jit
    psi = apply_gate(psi, U, [q])                     # rotate the measurement basis onto the Z basis
    outcome, psi = my_measure_z(key, psi, q)          # ordinary Z measurement
    return outcome, apply_gate(psi, U.conj().T, [q])  # rotate back

# %% [markdown]
# **Deterministic checkpoints.** A measurement is random in general, but an *eigenstate* of the measured observable must give its eigenvalue with certainty and must not be disturbed. This gives sharp tests:
# $|{+}\rangle$ measured in $X$ always yields 0, $|{-}\rangle$ always 1; the $Y$ eigenstate likewise; and for the product state $|{+}\rangle|0\rangle|1\rangle$ measured spin by spin in the bases $(X,Z,Z)$ the record must be $(0,0,1)$, every time.
# We run each test for 200 different keys at once — our first use of `jax.vmap` over keys.

# %%
# ==============================================================================
# CHECKPOINT: eigenstates give deterministic outcomes and are not disturbed (200 keys each, via vmap)
# ==============================================================================
keys = jax.random.split(jax.random.PRNGKey(0), 200)
for label, state, basis, expected in (("|+>", "+", "X", 0), ("|->", "-", "X", 1), ("|r> = (|0>+i|1>)/sqrt2", "r", "Y", 0),
                                      ("|l> = (|0>-i|1>)/sqrt2", "l", "Y", 1), ("|1>", "1", "Z", 1)):
    psi = product_state(state + "0")                  # the measured spin plus a spectator spin
    outcomes, posts = jax.vmap(lambda k: my_measure_qubit(k, psi, 0, basis))(keys)
    undisturbed = max_abs(posts - psi[None])
    print(f"{label:24s} measured in {basis}: all outcomes == {expected}: {bool(jnp.all(outcomes == expected))}, "
          f"state undisturbed: {undisturbed:.1e}")
    assert bool(jnp.all(outcomes == expected)) and undisturbed < TOL


def measure_xzz(key):
    k0, k1, k2 = jax.random.split(key, 3)             # one fresh key per random decision
    psi = product_state("+01")
    m0, psi = my_measure_qubit(k0, psi, 0, "X")
    m1, psi = my_measure_qubit(k1, psi, 1, "Z")
    m2, psi = my_measure_qubit(k2, psi, 2, "Z")
    return jnp.stack([m0, m1, m2])

records = jax.vmap(measure_xzz)(keys)
print("per-spin bases (X,Z,Z) on |+>|0>|1>: all records == (0,0,1):", bool(jnp.all(records == jnp.array([0, 0, 1]))))
assert bool(jnp.all(records == jnp.array([0, 0, 1])))

# %% [markdown]
# All outcomes are deterministic and the eigenstates come back unchanged (the rotate–project–rotate-back sequence acts as the identity on them).
#
# ## 7. Reset: measurement plus feedback, still without an `if`
#
# Many protocols need a spin to be returned to $|0\rangle$ in the middle of a computation (ancilla reuse, error correction, reservoir computing, autoencoders). **Active reset** does it by
# measuring in $Z$ and flipping the spin if the outcome was 1: the simplest example of *feedback* (a classical measurement result controls a later quantum operation). Again no Python branch is needed —
# the correction is `jnp.where(outcome == 1, X, I2)`.
#
# In each shot the *other* spins are left in the conditional state $|\psi_m\rangle$. If we do not look at the outcome (we only wanted a fresh spin), we must average over it.
# With $\rho=|\psi\rangle\langle\psi|$, the operation applied for outcome $m$ is $X^m\Pi_m$, and on spin $q$ one has $X^m|m\rangle\langle m|=|0\rangle\langle m|$. The average is therefore the **reset channel**
#
# $$\rho\;\longmapsto\;\sum_m\big(|0\rangle\langle m|\big)_q\,\rho\,\big(|m\rangle\langle0|\big)_q=|0\rangle\langle0|_q\otimes\sum_m\langle m|\rho|m\rangle_q=|0\rangle\langle0|_q\otimes\mathrm{Tr}_q\,\rho :$$
#
# the spin is replaced by a fresh $|0\rangle$, the rest keeps its reduced density matrix, and all correlations between them are erased.

# %%
# ==============================================================================
# STEP 4: active reset  =  Z measurement + conditional X
# ==============================================================================
def my_reset_qubit(key, psi, q):
    """Reset spin q to |0>: measure Z, apply X iff the outcome was 1 (selected with jnp.where, no branch)."""
    outcome, psi = my_measure_qubit(key, psi, q)
    return apply_gate(psi, jnp.where(outcome == 1, X, I2), [q])


PREP = {"0": I2, "1": X, "+": H, "-": H @ X}             # gate that maps |0> to the target state


def reset_qubit_to(key, psi, q, target="0"):
    """Reset spin q to |0>, |1>, |+> or |->:  reset to |0>, then one preparation gate."""
    return apply_gate(my_reset_qubit(key, psi, q), PREP[target], [q])


# --- CHECKPOINT 1: after reset the spin IS in the target state, in every single shot -----------------
psi_r = haar_state(jax.random.PRNGKey(8), 3)
keys = jax.random.split(jax.random.PRNGKey(1), 500)
targets = {"0": P0, "1": P1, "+": H @ P0 @ H, "-": H @ P1 @ H}           # projectors onto the target states
for target, proj in targets.items():
    posts = jax.vmap(lambda k: reset_qubit_to(k, psi_r, 1, target))(keys)
    worst = max_abs(jax.vmap(lambda p: rdm(p, (1,)))(posts) - proj[None])
    print(f"reset spin 1 of a random state to |{target}>: max |rho_1 - |{target}><{target}|| over 500 shots = {worst:.1e}")
    assert worst < 1e2 * TOL

# --- CHECKPOINT 2: the average over shots is the reset channel |0><0| (x) Tr_q rho --------------------
M = 4000
posts = jax.vmap(lambda k: my_reset_qubit(k, psi_r, 0))(jax.random.split(jax.random.PRNGKey(2), M))
rho_avg = jnp.mean(jax.vmap(lambda p: dm_matrix(to_dm(p)))(posts), axis=0)      # trajectory average of |psi><psi|
rho_exact = jnp.kron(P0, rdm(psi_r, (1, 2)))                                  # |0><0| on spin 0 (x) RDM of spins 1,2
rms_exact = np.sqrt((1 - float(jnp.real(jnp.trace(rho_exact @ rho_exact)))) / M)  # E||avg - rho'||_F^2 = (1 - Tr rho'^2)/M
err = float(jnp.linalg.norm(rho_avg - rho_exact))
err_wrong = float(jnp.linalg.norm(rho_avg - jnp.kron(rdm(psi_r, (1, 2)), P0)))  # WRONG CONTROL: tensor factors swapped
print(f"\nCHECKPOINT reset channel: ||average over {M} shots - exact channel||_F = {err:.4f} = {err / rms_exact:.2f} x the expected rms {rms_exact:.4f}")
print(f"wrong control (kron factors in the wrong order): {err_wrong:.3f} = {err_wrong / rms_exact:.0f} x the expected rms")
assert err < 4 * rms_exact and err_wrong > 10 * rms_exact

# %% [markdown]
# After the reset the spin is in the requested state in *every* shot (error at rounding level), while the shot-averaged state of the full register agrees with the reset channel up to a
# statistical error of order $1/\sqrt M$ — our first encounter with the scaling that governs all of Section 11. Each post-measurement state is pure, so $\mathbb E\|\bar\rho-\rho'\|_F^2=(1-\mathrm{Tr}\,\rho'^2)/M$ exactly;
# only two post-measurement states occur here, so the printed ratio is the $|z|$-score of the count $n_0$. The same register with the tensor factors of the reference swapped is rejected by two orders of magnitude.
#
# Here are the production versions of the two functions, verbatim from the course engine; later notebooks use these.

# %%
#@engine-show: measure_qubit, reset_qubit

# %%
# CHECKPOINT: engine versions reproduce our functions (same key -> same outcome, same state)
for seed, basis in itertools.product(range(3), "XYZ"):
    key = jax.random.PRNGKey(seed)
    m_a, p_a = measure_qubit(key, psi_r, 1, basis)
    m_b, p_b = my_measure_qubit(key, psi_r, 1, basis)
    assert int(m_a) == int(m_b) and max_abs(p_a - p_b) < TOL
    assert max_abs(reset_qubit(key, psi_r, 2) - my_reset_qubit(key, psi_r, 2)) < TOL
print("engine measure_qubit / reset_qubit agree with the hand-derived versions for all tested keys and bases.")

# %% [markdown]
# ## 8. Many shots at once: `vmap` over keys
#
# One call = one shot. For statistics we need many, and they are independent — the perfect job for `jax.vmap`: we split one master key into $M$ keys and map the single-shot function over them.
# `jax.jit` then compiles the *whole batch* into one XLA program. The batched call returns an array of $M$ outcomes and an array of $M$ post-measurement states of shape `(M, 2, ..., 2)`.
#
# **Experiment**: the 3-spin GHZ state $(|000\rangle+|111\rangle)/\sqrt2$, spin 0 measured in $Z$, 1000 shots. Expected: outcomes 0 and 1 with probability $1/2$ each; the state collapses to $|000\rangle$ or to $|111\rangle$.
# The estimate of a probability from $M$ shots, $\hat p=n_0/M$, has the binomial standard error $\sqrt{p(1-p)/M}$.

# %%
# ==============================================================================
# EXPERIMENT: 1000 shots of "measure spin 0 of GHZ_3 in Z"
# ==============================================================================
N_SHOTS = 1000
psi_ghz3 = ghz_state(3)
shot = jax.jit(jax.vmap(lambda k: measure_qubit(k, psi_ghz3, 0, "Z")))       # single-shot function -> batched, compiled
outcomes, posts = shot(jax.random.split(jax.random.PRNGKey(7), N_SHOTS))
print("outcomes:", outcomes.shape, outcomes.dtype, "| post-measurement states:", posts.shape)

p_hat = float(jnp.mean(outcomes == 0))
se = np.sqrt(p_hat * (1 - p_hat) / N_SHOTS)
print(f"P(outcome 0) = {p_hat:.3f} +- {se:.3f}   (exact 0.5; deviation = {(p_hat - 0.5) / se:+.2f} standard errors)")
assert abs(p_hat - 0.5) < 4 * se

# in EVERY shot the state must be |000> (outcome 0) or |111> (outcome 1)
fid_000 = jnp.abs(posts[:, 0, 0, 0]) ** 2
fid_111 = jnp.abs(posts[:, 1, 1, 1]) ** 2
fid = jnp.where(outcomes == 0, fid_000, fid_111)
print(f"CHECKPOINT collapse: min over shots of |<s s s|psi_post>|^2 = {float(jnp.min(fid)):.12f}")
assert float(jnp.min(fid)) > 1 - 1e2 * TOL

# %% [markdown]
# The frequency of outcome 0 is compatible with $1/2$ within the binomial error bar, and in each of the 1000 shots the register collapsed completely onto $|000\rangle$ or $|111\rangle$: reading *one* spin of a GHZ state fixes *all* of them.
#
# > **JAX practice.** Statistical checkpoints cannot use `TOL`. We compare the deviation with the *standard error* and accept up to 4 standard errors (a fluctuation that large has probability $<10^{-4}$);
# > because the keys are fixed, the notebook is nevertheless perfectly reproducible. The memory of a batched measurement is $M\times2^N$ amplitudes — for large $N$ process the keys in chunks or use `jax.lax.map`.

# %% [markdown]
# ## 9. Back-action: what one measurement does to the other spins
#
# ### 9.1 GHZ versus W, $Z$ versus $X$
#
# Take $N=6$ spins, measure spin 0, and ask how much entanglement survives *among the remaining spins* (for three qubits Dür, Vidal and Cirac (2000) showed that the W state, unlike GHZ, keeps maximal bipartite entanglement when one qubit is traced out). We quantify it by the entanglement entropy of the block $A=\{1,2\}$ (after the measurement spin 0 is in a
# product state with the rest, so this is the entanglement between $\{1,2\}$ and $\{3,4,5\}$). Pencil-and-paper expectations:
#
# * **GHZ, $Z$ basis**: collapse to $|0\dots0\rangle$ or $|1\dots1\rangle$ — a product state, $S=0$. All entanglement is destroyed.
# * **GHZ, $X$ basis**: write $|0\rangle=(|{+}\rangle+|{-}\rangle)/\sqrt2$, $|1\rangle=(|{+}\rangle-|{-}\rangle)/\sqrt2$; then GHZ$_N=\tfrac1{\sqrt2}\big[|{+}\rangle\,{\rm GHZ}^+_{N-1}+|{-}\rangle\,{\rm GHZ}^-_{N-1}\big]$ with
#   ${\rm GHZ}^\pm=(|0..0\rangle\pm|1..1\rangle)/\sqrt2$. The rest remains a GHZ state ($S=1$ bit) whose *sign depends on the outcome*.
# * **W, $Z$ basis**: with probability $1/N$ we find the excitation (outcome 1) and the rest collapses to $|0\dots0\rangle$; with probability $1-1/N$ we do not, and the rest is left in $W_{N-1}$, still entangled, with
#   $S=h(2/5)\approx0.971$ bit ($h$ = binary entropy, see notebook 06). On average $\tfrac56\times0.971\approx0.809$ bit survives.
# * **W, $X$ basis**: both outcomes have probability $1/2$ and leave the same entropy. Writing $|W_6\rangle=\tfrac1{\sqrt6}\big(|1\rangle|0^5\rangle+\sqrt5\,|0\rangle|W_5\rangle\big)$
#   and expanding spin 0 in $|{\pm}\rangle$ leaves the other five spins in $\big(\pm|0^5\rangle+\sqrt5|W_5\rangle\big)/\sqrt6$. Splitting that further as $|W_5\rangle=\big(\sqrt2|W_2\rangle|0^3\rangle+\sqrt3|0^2\rangle|W_3\rangle\big)/\sqrt5$
#   gives a reduced density matrix of the block $\{1,2\}$ that lives in the two-dimensional space spanned by $|0^2\rangle$ and $|W_2\rangle$, namely $\tfrac16\begin{pmatrix}4&\pm\sqrt2\\ \pm\sqrt2&2\end{pmatrix}$,
#   with eigenvalues $(3\pm\sqrt3)/6$ and $S=h\big((3+\sqrt3)/6\big)=0.74401$ bit.

# %%
# ==============================================================================
# EXPERIMENT: entanglement among the unmeasured spins after measuring spin 0 (N = 6, 2000 shots each)
# ==============================================================================
N, N_SHOTS, BLOCK = 6, 2000, (1, 2)
keys = jax.random.split(jax.random.PRNGKey(5), N_SHOTS)


def shot_entropy(key, psi, basis):
    outcome, post = measure_qubit(key, psi, 0, basis)
    return outcome, entanglement_entropy(post, BLOCK)


print(f"{'state':6s} {'basis':>5s} {'S before':>9s} {'P(outcome 0)':>16s} {'S | outcome 0':>14s} {'S | outcome 1':>14s} {'mean S after':>13s}")
results = {}
for name, psi in (("GHZ", ghz_state(N)), ("W", w_state(N))):
    for basis in "ZX":
        outcomes, S = jax.jit(jax.vmap(partial(shot_entropy, psi=psi, basis=basis)))(keys)
        outcomes, S = np.asarray(outcomes), np.asarray(S)
        p0 = np.mean(outcomes == 0); se = np.sqrt(p0 * (1 - p0) / N_SHOTS)
        S0, S1 = S[outcomes == 0].mean(), S[outcomes == 1].mean()
        results[(name, basis)] = (p0, se, S0, S1, S.mean())
        print(f"{name:6s} {basis:>5s} {float(entanglement_entropy(psi, BLOCK)):9.4f} {p0:9.3f} +- {se:.3f} {S0:14.4f} {S1:14.4f} {S.mean():13.4f}")

h = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x)
p0, se, S0, S1, _ = results[("W", "Z")]
print(f"\nCHECKPOINT W/Z: P(0) = {p0:.3f} vs 1-1/N = {1 - 1 / N:.3f} ({(p0 - (1 - 1 / N)) / se:+.2f} s.e.);  S|0 = {S0:.6f} vs h(2/5) = {h(0.4):.6f};  S|1 = {S1:.1e}")
assert abs(p0 - (1 - 1 / N)) < 4 * se and abs(S0 - h(0.4)) < 1e3 * TOL and S1 < 1e3 * TOL
assert results[("GHZ", "Z")][4] < 1e3 * TOL and abs(results[("GHZ", "X")][4] - 1) < 1e3 * TOL

# %% [markdown]
# The entropies confirm the four predictions to the printed digits, and the outcome frequencies agree with them within shot noise (the largest deviation, $-2.1$ SE, is for W in $Z$;
# the same 2000 keys serve all four rows, which is why the three rows with $P(0)=1/2$ print the same $0.490$). A $Z$ measurement of a single spin wipes out the GHZ entanglement completely, whereas the W state keeps about $0.8$ of its $0.92$ bits on average (and the full $0.97$ bits of $W_5$ in five shots out of six);
# in the $X$ basis the W state is left with $0.7440$ bit in both branches, which is $h\big((3+\sqrt3)/6\big)$ to fourteen digits.
# This is the precise sense in which **GHZ entanglement is fragile and W entanglement is robust**. It also shows that the damage depends on the *basis*: measured in $X$, the GHZ state survives as a smaller GHZ state —
# but with an outcome-dependent sign, so the outcome must be recorded to make use of it.
#
# ### 9.2 An unread measurement is dephasing
#
# If the measurement happens but nobody reads the result (the "observer" may be the environment), the register is described by the average over outcomes,
#
# $$\rho'=\sum_mp(m)|\psi_m\rangle\langle\psi_m|=\sum_m\Pi_m\,|\psi\rangle\langle\psi|\,\Pi_m .$$
#
# The derivation is one line ($p(m)|\psi_m\rangle\langle\psi_m|=\Pi_m|\psi\rangle\langle\psi|\Pi_m$ by the collapse rule). In the density matrix this *deletes all matrix elements between the sectors $s_q=0$ and $s_q=1$* and leaves the rest untouched: complete
# **dephasing** of spin $q$, one of the noise channels of notebook 07, so an unread measurement and decoherence by an environment are the same map. The same fact underlies the principle of implicit measurement (Nielsen & Chuang, Sec. 4.4): an unread measurement of one qubit does not change the reduced state of the others. We verify it by averaging projectors over shots; the exact channel is two `apply_gate` calls per term on the density *tensor*
# ($\Pi_m$ on the ket axis $q$, $\Pi_m^*$ on the bra axis $N+q$). The size of the statistical error is known exactly: every post-measurement state is pure, $\||\psi\rangle\langle\psi|\|_F=1$, so the
# average $\bar\rho$ of $M$ of them satisfies $\mathbb E\|\bar\rho-\rho'\|_F^2=(1-\mathrm{Tr}\,\rho'^2)/M$.

# %%
# ==============================================================================
# CHECKPOINT: average over outcomes = sum_m Pi_m rho Pi_m, with the 1/sqrt(M) convergence
# ==============================================================================
N, q = 3, 1
psi = haar_state(jax.random.PRNGKey(12), N)
rho = to_dm(psi)                                                              # density tensor, rank 2N
rho_exact = sum(apply_gate(apply_gate(rho, P, [q]), jnp.conj(P), [N + q]) for P in (P0, P1))
rho_exact = dm_matrix(rho_exact)

R = 20                                                                        # independent repetitions per M


def trajectory_error(key, M):
    """||(1/M) sum_shots |psi_post><psi_post|  -  exact dephased rho||_F  for one batch of M shots."""
    posts = jax.vmap(lambda k: measure_qubit(k, psi, q)[1])(jax.random.split(key, M))
    rho_avg = jnp.mean(jax.vmap(lambda p: dm_matrix(to_dm(p)))(posts), axis=0)
    return jnp.linalg.norm(rho_avg - rho_exact)


print(f"{'shots M':>8s} {'rms error over ' + str(R) + ' repetitions':>32s} {'rms * sqrt(M)':>14s}")
c_exact = np.sqrt(1 - float(jnp.real(jnp.trace(rho_exact @ rho_exact))))          # sqrt(1 - Tr rho'^2): the exact rms * sqrt(M)
pooled = []
for M in (100, 1000, 10000):
    # lax.map = sequential map over the R repetition keys (keeps memory at ONE batch of M shots; vmap would need R of them)
    errs = jax.jit(lambda ks: lax.map(partial(trajectory_error, M=M), ks))(jax.random.split(jax.random.PRNGKey(M), R))
    rms = float(jnp.sqrt(jnp.mean(errs ** 2)))
    print(f"{M:8d} {rms:32.5f} {rms * np.sqrt(M):14.3f}")
    pooled.append(rms ** 2 * M)
ratio_c = np.sqrt(np.mean(pooled)) / c_exact                                      # 60 squared errors: relative SE 1/sqrt(2*60)
print(f"exact sqrt(1 - Tr rho'^2) = {c_exact:.3f};  pooled rms * sqrt(M) / exact = {ratio_c:.3f} +- {1 / np.sqrt(2 * 3 * R):.3f}")
assert abs(ratio_c - 1) < 4 / np.sqrt(2 * 3 * R)
coh_before = max_abs(dm_matrix(rho).reshape((2,) * 6)[:, 0, :, :, 1, :])
coh_after = max_abs(rho_exact.reshape((2,) * 6)[:, 0, :, :, 1, :])
print(f"\nlargest coherence between the sectors s_1=0 and s_1=1:  before {coh_before:.3f}  ->  after {coh_after:.1e}")

# %% [markdown]
# The trajectory average converges to the dephased density matrix: the rms error (over 20 independent repetitions) falls by roughly $\sqrt{10}\approx3.2$ for every tenfold increase of $M$ — measured factors $2.95$ and $3.59$ — and the product
# rms$\,\times\sqrt M$ scatters by the $1/\sqrt{2\cdot20}\approx16\,\%$ that 20 repetitions allow around the exact value $\sqrt{1-\mathrm{Tr}\,\rho'^2}=0.697$; pooled over all 60 runs it agrees with it to $0.5\,\%$. The coherences between the two measurement sectors are gone.
# The same idea — *average of random pure-state trajectories = deterministic evolution of a density matrix* — is the basis of the Monte-Carlo wave-function method for open systems
# ([notebook 17, Chapter 6](../ch06_open_quantum_systems/17_monte_carlo_wave_function.ipynb)).

# %% [markdown]
# ## 10. Measuring all spins: sampling bit strings
#
# ### 10.1 Two equivalent procedures
#
# The standard final readout measures *every* spin in $Z$ and yields a bit string $s=s_0s_1\dots s_{N-1}$.
#
# **Sequential (what the postulate says).** Measure spin 0, collapse, measure spin 1 on the collapsed state, and so on. The probability of the full record is a product of conditional probabilities,
#
# $$p(s_0,s_1,\dots)=p(s_0)\,p(s_1|s_0)\,p(s_2|s_0s_1)\cdots,$$
#
# and it telescopes: after the first collapse the state is $\Pi_{s_0}\psi/\sqrt{p(s_0)}$, so $p(s_1|s_0)=\|\Pi_{s_1}\Pi_{s_0}\psi\|^2/p(s_0)$, etc.; all denominators cancel and
#
# $$p(s)=\|\Pi_{s_{N-1}}\cdots\Pi_{s_0}\psi\|^2=|\psi[s_0,\dots,s_{N-1}]|^2 .$$
#
# The order of the measurements does not matter (the projectors commute). Cost per shot: $N$ measurements of $O(2^N)$ each.
#
# **Joint (what a simulator can do).** Since we *know* all $2^N$ probabilities $|\psi_s|^2$, we can draw the flat index $i\in\{0,\dots,2^N-1\}$ directly from this distribution and unpack its bits,
# $s_q=(i\gg(N-1-q))\,\&\,1$ (shift and mask; spin 0 is the most significant bit). No collapse bookkeeping — and no post-measurement state, which for a complete readout is just $|s\rangle$ anyway.
# Other bases: rotate every spin first with its $U$ from Section 6 (a string like `"XZY"`).
#
# **How to draw from a discrete distribution in JAX.** `jax.random.categorical(key, logits, shape=(shots,))` takes *log*-probabilities and uses the Gumbel-max trick: it adds independent Gumbel noise (Gumbel 1958) $g_i$ to every
# $\log p_i$ and returns $\arg\max_i(\log p_i+g_i)$, which is distributed exactly according to $p$. It is simple and parallel, at a cost of **$2^N$ random numbers per shot**. We clip $p$ before the logarithm because $\log0=-\infty$.

# %%
# ==============================================================================
# STEP 5: the two samplers, by hand
# ==============================================================================
def measure_all_sequential(key, psi, bases=None):
    """Measure every spin one after another (chain rule). Returns (bits[N], final collapsed state).
    `bases`: optional string like 'XZY' (one letter per spin).  Cost O(N 2^N) per shot."""
    N = psi.ndim
    keys = jax.random.split(key, N)                                # one independent key per measurement
    bits = []
    for q in range(N):                                            # Python loop over STATIC spin labels: unrolled by jit
        m, psi = measure_qubit(keys[q], psi, q, "Z" if bases is None else bases[q])
        bits.append(m)
    return jnp.stack(bits), psi


def my_sample_bitstrings(key, psi, shots):
    """Draw `shots` bit strings from p(s) = |psi[s]|^2 (joint sampling, Z basis). Returns int array (shots, N)."""
    N = psi.ndim
    probs = jnp.abs(psi.reshape(-1)) ** 2                          # all 2^N Born probabilities
    logp = jnp.log(jnp.clip(probs, 1e-300, None))                  # categorical wants LOG-probabilities
    idx = jax.random.categorical(key, logp, shape=(shots,))        # flat indices i ~ p  (Gumbel-max trick)
    return (idx[:, None] >> jnp.arange(N - 1, -1, -1)[None, :]) & 1   # unpack bits, spin 0 = most significant


# --- a first look: GHZ_4 can only give 0000 or 1111 ------------------------------------------------
bits = my_sample_bitstrings(jax.random.PRNGKey(0), ghz_state(4), 1000)
strings, counts = np.unique([bits_to_str(b) for b in np.asarray(bits)], return_counts=True)   # to NumPy first: one transfer
print("GHZ_4, 1000 shots, joint sampling :", {str(a): int(n) for a, n in zip(strings, counts)})
bits_seq, finals = jax.jit(jax.vmap(lambda k: measure_all_sequential(k, ghz_state(4))))(jax.random.split(jax.random.PRNGKey(1), 1000))
strings, counts = np.unique([bits_to_str(b) for b in np.asarray(bits_seq)], return_counts=True)
print("GHZ_4, 1000 shots, sequential     :", {str(a): int(n) for a, n in zip(strings, counts)})
assert set(strings) <= {"0000", "1111"}

# the sequentially collapsed state is the basis state |s> (up to a phase)
flat = jnp.sum(bits_seq * 2 ** jnp.arange(3, -1, -1), axis=1)
overlap = jnp.abs(finals.reshape(1000, -1)[jnp.arange(1000), flat])
print(f"CHECKPOINT sequential readout ends in the basis state |s>: min |<s|psi_final>| = {float(jnp.min(overlap)):.12f}")
assert float(jnp.min(overlap)) > 1 - 1e2 * TOL

# %% [markdown]
# Both samplers return only `0000` and `1111`, roughly half and half, and the sequentially collapsed state is exactly the basis state named by the record. For a quantitative test we need a state with a non-trivial distribution and a
# proper statistical criterion (Press *et al.* 2007, §14.3). For $M$ shots distributed over $K$ bins with expected counts $Mp_s$, Pearson's
#
# $$\chi^2=\sum_s\frac{(n_s-Mp_s)^2}{Mp_s}$$
#
# fluctuates around $K-1$ with standard deviation $\sqrt{2(K-1)}$ if — and only if — the sampler draws from the right distribution. The number of degrees of freedom is $K-1$ and not $K$ because the counts obey the one constraint $\sum_sn_s=M$;
# it would drop further only if we had *fitted* parameters of $p$ to the same data, which we have not — the $p_s$ come from the exact state.
#
# > **Numerical practice.** The $\chi^2$ statistic follows the $\chi^2_{K-1}$ distribution only if every bin is well populated; the standard rule of thumb is an expected count $Mp_s\gtrsim5$ in each bin. Here $K=16$, $M=20000$ and the
# > smallest exact probability is $0.0126$, so the smallest expected count is $252$ — comfortably in the regime where the test is meaningful. For a state with many nearly empty bins (a GHZ state, say) the bins must be merged first.

# %%
# ==============================================================================
# CHECKPOINT: both samplers reproduce |psi_s|^2 for a random 4-spin state (chi-squared test + figure)
# ==============================================================================
N, M = 4, 20000
psi = haar_state(jax.random.PRNGKey(4), N)
p_exact = np.abs(np.asarray(psi).reshape(-1)) ** 2


def histogram(bits):
    flat = np.asarray(bits) @ (2 ** np.arange(bits.shape[1] - 1, -1, -1))
    return np.bincount(flat, minlength=2 ** bits.shape[1])


n_joint = histogram(my_sample_bitstrings(jax.random.PRNGKey(10), psi, M))
n_seq = histogram(jax.jit(jax.vmap(lambda k: measure_all_sequential(k, psi)[0]))(jax.random.split(jax.random.PRNGKey(11), M)))
dof = 2 ** N - 1
for label, n in (("joint (categorical)", n_joint), ("sequential collapse", n_seq)):
    chi2 = np.sum((n - M * p_exact) ** 2 / (M * p_exact))
    print(f"{label:20s}: chi^2 = {chi2:6.2f}   (expected {dof} +- {np.sqrt(2 * dof):.1f})")
    assert chi2 < dof + 5 * np.sqrt(2 * dof)
n_rev = histogram(np.asarray(my_sample_bitstrings(jax.random.PRNGKey(10), psi, M))[:, ::-1])   # WRONG CONTROL: spin 0 = least significant bit
chi2_rev = np.sum((n_rev - M * p_exact) ** 2 / (M * p_exact))
print(f"wrong control, bit order reversed: chi^2 = {chi2_rev:.0f}")
assert chi2_rev > dof + 20 * np.sqrt(2 * dof)

labels = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=N)]
x = np.arange(2 ** N)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(x, p_exact, width=0.8, color="0.85", edgecolor="0.5", label=r"exact $|\psi_s|^2$")
ax.errorbar(x - 0.15, n_joint / M, yerr=np.sqrt(p_exact * (1 - p_exact) / M), fmt="o", ms=4, capsize=2, color=PALETTE[0], label="joint sampling")
ax.errorbar(x + 0.15, n_seq / M, yerr=np.sqrt(p_exact * (1 - p_exact) / M), fmt="s", ms=4, capsize=2, color=PALETTE[1], label="sequential collapse")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_xlabel(r"bit string $s_0s_1s_2s_3$"); ax.set_ylabel("probability / frequency")
ax.set_title(f"Sampling a random 4-spin state, {M} shots (error bars: binomial standard error)"); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# Both $\chi^2$ values lie in the expected range $15\pm5.5$, while the same joint samples read with the bit order reversed (spin 0 as the least significant bit, a common bug) are rejected by orders of magnitude. In the figure the sampled frequencies scatter around the exact probabilities by about one error bar, as they should. The two procedures are statistically equivalent;
# they differ in cost (Section 14) and in what they return (the sequential one also gives the collapsed state, which matters when only *some* spins are measured).
#
# The production version, with the optional per-spin bases:

# %%
#@engine-show: sample_bitstrings

# %% [markdown]
# ### 10.2 A cheaper sampler for many shots: the inverse CDF
#
# The Gumbel-max trick spends $2^N$ random numbers *per shot*: for $N=12$ and $10^4$ shots that is $4\times10^7$ numbers and as many logarithms, and the memory grows as shots $\times\,2^N$.
# The classic alternative is **inverse-transform sampling** (Devroye 1986; Press *et al.* 2007, §7.3): compute the cumulative distribution $c_i=\sum_{j\le i}p_j$ once ($O(2^N)$), draw *one* uniform number $u$ per shot and find the first index with $c_i>u$
# by binary search (`jnp.searchsorted`, $O(N)$ per shot). Picture the interval $[0,1)$ cut into $2^N$ segments of lengths $p_i$: a uniformly thrown dart lands in segment $i$ with probability $p_i$.
# We write it with an engine-style docstring and use it whenever we need very many shots.

# %%
# ==============================================================================
# STEP 6: inverse-CDF sampler -- O(2^N + shots * N) instead of O(shots * 2^N)
# ==============================================================================
def sample_bitstrings_cdf(key, psi, shots, bases=None):
    """Sample `shots` bit strings from |psi[s]|^2 by inverse-transform sampling.

    MATH            c_i = sum_{j<=i} p_j ;  u ~ U[0,1) ;  i = #{j : c_j <= u}   =>   P(i) = p_i
                    (p_i after rotating spin q with U of bases[q], Section 6, if `bases` is given)
    IMPLEMENTATION  cumsum once, one uniform number per shot, binary search (`searchsorted`, side='right' so that
                    states with p_i = 0 are never returned); u is scaled by c_last to absorb rounding in the cumsum.
    COST            O(2^N + shots*N) time and memory  (Gumbel-max `categorical`: O(shots * 2^N) both).
    JAX             `shots` fixes an array shape -> static under jit; psi and key are traced.
    """
    N = psi.ndim
    if bases is not None:
        for q, b in enumerate(bases):
            psi = apply_gate(psi, BASIS_ROT[b], [q])
    cdf = jnp.cumsum(jnp.abs(psi.reshape(-1)) ** 2)
    u = jax.random.uniform(key, (shots,), dtype=cdf.dtype) * cdf[-1]
    idx = jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, 2 ** N - 1)
    return (idx[:, None] >> jnp.arange(N - 1, -1, -1)[None, :]) & 1


n_cdf = histogram(sample_bitstrings_cdf(jax.random.PRNGKey(12), psi, M))
chi2 = np.sum((n_cdf - M * p_exact) ** 2 / (M * p_exact))
print(f"inverse-CDF sampler : chi^2 = {chi2:6.2f}   (expected {dof} +- {np.sqrt(2 * dof):.1f})")
assert chi2 < dof + 5 * np.sqrt(2 * dof)

# deterministic test with per-spin bases: |+>|0>|1>|r> read out in the bases X,Z,Z,Y must always give 0010
for sampler in (sample_bitstrings, sample_bitstrings_cdf):
    rec = sampler(jax.random.PRNGKey(3), product_state("+01r"), 500, bases="XZZY")
    print(f"{sampler.__name__:22s} on |+>|0>|1>|r> in bases XZZY: all records == 0010: {bool(jnp.all(rec == jnp.array([0, 0, 1, 0])))}")
    assert bool(jnp.all(rec == jnp.array([0, 0, 1, 0])))

# GHZ has zero-probability strings: they must never appear
rec = sample_bitstrings_cdf(jax.random.PRNGKey(5), ghz_state(10), 5000)
print("GHZ_10, 5000 shots (CDF sampler): distinct records =", sorted(set(bits_to_str(b) for b in np.asarray(rec))))

# %% [markdown]
# The inverse-CDF sampler passes the same $\chi^2$ test, handles per-spin bases, and returns no string of probability zero: if $p_i=0$ then $c_{i-1}=c_i$, the interval that would select $i$ is empty, and `side="right"`
# makes the search land past it. The one loophole is the final `clip`, which is reached only when the rounded product $u\cdot c_{\rm last}$ comes out equal to $c_{\rm last}$ — a tie of probability $\lesssim2^{-53}$ per shot,
# and the price for being safe against the cumsum ending a rounding unit below 1.
#
# > **Physics insight.** Sampling *without collapsing* is a privilege of simulators: in the laboratory every shot destroys the state, which must be prepared again. In our functional code the state
# > tensor `psi` is never modified by any of these functions, so "measuring" costs nothing in preparation — but keep the distinction in mind when you count experimental resources: **one shot = one state preparation**.

# %% [markdown]
# ## 11. Shot noise: estimating expectation values with error bars
#
# ### 11.1 Estimators
#
# Each shot in the $Z$ basis gives a string $s$; convert bits to eigenvalues $z_q=1-2s_q=\pm1$. Then
#
# $$\widehat{\langle Z_q\rangle}=\frac1M\sum_{k=1}^Mz_q^{(k)},\qquad\widehat{\langle Z_iZ_j\rangle}=\frac1M\sum_kz_i^{(k)}z_j^{(k)} .$$
#
# Both are sample means of a random variable $x=\pm1$ with mean $\mu=\langle O\rangle$ and variance $\sigma^2=\langle x^2\rangle-\mu^2=1-\langle O\rangle^2$. By the central limit theorem the estimate is
# approximately Gaussian around the exact value with **standard error**
#
# $$\mathrm{SE}=\frac{\sigma}{\sqrt M}=\sqrt{\frac{1-\langle O\rangle^2}{M}}\;\approx\;\frac{\text{sample std}}{\sqrt M}.$$
#
# Three practical consequences: (i) one more digit costs $100\times$ more shots; (ii) the closer $|\langle O\rangle|$ is to 1, the smaller the noise (an eigenstate gives no noise at all); (iii) an estimate is reported together with its standard error.
# $X$- or $Y$-type observables need shots taken in another basis setting (`bases="XX..X"`) — **each measurement setting costs its own shots**, because $X_q$ and $Z_q$ cannot be read in the same run.
#
# ### 11.2 Experiment: measuring the energy of a ground state the way a laboratory would
#
# As a physically meaningful test state we take the ground state of the critical transverse-field Ising chain, $H=-J\sum_iZ_iZ_{i+1}-h\sum_iX_i$ with $J=h=1$ and $N=8$ (computed by the Lanczos
# method of the engine, used as a black box and checked via the residual $\|H\psi-E\psi\|$). Its energy needs two settings: all spins in $Z$ for the bond terms, all spins in $X$ for the field terms.
# For every shot we form the *single-shot energy contribution* $e^{ZZ}=-J\sum_iz_iz_{i+1}$ (or $e^X=-h\sum_ix_i$) and average afterwards. Summing first and averaging second is what makes the error bar right: the variance of a sum is
#
# $$\mathrm{Var}\Big(\sum_ie_i\Big)=\sum_i\mathrm{Var}(e_i)+\sum_{i\ne j}\mathrm{Cov}(e_i,e_j),$$
#
# and the bond terms $z_iz_{i+1}$ and $z_{i+1}z_{i+2}$ measured in the *same* shot are strongly correlated. For this ground state the covariances are positive and raise the variance of $e^{ZZ}$ by a factor $1.30$ over the sum of the individual
# variances, so estimating each bond separately and adding the standard errors in quadrature would report an error bar $\sqrt{1.30}=1.14$ times ($12\,\%$) too small; the cell below shows this on the samples. The sample standard deviation of the single-shot sum contains the covariances automatically.
# Across the two settings the situation is different: they use different keys and different shots, so their covariance really is
# zero and *their* errors do add in quadrature.

# %%
# ==============================================================================
# PARAMETERS
# ==============================================================================
N = 8                   # chain length
J_ISING, H_FIELD = 1.0, 1.0
N_SHOTS = 4000          # shots PER measurement setting

terms = heisenberg_terms(N, Jxx=0.0, Jyy=0.0, Jzz=-J_ISING, hx=-H_FIELD)
E0, psi_gs = lanczos_ground_state(terms, N, m=60, restarts=2)
residual = float(jnp.linalg.norm(apply_hamiltonian(terms, psi_gs) - E0 * psi_gs))
print(f"ground state: E0 = {E0:.8f},  residual |H psi - E psi| = {residual:.1e}")
assert residual < 1e4 * TOL


def mean_and_se(x):
    """Sample mean and its standard error  std/sqrt(M)  (ddof=1: unbiased variance estimate)."""
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


key_z, key_x = jax.random.split(jax.random.PRNGKey(2024))
z = 1 - 2 * sample_bitstrings_cdf(key_z, psi_gs, N_SHOTS)                       # setting 1: all spins in Z   -> (M, N) of +-1
x = 1 - 2 * sample_bitstrings_cdf(key_x, psi_gs, N_SHOTS, bases="X" * N)       # setting 2: all spins in X

print(f"\n{'observable':>12s} {'estimate':>10s} {'std.err.':>9s} {'exact':>10s} {'deviation/SE':>13s}")
rows = [("<Z_3>", z[:, 3], expect_local(psi_gs, Z, (3,))), ("<X_3>", x[:, 3], expect_local(psi_gs, X, (3,))),
        ("<Z_3 Z_4>", z[:, 3] * z[:, 4], expect_local(psi_gs, ZZ, (3, 4))), ("<Z_0 Z_7>", z[:, 0] * z[:, 7], expect_local(psi_gs, ZZ, (0, 7))),
        ("<X_3 X_4>", x[:, 3] * x[:, 4], expect_local(psi_gs, XX, (3, 4)))]
for label, samples, exact in rows:
    est, se = mean_and_se(samples)
    print(f"{label:>12s} {est:+10.4f} {se:9.4f} {float(exact):+10.4f} {(est - float(exact)) / max(se, 1e-12):+13.2f}")
    assert abs(est - float(exact)) < 4 * se + 1e-12

e_zz = -J_ISING * jnp.sum(z[:, :-1] * z[:, 1:], axis=1)                          # single-shot bond energy (open chain)
e_x = -H_FIELD * jnp.sum(x, axis=1)                                              # single-shot field energy
(E_zz, se_zz), (E_x, se_x) = mean_and_se(e_zz), mean_and_se(e_x)
E_est, E_se = E_zz + E_x, np.sqrt(se_zz ** 2 + se_x ** 2)
se_bonds = np.sqrt(sum(mean_and_se(-J_ISING * z[:, i] * z[:, i + 1])[1] ** 2 for i in range(N - 1)))   # WRONG: covariances dropped
print(f"bond energy: SE of the single-shot sum = {se_zz:.4f};  per-bond SEs added in quadrature = {se_bonds:.4f}  (ratio {se_zz / se_bonds:.2f})")
print(f"\nenergy from 2 x {N_SHOTS} shots: E = {E_est:.3f} +- {E_se:.3f}    exact E0 = {E0:.3f}    deviation = {(E_est - E0) / E_se:+.2f} SE")
assert abs(E_est - E0) < 4 * E_se

# %% [markdown]
# Every estimate agrees with its exact value within a few standard errors (the `assert`s allow four). By the spin-flip symmetry of the Hamiltonian the exact $\langle Z_3\rangle$ is 0, and its estimate is zero only within the error bar.
# The energy comes out with a relative error of a fraction of a percent after $8000$ state preparations. Because of the $1/\sqrt M$ law, a variational algorithm that needs the energy with a standard error of $10^{-2}$ at every optimisation step would need about 30 times more shots, and $10^{-3}$ about 3000 times more: this measurement overhead is a central practical issue of near-term quantum algorithms ([notebook 41, Chapter 11](../ch11_variational_quantum_circuits/41_optimizers.ipynb)).
#
# ### 11.3 The $1/\sqrt M$ law and the central limit theorem over $R=400$ repetitions
#
# To *see* the statistics we repeat the whole experiment $R=400$ times for each number of shots $M$ — `vmap` over $R$ independent keys — and look at the distribution of the estimates of $\langle Z_3Z_4\rangle$:
# its root-mean-square error against the exact value should follow $\sigma/\sqrt M$ with $\sigma=\sqrt{1-\langle Z_3Z_4\rangle^2}$ (no fit parameter), its histogram should be Gaussian, and the error bars
# should "cover" the truth in about 68 % of the repetitions (one standard error) and 95 % (two).

# %%
# ==============================================================================
# EXPERIMENT: R independent repetitions for each shot number M  (vmap over repetition keys)
# ==============================================================================
R = 400
SHOT_LIST = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
exact = float(expect_local(psi_gs, ZZ, (3, 4)))
sigma = np.sqrt(1 - exact ** 2)


def one_experiment(key, shots):
    """One complete experiment: `shots` shots -> (estimate of <Z_3 Z_4>, its standard error)."""
    zz = 1 - 2 * sample_bitstrings_cdf(key, psi_gs, shots)
    v = (zz[:, 3] * zz[:, 4]).astype(RDTYPE)
    return jnp.mean(v), jnp.std(v, ddof=1) / jnp.sqrt(shots)


rms, cover1, cover2, estimates = [], [], [], {}
for i, M in enumerate(SHOT_LIST):
    rep_keys = jax.random.split(jax.random.PRNGKey(1000 + i), R)
    est, se = jax.jit(jax.vmap(partial(one_experiment, shots=M)))(rep_keys)       # shots is static: one compilation per M
    est, se = np.asarray(est), np.asarray(se)
    estimates[M] = est
    rms.append(np.sqrt(np.mean((est - exact) ** 2)))
    cover1.append(np.mean(np.abs(est - exact) < se)); cover2.append(np.mean(np.abs(est - exact) < 2 * se))
print(f"exact <Z_3 Z_4> = {exact:.5f},  single-shot sigma = {sigma:.4f}")
print(f"{'M':>6s} {'rms error':>10s} {'sigma/sqrt(M)':>14s} {'ratio':>6s} {'within 1 SE':>12s} {'within 2 SE':>12s}")
for M, r_, c1, c2 in zip(SHOT_LIST, rms, cover1, cover2):
    print(f"{M:6d} {r_:10.5f} {sigma / np.sqrt(M):14.5f} {r_ / (sigma / np.sqrt(M)):6.2f} {c1:12.2f} {c2:12.2f}")
ratio = np.array(rms) / (sigma / np.sqrt(np.array(SHOT_LIST)))
assert np.all(np.abs(ratio - 1) < 0.15), "rms error deviates from sigma/sqrt(M) by more than 15%"
q_pool = np.mean([np.mean((estimates[M] - exact) ** 2) * M for M in SHOT_LIST])     # pooled (error / (1/sqrt(M)))^2
q_se = np.sqrt(2 / (R * len(SHOT_LIST)))
print(f"pooled over {R * len(SHOT_LIST)} experiments: rms/(sigma/sqrt M) = {np.sqrt(q_pool) / sigma:.3f};  (mean sq. error)/(sigma^2/M) = {q_pool / sigma ** 2:.3f} +- {q_se:.3f}"
      f"  ->  {(q_pool / sigma ** 2 - 1) / q_se:+.1f} SE;  wrong control sigma = 1: {(q_pool - 1) / q_se:+.1f} SE")
assert abs(q_pool / sigma ** 2 - 1) < 4 * q_se and abs(q_pool - 1) > 4 * q_se

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.loglog(SHOT_LIST, rms, "o", color=PALETTE[0], ms=7, label=f"rms error over {R} repetitions")
ax.loglog(SHOT_LIST, sigma / np.sqrt(SHOT_LIST), "-", color=PALETTE[1], label=r"prediction $\sqrt{(1-\langle ZZ\rangle^2)/M}$  (slope $-1/2$)")
ax.set_xlabel("shots $M$"); ax.set_ylabel(r"error of $\widehat{\langle Z_3Z_4\rangle}$"); ax.set_title("(a) shot noise decreases as $1/\\sqrt{M}$"); ax.legend(fontsize=8)

ax = axes[1]
M_show = 1000
edges = -1 - 1 / M_show + (8 / M_show) * np.arange(2 * M_show)          # estimates live on a lattice of spacing 2/M: 4 lattice points per bin
edges = edges[(edges > estimates[M_show].min() - 8 / M_show) & (edges < estimates[M_show].max() + 8 / M_show)]
ax.hist(estimates[M_show], bins=edges, density=True, color=PALETTE[0], alpha=0.6, label=f"{R} experiments with M = {M_show}")
grid = np.linspace(estimates[M_show].min(), estimates[M_show].max(), 200)
s_ = sigma / np.sqrt(M_show)
ax.plot(grid, np.exp(-(grid - exact) ** 2 / (2 * s_ ** 2)) / np.sqrt(2 * np.pi * s_ ** 2), "-", color=PALETTE[1], lw=2, label=r"Gaussian, width $\sigma/\sqrt{M}$")
ax.axvline(exact, color="k", ls=":", label="exact value")
ax.set_xlabel(r"estimate of $\langle Z_3Z_4\rangle$"); ax.set_ylabel("probability density"); ax.set_title("(b) central limit theorem"); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# **(a)** Over three and a half decades of $M$ the measured rms error sits on the parameter-free line $\sigma/\sqrt M$ (ratios close to 1 in the table; an rms from 400 repetitions has a relative uncertainty of $1/\sqrt{800}=3.5\,\%$, so the $1.10$ at $M=30000$ is a $2.8\sigma$ fluctuation, and the pooled test below the table uses all $3200$ experiments at once).
# Misreading the single-shot variance as 1, the bound for a $\pm1$ variable, would predict errors $1/0.8155=1.23$ times larger; the pooled test passes the correct $\sigma$ at $+0.8$ SE and rejects $\sigma=1$ at $-12.8$ SE.
# **(b)** The histogram of the estimates follows the Gaussian of that width centred on the exact value. The coverage columns show that the *estimated* error bars cover the exact value at the stated rate: about two thirds of the experiments land within one standard error, about 95 % within two.
# The one-standard-error column sits a little below the Gaussian $68.3\,\%$, as it should: for a $\pm1$ observable the coverage can be computed exactly from the binomial distribution, and with $\langle Z_3Z_4\rangle=0.5787$ it is
# $0.52$ at $M=10$, $0.62$ at $M=30$ and then $0.67$–$0.69$, approaching $0.683$ only slowly. Two effects push it down: the estimate lives on a lattice of spacing $2/M$, and the estimated error bar $\sqrt{(1-\widehat{\langle O\rangle}^2)/M}$ *shrinks*
# exactly when the estimate wanders towards $\pm1$. At $M=10$ the last effect is extreme: all ten shots agree in 9 % of the experiments, and those get an error bar of zero, which can never cover anything.
#
# > **Numerical practice.** For any stochastic estimate report *mean $\pm$ standard error*, check the $1/\sqrt M$ scaling once, and when comparing with a reference look at the deviation in units of the standard error.
# > A deviation of $0.5\,$SE is not "better" than one of $1.5\,$SE; consistent deviations above $3\,$SE mean a bug or a bias.

# %% [markdown]
# ## 12. A Bell pair measured along different axes
#
# The cleanest demonstration that measurement outcomes on entangled spins are correlated *in every basis* — the raw material of Bell inequalities (Bell 1964; Clauser, Horne, Shimony and Holt 1969; textbook treatment: Sakurai and Napolitano, Ch. 3) and quantum key distribution. For two spins define the correlation of outcomes
#
# $$E(a,b)=\langle\sigma_a\otimes\sigma_b\rangle=P(\text{same})-P(\text{different}),$$
#
# estimated from shots as the mean of $z_0z_1$ in the setting $(a,b)$. For $|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2$ one finds $E(Z,Z)=+1$, $E(X,X)=+1$, $E(Y,Y)=-1$, and zero for all mixed settings;
# for the singlet $|\Psi^-\rangle=(|01\rangle-|10\rangle)/\sqrt2$ the outcomes are perfectly **anti**-correlated in *every* common basis, $E(a,a)=-1$ — it is rotationally invariant (total spin zero).
# Yet each spin alone is a fair coin in every basis ($\rho_q=\mathbb 1/2$). We first fill the $3\times3$ table from samples.

# %%
# ==============================================================================
# EXPERIMENT: 3x3 table of sampled Bell correlations E(a,b), 2000 shots per setting
# ==============================================================================
N_SHOTS = 2000
for kind in ("phi+", "psi-"):
    psi_bell = bell_state(kind)
    print(f"Bell state {kind}:   rows = basis of spin 0, columns = basis of spin 1;   entries: sampled E +- SE  [exact]")
    key = jax.random.PRNGKey(77 if kind == "phi+" else 78)
    for a in "XYZ":
        row = []
        for b in "XYZ":
            key, sub = jax.random.split(key)                       # a fresh key for every setting
            zz = 1 - 2 * sample_bitstrings(sub, psi_bell, N_SHOTS, bases=a + b)
            est, se = mean_and_se(zz[:, 0] * zz[:, 1])
            exact = float(expect_pauli_string(psi_bell, a + b))
            assert abs(est - exact) < 4 * se + 1e-12
            row.append(f"{est:+.3f}+-{se:.3f} [{exact:+.0f}]")
        print(f"   {a}:  " + "   ".join(row))
    single = 1 - 2 * sample_bitstrings(jax.random.PRNGKey(5), psi_bell, N_SHOTS, bases="XX")
    print(f"   single-spin average of spin 0 in the X basis: {mean_and_se(single[:, 0])[0]:+.3f} +- {mean_and_se(single[:, 0])[1]:.3f}  (a fair coin)\n")

# %% [markdown]
# The diagonal entries are *exactly* $\pm1$ with zero error bar: all 2000 shots agree (or all disagree) — perfect correlations have no shot noise. Mixed settings give zero within two standard errors.
#
# **Arbitrary axes.** Let spin 0 be measured along $z$ and spin 1 along the axis $\hat n(\theta)=(\sin\theta,0,\cos\theta)$ in the $x$–$z$ plane, i.e. the observable $\sigma_\theta=\cos\theta\,Z+\sin\theta\,X$. Since
# $R_y(\theta)ZR_y(\theta)^\dagger=\cos\theta Z+\sin\theta X$, the rule of Section 6 applies with $U=R_y(\theta)^\dagger=R_y(-\theta)$: rotate spin 1 with $R_y(-\theta)$, then read out in $Z$. Quantum mechanics predicts
#
# $$E(\theta)=\langle Z\otimes\sigma_\theta\rangle=\cos\theta\ \ (\Phi^+),\qquad E(\theta)=-\cos\theta\ \ (\Psi^-).$$
#
# The angle is a *traced* number, so the whole scan over $\theta$ is one `vmap` (over angles and keys simultaneously).

# %%
# ==============================================================================
# EXPERIMENT: E(theta) for phi+ and psi-, 1000 shots per angle, vmapped over the angles
# ==============================================================================
theta_test = 0.7
err = max_abs(ry(theta_test) @ Z @ ry(theta_test).conj().T - (np.cos(theta_test) * Z + np.sin(theta_test) * X))
print(f"CHECKPOINT Ry(t) Z Ry(t)^dag = cos(t) Z + sin(t) X : error = {err:.1e}")
assert err < TOL

N_SHOTS, N_ANGLES = 1000, 25
thetas = jnp.linspace(0.0, 2 * jnp.pi, N_ANGLES)


def correlation_at_angle(theta, key, psi):
    """Sampled E(theta) +- SE: spin 0 along z, spin 1 along (sin t, 0, cos t)."""
    rotated = apply_gate(psi, ry(-theta), [1])                     # U = Ry(theta)^dagger on spin 1, then Z readout
    zz = 1 - 2 * sample_bitstrings(key, rotated, N_SHOTS)
    v = (zz[:, 0] * zz[:, 1]).astype(RDTYPE)
    return jnp.mean(v), jnp.std(v, ddof=1) / jnp.sqrt(N_SHOTS)


fig, ax = plt.subplots(figsize=(7.5, 4))
fine = np.linspace(0, 2 * np.pi, 300)
for kind, sign, c, m in (("phi+", +1, PALETTE[0], "o"), ("psi-", -1, PALETTE[1], "s")):
    keys = jax.random.split(jax.random.PRNGKey(31 if sign > 0 else 32), N_ANGLES)
    E, SE = jax.jit(jax.vmap(partial(correlation_at_angle, psi=bell_state(kind))))(thetas, keys)
    E, SE = np.asarray(E), np.asarray(SE)
    pull = (E - sign * np.cos(np.asarray(thetas))) / np.maximum(SE, 1e-3)
    print(f"{kind}: largest deviation from {'+' if sign > 0 else '-'}cos(theta) = {np.max(np.abs(pull)):.2f} standard errors")
    assert np.max(np.abs(pull)) < 4.5
    ax.plot(fine / np.pi, sign * np.cos(fine), "-", color=c, lw=1.5, label=rf"theory ${'+' if sign > 0 else '-'}\cos\theta$")
    ax.errorbar(np.asarray(thetas) / np.pi, E, yerr=SE, fmt=m, color=c, ms=5, capsize=2, label=f"{kind}: {N_SHOTS} shots per angle")
ax.set_xlabel(r"relative angle of the measurement axes $\theta/\pi$"); ax.set_ylabel(r"correlation $E(\theta)$")
ax.set_title("Bell-pair correlations versus measurement angle"); ax.legend(fontsize=8, ncol=2, loc="upper center")
ax.set_ylim(-1.15, 1.6)
fig.tight_layout(); plt.show()

# %% [markdown]
# The sampled correlations follow $\pm\cos\theta$ within three standard errors (largest deviation $2.7$ SE among 50 points); the error bars shrink to zero where $|E|=1$ and are largest ($1/\sqrt M\approx0.03$) where $E=0$ — the $\sqrt{1-E^2}$ law of Section 11.
# The smooth cosine is what makes quantum correlations stronger than any classical "hidden instruction set" could produce: such models are constrained by the CHSH inequality $|S|\le2$, while four well-chosen angles on this curve give $2\sqrt2$.
# The derivation and a sample-based CHSH test are the subject of [notebook 19, Chapter 8](../ch08_quantum_information_protocols/19_bell_states_and_chsh.ipynb) (and of Exercise 6).

# %% [markdown]
# ## 13. Measurements inside compiled loops: mid-circuit measurement
#
# In many protocols a measurement sits *in the middle* of the dynamics and is repeated many times. The JAX pattern is `lax.scan` (notebook 01): the loop body is compiled once; the **carry** holds the
# state (and any classical memory), and the **scanned input is an array of keys**, one per round, prepared beforehand with `jax.random.split`. `vmap` over a second batch of keys then runs many independent realisations in parallel.
#
# ### 13.1 The quantum Zeno effect
#
# A spin starts in $|0\rangle$ and is rotated about $x$ by a total angle $\pi$, which would take it to $|1\rangle$ with certainty. Now split the rotation into $n$ equal steps $R_x(\pi/n)$ and measure $Z$ after each step.
# After one small step the probability to still find $0$ is $\cos^2(\pi/2n)$, and if we do, the state is *reset to $|0\rangle$ by the collapse*. The probability of finding 0 in **all** $n$ measurements is therefore
#
# $$P_{\rm survive}(n)=\Big[\cos^2\frac{\pi}{2n}\Big]^n .$$
#
# Taking the logarithm, $\ln P_{\rm survive}=2n\ln\cos(\pi/2n)=-n\big[(\pi/2n)^2+O(n^{-4})\big]$, so
#
# $$P_{\rm survive}(n)=\exp\Big(-\frac{\pi^2}{4n}+O(n^{-3})\Big)=1-\frac{\pi^2}{4n}+O(n^{-2})\;\longrightarrow\;1 .$$
#
# A watched spin does not flip: frequent measurement freezes the dynamics, and the deficit closes like $1/n$. This is the quantum Zeno effect, named by Misra and Sudarshan (1977) and observed with trapped ions by Itano, Heinzen, Bollinger and Wineland (1990). The reason is that for short times transition probabilities grow *quadratically*, $\sin^2(\epsilon/2)\approx\epsilon^2/4$,
# so $n$ interruptions cost only $n\cdot(\pi/2n)^2=\pi^2/4n\to0$ instead of the $O(1)$ they would cost if the probability grew linearly in time. Per unit of the swept angle the effective flip rate therefore vanishes as $1/n$.

# %%
# ==============================================================================
# EXPERIMENT: quantum Zeno effect -- scan over measurement rounds, vmap over shots
# ==============================================================================
N_SHOTS = 4000


def zeno_shot(key, n_steps):
    """One realisation: n_steps x [rotate by pi/n_steps about x, measure Z]. Returns True if ALL outcomes were 0."""
    U = rx(jnp.pi / n_steps)

    def one_round(carry, round_key):
        psi, survived = carry
        psi = apply_gate(psi, U, [0])                              # coherent evolution
        outcome, psi = measure_qubit(round_key, psi, 0)            # mid-circuit measurement (branch-free -> scan-able)
        return (psi, survived & (outcome == 0)), outcome

    round_keys = jax.random.split(key, n_steps)                    # one key per round, prepared up front
    (_, survived), _ = lax.scan(one_round, (zero_state(1), jnp.array(True)), round_keys)
    return survived


steps = [1, 2, 4, 8, 16, 32, 64, 128]
P_hat, P_se = [], []
for n in steps:
    shots = jax.jit(jax.vmap(partial(zeno_shot, n_steps=n)))(jax.random.split(jax.random.PRNGKey(n), N_SHOTS))
    p = float(jnp.mean(shots)); P_hat.append(p); P_se.append(np.sqrt(max(p * (1 - p), 1e-12) / N_SHOTS))
P_hat, P_se = np.array(P_hat), np.array(P_se)
P_theory = np.cos(np.pi / (2 * np.array(steps))) ** (2 * np.array(steps))
print(f"{'n':>4s} {'P_survive (sampled)':>22s} {'theory':>8s} {'deviation/SE':>13s}")
for n, p, s, t in zip(steps, P_hat, P_se, P_theory):
    print(f"{n:4d} {p:12.4f} +- {s:.4f} {t:8.4f} {(p - t) / max(s, 1e-6):+13.2f}")
assert np.all(np.abs(P_hat - P_theory) < 4 * P_se + 1e-9)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
nn = np.arange(1, 129)
ax.semilogx(nn, np.cos(np.pi / (2 * nn)) ** (2 * nn), "-", color=PALETTE[1], label=r"theory $[\cos^2(\pi/2n)]^n$")
ax.errorbar(steps, P_hat, yerr=P_se, fmt="o", color=PALETTE[0], capsize=3, label=f"simulation, {N_SHOTS} shots")
ax.set_xlabel("number of intermediate measurements $n$"); ax.set_ylabel("probability that the spin never flips")
ax.set_title("Quantum Zeno effect"); ax.legend(fontsize=8); fig.tight_layout(); plt.show()

# %% [markdown]
# With a single final measurement ($n=1$) the spin has always flipped ($P=0$, with no shot noise at all); with 128 interruptions it survives in about 98 % of the runs, in agreement with the formula for every $n$.
# Technically, each point is one compiled program containing a `scan` over $n$ rounds inside a `vmap` over 4000 shots — $5\times10^5$ simulated measurements for $n=128$, and no Python loop over either.
#
# ### 13.2 Entanglement *created* by measurement
#
# Section 9 showed measurements destroying entanglement. They can also **create** it between spins that never interacted directly. Take the 1D cluster state of notebook 06 ($|{+}\rangle^{\otimes N}$ followed by controlled-$Z$ on every bond; Briegel and Raussendorf 2001), the resource of the one-way quantum computer, in which single-spin measurements are the whole computation (Raussendorf and Briegel 2001). Its stabilisers
# connect a site only to its neighbours, and in fact **no pair of its spins is entangled at all**: every two-spin reduced state is separable, either exactly $\mathbb 1/4$ in the bulk or, at the two ends of the open chain, the classically correlated
# $\rho_{01}=(\mathbb 1+X_0Z_1)/4$ and $\rho_{N-2,N-1}=(\mathbb 1+Z_{N-2}X_{N-1})/4$ that the end stabilisers $X_0Z_1$ and $Z_{N-2}X_{N-1}$ produce. The entanglement of the cluster state is genuinely multipartite, which is exactly why measurements can
# concentrate it into a pair. Its two end spins in particular are **not entangled with each other**: for $N\ge4$ their reduced state is $\mathbb 1/4$ (no correlations whatsoever), and for $N=3$ it is the classical mixture
# $\tfrac12(|{+}{+}\rangle\langle{+}{+}|+|{-}{-}\rangle\langle{-}{-}|)=(\mathbb 1+X_0X_2)/4$ — correlated in $X$, but separable (the code below checks both statements). Now measure all *interior* spins in the $X$ basis. For $N=3$ the argument is short: the state is stabilised by $K_0=X_0Z_1$, $K_1=Z_0X_1Z_2$, $K_2=Z_1X_2$
# (eigenvalue $+1$ each). The products $K_0K_2=X_0X_2$ and $K_1$ commute with the measured $X_1$, so they remain valid after the measurement, with $X_1$ replaced by its outcome $(-1)^m$:
# the ends are left in the state with $X_0X_2=+1$ and $Z_0Z_2=(-1)^m$ — a Bell state, $|\Phi^+\rangle$ or $|\Psi^+\rangle$ depending on the outcome. For $N=7$ the same reasoning uses
# $K_0K_2K_4K_6=X_0X_2X_4X_6$ and $K_1K_3K_5=Z_0X_1X_3X_5Z_6$, which give $X_0X_6=(-1)^{m_2+m_4}$ and $Z_0Z_6=(-1)^{m_1+m_3+m_5}$. Measuring the middle spin in $Z$ instead *cuts* the chain and leaves a product state.
# This is the elementary step of measurement-based quantum computing and of entanglement swapping in quantum repeaters.

# %%
# ==============================================================================
# EXPERIMENT: measure the interior of a cluster chain; entanglement between the two END spins
# ==============================================================================
def measure_interior(key, N, basis):
    """Measure spins 1..N-2 of the N-spin cluster state in `basis`; return outcomes, S(end spin), <X_0 X_{N-1}>, <Z_0 Z_{N-1}>."""
    psi = cluster_state(N)
    keys = jax.random.split(key, N)
    outs = []
    for q in range(1, N - 1):
        m, psi = measure_qubit(keys[q], psi, q, basis)
        outs.append(m)
    ends = {0: "X", N - 1: "X"}, {0: "Z", N - 1: "Z"}
    return jnp.stack(outs), entanglement_entropy(psi, (0,)), expect_pauli_string(psi, ends[0]), expect_pauli_string(psi, ends[1])


keys = jax.random.split(jax.random.PRNGKey(9), 1000)
rho_ends_3 = rdm(cluster_state(3), (0, 2))
rho_ends_7 = rdm(cluster_state(7), (0, 6))
print(f"before any measurement, N = 3: |rho_ends - (1 + X0 X2)/4| = {max_abs(rho_ends_3 - (jnp.eye(4) + XX) / 4):.1e}   (a classical mixture of |++> and |-->)")
print(f"before any measurement, N = 7: |rho_ends - 1/4|           = {max_abs(rho_ends_7 - jnp.eye(4) / 4):.1e}   (no correlations at all)\n")
assert max_abs(rho_ends_3 - (jnp.eye(4) + XX) / 4) < TOL and max_abs(rho_ends_7 - jnp.eye(4) / 4) < TOL
for N_c, basis in ((3, "X"), (3, "Z"), (7, "X")):
    outs, S_end, xx, zz = jax.jit(jax.vmap(partial(measure_interior, N=N_c, basis=basis)))(keys)
    outs, S_end, xx, zz = map(np.asarray, (outs, S_end, xx, zz))
    print(f"N = {N_c}, interior measured in {basis}:  S(end spin | record) min/max over 1000 shots = {S_end.min():.6f} / {S_end.max():.6f} bits")
    if N_c == 3:
        for m in (0, 1):
            sel = outs[:, 0] == m
            print(f"      outcome {m} ({sel.mean():.3f} of the shots):  <X_0 X_2> = {xx[sel].mean():+.3f},  <Z_0 Z_2> = {zz[sel].mean():+.3f}")
        print(f"      averaged over outcomes (record discarded):  <X_0 X_2> = {xx.mean():+.3f},  <Z_0 Z_2> = {zz.mean():+.3f}")
    if basis == "X":
        assert abs(S_end.min() - 1) < 1e3 * TOL and abs(S_end.max() - 1) < 1e3 * TOL
    else:
        assert S_end.max() < 1e3 * TOL

# %% [markdown]
# * $X$ measurements: in **every** shot the end spins share exactly 1 bit of entanglement — for $N=3$ and also for $N=7$, where the two ends are six sites apart and five spins were measured. For $N=3$ the correlations are $X_0X_2=+1$ and
#   $Z_0Z_2=(-1)^m$, as derived.
# * $Z$ measurement: the ends are left in a product state ($S=0$), and the printed $\langle X_0X_2\rangle=+1$ is a classical correlation: $K_0=X_0Z_1$ and $K_2=Z_1X_2$ both commute with the measured $Z_1$, so the
#   outcome fixes $X_0=X_2=(-1)^m$ separately on each end, and the state is the product $|x_0\rangle|z_1\rangle|x_2\rangle$.
# * If the record is discarded, $\langle Z_0Z_2\rangle$ averages to zero (within shot noise): the entanglement is only *useful* together with the classical outcome. Turning "a random one of several Bell states" into "always $|\Phi^+\rangle$" requires a correction conditioned on the outcome — feedback,
#   exactly as in `reset_qubit`, and exactly as in quantum teleportation ([notebook 20, Chapter 8](../ch08_quantum_information_protocols/20_quantum_teleportation.ipynb)).

# %% [markdown]
# ## 14. Performance
#
# We time (i) single-spin measurement shots at $N=12$: a Python loop over a jitted single-shot function versus one `jit(vmap(...))` call; and (ii) a complete readout at $N=12$: sequential collapse versus
# `categorical` (Gumbel-max) versus the inverse CDF. Timing rules as always: block until the result is ready; the first call includes compilation and is reported separately.
#
# | method | time per batch | memory |
# |---|---|---|
# | `measure_qubit`, $M$ shots (vmap) | $O(M\,2^N)$ | $O(M\,2^N)$ — the batch of post-measurement states |
# | full readout, sequential | $O(M\,N\,2^N)$ | $O(M\,2^N)$ |
# | full readout, `categorical` | $O(M\,2^N)$ random numbers | $O(M\,2^N)$ |
# | full readout, inverse CDF | $O(2^N+M\,N)$ | $O(2^N+M\,N)$ — the returned bits |

# %%
# ==============================================================================
# BENCHMARK at N = 12
# ==============================================================================
def timed(fn, *args, repeats=3):
    """(first call incl. compilation, best of `repeats` further calls) in seconds."""
    t0 = time.perf_counter(); jax.block_until_ready(fn(*args)); first = time.perf_counter() - t0
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*args)); best = min(best, time.perf_counter() - t0)
    return first, best


N, M = 12, 1000
psi = haar_state(jax.random.PRNGKey(0), N)
keys = jax.random.split(jax.random.PRNGKey(1), M)

# (i) single-spin measurement: python loop over shots vs vmap
single = jax.jit(lambda k: measure_qubit(k, psi, 0))                          # returns (outcome, post-measurement state)
batched = jax.jit(jax.vmap(lambda k: measure_qubit(k, psi, 0)))
jax.block_until_ready(single(keys[0]))                                        # compile
t0 = time.perf_counter()
for k in keys[:200]:                                                          # 200 shots, one Python iteration each
    jax.block_until_ready(single(k))
t_loop = (time.perf_counter() - t0) / 200
first, t_vmap = timed(batched, keys)
print(f"measure_qubit, N={N}:  python loop {t_loop * 1e6:9.1f} us/shot  |  jit(vmap) {t_vmap / M * 1e6:9.1f} us/shot"
      f"  (compile {first:.2f} s)  ->  speed-up x{t_loop / (t_vmap / M):.0f}")

# (ii) complete readout, three ways
print(f"\ncomplete readout of N={N} spins:")
print(f"{'method':>24s} {'shots':>7s} {'first call [s]':>15s} {'run [ms]':>10s} {'us/shot':>9s}")
for shots in (1000, 10000):
    ks = jax.random.split(jax.random.PRNGKey(2), shots)
    methods = {"sequential collapse": (jax.jit(jax.vmap(lambda k: measure_all_sequential(k, psi)[0])), (ks,)),
               "categorical (Gumbel-max)": (jax.jit(partial(sample_bitstrings, shots=shots)), (ks[0], psi)),
               "inverse CDF": (jax.jit(partial(sample_bitstrings_cdf, shots=shots)), (ks[0], psi))}
    for name, (fn, args) in methods.items():
        if name.startswith("sequential") and shots > 1000:
            continue                                                          # 10^4 x 2^12 post-states: skip to stay in budget
        first, run = timed(fn, *args, repeats=2)
        print(f"{name:>24s} {shots:7d} {first:15.2f} {run * 1e3:10.2f} {run / shots * 1e6:9.2f}")

# %% [markdown]
# **Interpreting the benchmark** (absolute numbers depend on the machine and its load; look at the ratios).
#
# * Batching the shots with `vmap` is one to two orders of magnitude faster per shot than calling a jitted single-shot function from a Python loop: the loop pays the Python-side dispatch of one XLA program per shot, and a batch of 1000 shots keeps
#   the machine busy where a single $2^{12}$ tensor does not. Real analysis code is slower still, because inspecting the outcome with `int(...)` adds a device-to-host transfer per shot.
# * For a complete readout the three samplers differ by orders of magnitude, in the order predicted by the cost table: the sequential collapse does $N$ projections of the full state per shot, the Gumbel-max sampler generates $2^N$ random numbers per shot,
#   and the inverse CDF needs one random number and an $N$-step binary search per shot after a single pass over the state.
# * Rule of thumb: use `sample_bitstrings` for moderate `shots`$\times2^N$; switch to the inverse CDF for many shots or larger $N$; use sequential `measure_qubit` only when you need the *post-measurement state* (partial or mid-circuit measurements).

# %% [markdown]
# ## 15. Key takeaways
#
# * **Born rule on a subsystem** = diagonal of its reduced density matrix; **collapse** = `apply_gate` with a $2\times2$ projector on one axis + renormalisation. Cost $O(2^N)$, no large matrices.
# * Other bases: **rotate – measure $Z$ – rotate back** ($U=H$ for $X$, $U=HS^\dagger$ for $Y$, $U=R_y(-\theta)$ for an axis in the $x$–$z$ plane).
# * Stochastic JAX code: explicit keys (split, never reuse), **no Python `if` on random values** — select with `jnp.where`; then `jit`, `vmap` (shots) and `lax.scan` (repeated rounds) just work. Feedback (reset, corrections) is the same trick.
# * Measuring part of an entangled state changes the rest: GHZ is destroyed by one $Z$ measurement, W mostly survives; an unread measurement is dephasing; reset is the channel $\rho\to|0\rangle\langle0|\otimes\mathrm{Tr}_q\rho$; $X$ measurements on a cluster chain create a Bell pair between its ends.
# * A complete readout can be sampled jointly from $|\psi_s|^2$ (chain rule $\Rightarrow$ same statistics as sequential collapse). Know the cost of your sampler: Gumbel-max $O(\text{shots}\times2^N)$, inverse CDF $O(2^N+\text{shots}\times N)$.
# * **Shot noise**: standard error $\sqrt{(1-\langle O\rangle^2)/M}$; every measurement setting needs its own shots; always report mean $\pm$ SE and validate stochastic code with $z$-scores and $\chi^2$ (`TOL` is for deterministic checks).
# * Frequent measurements freeze dynamics (Zeno); measurement + classical record + feedback is a universal primitive of quantum protocols.

# %% [markdown]
# ## 16. Exercises
#
# 1. ★ **By-hand einsum.** For $N=4$ write the einsum strings that give $p(m)$ for spin $q=2$ and the projected state for outcome $m=1$. Check against `rdm` and `apply_gate` with `assert max_abs(...) < TOL`.
# 2. ★ **Expectation value from single-spin shots.** Using `vmap` over `measure_qubit`, estimate $\langle Y_0\rangle$ with its standard error for the state `product_state("r0")` rotated by `apply_gate(psi, rx(0.4), [0])`. Compare with `expect_local`;
#    the exact answer is $\cos(0.4)=0.92106$. Why would `ry(0.4)` instead of `rx(0.4)` make the exercise pointless?
# 3. ★★ **Measuring an arbitrary single-spin observable (extend the code).** Write `measure_observable(key, psi, q, O)` for any Hermitian $2\times2$ matrix $O$: diagonalise $O=V\,\mathrm{diag}(\lambda_0,\lambda_1)V^\dagger$ with `jnp.linalg.eigh`, use $U=V^\dagger$ as the basis rotation, and return the
#    *eigenvalue* and the post-measurement state. Verify with $O=(X+Z)/\sqrt2$ on $|0\rangle$: the mean of the returned eigenvalues must approach $\langle O\rangle=1/\sqrt2$. Why does this work under `jit` although `eigh` is called inside?
# 4. ★★ **Measurements on a density tensor (extend the code).** Write `measure_qubit_dm(key, rho, q)` for a rank-$2N$ density tensor: $p(0)=\mathrm{Tr}(P_0\rho_q)$ from `rdm_dm`-style contraction (or from the diagonal), collapse $\rho\to\Pi_m\rho\Pi_m/p(m)$ with two `apply_gate` calls.
#    Check on `to_dm(psi)` that, for the same key, it reproduces `to_dm` of the pure-state result.
# 5. ★★ **Shot budget (physics/practice).** For the ground state of Section 11 estimate, using the single-shot variances measured there, how many shots per setting are needed to obtain the energy per site with a standard error of $10^{-3}$. Verify by simulation.
#    Would it be better to distribute the shots unequally between the two settings? (Minimise $\sigma_{ZZ}^2/M_{ZZ}+\sigma_X^2/M_X$ at fixed $M_{ZZ}+M_X$.)
# 6. ★★ **CHSH from samples (physics).** With `correlation_at_angle`-style code estimate $S=E(a,b)+E(a,b')+E(a',b)-E(a',b')$ for $|\Phi^+\rangle$ with spin 0 measured along angles $a=0$, $a'=\pi/2$ and spin 1 along $b=\pi/4$, $b'=-\pi/4$ (you need a rotation on spin 0 as well).
#    Propagate the four standard errors. How many shots per setting are needed to violate $|S|\le2$ by five standard errors?
# 7. ★★★ **Zeno with a general measurement rate (extend the code).** Modify `zeno_shot` so that in each round the measurement happens only with probability $p$ (draw a second random number per round and select between the measured and the unmeasured state with `jnp.where` — think about how to do this for a state *tensor*).
#    Plot the probability to find the spin in $|0\rangle$ at the end (not the survival of the whole record) versus $p$ for $n=64$ rounds.
# 8. ★★★ **Measurement-induced disentangling in a many-body state (physics).** Take a Haar-random state of $N=10$ spins (volume-law entanglement, notebook 06). Measure a fraction $f$ of randomly chosen spins in $Z$ and compute the average half-chain entanglement entropy *of the post-measurement state*
#    as a function of $f$ (average over shots and over the choice of spins with `vmap`; the measured positions must be static, so loop over a few random choices). Compare with the same experiment on the cluster state and on GHZ.
#
# ## References
#
# * M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information* (Cambridge University Press, 2000) — measurement postulate and projective measurements (Sec. 2.2), principle of deferred/implicit measurement (Sec. 4.4).
# * J. J. Sakurai and J. Napolitano, *Modern Quantum Mechanics*, 3rd ed. (Cambridge University Press, 2020) — measurement, spin-1/2 and Stern–Gerlach sequences (Ch. 1, *Fundamental Concepts*), spin correlation measurements and Bell's inequality (Ch. 3, *Theory of Angular Momentum*).
# * J. S. Bell, *On the Einstein Podolsky Rosen paradox*, Physics **1**, 195 (1964); J. F. Clauser, M. A. Horne, A. Shimony, R. A. Holt, *Proposed experiment to test local hidden-variable theories*, Phys. Rev. Lett. **23**, 880 (1969).
# * B. Misra and E. C. G. Sudarshan, *The Zeno's paradox in quantum theory*, J. Math. Phys. **18**, 756 (1977); W. M. Itano, D. J. Heinzen, J. J. Bollinger, D. J. Wineland, *Quantum Zeno effect*, Phys. Rev. A **41**, 2295 (1990).
# * W. Dür, G. Vidal, J. I. Cirac, *Three qubits can be entangled in two inequivalent ways*, Phys. Rev. A **62**, 062314 (2000) — robustness of W versus GHZ.
# * H. J. Briegel and R. Raussendorf, *Persistent entanglement in arrays of interacting particles*, Phys. Rev. Lett. **86**, 910 (2001); R. Raussendorf and H. J. Briegel, *A one-way quantum computer*, Phys. Rev. Lett. **86**, 5188 (2001).
# * E. J. Gumbel, *Statistics of Extremes* (Columbia University Press, 1958) — the distribution behind the Gumbel-max trick; L. Devroye, *Non-Uniform Random Variate Generation* (Springer, 1986) — inverse-transform sampling.
# * W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes: The Art of Scientific Computing*, 3rd ed. (Cambridge University Press, 2007) — Section 7.3 (*Deviates from Other Distributions*: the transformation
#   and inverse-CDF methods used in Section 10.2) and Section 14.3 (*Are Two Distributions Different?*: the $\chi^2$ test and its degrees of freedom); the same algorithms in *Numerical Recipes in Fortran 90*, 2nd ed. (Cambridge University Press, 1996).
# * JAX documentation, *Pseudorandom numbers* (https://docs.jax.dev) — design of the explicit-key PRNG.

