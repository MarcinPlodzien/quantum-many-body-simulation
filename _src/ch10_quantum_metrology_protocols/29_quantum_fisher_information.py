#@title: Quantum Fisher information — how fast a quantum state changes under a small parameter shift
#@part: Chapter 10 — Quantum metrology protocols
#@description: A from-scratch course on parameter estimation and its quantum limit: estimators, the classical Fisher information and the Cramér–Rao bound derived by Cauchy–Schwarz, the quantum Fisher information as the best measurement's Fisher information, the pure-state formula 4 Var(G), the standard quantum limit versus the Heisenberg limit with proofs, a zoo of many-body states, the 3x3 collective QFI matrix and its optimal direction, QFI as an entanglement witness, and QFI under noise and particle loss.

# %% [markdown]
# ## 1. Introduction and motivation
#
# Optical atomic clocks, gravitational-wave interferometers and atomic magnetometers are among the most precise
# instruments in physics. All three do the same thing:
#
# 1. **prepare** a quantum state,
# 2. let an unknown physical quantity $\theta$ (a frequency, a phase, a magnetic field) **imprint** itself on that state,
# 3. **measure** the state and **estimate** $\theta$ from the clicks.
#
# This notebook determines the best precision that any measurement on a given prepared state can reach. It is set by a
# single number attached to the state and to the way $\theta$ enters — the
# **quantum Fisher information** $F_Q$ — and it bounds the error of every conceivable experiment through the
# **quantum Cramér–Rao bound**
#
# $$\Delta\theta \;\ge\; \frac{1}{\sqrt{M\,F_Q}},$$
#
# where $M$ is the number of repetitions and $\Delta\theta$ the standard deviation of an unbiased estimator. Everything else in Chapter 10 — Ramsey interferometry, GHZ interferometry,
# spin squeezing — is an attempt to make $F_Q$ as large as possible and then to actually reach the bound with a
# realistic readout.
#
# There is a second reason physicists care about $F_Q$. For $N$ qubits prepared independently $F_Q$ cannot exceed $N$;
# any state that beats this value *must* be entangled. The quantum Fisher information is therefore an **entanglement
# witness with an operational meaning**: besides certifying entanglement, it states how much that entanglement is worth
# in a laboratory.
#
# **Road map.**
#
# * **Sections 3–5 — classical estimation theory.** What an estimator is, what bias and variance mean, the Fisher
#   information of a probability distribution, and the Cramér–Rao bound *derived* from the Cauchy–Schwarz inequality.
#   We check every statement with simulated coin flips and show, from sampled data with error bars and from an exact
#   enumeration, that the maximum-likelihood estimator reaches the bound as the sample grows.
# * **Sections 6–8 — the quantum step.** A measurement turns the state $\rho_\theta$ into a probability distribution
#   $p(x\vert\theta)$, so every measurement has a classical Fisher information. The quantum Fisher information is the
#   maximum over all measurements (Braunstein and Caves). For a pure state and unitary encoding we derive in full that
#   $F_Q = 4\,\mathrm{Var}(G)$, and we implement it matrix-free.
# * **Sections 9–10 — the two limits and a zoo of states.** Proofs of the standard quantum limit $F_Q\le N$ for product
#   states and of the Heisenberg limit $F_Q\le N^2$ for everything, then measured values for product, GHZ, W, Dicke,
#   cluster, Haar-random and ground states.
# * **Sections 11–12 — directions and entanglement.** The $3\times3$ QFI matrix over $J_x,J_y,J_z$, the optimal
#   generator direction as its largest eigenvector, and the entanglement-witness inequalities.
# * **Sections 13–14 — reality.** QFI of noisy states (the mixed-state formula used here as a black box and derived
#   in the next notebook) and QFI of a subsystem after particle loss, with a Schmidt-compression algorithm that makes
#   losing one qubit as cheap as losing almost all of them.
#
# ### What you will learn
#
# *Physics*
# * what "precision" means mathematically, and why an unbiased estimator cannot do better than the Fisher information allows;
# * why a product state of $N$ qubits gives $F_Q\le N$ (the standard quantum limit) and why no state gives more than $N^2$
#   (the Heisenberg limit), with both proofs;
# * which many-body states are metrologically useful, which are not, and why a highly entangled state can offer no
#   gain over a product state (Haar-random states, and cluster states up to a boundary term, are the instructive
#   counter-examples);
# * how noise and the loss of a single particle destroy the quantum advantage of a GHZ state.
#
# *Numerical methods*
# * the Cauchy–Schwarz proof of the Cramér–Rao bound, additivity of Fisher information, asymptotic efficiency of maximum likelihood;
# * variance of a collective operator computed matrix-free in $O(N2^N)$ instead of $O(4^N)$;
# * a $3\times3$ eigenvalue problem that replaces an optimisation over the sphere of generator directions;
# * the symmetric-logarithmic-derivative formula as a black box, plus a rank-revealing **QR compression** that reduces an
#   $O(8^{K})$ eigendecomposition to $O(8^{N-K})$ for the reduced state of $K$ kept qubits.
#
# *Implementation practice*
# * `jax.vmap` over Monte-Carlo repetitions, random states and parameter sweeps; explicit PRNG keys;
# * `jax.jit` with static qubit indices and traced arrays; compile time separated from run time;
# * validating a physical quantity three ways (matrix-free, dense, analytic) before trusting a single plot.
#
# ### Prerequisites
# * [01 — JAX from scratch](../ch01_computational_toolbox/01_jax_from_scratch.ipynb): `jit`, `vmap`, PRNG keys;
# * [02 — einsum from scratch](../ch01_computational_toolbox/02_einsum_from_scratch.ipynb): index notation as executable code;
# * [05 — matrix-free operators](../ch03_matrix_free_engine/05_matrix_free_operators.ipynb): states as rank-$N$ tensors, `apply_gate`;
# * [06 — states, observables, entanglement](../ch03_matrix_free_engine/06_states_observables_entanglement.ipynb): reduced density
#   matrices, Schmidt decomposition;
# * [07 — density matrices and quantum channels](../ch03_matrix_free_engine/07_density_matrices_and_quantum_channels.ipynb):
#   density tensors, Kraus channels;
# * [08 — measurements](../ch03_matrix_free_engine/08_measurements.ipynb): Born rule, sampling, shot noise;
# * helpful: [22 — GHZ states and decoherence](../ch08_quantum_information_protocols/22_ghz_states_and_decoherence.ipynb).
#
# **What comes next.** The notebook that follows,
# [30 — QFI from the SLD: prepare, encode, estimate](../ch10_quantum_metrology_protocols/30_qfi_from_the_sld_prepare_encode_estimate.ipynb),
# derives the mixed-state formula we use here as a black box and builds the optimal measurement explicitly.
# Then come the protocols: [31 — Ramsey interferometry](../ch10_quantum_metrology_protocols/31_ramsey_interferometry.ipynb),
# [32 — GHZ interferometry and the Heisenberg limit](../ch10_quantum_metrology_protocols/32_ghz_interferometry_heisenberg_limit.ipynb),
# [33 — spin squeezing by one-axis twisting](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb) and
# [34 — the full one-axis-twisting to GHZ protocol](../ch10_quantum_metrology_protocols/34_oat_to_ghz_full_metrology_protocol.ipynb).
# This notebook supplies the theory they all use.
#
# **Conventions.** Qubit $q$ = tensor axis $q$; $\vert 0\rangle$ is the $+1$ eigenstate of $Z$. Collective spin operators are
# $J_a=\tfrac12\sum_{q}\sigma^a_q$ for $a=x,y,z$, so a single qubit carries spin $1/2$ and $J_a$ has eigenvalues between
# $-N/2$ and $+N/2$. The parameter is imprinted by $U(\theta)=e^{-i\theta G}$ with a Hermitian **generator** $G$; unless
# said otherwise $G=J_{\mathbf n}=\mathbf n\cdot\mathbf J$ for a unit vector $\mathbf n$. With this convention the two
# famous limits read $F_Q\le N$ and $F_Q\le N^2$ — no stray factors of $4$.

# %% [markdown]
# ## 2. Engine recap and notebook helpers
#
# From the engine we reuse the state constructors, `apply_gate` / `apply_gate_dm` (the einsum that applies a small matrix to
# chosen axes of a state or density tensor), the Kraus channels, `apply_collective` (the matrix-free
# $\sum_q P_q$), `qfi_pure`, `qfi_mixed`, `collective_dense`, `spin_moments`, and the Lanczos ground-state solver.
# Everything specific to this notebook is built below from scratch.

# %%
#@engine: apply_gate, apply_gate_dm, apply_kraus_dm, rdm, dm_matrix, to_dm, product_state, ghz_state, w_state, dicke_state, cluster_state, haar_state, I2, X, Y, Z, H, apply_collective, qfi_pure, qfi_mixed, collective_dense, spin_moments, purity, entanglement_entropy, kraus_depolarizing, kraus_dephasing, kraus_amplitude_damping, heisenberg_terms, lanczos_ground_state, expect_pauli_string, oat_evolve

# %%
# ==============================================================================
# PLOT STYLE + small helpers used throughout this notebook
# ==============================================================================
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # colour-blind-friendly, fixed order
MARKERS = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=PALETTE), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "legend.frameon": False})


def max_abs(a):
    """Largest absolute entry of an array, as a Python float."""
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def timed(f, *args, budget=0.3, min_reps=5, max_reps=500):
    """Return (result, compile_time, best_run_time): the first call includes tracing+compilation, the rest do not.

    JAX is asynchronous, so every timed value must be forced with `block_until_ready` -- otherwise one
    measures the time to *dispatch* the work, not to do it.  We repeat for a fixed time `budget` and report
    the MINIMUM: on a shared machine the mean measures the operating system, the minimum measures the code.
    """
    t0 = time.time()
    out = jax.block_until_ready(f(*args))
    t_compile = time.time() - t0
    best, n, t_start = np.inf, 0, time.time()
    while n < min_reps or (time.time() - t_start < budget and n < max_reps):
        t0 = time.time()
        out = jax.block_until_ready(f(*args))
        best = min(best, time.time() - t0)
        n += 1
    return out, t_compile, best

# %% [markdown]
# ## 3. Estimation theory: estimators, bias, variance
#
# Before any quantum mechanics we need the classical machinery, because the quantum answer is a statement *about* it.
#
# ### 3.1 The setting
#
# Nature holds a number $\theta$ that we cannot see. We can run an experiment whose outcome $x$ is random, with a
# probability distribution $p(x\vert\theta)$ that we know as a *function of* $\theta$ — this function is the **statistical
# model**. We run the experiment $M$ times, collect $x_1,\dots,x_M$, and compute a number
#
# $$\hat\theta \;=\; T(x_1,\dots,x_M)$$
#
# which we call an **estimator**. An estimator is a formula applied to data. Being a function of random data, it is itself
# a random variable, distinct from the true value $\theta$. Two numbers describe how good it is:
#
# $$\mathrm{bias}(\hat\theta)=\mathbb{E}_\theta[\hat\theta]-\theta,\qquad
#   \mathrm{Var}(\hat\theta)=\mathbb{E}_\theta\!\left[(\hat\theta-\mathbb{E}_\theta[\hat\theta])^2\right],$$
#
# and they combine into the **mean squared error**
#
# $$\mathrm{MSE}(\hat\theta)=\mathbb{E}_\theta\!\left[(\hat\theta-\theta)^2\right]
#   =\mathrm{Var}(\hat\theta)+\mathrm{bias}(\hat\theta)^2 . \tag{1}$$
#
# *Proof of Eq. (1).* Write $\hat\theta-\theta=(\hat\theta-\mathbb{E}[\hat\theta])+(\mathbb{E}[\hat\theta]-\theta)$, square, and
# take the expectation; the cross term is $2\,\mathbb{E}[\hat\theta-\mathbb{E}[\hat\theta]]\cdot\mathrm{bias}=0$. $\square$
#
# An estimator is **unbiased** if its bias vanishes for *every* $\theta$. Unbiasedness is a convenient property rather
# than a requirement: Eq. (1) shows that a slightly biased estimator with a much smaller variance can be the better one. But the bound we are
# heading for — the Cramér–Rao bound — is a statement about unbiased estimators, so we keep track of the distinction.
#
# ### 3.2 The running example: a biased coin
#
# The simplest statistical model is a coin that lands heads ($x=1$) with unknown probability $\theta$ and tails ($x=0$) with
# probability $1-\theta$, so $p(x\vert\theta)=\theta^x(1-\theta)^{1-x}$. Flip it $M$ times, count the heads $k$, and use
#
# $$\hat\theta_{\text{mean}}=\frac{k}{M},\qquad\text{or}\qquad
#   \hat\theta_{\text{Lap}}=\frac{k+1}{M+2}\quad\text{(Laplace's rule of succession)} .$$
#
# The first is unbiased because $\mathbb{E}[k]=M\theta$; its variance is $\mathrm{Var}(k)/M^2=\theta(1-\theta)/M$.
# The second is biased towards $1/2$. Let us simulate both.

# %%
# ==============================================================================
# STEP 1: the coin experiment -- two estimators, measured bias and variance
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
THETA_COIN = 0.30          # the "unknown" probability of heads
N_REPEAT   = 20_000        # how many independent experiments we simulate (Monte-Carlo sample size)
M_LIST     = (10, 100, 1000)   # number of flips per experiment
# -----------------------------------------------------------------------------


def coin_experiments(key, theta, M, n_repeat):
    """Simulate `n_repeat` independent experiments of `M` Bernoulli(theta) flips; return the head counts.

    MATH   k ~ Binomial(M, theta).
    JAX    one (n_repeat, M) array of booleans, summed along the flip axis: no Python loop at all.
    """
    flips = jax.random.bernoulli(key, theta, (n_repeat, M))
    return jnp.sum(flips, axis=1)


print(f"true theta = {THETA_COIN}\n")
print(f"{'M':>6s} | {'bias(k/M)':>12s} {'Var(k/M)':>12s} {'theta(1-theta)/M':>18s} | "
      f"{'bias(Laplace)':>14s} {'Var(Laplace)':>13s}")
for M in M_LIST:
    k = coin_experiments(jax.random.PRNGKey(0), THETA_COIN, M, N_REPEAT)
    mean_est = k / M
    lap_est = (k + 1) / (M + 2)
    var_theory = THETA_COIN * (1 - THETA_COIN) / M
    print(f"{M:6d} | {float(jnp.mean(mean_est)) - THETA_COIN:+12.5f} {float(jnp.var(mean_est)):12.3e} "
          f"{var_theory:18.3e} | {float(jnp.mean(lap_est)) - THETA_COIN:+14.5f} {float(jnp.var(lap_est)):13.3e}")
    # the sample mean is unbiased; the Monte-Carlo error of a mean over N_REPEAT samples is sqrt(Var/N_REPEAT)
    assert abs(float(jnp.mean(mean_est)) - THETA_COIN) < 5 * np.sqrt(var_theory / N_REPEAT)

# %% [markdown]
# The measured variance of the sample mean matches $\theta(1-\theta)/M$ to about $1\%$ (the printed values are $1.3\%$,
# $1.1\%$ and $0.1\%$ off), which is the statistical precision of a variance estimated from $20\,000$ experiments,
# $\sqrt{2/20\,000}=1\%$; its bias is zero within the Monte-Carlo error, as the assert requires. Laplace's estimator is
# visibly biased: its exact bias is $\mathbb{E}[(k+1)/(M+2)]-\theta=(1-2\theta)/(M+2)$, i.e. $+0.033$ at $M=10$ (the
# printed $+0.036$ also contains the sampling fluctuation of $k$ that shows up as $+0.003$ in the first column), and it
# shrinks like $1/M$. Its variance, exactly $M\theta(1-\theta)/(M+2)^2$, is *smaller* than that of the sample mean at every $M$. That is the bias–variance trade-off of Eq. (1) in one line
# of output, and the reason the coming bound has to say "unbiased" out loud.

# %% [markdown]
# ## 4. The classical Fisher information
#
# ### 4.1 The score
#
# Fix the data $x$ and vary $\theta$: the function $\theta\mapsto p(x\vert\theta)$ tells us how *surprised* we should be by
# the data we got, for each hypothetical $\theta$. Its logarithmic derivative,
#
# $$s(x,\theta)=\frac{\partial}{\partial\theta}\log p(x\vert\theta)=\frac{\partial_\theta p(x\vert\theta)}{p(x\vert\theta)},$$
#
# is called the **score**. It measures how strongly the outcome $x$ discriminates between $\theta$ and $\theta+d\theta$:
# if $p(x\vert\theta)$ barely changes with $\theta$, the outcome $x$ carries almost no information about $\theta$.
#
# The score has mean zero:
#
# $$\mathbb{E}_\theta[s]=\sum_x p(x\vert\theta)\,\frac{\partial_\theta p(x\vert\theta)}{p(x\vert\theta)}
#   =\sum_x\partial_\theta p(x\vert\theta)=\partial_\theta\!\sum_x p(x\vert\theta)=\partial_\theta 1=0 . \tag{2}$$
#
# (We may exchange sum and derivative because the sum is finite, or under mild regularity for continuous $x$.)
#
# ### 4.2 Definition and the two formulas
#
# The **Fisher information** of the model at $\theta$ is the variance of the score, which by Eq. (2) is its second moment:
#
# $$I(\theta)\;=\;\mathrm{Var}_\theta[s]\;=\;\mathbb{E}_\theta\!\left[s^2\right]
#   \;=\;\sum_x\frac{\left(\partial_\theta p(x\vert\theta)\right)^2}{p(x\vert\theta)} . \tag{3}$$
#
# Equation (3) is the working formula: sum the squared sensitivity of each outcome probability, divided by that probability.
# There is an equivalent "curvature" formula,
#
# $$I(\theta)=-\,\mathbb{E}_\theta\!\left[\frac{\partial^2}{\partial\theta^2}\log p(x\vert\theta)\right], \tag{4}$$
#
# *Proof of Eq. (4).* $\partial_\theta^2\log p=\partial_\theta(\partial_\theta p/p)=\partial^2_\theta p/p-(\partial_\theta p/p)^2$.
# Taking the expectation, the first term is $\sum_x\partial^2_\theta p=\partial_\theta^2\sum_x p=0$ and the second is $\mathbb{E}[s^2]=I$. $\square$
#
# Equation (4) explains the name: $I(\theta)$ is the average curvature of the log-likelihood at its peak. A sharply peaked
# likelihood means a well-determined $\theta$.
#
# ### 4.3 Additivity over repetitions
#
# If we repeat the experiment $M$ times independently, the data are $x_1,\dots,x_M$ and
# $p(x_1..x_M\vert\theta)=\prod_j p(x_j\vert\theta)$, so the score is a **sum of independent scores**,
# $s_{\text{tot}}=\sum_j s(x_j,\theta)$. Variances of independent variables add:
#
# $$I_M(\theta)=M\,I(\theta). \tag{5}$$
#
# Equation (5) is the origin of the $1/\sqrt M$ improvement of precision with the number of repetitions.
#
# ### 4.4 The coin, again
#
# For the Bernoulli model $\log p=x\log\theta+(1-x)\log(1-\theta)$, hence $s=x/\theta-(1-x)/(1-\theta)$ and
#
# $$I(\theta)=\frac{(1-\theta)^2}{\theta^2}\theta+\frac{\theta^2}{(1-\theta)^2}(1-\theta)
#   =\frac{1-\theta}{\theta}+\frac{\theta}{1-\theta}=\frac{1}{\theta(1-\theta)} . \tag{6}$$
#
# Two sanity checks: $I$ blows up as $\theta\to0$ or $1$ (a coin that almost never shows heads pins its bias down very
# precisely once it does), and it is smallest at $\theta=1/2$ (a fair-looking coin is the hardest to characterise).
# We verify Eq. (3) numerically, once with a finite-difference derivative and once with automatic differentiation.

# %%
# ==============================================================================
# STEP 2: classical Fisher information from its definition, Eq. (3)
# ==============================================================================
def fisher_information(prob_fn, theta, eps=1e-6, tol=1e-14):
    """Classical Fisher information of a discrete model, Eq. (3), with a central-difference derivative.

    MATH   I(theta) = sum_x (d p(x|theta)/d theta)^2 / p(x|theta)      [terms with p = 0 are dropped]
    ARGS   prob_fn: theta -> array of probabilities p(x|theta) summing to 1.
    JAX    `jnp.where` guards the division so that zero-probability outcomes contribute exactly 0
           (and produce no NaN inside a jit-ted/vmapped computation).
    """
    p = prob_fn(theta)
    dp = (prob_fn(theta + eps) - prob_fn(theta - eps)) / (2 * eps)
    ok = p > tol
    return jnp.sum(jnp.where(ok, dp ** 2 / jnp.where(ok, p, 1.0), 0.0))


def fisher_information_ad(prob_fn, theta, tol=1e-14):
    """The same quantity with an EXACT derivative from forward-mode automatic differentiation.

    JAX   `jax.jacfwd` pushes a tangent of the (real) input through `prob_fn`; no step size, no truncation error.
    """
    p = prob_fn(theta)
    dp = jax.jacfwd(prob_fn)(theta)
    ok = p > tol
    return jnp.sum(jnp.where(ok, dp ** 2 / jnp.where(ok, p, 1.0), 0.0))


def coin_probs(theta):
    """p(x|theta) for one Bernoulli flip, as the array [p(tails), p(heads)]."""
    return jnp.stack([1 - theta, theta])


# --- CHECKPOINT: numerical Fisher information vs the analytic Eq. (6) ------------------
print(f"{'theta':>6s} | {'I (finite diff)':>16s} {'I (jacfwd)':>13s} {'1/(theta(1-theta))':>20s}")
for th in (0.1, 0.3, 0.5, 0.7, 0.9):
    i_fd = float(fisher_information(coin_probs, th))
    i_ad = float(fisher_information_ad(coin_probs, th))
    i_ex = 1.0 / (th * (1 - th))
    print(f"{th:6.2f} | {i_fd:16.8f} {i_ad:13.8f} {i_ex:20.8f}")
    assert abs(i_ad - i_ex) < 1e3 * TOL and abs(i_fd - i_ex) < 1e-5

# %% [markdown]
# Both derivatives reproduce Eq. (6). Automatic differentiation is exact to machine precision; the finite difference agrees
# to a few times $10^{-10}$, which is the usual compromise of a central difference (truncation error $O(\epsilon^2)$ plus
# round-off $O(\epsilon^{-1}\cdot 10^{-16})$; the sum of the two is smallest near $\epsilon\approx10^{-5}$, so the
# $\epsilon=10^{-6}$ used above is already on the round-off side of the optimum).
#
# > **Numerical practice.** Whenever you need $\partial_\theta$ of something a JAX function computes, reach for
# > `jax.jacfwd` / `jax.grad` first: no step size to tune, no cancellation. Keep the finite difference as an *independent*
# > check — the two routes fail in completely different ways, so their agreement is real evidence.

# %% [markdown]
# ## 5. The Cramér–Rao bound, derived
#
# ### 5.1 Statement and proof
#
# **Theorem (Cramér–Rao).** Let $\hat\theta=T(x)$ be an unbiased estimator of $\theta$ in a model with Fisher
# information $I(\theta)>0$. Then
#
# $$\mathrm{Var}(\hat\theta)\;\ge\;\frac{1}{I(\theta)} . \tag{7}$$
#
# *Proof.* Consider the covariance of the estimator with the score. Since $\mathbb{E}[s]=0$ by Eq. (2),
#
# $$\mathrm{Cov}(T,s)=\mathbb{E}[T\,s]-\mathbb{E}[T]\,\mathbb{E}[s]=\sum_x T(x)\,p(x\vert\theta)\,\frac{\partial_\theta p(x\vert\theta)}{p(x\vert\theta)}
#   =\sum_x T(x)\,\partial_\theta p(x\vert\theta)=\partial_\theta\!\sum_x T(x)p(x\vert\theta)=\partial_\theta\,\mathbb{E}[T] .$$
#
# Unbiasedness means $\mathbb{E}[T]=\theta$ for all $\theta$, so the right-hand side is exactly $1$:
#
# $$\mathrm{Cov}(T,s)=1 .$$
#
# Now apply the Cauchy–Schwarz inequality to the two zero-mean random variables $T-\mathbb{E}[T]$ and $s$:
#
# $$\big(\mathrm{Cov}(T,s)\big)^2\;\le\;\mathrm{Var}(T)\,\mathrm{Var}(s) .$$
#
# The left side is $1$ and $\mathrm{Var}(s)=I(\theta)$ by definition, so $1\le\mathrm{Var}(T)\,I(\theta)$, which is Eq. (7). $\square$
#
# The proof also tells us **when the bound is tight**: Cauchy–Schwarz is an equality precisely when the two variables are
# proportional, i.e. when
#
# $$T(x)-\theta \;=\; \frac{1}{I(\theta)}\,s(x,\theta)\qquad\text{for all }x . \tag{8}$$
#
# Such an estimator is called *efficient*. Applying Eq. (7) to the $M$-sample model, whose Fisher
# information is $M I(\theta)$ by Eq. (5), gives the form every experimentalist quotes:
#
# $$\Delta\theta\;=\;\sqrt{\mathrm{Var}(\hat\theta)}\;\ge\;\frac{1}{\sqrt{M\,I(\theta)}} , \tag{9}$$
#
# for any estimator $\hat\theta=T(x_1,\dots,x_M)$ that is unbiased as a function of the whole data set.
#
# ### 5.2 The small print: regularity and local unbiasedness
#
# Two steps of the proof are assumptions in disguise, and both are violated by models a student will meet.
#
# 1. **Regularity.** We differentiated under the sum twice, in Eq. (2) and in the covariance computation. This is legitimate
#    when the *set of possible outcomes does not depend on $\theta$* and $p(x\vert\theta)$ is differentiable there. If some
#    $p(x\vert\theta)$ hits $0$ — as the fringe model of Eq. (10) does at $\theta=0$ and $\theta=\pi$ — the score is undefined
#    for that outcome, $I(\theta)$ need not be the limit of its neighbours, and the bound does not apply at that point.
#    Exercise 2 examines exactly this situation.
# 2. **How much unbiasedness.** The proof used $\mathbb{E}[T]=\theta$ only to conclude $\partial_\theta\mathbb{E}[T]=1$ at the
#    one value of $\theta$ we are bounding. So *local* unbiasedness — $\mathbb{E}[T]=\theta$ and $\partial_\theta\mathbb{E}[T]=1$
#    at that point — already suffices; global unbiasedness for all $\theta$ is a stronger condition than the theorem needs.
#    This matters in practice: an estimator that is unbiased only near the working point, like the maximum-likelihood
#    estimator at large $M$, is still constrained by Eq. (9) there.
#
# ### 5.3 The coin is efficient; a nonlinear model is only asymptotically efficient
#
# For the coin, the score of $M$ flips is $s_{\text{tot}}=k/\theta-(M-k)/(1-\theta)=(k-M\theta)/(\theta(1-\theta))$, and
# $I_M=M/(\theta(1-\theta))$, so $s_{\text{tot}}/I_M=k/M-\theta$: Eq. (8) holds exactly with $T=k/M$. That is why the
# measured variance in Section 3 sat *on* the bound rather than above it.
#
# Efficiency at finite $M$ is the exception. Take the model we will meet again in every interferometer of this chapter: a binary outcome
# whose probability is
#
# $$p(+\vert\theta)=\frac{1+\cos\theta}{2},\qquad p(-\vert\theta)=\frac{1-\cos\theta}{2} . \tag{10}$$
#
# Its Fisher information is, from Eq. (3),
#
# $$I(\theta)=\frac{\left(\tfrac12\sin\theta\right)^2}{p_+}+\frac{\left(\tfrac12\sin\theta\right)^2}{p_-}
#   =\frac{\sin^2\theta}{4}\cdot\frac{p_++p_-}{p_+p_-}
#   =\frac{\sin^2\theta}{4}\cdot\frac{4}{\sin^2\theta}=1 \tag{11}$$
#
# (using $\partial_\theta p_\pm=\mp\tfrac12\sin\theta$ and $p_+p_-=(1-\cos^2\theta)/4=\sin^2\theta/4$) — independent of
# $\theta$, except at $\theta=0$ and $\theta=\pi$, where one outcome has probability zero and the regularity condition of
# Section 5.2 fails. The **maximum-likelihood
# estimator** maximises $\log p(\text{data}\vert\theta)=k\log p_++(M-k)\log p_-$ over $\theta$; setting the derivative to zero
# gives $\cos\hat\theta=2k/M-1$, i.e.
#
# $$\hat\theta_{\text{ML}}=\arccos\!\left(\frac{2k}{M}-1\right) .$$
#
# Because $\arccos$ is nonlinear, $\hat\theta_{\text{ML}}$ is in general *biased* at finite $M$ and does not satisfy
# Eq. (8) exactly. The size of the bias follows from a second-order Taylor expansion (the "delta method") of
# $g(\hat p)=\arccos(2\hat p-1)$ around $p=p_+$, with $\hat p=k/M$, $\mathbb{E}[\hat p-p]=0$ and $\mathrm{Var}(\hat p)=p(1-p)/M$:
#
# $$\mathbb{E}[\hat\theta_{\text{ML}}]-\theta\;\approx\;\tfrac12 g''(p)\,\frac{p(1-p)}{M}
#   =\frac{1-2p}{4M\sqrt{p(1-p)}}=-\frac{\cot\theta}{2M}, \tag{11a}$$
#
# using $g'(p)=-1/\sqrt{p(1-p)}$, $g''(p)=(1-2p)/\left(2[p(1-p)]^{3/2}\right)$, $p(1-p)=\sin^2\theta/4$ and $1-2p=-\cos\theta$.
# The bias vanishes at $\theta=\pi/2$, and there it vanishes *exactly* for every $M$: at $p_+=\tfrac12$ the distribution
# of $k$ is symmetric under $k\to M-k$, which maps $\hat\theta_{\text{ML}}\to\pi-\hat\theta_{\text{ML}}$, so
# $\mathbb{E}[\hat\theta_{\text{ML}}]=\pi/2$. To see a bias at all we must work away from the symmetric point, so the
# simulation uses two working points, $\theta=\pi/2$ and $\theta=\pi/3$.
#
# The general theorem — quoted, not proved here — says that under regularity conditions the maximum-likelihood estimator is
# **consistent and asymptotically efficient**: its bias falls like $1/M$ and $M\,\mathrm{Var}(\hat\theta_{\text{ML}})\to1/I(\theta)$.
# The simulation below measures both, with Monte-Carlo error bars, and compares them with the *exact* finite-$M$ moments,
# which here are a finite sum over the binomial distribution of $k$.

# %%
# ==============================================================================
# STEP 3: maximum likelihood saturates the Cramer-Rao bound as M grows
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
THETA_LIST = (np.pi / 2, np.pi / 3)   # symmetric working point (bias exactly 0) and an asymmetric one
M_GRID     = (4, 10, 30, 100, 300, 1000, 3000)
N_EXP      = 40_000                   # independent experiments per (theta, M)
# -----------------------------------------------------------------------------


def ramsey_probs(theta):
    """The two-outcome model of Eq. (10) as an array [p(+), p(-)]."""
    return jnp.stack([(1 + jnp.cos(theta)) / 2, (1 - jnp.cos(theta)) / 2])


def ml_exact_moments(theta, M):
    """EXACT bias and variance of theta_ML = arccos(2k/M - 1) at finite M.

    MATH   E[f(k)] = sum_{k=0}^{M} C(M,k) p^k (1-p)^(M-k) f(k),   p = (1 + cos theta)/2,
           with log C(M,k) from the log-Gamma function so that M = 3000 does not overflow.
    """
    p = (1 + np.cos(theta)) / 2
    k = np.arange(M + 1)
    logw = (math.lgamma(M + 1) - np.array([math.lgamma(j + 1) + math.lgamma(M - j + 1) for j in k])
            + k * np.log(p) + (M - k) * np.log1p(-p))
    w = np.exp(logw)
    est = np.arccos(np.clip(2 * k / M - 1, -1.0, 1.0))
    mean = w @ est
    return mean - theta, w @ (est - mean) ** 2


ml_rows = {}
print(f"{'theta':>6s} {'M':>5s} | {'bias (MC)':>20s} {'bias (exact)':>13s} | {'M Var (MC)':>17s} {'M Var (exact)':>14s}")
for j, theta in enumerate(THETA_LIST):
    I_th = float(fisher_information_ad(ramsey_probs, theta))
    assert abs(I_th - 1.0) < 1e3 * TOL                                      # Eq. (11): I = 1 at both working points
    p_plus = float(ramsey_probs(theta)[0])
    rows = []
    for i, M in enumerate(M_GRID):
        key = jax.random.fold_in(jax.random.PRNGKey(1), 100 * j + i)       # a fresh key for every (theta, M)
        k = jnp.sum(jax.random.bernoulli(key, p_plus, (N_EXP, M)), axis=1)  # number of '+' outcomes
        est = np.asarray(jnp.arccos(jnp.clip(2 * k / M - 1, -1.0, 1.0)))     # maximum-likelihood estimate
        bias, var = est.mean() - theta, est.var()
        se_bias = est.std() / np.sqrt(N_EXP)                                 # standard error of the mean
        se_var = np.sqrt(np.mean((est - est.mean()) ** 4) - var ** 2) / np.sqrt(N_EXP)   # ... of the variance
        b_ex, v_ex = ml_exact_moments(theta, M)
        rows.append((M, bias, se_bias, M * var, M * se_var, b_ex, M * v_ex))
        print(f"{theta:6.4f} {M:5d} | {bias:+10.5f} +- {se_bias:7.5f} {b_ex:+13.5f} | "
              f"{M * var:7.4f} +- {M * se_var:6.4f} {M * v_ex:14.5f}")
        # CHECKPOINT: the sampled moments agree with the exact ones within 4 standard errors
        assert abs(bias - b_ex) < 4 * se_bias and abs(var - v_ex) < 4 * se_var
    ml_rows[theta] = rows
    print()

# CHECKPOINT: the exact moments obey the asymptotic theory
for theta, rows in ml_rows.items():
    assert abs(rows[-1][6] - 1.0) < 1e-3                                    # M Var -> 1/I = 1 (efficiency)
for M, _, _, _, _, b_ex, _ in ml_rows[np.pi / 3]:
    if M >= 300:                                                           # bias -> -cot(theta)/(2M), Eq. (11a)
        assert abs(M * b_ex / (-0.5 / np.tan(np.pi / 3)) - 1.0) < 0.01
assert all(abs(r[5]) < 1e-10 for r in ml_rows[np.pi / 2])                  # exactly unbiased at pi/2 (round-off only)
# WRONG CONTROL: "the ML estimator is unbiased" must be rejected by the sampled data at theta = pi/3, M = 10
_, b10, se10, *_ = ml_rows[np.pi / 3][1]
print(f"wrong control: at theta = pi/3, M = 10 the hypothesis 'bias = 0' is off by {abs(b10) / se10:.0f} standard errors")
assert abs(b10) > 8 * se10

# %%
# ==============================================================================
# FIGURE: convergence of the maximum-likelihood estimator to the Cramer-Rao bound
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
Ms = np.array(M_GRID, dtype=float)
Mfine = np.unique(np.round(np.logspace(np.log10(4), np.log10(3000), 60)).astype(int))
for j, (theta, lab) in enumerate(zip(THETA_LIST, (r"\pi/2", r"\pi/3"))):
    r = np.array(ml_rows[theta])
    exact_fine = np.array([ml_exact_moments(theta, int(M)) for M in Mfine])
    axes[0].errorbar(Ms, Ms * r[:, 1], yerr=Ms * r[:, 2], fmt=MARKERS[j], color=PALETTE[j], ms=5, capsize=2,
                     label=rf"sampled, $\theta={lab}$")
    axes[0].plot(Mfine, Mfine * exact_fine[:, 0], "-", color=PALETTE[j], lw=1, label=rf"exact, $\theta={lab}$")
    axes[1].errorbar(Ms, r[:, 3], yerr=r[:, 4], fmt=MARKERS[j], color=PALETTE[j], ms=5, capsize=2,
                     label=rf"sampled, $\theta={lab}$")
    axes[1].plot(Mfine, Mfine * exact_fine[:, 1], "-", color=PALETTE[j], lw=1, label=rf"exact, $\theta={lab}$")
axes[0].axhline(-0.5 / np.tan(np.pi / 3), color="k", ls="--", lw=1, label=r"$-\cot\theta/2$ at $\theta=\pi/3$, Eq. (11a)")
axes[0].set_xscale("log")
axes[0].set_xlabel("number of repetitions $M$"); axes[0].set_ylabel(r"$M\,(\mathbb{E}[\hat\theta]-\theta)$")
axes[0].set_title("Bias of the maximum-likelihood estimator, times $M$"); axes[0].legend(fontsize=8)
axes[1].axhline(1.0, color="k", ls="--", lw=1, label=r"Cramér–Rao bound $1/I(\theta)=1$")
axes[1].set_xscale("log")
axes[1].set_xlabel("number of repetitions $M$"); axes[1].set_ylabel(r"$M\,\mathrm{Var}(\hat\theta_{\rm ML})$")
axes[1].set_ylim(0.8, 2.0)
axes[1].set_title("Asymptotic efficiency"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# Every sampled point agrees with the exact finite-$M$ value within four standard errors (asserted), so the Monte-Carlo
# data and the binomial sum are two independent views of the same estimator. The left panel plots the bias multiplied by
# $M$, so that a $1/M$ law appears as a plateau. At $\theta=\pi/3$ the exact curve settles on the delta-method value
# $-\cot\theta/2=-0.289$ of Eq. (11a) (within $1\%$ from $M=300$ on, asserted), so the bias falls like $1/M$; the sampled
# points follow it with error bars that grow with $M$, because the bias shrinks faster than the Monte-Carlo noise of
# $40\,000$ experiments. At $\theta=\pi/2$ the exact bias is zero at every $M$, and the sampled values scatter around zero
# within their error bars. The wrong control makes the same point quantitatively: at $\theta=\pi/3$, $M=10$ the sampled
# bias excludes "unbiased" by the number of standard errors printed above.
#
# The right panel shows the efficiency. The scaled variance $M\,\mathrm{Var}$ starts well *above* the bound at $M=4$
# (exactly $1.78$ at $\theta=\pi/2$ and $1.79$ at $\theta=\pi/3$) — a handful of outcomes is not enough to invert a cosine reliably — and
# approaches $1/I(\theta)=1$ from above: at $\theta=\pi/2$ the exact excess is $3.6\%$ at $M=30$, $1.0\%$ at $M=100$ and
# $0.03\%$ at $M=3000$; at $\theta=\pi/3$ the approach is slower at small $M$ ($40\%$ excess at $M=10$). For a regular
# model the Cramér–Rao bound is therefore the *achievable* precision once enough data have been collected.
#
# > **Common pitfall.** "My estimator beat the Cramér–Rao bound" almost always means "my estimator is biased". At $M=4$
# > above one can indeed construct estimators with a smaller variance than $1/(MI)$ — they shrink towards a fixed guess and
# > pay for it with bias. Eq. (7) constrains the variance only for *unbiased* estimators; Eq. (1) is what you should minimise.

# %% [markdown]
# ## 6. Going quantum: a measurement turns a state into a probability distribution
#
# Now the model $p(x\vert\theta)$ stops being given and starts being *chosen*. The experiment is:
#
# $$\rho\;\xrightarrow[\text{encoding}]{\;U(\theta)=e^{-i\theta G}\;}\;\rho_\theta=U(\theta)\,\rho\,U^\dagger(\theta)
#   \;\xrightarrow[\text{measurement}]{\;\{E_x\}\;}\;p(x\vert\theta)=\mathrm{Tr}\!\left(\rho_\theta E_x\right).$$
#
# The middle step is fixed by the physics of the device: a magnetic field $B$ acting for a time $t$ rotates the spins, so
# $\theta = \gamma B t$ with $G=J_z$; an optical path-length difference gives $\theta=2\pi \Delta L/\lambda$; and a clock
# compares an atomic frequency with a laser, so $\theta=(\omega_{\text{atom}}-\omega_{\text{laser}})t$. The last step is what
# we choose: a **POVM**, i.e. a set of positive operators $E_x\succeq0$ with $\sum_x E_x=\mathbb 1$ (see
# [08 — measurements](../ch03_matrix_free_engine/08_measurements.ipynb)). A projective measurement in some basis
# $\{\vert x\rangle\}$ is the special case $E_x=\vert x\rangle\langle x\vert$.
#
# Every choice of POVM yields a classical model and therefore a classical Fisher information
#
# $$I(\theta;\{E_x\})=\sum_x\frac{\left(\partial_\theta\mathrm{Tr}(\rho_\theta E_x)\right)^2}{\mathrm{Tr}(\rho_\theta E_x)},$$
#
# and, by the Cramér–Rao bound, a precision limit $\Delta\theta\ge1/\sqrt{M\,I(\theta;\{E_x\})}$. A *bad* measurement gives
# $I=0$: for a GHZ state with $G=J_z$, measuring every qubit along $z$ produces outcome statistics that do not depend on
# $\theta$ at all, so no estimator can extract the phase from that data. A *good* measurement makes $I$ as large as
# possible. The natural question is how large it can be.
#
# ### 6.1 The quantum Fisher information
#
# **Definition.** The **quantum Fisher information** of the family $\rho_\theta$ is
#
# $$F_Q[\rho_\theta]\;=\;\max_{\{E_x\}}\;I(\theta;\{E_x\}), \tag{12}$$
#
# the maximum of the classical Fisher information over all POVMs. It depends on the state and on how $\theta$ enters, but
# **not** on the measurement — the optimisation has been done once and for all.
#
# **Theorem (Braunstein and Caves 1994).** The maximum in Eq. (12) exists, equals
#
# $$F_Q=\mathrm{Tr}\!\left(\rho_\theta L_\theta^2\right)
#   \qquad\text{where } L_\theta \text{ solves } \partial_\theta\rho_\theta=\tfrac12\left(L_\theta\rho_\theta+\rho_\theta L_\theta\right),$$
#
# and is attained by the projective measurement in the eigenbasis of the Hermitian operator $L_\theta$, called the
# **symmetric logarithmic derivative** (SLD). The optimal measurement is *local* in the parameter: $L_\theta$ depends on
# $\theta$, so the basis that saturates the bound at one working point need not saturate it at another. An experiment
# reaches $F_Q$ by using a rough estimate of $\theta$ to choose the basis and then refining — adaptively, or by working at
# a fixed operating point as the interferometers of notebooks 31 and 32 do. Consequently
#
# $$\Delta\theta\;\ge\;\frac{1}{\sqrt{M\,I(\theta;\{E_x\})}}\;\ge\;\frac{1}{\sqrt{M\,F_Q}} , \tag{13}$$
#
# the **quantum Cramér–Rao bound**. It inherits the conditions of Eq. (9): $\hat\theta$ is (locally) unbiased as a
# function of all $M$ outcomes, and the model is regular at $\theta$. Measuring the $M$ copies *jointly* with an
# entangled POVM does not beat it either, because the quantum Fisher information is additive over independent copies,
# $F_Q[\rho_\theta^{\otimes M}]=M\,F_Q[\rho_\theta]$, so Eq. (12) applied to the $M$-copy state gives the same right-hand
# side. Equality in Eq. (13) is reached asymptotically, for large $M$, by measuring each copy in the optimal basis and
# processing the outcomes with the maximum-likelihood estimator of Section 5.3.
#
# We take this theorem on trust *here* and pay the debt in the next notebook,
# [30 — QFI from the SLD](../ch10_quantum_metrology_protocols/30_qfi_from_the_sld_prepare_encode_estimate.ipynb), which
# solves the equation for $L_\theta$ line by line, evaluates $\mathrm{Tr}(\rho L^2)$, builds the optimal measurement
# explicitly and shows numerically that its classical Fisher information equals $F_Q$ while other measurements fall short.
# For the rest of *this* notebook we only need one consequence of it, which we derive from scratch right now.

# %% [markdown]
# ## 7. Pure states and unitary encoding: $F_Q=4\,\mathrm{Var}(G)$
#
# Take a pure initial state $\rho=\vert\psi\rangle\langle\psi\vert$ and the unitary encoding
# $\vert\psi_\theta\rangle=e^{-i\theta G}\vert\psi\rangle$. We prove
#
# $$\boxed{\;F_Q\;=\;4\left(\langle\psi\vert G^2\vert\psi\rangle-\langle\psi\vert G\vert\psi\rangle^2\right)\;=\;4\,\mathrm{Var}_\psi(G)\;} \tag{14}$$
#
# in two ways: from the SLD theorem, and from the fidelity relation quoted in Eq. (15).
#
# ### 7.1 Route 1: straight from the SLD
#
# For a pure state $\rho_\theta^2=\rho_\theta$. Differentiate:
#
# $$\partial_\theta\rho=\left(\partial_\theta\rho\right)\rho+\rho\left(\partial_\theta\rho\right)
#   =\tfrac12\left[(2\partial_\theta\rho)\rho+\rho(2\partial_\theta\rho)\right],$$
#
# which is exactly the defining equation of the SLD with
#
# $$L=2\,\partial_\theta\rho .$$
#
# The derivative of the encoded state is a commutator,
#
# $$\partial_\theta\rho_\theta=\partial_\theta\left(e^{-i\theta G}\rho\,e^{i\theta G}\right)=-i\left[G,\rho_\theta\right],$$
#
# so, writing everything at $\theta=0$ (nothing depends on $\theta$, as we check below) and abbreviating
# $g=\langle G\rangle$, $g_2=\langle G^2\rangle$, $\rho=\vert\psi\rangle\langle\psi\vert$:
#
# $$\begin{aligned}
# (\partial_\theta\rho)^2 &= -\left(G\rho-\rho G\right)\left(G\rho-\rho G\right)
#   = -\left(G\rho G\rho-G\rho\rho G-\rho G G\rho+\rho G\rho G\right)\\
#  &= -\left(g\,G\rho-G\rho G-g_2\,\rho+g\,\rho G\right),
# \end{aligned}$$
#
# where we used $\rho\rho=\rho$ and $\rho G\rho=g\,\rho$. Taking $\mathrm{Tr}(\rho\;\cdot\;)$ term by term, with
# $\mathrm{Tr}(\rho G\rho)=g$, $\mathrm{Tr}(\rho G\rho G)=g^2$, $\mathrm{Tr}(\rho\rho)=1$ and $\mathrm{Tr}(\rho\rho G)=g$:
#
# $$\mathrm{Tr}\!\left(\rho\,(\partial_\theta\rho)^2\right)=-\left(g^2-g^2-g_2+g^2\right)=g_2-g^2=\mathrm{Var}(G).$$
#
# Hence $F_Q=\mathrm{Tr}(\rho L^2)=4\,\mathrm{Tr}\!\left(\rho(\partial_\theta\rho)^2\right)=4\,\mathrm{Var}(G)$, which is Eq. (14). $\square$
#
# ### 7.2 Route 2: the overlap (fidelity) expansion
#
# The quantum Fisher information is also the curvature of the **fidelity** between neighbouring states. This standard
# relation, which identifies $F_Q/4$ with the Bures metric (Braunstein and Caves 1994), is quoted here; notebook 30
# verifies it numerically for mixed states,
#
# $$F_Q=\lim_{d\theta\to0}\frac{8\left(1-\sqrt{F(\rho_\theta,\rho_{\theta+d\theta})}\right)}{d\theta^2},\qquad
#   F(\vert\psi\rangle,\vert\phi\rangle)=\left\vert\langle\psi\vert\phi\rangle\right\vert^2 . \tag{15}$$
#
# For pure states with unitary encoding the overlap is computable by hand. Expand the exponential to second order:
#
# $$\langle\psi\vert e^{-i\,d\theta\,G}\vert\psi\rangle
#  =1-i\,d\theta\,\langle G\rangle-\frac{d\theta^2}{2}\langle G^2\rangle+O(d\theta^3).$$
#
# Its squared modulus is (real part squared plus imaginary part squared)
#
# $$\begin{aligned}
# \left\vert\langle\psi\vert\psi_{d\theta}\rangle\right\vert^2
#  &=\left(1-\frac{d\theta^2}{2}\langle G^2\rangle\right)^2+d\theta^2\langle G\rangle^2+O(d\theta^3)\\
#  &=1-d\theta^2\left(\langle G^2\rangle-\langle G\rangle^2\right)+O(d\theta^3)
#   =1-d\theta^2\,\mathrm{Var}(G)+O(d\theta^3).
# \end{aligned}$$
#
# Therefore $\sqrt{F}=1-\tfrac12 d\theta^2\mathrm{Var}(G)+O(d\theta^3)$ and Eq. (15) gives
# $F_Q=8\cdot\tfrac12\mathrm{Var}(G)=4\,\mathrm{Var}(G)$, the same answer. $\square$
#
# ### 7.3 Two immediate corollaries
#
# **$F_Q$ does not depend on $\theta$.** $\mathrm{Var}_{\psi_\theta}(G)=\mathrm{Var}_\psi(e^{i\theta G}Ge^{-i\theta G})=\mathrm{Var}_\psi(G)$
# because $G$ commutes with its own exponential. So we may evaluate everything at $\theta=0$ and never encode at all when
# we only want $F_Q$ — a large practical saving.
#
# **A global phase is invisible.** Replacing $G\to G+c\,\mathbb 1$ changes $U$ by a phase and leaves $\mathrm{Var}(G)$
# unchanged. Only the *spread* of the generator matters, never its mean.
#
# ### 7.4 A worked check on one qubit
#
# One qubit, $\vert\psi\rangle=\vert+\rangle=(\vert0\rangle+\vert1\rangle)/\sqrt2$, generator $G=J_z=Z/2$. Then
# $\langle J_z\rangle=0$ and $J_z^2=\mathbb 1/4$, so $\mathrm{Var}(J_z)=1/4$ and Eq. (14) gives $F_Q=1$.
# A measurement along $x$ reaches this value. Encode and measure:
#
# $$\vert\psi_\theta\rangle=\frac{e^{-i\theta/2}\vert0\rangle+e^{+i\theta/2}\vert1\rangle}{\sqrt2},\qquad
# p(\pm\vert\theta)=\left\vert\langle\pm\vert\psi_\theta\rangle\right\vert^2=\frac{1\pm\cos\theta}{2},$$
#
# which is precisely the model of Eq. (10), whose Fisher information we computed to be $I=1$ for every $\theta$ except the
# two irregular points $\theta=0,\pi$. So the single-qubit interferometer saturates the quantum Cramér–Rao bound at every
# regular working point. That is the
# Ramsey protocol, and it is the subject of
# [31 — Ramsey interferometry](../ch10_quantum_metrology_protocols/31_ramsey_interferometry.ipynb).

# %% [markdown]
# ## 8. From formula to code: the QFI without any big matrix
#
# Equation (14) asks for two numbers, $\langle G^2\rangle$ and $\langle G\rangle$, for a collective generator
# $G=\mathbf n\cdot\mathbf J$ with $J_a=\tfrac12\sum_q\sigma^a_q$. The textbook route would build a $2^N\times2^N$ matrix
# (memory $16\cdot4^N$ bytes in complex double precision: $4.3$ GB at $N=14$, $69$ GB at $N=16$). The matrix-free route needs only the *action* of $G$:
#
# * apply $\sigma^a$ to each axis in turn and add up, $\;\vert\phi\rangle=G\vert\psi\rangle=\tfrac12\sum_q\sigma^a_q\vert\psi\rangle$ — that is
#   `apply_collective`, $N$ einsums, cost $O(N2^N)$;
# * then $\langle G\rangle=\langle\psi\vert\phi\rangle$ and $\langle G^2\rangle=\langle\phi\vert\phi\rangle$, because $G$ is Hermitian:
#   $\langle\psi\vert G^2\vert\psi\rangle=\langle G\psi\vert G\psi\rangle$.
#
# Only two inner products are needed and no operator is stored. This is the engine's `qfi_pure`, whose convention we must read carefully: it applies
# $P_q$ *without* the factor $\tfrac12$, i.e. it returns
#
# $$\texttt{qfi\_pure}(\psi,P)=\langle \tilde G^2\rangle-\langle \tilde G\rangle^2\quad\text{with }\tilde G=\sum_q P_q=2J_a,$$
#
# which is $4\,\mathrm{Var}(J_a)=F_Q$ in our convention: the engine already returns the quantum Fisher information for
# $G=J_a$, and the two factors of $2$ cancel exactly as they should.
#
# For a direction $\mathbf n=(n_x,n_y,n_z)$ we simply feed it the single-qubit matrix $P=n_x X+n_y Y+n_z Z$, since
# $\sum_q(\mathbf n\cdot\vec\sigma)_q=2\,\mathbf n\cdot\mathbf J$.

# %%
# ==============================================================================
# STEP 4: QFI of a pure state, matrix-free, and its dense validation
# ==============================================================================
def pauli_direction(n):
    """Single-qubit matrix n.sigma = n_x X + n_y Y + n_z Z for a unit vector n (used as the generator direction)."""
    n = jnp.asarray(n, dtype=CDTYPE)
    return n[0] * X + n[1] * Y + n[2] * Z


def qfi_pure_direction(psi, n):
    """QFI of a pure state for the collective generator G = n.J = (1/2) sum_q (n.sigma)_q.

    MATH   F_Q = 4 Var(G) = <(2G)^2> - <2G>^2   with  2G = sum_q (n.sigma)_q.
    COST   O(N 2^N) time, O(2^N) memory -- no 2^N x 2^N matrix is ever created.
    """
    return qfi_pure(psi, pauli_direction(n))


def qfi_dense_reference(psi, n):
    """Independent DENSE reference (validation only, small N): build J = (1/2) sum_q (n.sigma)_q as a matrix
    and evaluate 4(<J^2> - <J>^2) with ordinary linear algebra."""
    N = psi.ndim
    J = collective_dense(pauli_direction(n), N)
    v = psi.reshape(-1)
    m1 = jnp.real(jnp.vdot(v, J @ v))
    m2 = jnp.real(jnp.vdot(v, J @ (J @ v)))
    return 4 * (m2 - m1 ** 2)


# --- CHECKPOINT: matrix-free vs dense vs the mixed-state formula on a pure density matrix ---------
DIRS = {"z": (0.0, 0.0, 1.0), "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0),
        "tilted": tuple(np.array([1.0, 2.0, -2.0]) / 3.0)}
print(f"{'state':>10s} {'dir':>7s} | {'matrix-free':>13s} {'dense 4Var(J)':>15s} {'SLD formula':>13s} {'max err':>10s}")
for name, psi in (("GHZ(5)", ghz_state(5)), ("W(5)", w_state(5)), ("Haar(5)", haar_state(jax.random.PRNGKey(0), 5))):
    for dname, n in DIRS.items():
        f_free = float(qfi_pure_direction(psi, n))
        f_dense = float(qfi_dense_reference(psi, n))
        f_sld = float(qfi_mixed(dm_matrix(to_dm(psi)), collective_dense(pauli_direction(n), psi.ndim)))
        err = max(abs(f_free - f_dense), abs(f_free - f_sld))
        print(f"{name:>10s} {dname:>7s} | {f_free:13.8f} {f_dense:15.8f} {f_sld:13.8f} {err:10.2e}")
        assert err < 1e4 * TOL

# %% [markdown]
# Three independent routes — the matrix-free variance, the dense variance, and the general mixed-state
# (symmetric-logarithmic-derivative) formula applied to a pure density matrix — agree to $10^{-14}$ or better for every state and
# direction. From here on we trust `qfi_pure_direction` and use it freely, with one reservation: all three routes obtain
# the generator from the same primitive `apply_collective`, so this test proves that the three *formulas* agree; it does
# not test whether the collective operator itself is built correctly. That gap is closed in Section 14, where the same quantity is
# recomputed from a state encoded gate by gate with `apply_gate` and differentiated by automatic differentiation.
#
# > **JAX practice.** `apply_collective` loops over qubits in *Python*, but the loop index is a static integer that only
# > builds einsum strings: after `jax.jit` the whole sum is a single fused XLA program with no Python left in it. Static
# > structure and traced data are what make this simulator fast.

# %% [markdown]
# ## 9. The two limits: standard quantum limit and Heisenberg limit
#
# ### 9.1 Product states: $F_Q\le N$ (the standard quantum limit)
#
# **Claim.** If $\vert\psi\rangle=\vert\psi_0\rangle\otimes\cdots\otimes\vert\psi_{N-1}\rangle$ is a product state and
# $G=J_{\mathbf n}=\tfrac12\sum_q(\mathbf n\cdot\vec\sigma)_q$, then $F_Q\le N$, with equality when every qubit lies on the
# equator perpendicular to $\mathbf n$.
#
# *Proof.* The generator is a **sum of single-qubit terms**, $G=\sum_q g_q$ with $g_q=\tfrac12(\mathbf n\cdot\vec\sigma)_q$.
# For a product state the qubits are statistically independent, so the variance of a sum is the sum of variances:
#
# $$\mathrm{Var}(G)=\sum_q\mathrm{Var}(g_q).$$
#
# (Write $\mathrm{Var}(G)=\sum_{q,q'}\left[\langle g_qg_{q'}\rangle-\langle g_q\rangle\langle g_{q'}\rangle\right]$; for
# $q\neq q'$ the two operators act on different factors of a product state, so $\langle g_qg_{q'}\rangle=\langle g_q\rangle\langle g_{q'}\rangle$
# and the term vanishes.) For a single qubit, $(\mathbf n\cdot\vec\sigma)^2=\mathbb 1$ for a unit vector $\mathbf n$, hence
# $g_q^2=\tfrac14\mathbb 1$ and
#
# $$\mathrm{Var}(g_q)=\tfrac14-\langle g_q\rangle^2\le\tfrac14 .$$
#
# Summing, $\mathrm{Var}(G)\le N/4$ and $F_Q=4\mathrm{Var}(G)\le N$. Equality needs $\langle g_q\rangle=0$ for all $q$,
# i.e. every Bloch vector perpendicular to $\mathbf n$. $\square$
#
# The same bound holds for every **separable** state, pure or mixed, by the convexity of the QFI (quoted here, demonstrated in
# notebook 30): a separable state is a mixture of product states, and a mixture cannot have more QFI than the average of its
# parts. Hence the contrapositive, which is the entanglement witness of Section 12:
#
# $$F_Q>N\;\Longrightarrow\;\text{the state is entangled.}$$
#
# Since $\Delta\theta\ge1/\sqrt{MF_Q}$, the best classical-resource precision is $\Delta\theta=1/\sqrt{MN}$ — the
# **standard quantum limit** (SQL), also called the shot-noise limit.
#
# ### 9.2 Everything: $F_Q\le N^2$ (the Heisenberg limit)
#
# **Claim.** For *any* state of $N$ qubits and $G=J_{\mathbf n}$, $\;F_Q\le N^2$.
#
# *Proof.* $J_{\mathbf n}$ is a sum of $N$ commuting single-qubit operators with eigenvalues $\pm\tfrac12$, so its spectrum
# lies in $[-N/2,+N/2]$. For any random variable $A$ bounded by $a_{\min}\le A\le a_{\max}$,
#
# $$\mathrm{Var}(A)\le\left(\frac{a_{\max}-a_{\min}}{2}\right)^2 ,$$
#
# (Popoviciu's inequality; the quickest argument: $\mathrm{Var}(A)=\mathbb{E}[(A-c)^2]-(\mathbb{E}[A]-c)^2\le\mathbb{E}[(A-c)^2]$
# for the midpoint $c=(a_{\max}+a_{\min})/2$, and $\vert A-c\vert\le(a_{\max}-a_{\min})/2$ pointwise). With
# $a_{\max}-a_{\min}=N$ we get $\mathrm{Var}(J_{\mathbf n})\le N^2/4$ and $F_Q\le N^2$. $\square$
#
# Equality requires the state to put all its weight on the two extreme eigenvalues $\pm N/2$ with equal probability — which
# is exactly what the **GHZ state** does for $\mathbf n=\hat z$:
#
# $$\vert\mathrm{GHZ}\rangle=\frac{\vert0\cdots0\rangle+\vert1\cdots1\rangle}{\sqrt2},\qquad
#   J_z\vert0\cdots0\rangle=+\frac N2\vert0\cdots0\rangle,\quad J_z\vert1\cdots1\rangle=-\frac N2\vert1\cdots1\rangle,$$
#
# so $\langle J_z\rangle=0$, $\langle J_z^2\rangle=N^2/4$, $F_Q=N^2$. The precision limit becomes $\Delta\theta=1/(N\sqrt M)$,
# the **Heisenberg limit** — a factor $\sqrt N$ better than any unentangled state can do.
#
# **What is counted.** Both limits compare strategies at a fixed number of uses of the phase-imprinting interaction. In
# the parallel scheme used throughout this notebook each of the $N$ qubits acquires the phase exactly once per run
# (the generator $J_{\mathbf n}$ has one term $\tfrac12\mathbf n\cdot\vec\sigma_q$ per qubit), and the run is repeated $M$
# times, so the resource is $\nu=MN$ single-qubit phase imprints. The SQL is then $\Delta\theta=1/\sqrt{\nu}$ and the
# Heisenberg limit $\Delta\theta=\sqrt M/\nu$. Neither the interrogation time, nor ancilla qubits that do not see the
# phase, nor the cost of preparing the state enters the count; ancillas leave the spectral width of the generator, and
# therefore the bound $F_Q\le N^2$, unchanged. Giovannetti, Lloyd and Maccone (2006) count resources in exactly this way,
# as the number of times the system is sampled, and also discuss a sequential protocol in which a single probe passes
# through the phase shift $N$ times; it reaches the same $1/N$ scaling with the same number of uses and no entanglement
# between probes.
#
# Let us measure both limits.

# %%
# ==============================================================================
# STEP 5: the two limits, measured -- product state vs GHZ, N = 2 ... 14
# ==============================================================================
N_SCALE = range(2, 15)
rows_scale = []
for N in N_SCALE:
    f_prod = float(qfi_pure_direction(product_state("+" * N), (0.0, 1.0, 0.0)))   # |+>^N, generator J_y (perpendicular)
    f_ghz = float(qfi_pure_direction(ghz_state(N), (0.0, 0.0, 1.0)))              # GHZ, generator J_z
    rows_scale.append((N, f_prod, f_ghz))
    assert abs(f_prod - N) < 1e3 * TOL and abs(f_ghz - N ** 2) < 1e3 * TOL

print(f"{'N':>3s} | {'F_Q(product, J_y)':>18s} {'N':>6s} | {'F_Q(GHZ, J_z)':>14s} {'N^2':>6s} | {'ratio':>6s}")
for N, fp, fg in rows_scale:
    print(f"{N:3d} | {fp:18.6f} {N:6d} | {fg:14.6f} {N ** 2:6d} | {fg / fp:6.2f}")

# %%
# ==============================================================================
# FIGURE: standard quantum limit vs Heisenberg limit
# ==============================================================================
Ns = np.array([r[0] for r in rows_scale], dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
axes[0].loglog(Ns, [r[1] for r in rows_scale], "o", color=PALETTE[0], ms=6, label=r"$\vert+\rangle^{\otimes N}$, $G=J_y$")
axes[0].loglog(Ns, [r[2] for r in rows_scale], "s", color=PALETTE[1], ms=6, label=r"GHZ$_N$, $G=J_z$")
axes[0].loglog(Ns, Ns, "--", color="0.4", lw=1, label=r"$N$ (standard quantum limit)")
axes[0].loglog(Ns, Ns ** 2, ":", color="0.2", lw=1.2, label=r"$N^2$ (Heisenberg limit)")
axes[0].set_xlabel("number of qubits $N$"); axes[0].set_ylabel(r"$F_Q$")
axes[0].set_title("Quantum Fisher information"); axes[0].legend(fontsize=9)

axes[1].loglog(Ns, 1 / np.sqrt(np.array([r[1] for r in rows_scale])), "o-", color=PALETTE[0], ms=5,
               label=r"product: $\Delta\theta=1/\sqrt{N}$")
axes[1].loglog(Ns, 1 / np.sqrt(np.array([r[2] for r in rows_scale])), "s-", color=PALETTE[1], ms=5,
               label=r"GHZ: $\Delta\theta=1/N$")
axes[1].set_xlabel("number of qubits $N$"); axes[1].set_ylabel(r"$\Delta\theta$ at $M=1$")
axes[1].set_title("Best achievable phase uncertainty"); axes[1].legend(fontsize=9)
fig.tight_layout(); plt.show()

# %% [markdown]
# The measured values are exactly $N$ and $N^2$ (the asserts demand agreement to $10^{-7}$), so both proofs of Section 9 are
# confirmed and both bounds are tight. The right panel translates them into the quantity an experimentalist cares about:
# with $N=14$ qubits the GHZ state promises a phase uncertainty $3.7$ times smaller than the best product state, and the
# advantage grows as $\sqrt N$.
#
# > **Physics insight.** $F_Q=N^2$ is the *ideal* value of the Heisenberg limit; Sections 13
# > and 14 show how quickly it collapses under noise and particle loss, and notebook 32 shows that even the ideal GHZ phase
# > estimate suffers from an $N$-fold ambiguity, because its signal oscillates $N$ times faster.

# %% [markdown]
# ## 10. A zoo of many-body states
#
# This section sorts the standard many-body states by metrological usefulness. We evaluate $F_Q$ for the standard family of $N$-qubit states, each with the
# generator that suits it, and compare against $N$ and $N^2$. Beyond the two extremes we already know, four entries deserve a
# prediction before we look:
#
# * **W state** $\vert W\rangle=\vert D_N^1\rangle$ (one excitation shared by all qubits). It is an eigenstate of $J_z$
#   (eigenvalue $N/2-1$), so $F_Q=0$ for $G=J_z$ — a perfect illustration that the generator matters as much as the state.
#   For $G=J_x$ we use the angular-momentum identity $\langle J_x^2\rangle=\tfrac12\left(\langle\mathbf J^2\rangle-\langle J_z^2\rangle\right)$
#   valid in the symmetric sector (where $\langle J_x^2\rangle=\langle J_y^2\rangle$), with $j=N/2$, $m=N/2-1$:
#
#   $$F_Q=4\cdot\tfrac12\left[\tfrac N2\left(\tfrac N2+1\right)-\left(\tfrac N2-1\right)^2\right]=3N-2 .$$
#
# * **Dicke state at half filling** $\vert D_N^{N/2}\rangle$ ($m=0$, $N$ even). The same identity gives
#
#   $$F_Q=4\cdot\tfrac12\cdot\tfrac N2\left(\tfrac N2+1\right)=\frac{N(N+2)}{2},$$
#
#   i.e. Heisenberg *scaling* (asymptotically $N^2/2$). Section 14 shows that, unlike GHZ, this state keeps a sizeable
#   part of its QFI when a particle is lost.
# * **Cluster (graph) state.** A genuinely multipartite entangled stabiliser state, yet all its $\langle\sigma^a_q\rangle$ and most
#   two-body correlators vanish — we will see what that does to $F_Q$.
# * **Haar-random pure state.** For a random $\vert\psi\rangle$ the standard unitary-averaging identities
#   $\mathbb{E}\langle A\rangle=\mathrm{Tr}A/d$ and $\mathbb{E}\langle A\rangle^2=\left[(\mathrm{Tr}A)^2+\mathrm{Tr}A^2\right]/[d(d+1)]$
#   with $d=2^N$, $\mathrm{Tr}J_z=0$, $\mathrm{Tr}J_z^2=dN/4$ give the exact average
#
#   $$\mathbb{E}\left[F_Q\right]=4\left(\frac N4-\frac{N}{4(d+1)}\right)=N\,\frac{d}{d+1}\;\xrightarrow[N\to\infty]{}\;N .$$
#
#   A typical state of the Hilbert space is nearly maximally entangled (its half-chain entropy is close to the Page value)
#   and yet, on average, worth slightly *less* than a product state.

# %%
# ==============================================================================
# STEP 6: the zoo -- QFI of standard many-body states, and the Haar average
# ==============================================================================
N_ZOO = 8
N_HAAR_SAMPLES = 64

# Printed tables use ASCII names; figures use the LaTeX labels of LBL below.
LBL = {"|0>^N": r"$\vert 0\rangle^{\otimes N}$", "|+>^N": r"$\vert +\rangle^{\otimes N}$", "GHZ": "GHZ",
       "W": "W", "Dicke N/2": r"Dicke $N/2$", "cluster": "cluster", "Haar": "Haar random"}
zoo = {
    "|0>^N": (product_state("0" * N_ZOO), (0.0, 1.0, 0.0), "J_y"),
    "|+>^N": (product_state("+" * N_ZOO), (0.0, 1.0, 0.0), "J_y"),
    "GHZ": (ghz_state(N_ZOO), (0.0, 0.0, 1.0), "J_z"),
    "W": (w_state(N_ZOO), (1.0, 0.0, 0.0), "J_x"),
    "Dicke N/2": (dicke_state(N_ZOO, N_ZOO // 2), (1.0, 0.0, 0.0), "J_x"),
    "cluster": (cluster_state(N_ZOO), (0.0, 0.0, 1.0), "J_z"),
}
print(f"N = {N_ZOO}   (SQL: F_Q = {N_ZOO}, Heisenberg: F_Q = {N_ZOO ** 2})\n")
print(f"{'state':>12s} {'G':>5s} | {'F_Q':>10s} {'F_Q/N':>8s} {'F_Q/N^2':>9s} {'entangled?':>11s}")
for name, (psi, n, gname) in zoo.items():
    f = float(qfi_pure_direction(psi, n))
    print(f"{name:>12s} {gname:>5s} | {f:10.4f} {f / N_ZOO:8.3f} {f / N_ZOO ** 2:9.4f} "
          f"{'YES' if f > N_ZOO + 1e-6 else 'not by QFI':>11s}")

# Haar average: vmap over 64 independent random states (one PRNG key each)
keys = jax.random.split(jax.random.PRNGKey(0), N_HAAR_SAMPLES)
f_haar = jax.vmap(lambda k: qfi_pure_direction(haar_state(k, N_ZOO), (0.0, 0.0, 1.0)))(keys)
d, mean_haar, std_haar = 2 ** N_ZOO, float(jnp.mean(f_haar)), float(jnp.std(f_haar))
print(f"{'Haar (mean)':>12s} {'J_z':>5s} | {mean_haar:10.4f} {mean_haar / N_ZOO:8.3f} "
      f"{mean_haar / N_ZOO ** 2:9.4f} {'':>11s}")
sem_haar = std_haar / np.sqrt(N_HAAR_SAMPLES)
print(f"\nHaar average: measured {mean_haar:.4f} +- {sem_haar:.4f} (standard error of the mean "
      f"over {N_HAAR_SAMPLES} states, sample std {std_haar:.3f});  exact N d/(d+1) = {N_ZOO * d / (d + 1):.4f}")
# the measured mean must agree with the exact average within four standard errors
assert abs(mean_haar - N_ZOO * d / (d + 1)) < 4 * sem_haar

# --- CHECKPOINT: the analytic values predicted above ---------------------------------
assert abs(float(qfi_pure_direction(w_state(N_ZOO), (1.0, 0.0, 0.0))) - (3 * N_ZOO - 2)) < 1e3 * TOL
assert abs(float(qfi_pure_direction(dicke_state(N_ZOO, N_ZOO // 2), (1.0, 0.0, 0.0)))
           - N_ZOO * (N_ZOO + 2) / 2) < 1e3 * TOL
print("\nanalytic checks passed:  W -> 3N-2 = %d   Dicke(N/2) -> N(N+2)/2 = %d"
      % (3 * N_ZOO - 2, N_ZOO * (N_ZOO + 2) // 2))

# %%
# ==============================================================================
# FIGURE: how the zoo scales with N
# ==============================================================================
N_RANGE = np.arange(2, 13)
curves = {
    r"$\vert+\rangle^{\otimes N}$, $J_y$": lambda N: qfi_pure_direction(product_state("+" * N), (0, 1, 0)),
    r"GHZ, $J_z$": lambda N: qfi_pure_direction(ghz_state(N), (0, 0, 1)),
    r"W, $J_x$": lambda N: qfi_pure_direction(w_state(N), (1, 0, 0)),
    r"Dicke $N/2$, $J_x$": lambda N: qfi_pure_direction(dicke_state(N, N // 2), (1, 0, 0)),
    r"cluster, $J_z$": lambda N: qfi_pure_direction(cluster_state(N), (0, 0, 1)),
}
haar_mean = np.array([float(jnp.mean(jax.vmap(lambda kk: qfi_pure_direction(haar_state(kk, N), (0, 0, 1)))(
    jax.random.split(jax.random.PRNGKey(N), 32)))) for N in N_RANGE])

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
for k, (lab, fn) in enumerate(curves.items()):
    Nl = np.array([N for N in N_RANGE if (N % 2 == 0 or "Dicke" not in lab)])   # Dicke N/2 needs even N
    vals = np.array([float(fn(N)) for N in Nl])
    axes[0].loglog(Nl, vals, MARKERS[k] + "-", color=PALETTE[k], ms=5, label=lab)
    axes[1].semilogx(Nl, vals / Nl, MARKERS[k] + "-", color=PALETTE[k], ms=5, label=lab)
axes[0].loglog(N_RANGE, haar_mean, "*", color=PALETTE[5], ms=9, label="Haar random (mean of 32)")
axes[1].semilogx(N_RANGE, haar_mean / N_RANGE, "*", color=PALETTE[5], ms=9, label="Haar random (mean of 32)")
axes[0].loglog(N_RANGE, N_RANGE, "--", color="0.4", lw=1.2, label=r"$N$ (SQL)")
axes[0].loglog(N_RANGE, N_RANGE ** 2.0, ":", color="0.2", lw=1.4, label=r"$N^2$ (Heisenberg)")
axes[0].set_xlabel("number of qubits $N$"); axes[0].set_ylabel(r"$F_Q$")
axes[0].set_title("Quantum Fisher information"); axes[0].legend(fontsize=8, ncol=2)
axes[1].axhline(1.0, color="0.4", ls="--", lw=1.2)
axes[1].text(6.0, 1.06, "standard quantum limit", fontsize=8, color="0.4")
axes[1].set_yscale("log"); axes[1].set_xlabel("number of qubits $N$"); axes[1].set_ylabel(r"$F_Q/N$")
axes[1].set_title("QFI density relative to the SQL"); axes[1].legend(fontsize=8)
fig.suptitle("Metrological usefulness of standard many-body states", y=1.02)
fig.tight_layout(); plt.show()

# %% [markdown]
# The figure, read from the bottom up:
#
# * $\vert+\rangle^{\otimes N}$ with $G=J_y$ sits exactly on the SQL line: the best an unentangled state can do.
# * The **cluster state** also sits exactly on $N$, even though it is a genuinely multipartite entangled stabiliser state —
#   the standard resource for measurement-based quantum computation. Its collective fluctuations are precisely those of independent
#   qubits, because all one-body expectation values and all same-Pauli two-body correlators vanish. **Entanglement is
#   necessary but far from sufficient for metrological gain.** (Section 11 finds a tilted direction where the
#   cluster state does a little better than $N$; the excess is a boundary term of size $2$ that does not change the scaling.)
# * The **Haar-random** points also collapse onto $N$, with a scatter that shrinks as $N$ grows; the measured mean at
#   $N=8$, $7.94\pm0.08$ over $64$ states, is consistent with the exact prediction $N d/(d+1)=7.97$. A state drawn at
#   random from the Hilbert space is nearly maximally entangled and offers no gain over a product state — the useful
#   states form a vanishingly small, highly structured subset.
# * The **W state** at $3N-2$ beats the SQL but still scales linearly: $F_Q/N=3-2/N\to3$, a constant-factor gain over the
#   SQL that does not grow with $N$, while its share of the Heisenberg value, $(3N-2)/N^2$, vanishes.
# * The **Dicke state** at $N(N+2)/2$ and the **GHZ state** at $N^2$ are the two Heisenberg-scaling families, differing by
#   a factor of about $2$.
#
# > **Physics insight.** The QFI is a property of the *pair* (state, generator). The same W state is worth $3N-2$ with
# > $G=J_x$ and exactly $0$ with $G=J_z$. Section 11 turns this observation into an algorithm for finding the best generator.

# %% [markdown]
# ### 10.1 Ground states as a metrological resource
#
# Many-body ground states are states a laboratory can reach by preparing the lowest-energy state of a tunable Hamiltonian
# (for instance by an adiabatic ramp of a field) rather than by circuit engineering, which makes them attractive resources. Take the transverse-field Ising chain (our Pauli convention, see
# [11 — Hamiltonians and ground states](../ch05_ground_states_and_unitary_dynamics/11_hamiltonians_and_ground_states.ipynb)):
#
# $$H(h)=-\sum_{q=0}^{N-2}Z_qZ_{q+1}-h\sum_{q=0}^{N-1}X_q .$$
#
# For $h\ll1$ the ground state is the symmetric superposition of the two ferromagnetic configurations — a macroscopic
# Schrödinger-cat state, essentially a GHZ state dressed by quantum fluctuations. For $h\gg1$ it is $\vert+\rangle^{\otimes N}$,
# a product state. Somewhere in between (at $h=1$ in the limit $N\to\infty$) sits a quantum critical point. We compute
# $F_Q$ for $G=J_z$ along the way with the Lanczos ground-state solver.

# %%
# ==============================================================================
# STEP 7: QFI of transverse-field Ising ground states across the transition
# ==============================================================================
H_FIELDS = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
N_GS = (8, 10)
gs_rows = {}
t0 = time.time()
for N in N_GS:
    row = []
    for h in H_FIELDS:
        terms = heisenberg_terms(N, Jxx=0.0, Jyy=0.0, Jzz=-1.0, hx=-h)     # -ZZ bonds, -h X fields
        E, psi = lanczos_ground_state(terms, N, m=80, key=jax.random.PRNGKey(7), restarts=3)
        row.append((h, float(E), float(qfi_pure_direction(psi, (0, 0, 1))),
                    float(qfi_pure_direction(psi, (0, 1, 0)))))
    gs_rows[N] = row
print(f"(ground states computed in {time.time() - t0:.1f} s)\n")
for N in N_GS:
    print(f"N = {N}:")
    print(f"  {'h':>5s} {'E_0':>10s} | {'F_Q(J_z)':>10s} {'F_Q/N':>8s} {'F_Q/N^2':>9s} | {'F_Q(J_y)':>10s} {'F_Q/N':>8s}")
    for h, E, fz, fy in gs_rows[N]:
        print(f"  {h:5.2f} {E:10.4f} | {fz:10.4f} {fz / N:8.3f} {fz / N ** 2:9.4f} | {fy:10.4f} {fy / N:8.3f}")
    print()

# %%
# ==============================================================================
# FIGURE: QFI density of the Ising ground state across the quantum phase transition
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
for k, N in enumerate(N_GS):
    hs = [r[0] for r in gs_rows[N]]
    axes[0].plot(hs, [r[2] / N ** 2 for r in gs_rows[N]], MARKERS[k] + "-", color=PALETTE[k], label=f"$N={N}$")
    axes[1].plot(hs, [r[2] / N for r in gs_rows[N]], MARKERS[k] + "-", color=PALETTE[k], label=f"$N={N}$, $G=J_z$")
    axes[1].plot(hs, [r[3] / N for r in gs_rows[N]], MARKERS[k] + "--", color=PALETTE[k + 2], label=f"$N={N}$, $G=J_y$")
axes[0].axvline(1.0, color="0.5", ls=":", lw=1.2)
axes[0].text(1.03, 0.9, "$h=1$", fontsize=9, color="0.4")
axes[0].set_xlabel("transverse field $h$"); axes[0].set_ylabel(r"$F_Q/N^2$")
axes[0].set_title("Ising ground state: Heisenberg fraction"); axes[0].legend(fontsize=9)
axes[1].axhline(1.0, color="0.5", ls="--", lw=1.2)
axes[1].text(2.2, 1.05, "standard quantum limit", fontsize=8, color="0.4")
axes[1].set_xlabel("transverse field $h$"); axes[1].set_ylabel(r"$F_Q/N$")
axes[1].set_yscale("log"); axes[1].set_title("QFI density"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# Deep in the ferromagnetic phase ($h=0.2$) the ground state reaches $F_Q/N^2\approx0.98$: the finite chain really is a
# near-perfect GHZ state, and preparing the ground state of an Ising magnet is an alternative way to make one. As $h$ grows the cat is destroyed —
# the ratio $F_Q/N^2$ falls monotonically through the transition region and, by $h=3$, $F_Q$ has dropped to about $1.4N$
# ($1.38N$ at $N=8$, $1.39N$ at $N=10$). $F_Q/N$ for $G=J_z$ stays above $1$ everywhere we look: the Ising ground state is entangled at every
# finite field, and the QFI witnesses it.
#
# The perpendicular generator $J_y$ tells a different and initially surprising story: its QFI density is *not* monotonic.
# It starts at $F_Q(J_y)/N\approx0.98$ at $h=0.2$, falls to a minimum of $\approx0.32$ at $h=1$, and climbs back to
# $\approx0.7$ at $h=3$. Both ends are easy: at $h\to0$ the state is the cat $(\vert0\cdots0\rangle+\vert1\cdots1\rangle)/\sqrt2$,
# for which $\mathrm{Var}(J_y)=N/4$ exactly, the same as for a product state; at $h\to\infty$ the state is
# $\vert+\rangle^{\otimes N}$, for which $\mathrm{Var}(J_y)=N/4$ as well. In between, the transverse fluctuations are
# reduced by the interaction. Since $\langle Y_q\rangle=0$ by symmetry, Eq. (14) reads
# $F_Q(J_y)=N+\sum_{q\neq q'}\langle Y_qY_{q'}\rangle$, so a density below $1$ means a net *negative* $y$-$y$ correlation;
# it is largest in magnitude in the critical region near $h=1$, where the correlation length is longest. Reading the two generators together: the transition transfers
# metrological weight from $J_y$ into $J_z$ as $h$ decreases, and the critical region is where the transfer happens.
#
# > **Numerical practice.** In the ordered phase the finite chain has two almost degenerate lowest states of opposite
# > $\prod_q X_q$ parity — the symmetric and the antisymmetric cat — split by about $5\times10^{-6}$ at $N=8$, $h=0.2$.
# > Which of the two a solver returns does not matter here: both are parity eigenstates and both give the same
# > $F_Q\approx0.98N^2$. What *would* matter is landing on a symmetry-broken superposition
# > $\alpha\vert\!\uparrow\cdots\uparrow\rangle+\beta\vert\!\downarrow\cdots\downarrow\rangle$: then
# > $F_Q\approx N^2\left(1-(\vert\alpha\vert^2-\vert\beta\vert^2)^2\right)$, anywhere between $N^2$ and $0$, and for the
# > fully polarised choice it collapses to a number of order $1$. Lanczos with a fixed seed and several restarts stayed on
# > a parity eigenstate here — the printed energies and the QFI are smooth in $h$ — but with a poorer solver one can easily
# > land anywhere in that family. Always check that a computed quantity is smooth in the parameter before interpreting it.

# %% [markdown]
# ## 11. The $3\times3$ QFI matrix and the optimal generator direction
#
# So far we chose the generator direction $\mathbf n$ by hand. We can do better: for collective generators the dependence on
# $\mathbf n$ is *quadratic*, so the optimisation over the whole sphere reduces to a $3\times3$ eigenvalue problem.
#
# ### 11.1 Derivation
#
# Let $G=\mathbf n\cdot\mathbf J=\sum_a n_aJ_a$ with $\mathbf n$ a real unit vector. Then
#
# $$\mathrm{Var}(G)=\langle G^2\rangle-\langle G\rangle^2
# =\sum_{a,b}n_an_b\left[\tfrac12\langle J_aJ_b+J_bJ_a\rangle-\langle J_a\rangle\langle J_b\rangle\right],$$
#
# where the symmetrisation is free: $n_an_b$ is symmetric under $a\leftrightarrow b$, so only the symmetric part of
# $\langle J_aJ_b\rangle$ survives the double sum. Defining the real symmetric **covariance matrix**
#
# $$C_{ab}=\tfrac12\langle J_aJ_b+J_bJ_a\rangle-\langle J_a\rangle\langle J_b\rangle \tag{16}$$
#
# and the **QFI matrix**
#
# $$\mathcal{F}_{ab}=4\,C_{ab}, \tag{17}$$
#
# we get the compact statement
#
# $$F_Q(\mathbf n)=\mathbf n^{\mathsf T}\,\mathcal{F}\,\mathbf n . \tag{18}$$
#
# Maximising a quadratic form over unit vectors is the Rayleigh-quotient problem: since $\mathcal F$ is real symmetric it has
# an orthonormal eigenbasis $\mathcal F\mathbf v_k=\lambda_k\mathbf v_k$ with $\lambda_1\le\lambda_2\le\lambda_3$; expanding
# $\mathbf n=\sum_k c_k\mathbf v_k$ with $\sum_kc_k^2=1$ gives $\mathbf n^{\mathsf T}\mathcal F\mathbf n=\sum_k\lambda_kc_k^2\le\lambda_3$,
# with equality for $\mathbf n=\mathbf v_3$. Hence
#
# $$\max_{\mathbf n}F_Q(\mathbf n)=\lambda_{\max}(\mathcal F),\qquad
#   \mathbf n_{\text{opt}}=\text{eigenvector of }\mathcal F\text{ for }\lambda_{\max} . \tag{19}$$
#
# Three matrix elements per pair of axes, one $3\times3$ `eigh`, and the optimisation over the sphere is done exactly.
#
# The identification $\mathcal F=4C$ of Eq. (17) uses $F_Q=4\,\mathrm{Var}(G)$ and is therefore valid **only for pure
# states**. The eigenvalue argument itself survives for mixed states, because the mixed-state formula of Eq. (22) below is
# also quadratic in the generator: $F_Q(\mathbf n)=\mathbf n^{\mathsf T}\Gamma\mathbf n$ with
#
# $$\Gamma_{ab}=2\sum_{m,n\,:\,\lambda_m+\lambda_n>0}\frac{(\lambda_m-\lambda_n)^2}{\lambda_m+\lambda_n}\,
#   \mathrm{Re}\!\left[\langle m\vert J_a\vert n\rangle\langle n\vert J_b\vert m\rangle\right].$$
#
# For a mixed state $\Gamma$ is in general *smaller* than $4C$, so feeding a covariance matrix into Eq. (19) returns an
# upper bound on the optimal QFI rather than the optimal QFI. Notebook 30 makes that gap explicit. Everything in this
# section is applied to pure states only, where the two coincide.
#
# ### 11.2 Code
#
# The engine's `spin_moments` returns exactly $\langle\mathbf J\rangle$ and the symmetrised covariance $C$ of Eq. (16),
# computed matrix-free: it forms $\vert\phi_a\rangle=J_a\vert\psi\rangle$ for $a=x,y,z$ (three calls of `apply_collective`) and
# reads off $\tfrac12\langle J_aJ_b+J_bJ_a\rangle=\mathrm{Re}\,\langle\phi_a\vert\phi_b\rangle$ — the real part of an inner
# product *is* the symmetrised moment, because $\langle\phi_a\vert\phi_b\rangle=\langle J_aJ_b\rangle$ and
# $\langle J_bJ_a\rangle=\overline{\langle J_aJ_b\rangle}$.

# %%
# ==============================================================================
# STEP 8: the 3x3 QFI matrix and the optimal generator direction
# ==============================================================================
def qfi_matrix(psi):
    """3x3 QFI matrix of a pure state over the collective generators (J_x, J_y, J_z).

    MATH   Fcal[a,b] = 4 * ( (1/2)<J_a J_b + J_b J_a> - <J_a><J_b> ),   F_Q(n) = n^T Fcal n   (Eq. 18)
    COST   three matrix-free applications of J_a: O(N 2^N).
    """
    _, cov = spin_moments(psi)
    return 4.0 * cov


def optimal_direction(psi):
    """Best generator direction and the QFI it delivers: (F_max, n_opt) from Eq. (19)."""
    F = qfi_matrix(psi)
    w, v = jnp.linalg.eigh(F)                 # eigenvalues ascending
    return w[-1], v[:, -1]


# --- CHECKPOINT: the quadratic form, Eq. (18), reproduces the direct matrix-free QFI ------------
N_DIR = 6
states_dir = {"GHZ": ghz_state(N_DIR), "W": w_state(N_DIR), "Dicke N/2": dicke_state(N_DIR, N_DIR // 2),
              "|+>^N": product_state("+" * N_DIR), "cluster": cluster_state(N_DIR),
              "Haar": haar_state(jax.random.PRNGKey(3), N_DIR)}
key_dirs = jax.random.normal(jax.random.PRNGKey(11), (12, 3))
key_dirs = key_dirs / jnp.linalg.norm(key_dirs, axis=1, keepdims=True)     # 12 random unit directions
err_quad = 0.0
for name, psi in states_dir.items():
    F = qfi_matrix(psi)
    for n in key_dirs:
        err_quad = max(err_quad, abs(float(n @ F @ n) - float(qfi_pure_direction(psi, n))))
print(f"max |n^T Fcal n  -  4 Var(n.J)| over 6 states x 12 random directions = {err_quad:.2e}")
assert err_quad < 1e4 * TOL

print(f"\nN = {N_DIR}   (SQL: {N_DIR},  Heisenberg: {N_DIR ** 2})\n")
print(f"{'state':>12s} | {'eigenvalues of Fcal':>32s} | {'F_max':>9s} {'n_opt':>24s} {'F(J_z)':>9s}")
for name, psi in states_dir.items():
    F = qfi_matrix(psi)
    w = np.array(jnp.linalg.eigvalsh(F))
    fmax, nopt = optimal_direction(psi)
    nopt = np.array(nopt) * np.sign(np.array(nopt)[np.argmax(np.abs(np.array(nopt)))])   # fix the global sign
    print(f"{name:>12s} | {np.array2string(w, precision=3, floatmode='fixed'):>32s} | {float(fmax):9.4f} "
          f"{np.array2string(nopt, precision=3, floatmode='fixed'):>24s} {float(qfi_pure_direction(psi, (0, 0, 1))):9.4f}")

# %%
# ==============================================================================
# FIGURE: F_Q as a function of the generator direction, in two great circles
# ==============================================================================
angles = np.linspace(0, 2 * np.pi, 361)
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), subplot_kw={"projection": "polar"})
for k, (name, psi) in enumerate(list(states_dir.items())):
    F = np.array(qfi_matrix(psi))
    xz = np.array([[np.sin(a), 0.0, np.cos(a)] for a in angles])          # great circle in the x-z plane
    xy = np.array([[np.cos(a), np.sin(a), 0.0] for a in angles])          # great circle in the x-y plane
    axes[0].plot(angles, np.einsum("ia,ab,ib->i", xz, F, xz), color=PALETTE[k % 6], lw=1.6, label=LBL[name])
    axes[1].plot(angles, np.einsum("ia,ab,ib->i", xy, F, xy), color=PALETTE[k % 6], lw=1.6, label=LBL[name])
for ax, ttl in zip(axes, [r"$x$–$z$ plane (angle from $+z$)", r"$x$–$y$ plane (angle from $+x$)"]):
    ax.set_title(ttl + f"\n$F_Q(\\mathbf{{n}})$ for $N={N_DIR}$", fontsize=10, pad=16)
    ax.grid(alpha=0.4)
axes[1].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.32, 1.12))
fig.tight_layout(); plt.show()

# %% [markdown]
# The polar plots are the quadratic form of Eq. (18) drawn as a radius: an ellipse-like curve (strictly, the restriction of a
# quadric to a great circle) whose longest axis is the optimal direction.
#
# * **GHZ** is a two-lobed figure along $\hat z$ with $F_Q=N^2=36$ there and only $F_Q=N=6$ in the perpendicular plane.
#   Point the generator the wrong way and a Heisenberg-limited state degrades to the standard quantum limit.
# * **W** and **Dicke** are the opposite: flat and *isotropic* in the $x$–$y$ plane (they are $J_z$ eigenstates, so the
#   $z$ direction carries nothing) — any equatorial direction is optimal.
# * $\vert+\rangle^{\otimes N}$ is a circle of radius $N$ in the $y$–$z$ plane and vanishes along $\hat x$, its own polarisation axis.
# * The **cluster state** is the interesting one. Its three eigenvalues at $N=6$ come out as $(4,6,8)$: unlike the naive
#   $F_Q=N$ along every Cartesian axis, the optimal direction is the diagonal $(\mathbf{\hat x}+\mathbf{\hat z})/\sqrt2$ and
#   gives $F_Q=8>N=6$. The off-diagonal entry comes from the two *boundary* stabilisers of the open chain, $K_0=X_0Z_1$ and
#   $K_{N-1}=Z_{N-2}X_{N-1}$. Since every $\langle J_a\rangle$ vanishes and $\sigma^x_q$ anticommutes with $\sigma^z_q$ on the
#   same site, only $q\neq q'$ survives in Eq. (16) and $\mathcal{F}_{xz}=\sum_{q\neq q'}\langle X_qZ_{q'}\rangle
#   =\langle X_0Z_1\rangle+\langle X_{N-1}Z_{N-2}\rangle=1+1=2$, while the diagonal stays at $N$; the eigenvalues are then
#   $N-2,\,N,\,N+2$, exactly the $(4,6,8)$ printed above. So the cluster state *is* detected as entangled — but only if you
#   look along the right axis, and the excess is a boundary effect of size $2$ (we measure $F_{\max}=N+2$ at both $N=6$
#   and $N=8$) that leaves the scaling linear.
# * The **Haar-random** state is a generic anisotropic blob; its three eigenvalues $(4.08,6.29,7.07)$ scatter around
#   $N=6$, and its optimal direction is a random-looking unit vector.
#
# Where $\lambda_{\max}$ is degenerate (W, Dicke, $\vert+\rangle^{\otimes N}$) the printed $\mathbf n_{\text{opt}}$ is one
# arbitrary vector of the degenerate plane returned by `eigh`; every unit vector in that plane is equally optimal.
#
# > **Numerical practice.** Replacing an optimisation over a continuum (here: the unit sphere of directions) by an exact
# > eigenvalue problem is always worth hunting for. It costs one `eigh` of a $3\times3$ matrix instead of a gradient descent
# > that can stall in local maxima, and it returns the *global* optimum with a certificate. Notebook 34 uses exactly this
# > $3\times3$ matrix to track the optimal measurement axis along a one-axis-twisting evolution.

# %% [markdown]
# ## 12. The QFI as an entanglement witness
#
# Section 9 proved $F_Q\le N$ for separable states. Turned around, this is one of the most useful entanglement criteria in
# experimental many-body physics, because it is *operational*: a violation certifies entanglement that is strong enough to beat the
# shot-noise limit in an interferometer.
#
# **Criterion 1 (Pezzè and Smerzi 2009).** For every separable state of $N$ qubits and every collective generator
# $G=\mathbf n\cdot\mathbf J$,
#
# $$F_Q[\rho,G]\le N;\qquad\text{therefore}\quad F_Q>N\;\Longrightarrow\;\rho\text{ is entangled.} \tag{20}$$
#
# **Criterion 2 (Hyllus et al. 2012; Tóth 2012).** Call a state **$k$-producible** if it is a mixture of products of blocks of
# at most $k$ qubits (so $k=1$ means separable). Writing $N=sk+r$ with $s=\lfloor N/k\rfloor$ and $0\le r<k$, every
# $k$-producible state obeys
#
# $$F_Q[\rho,G]\;\le\;s\,k^2+r^2\;\le\;kN . \tag{21}$$
#
# (The second inequality holds because $kN=sk^2+kr$ and $r<k$.) Consequently $F_Q>s\,k^2+r^2$ certifies **entanglement
# depth** at least $k+1$: some block of at least $k+1$ qubits is genuinely entangled. We always use the *sharp* middle
# expression, never the weaker $kN$ — for $N=8$, $k=3$ the two differ by $2$, and that difference decides the verdict for
# the W state below. Both criteria are quoted here; their proofs use the convexity of the QFI plus the single-block bound
# $F_Q\le k^2$, which is the Heisenberg-limit argument of Section 9.2 applied to a block of $k$ qubits.
#
# Let us apply the criteria to the zoo.

# %%
# ==============================================================================
# STEP 9: entanglement depth certified by the QFI, Eq. (21)
# ==============================================================================
def entanglement_depth(F_value, N):
    """Largest k+1 such that F_Q > s k^2 + r^2 with s = N//k, r = N%k  --  the certified entanglement depth.

    Returns 1 if the QFI certifies nothing (F_Q <= N), otherwise the smallest block size that must be
    genuinely entangled.  MATH: Eq. (21), k-producible states obey F_Q <= s k^2 + r^2.
    """
    depth = 1
    for k in range(1, N + 1):
        s, r = divmod(N, k)
        if F_value > s * k ** 2 + r ** 2 + 1e-9:     # k-producible is EXCLUDED
            depth = k + 1
    return depth


N_W = 8
print(f"N = {N_W}:  bound of Eq. (21) for k-producible states")
print("   " + "  ".join(f"k={k}: {N_W // k * k ** 2 + (N_W % k) ** 2:3d}" for k in range(1, N_W + 1)))
print(f"\n{'state':>20s} {'G':>7s} | {'F_Q':>9s} {'F_Q/N':>7s} | {'entangled':>10s} {'depth >=':>9s} {'S(N/2 cut)':>11s}")
wit_states = [
    ("|+>^N", product_state("+" * N_W), (0, 1, 0), "J_y"),
    ("W", w_state(N_W), (1, 0, 0), "J_x"),
    ("cluster (opt.dir.)", cluster_state(N_W), None, "n_opt"),
    ("Dicke N/2", dicke_state(N_W, N_W // 2), (1, 0, 0), "J_x"),
    ("GHZ", ghz_state(N_W), (0, 0, 1), "J_z"),
    ("Haar random", haar_state(jax.random.PRNGKey(5), N_W), None, "n_opt"),
]
for name, psi, n, gname in wit_states:
    f = float(optimal_direction(psi)[0]) if n is None else float(qfi_pure_direction(psi, n))
    S = float(entanglement_entropy(psi, range(N_W // 2)))
    print(f"{name:>20s} {gname:>7s} | {f:9.4f} {f / N_W:7.3f} | {'YES' if f > N_W else 'no':>10s} "
          f"{entanglement_depth(f, N_W):9d} {S:11.3f}")

# %% [markdown]
# The table puts the two halves of the story side by side. The last column is the entanglement entropy across the middle cut —
# the standard *quantitative* entanglement measure for pure states — and it is uncorrelated with metrological
# usefulness. The Haar-random state has by far the largest entropy ($3.21$ bits out of a maximum of $4$, close to the Page
# value) and yet, even in its optimal direction, it reaches only $F_Q=8.20$ against the separable bound $N=8$: the QFI
# certifies only that the state is entangled (depth $2$). The cluster state stops at depth $2$ as well, although it is
# a genuinely $8$-partite entangled stabiliser state. The GHZ state has just **one** bit of entropy across the cut — the
# same as the cluster state — and is certified as genuinely $8$-partite entangled; the Dicke state, with $1.64$ bits,
# reaches depth $6$.
#
# The W state is worth a second look because it lands exactly on a bound: $F_Q=22$ and the $k=3$ bound of Eq. (21) is
# $s k^2+r^2=2\cdot 9+2^2=22$. The inequality in Eq. (21) is not strict, so a value *equal* to the bound excludes nothing:
# $3$-producibility survives and the certified depth is $3$, not $4$. (That $\vert W_8\rangle$ is in fact genuinely
# $8$-partite entangled is true but invisible to this witness — a reminder that Eq. (21) is a one-sided criterion, and that
# the exact-equality case has to be decided by the inequality as written, not by a floating-point comparison. The
# `+1e-9` in `entanglement_depth` is what implements "not strictly greater".)
#
# > **Physics insight.** "How much entanglement" and "how useful the entanglement is" are different questions with different
# > answers. Metrology rewards *collective, coherent* superpositions of configurations with very different values of the
# > generator; the amount of entanglement across a cut is a different quantity. This is why the phrase "useful entanglement" appears so often in the
# > metrology literature, and why $F_Q$ rather than an entropy is the figure of merit.

# %% [markdown]
# ## 13. Noise: QFI of mixed states
#
# Real devices produce mixed states, and Eq. (14) no longer applies — a mixed state has no single $\vert\psi\rangle$ whose
# variance we could take. The general answer is the symmetric-logarithmic-derivative formula. Written in the eigenbasis
# $\rho=\sum_m\lambda_m\vert m\rangle\langle m\vert$, for unitary encoding with generator $G$, it reads
#
# $$F_Q[\rho,G]\;=\;2\sum_{m,n\,:\,\lambda_m+\lambda_n>0}\frac{(\lambda_m-\lambda_n)^2}{\lambda_m+\lambda_n}
#   \left\vert\langle m\vert G\vert n\rangle\right\vert^2 . \tag{22}$$
#
# This is what the engine's `qfi_mixed` implements. Like the Braunstein–Caves theorem of Section 6.1, it is used here
# without proof: the full derivation — the Lyapunov equation for the SLD, its solution in the eigenbasis, the treatment of the
# kernel, the reduction of Eq. (22) to $4\mathrm{Var}(G)$ for pure states, and the explicit optimal measurement — is the
# subject of the next notebook,
# [30 — QFI from the SLD](../ch10_quantum_metrology_protocols/30_qfi_from_the_sld_prepare_encode_estimate.ipynb).
# Two features of Eq. (22) matter here: it costs a full eigendecomposition, $O(8^N)$, so it is a
# small-system tool ($N\le7$ in practice); and every term is non-negative, so mixing can only be understood by looking at
# how *both* the eigenvalues and the matrix elements change.
#
# One inequality is worth recording now, because it will be used repeatedly later in the chapter. Since
# $(\lambda_m-\lambda_n)^2\le(\lambda_m+\lambda_n)^2$ for non-negative eigenvalues, Eq. (22) is bounded by
# $2\sum_{m,n}(\lambda_m+\lambda_n)\left\vert\langle m\vert G\vert n\rangle\right\vert^2=4\langle G^2\rangle$. Eq. (22) is
# unchanged by the shift $G\to G-\langle G\rangle\mathbb 1$ (the diagonal terms $m=n$ carry the factor
# $(\lambda_m-\lambda_m)^2=0$, and the off-diagonal matrix elements do not see the shift), so the same bound applied to
# the shifted generator gives
#
# $$F_Q[\rho,G]\;\le\;4\,\mathrm{Var}_\rho(G).$$
#
# Equality holds for every pure state (Section 7). For mixed states the inequality is in general strict, although not
# always: $\rho=\tfrac12(\vert00\rangle\langle00\vert+\vert11\rangle\langle11\vert)$ with $G=\tfrac12X_1$ has
# $F_Q=4\mathrm{Var}(G)=1$, because the two components rotate in orthogonal subspaces and stay perfectly distinguishable.
# So $4\,\mathrm{Var}(G)$ never *under*-estimates the quantum Fisher information — it is the right answer for pure states
# and an upper bound for mixed ones. Notebook 30 shows the gap numerically (a proof of the inequality is in Tóth and
# Apellaniz 2014), and notebook 36 measures how large it becomes for noisy states.
#
# ### 13.1 An analytic test case: dephased GHZ
#
# Apply single-qubit dephasing (probability $p$ of a $Z$ error) to each of the $N$ qubits of a GHZ state. Dephasing leaves
# the populations alone and multiplies each coherence $\rho_{ss'}$ by $(1-2p)$ for every qubit at which $s$ and $s'$ differ.
# The GHZ state has exactly one coherence, between $\vert0\cdots0\rangle$ and $\vert1\cdots1\rangle$, and those two strings
# differ at **all** $N$ qubits, so
#
# $$\rho(p)=\frac12\Big(\vert0\cdots0\rangle\langle0\cdots0\vert+\vert1\cdots1\rangle\langle1\cdots1\vert\Big)
#   +c\Big(\vert0\cdots0\rangle\langle1\cdots1\vert+\text{h.c.}\Big),\qquad c=\tfrac12(1-2p)^N .$$
#
# Everything lives in the two-dimensional space spanned by $\vert0\cdots0\rangle$ and $\vert1\cdots1\rangle$, where
# $\rho=\begin{pmatrix}1/2&c\\c&1/2\end{pmatrix}$ has eigenvalues $\lambda_\pm=\tfrac12\pm c$ with eigenvectors
# $(\vert0\cdots0\rangle\pm\vert1\cdots1\rangle)/\sqrt2$, and $J_z=\tfrac N2\,\mathrm{diag}(1,-1)$ becomes purely
# off-diagonal, $\langle+\vert J_z\vert-\rangle=N/2$. Equation (22) then has just the two terms $(m,n)=(+,-)$ and $(-,+)$:
#
# $$F_Q=2\cdot2\cdot\frac{(2c)^2}{1}\cdot\frac{N^2}{4}=4c^2N^2=N^2(1-2p)^{2N} . \tag{23}$$
#
# The Heisenberg value $N^2$ is multiplied by $(1-2p)^{2N}$: the advantage dies **exponentially in $N$** at fixed error
# probability. Equation (23) is the quantitative statement of GHZ fragility. We check it, and then compare the three
# standard channels.

# %%
# ==============================================================================
# STEP 10: QFI of noisy states -- density tensors and the mixed-state formula
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
N_NOISE = 6                                   # density tensor of 2^6 x 2^6 = 64 x 64 -- comfortable
P_GRID = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
# -----------------------------------------------------------------------------
CHANNELS = {"depolarising": kraus_depolarizing, "dephasing": kraus_dephasing,
            "amplitude damping": kraus_amplitude_damping}


def apply_local_channel(rho, kraus_fn, p, qubits=None):
    """Apply the SAME single-qubit channel to each listed qubit of a density TENSOR.

    MATH   rho -> (E_p (x) E_p (x) ... ) rho,  each E_p(sigma) = sum_m K_m sigma K_m^dagger.
    COST   O(N 4^N) -- one `apply_kraus_dm` einsum per qubit; the 4^N x 4^N superoperator never exists.
    """
    N = rho.ndim // 2
    qubits = range(N) if qubits is None else qubits
    K = kraus_fn(p)
    for q in qubits:
        rho = apply_kraus_dm(rho, K, [q])
    return rho


def qfi_noisy(rho_tensor, n, N):
    """QFI of a mixed state for the generator n.J, via the engine's mixed-state (SLD) formula, Eq. (22)."""
    return qfi_mixed(dm_matrix(rho_tensor), collective_dense(pauli_direction(n), N))


# --- CHECKPOINT: the analytic dephasing formula, Eq. (23) -----------------------------
print("dephased GHZ: numerical Eq. (22) vs analytic Eq. (23)   F_Q = N^2 (1-2p)^{2N}\n")
print(f"{'N':>3s} {'p':>6s} | {'F_Q (numeric)':>14s} {'F_Q (analytic)':>15s} {'rel. err':>10s}")
for N in (3, 5, 6):
    for p in (0.02, 0.10, 0.25):
        rho = apply_local_channel(to_dm(ghz_state(N)), kraus_dephasing, p)
        f_num = float(qfi_noisy(rho, (0, 0, 1), N))
        f_ana = N ** 2 * (1 - 2 * p) ** (2 * N)
        print(f"{N:3d} {p:6.2f} | {f_num:14.8f} {f_ana:15.8f} {abs(f_num - f_ana) / f_ana:10.2e}")
        assert abs(f_num - f_ana) < 1e4 * TOL

# %%
# ==============================================================================
# STEP 11: three channels, GHZ vs product state, N = 2 ... 6
# ==============================================================================
noise_data = {}
t0 = time.time()
for cname, kfn in CHANNELS.items():
    for N in (2, 3, 4, 5, 6):
        vals = []
        for p in P_GRID:
            rho = apply_local_channel(to_dm(ghz_state(N)), kfn, float(p))
            vals.append(float(qfi_noisy(rho, (0, 0, 1), N)))
        noise_data[(cname, N)] = np.array(vals)
print(f"(noise sweep computed in {time.time() - t0:.1f} s)\n")

print(f"GHZ_N, generator J_z, QFI normalised by the ideal value N^2\n")
for cname in CHANNELS:
    print(f"  {cname}:")
    print("    " + f"{'N':>3s} | " + " ".join(f"p={p:<5.3f}" for p in P_GRID))
    for N in (2, 4, 6):
        v = noise_data[(cname, N)] / N ** 2
        print("    " + f"{N:3d} | " + " ".join(f"{x:7.4f}" for x in v))
    print()

# %%
# ==============================================================================
# FIGURE: fragility of the Heisenberg limit
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
for j, cname in enumerate(CHANNELS):
    for k, N in enumerate((2, 3, 4, 5, 6)):
        axes[j].semilogy(P_GRID, noise_data[(cname, N)] / N ** 2, MARKERS[k] + "-",
                         color=PALETTE[k], ms=4, label=f"$N={N}$")
    if cname == "dephasing":
        pp = np.linspace(0, 0.3, 200)
        for k, N in enumerate((2, 6)):
            axes[j].semilogy(pp, (1 - 2 * pp) ** (2 * N), "k--", lw=1)
        axes[j].text(0.12, 2e-3, r"$(1-2p)^{2N}$", fontsize=9)
    axes[j].axhline(1.0, color="0.6", ls=":", lw=1)
    axes[j].set_xlabel("error probability $p$ per qubit"); axes[j].set_title(cname)
axes[0].set_ylabel(r"$F_Q/N^2$"); axes[0].set_ylim(1e-3, 2.0); axes[0].legend(fontsize=8)
fig.suptitle(r"GHZ$_N$ under local noise before phase encoding, generator $G=J_z$", y=1.02)
fig.tight_layout(); plt.show()

# %%
# ==============================================================================
# FIGURE: decay of the advantage with N at fixed p
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
Nn = np.array([2, 3, 4, 5, 6])
for j, p_show in enumerate((0.02, 0.10)):
    ip = int(np.argmin(np.abs(P_GRID - p_show)))
    for k, cname in enumerate(CHANNELS):
        axes[j].semilogy(Nn, [noise_data[(cname, N)][ip] for N in Nn], MARKERS[k] + "-",
                         color=PALETTE[k], ms=5, label=cname)
    axes[j].semilogy(Nn, Nn ** 2.0, ":", color="0.2", lw=1.2, label=r"$N^2$ (ideal GHZ)")
    axes[j].semilogy(Nn, Nn * 1.0, "--", color="0.5", lw=1.2, label=r"$N$ (SQL)")
    axes[j].set_xlabel("number of qubits $N$"); axes[j].set_ylabel(r"$F_Q$")
    axes[j].set_title(f"$p = {P_GRID[ip]:.3f}$ per qubit"); axes[j].set_xticks(Nn)
axes[0].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %%
# ==============================================================================
# STEP 11b: comparing two channels at the SAME physical strength, not the same label p
# ==============================================================================
# A single qubit under dephasing(p) keeps a fraction (1 - 2p) of its transverse Bloch components;
# under depolarising(p) it keeps (1 - 4p/3).  Matching those two contractions to a common value eta
# removes the parametrisation convention from the comparison and leaves only physics.
print("GHZ_N, G = J_z: dephasing vs depolarising at EQUAL transverse Bloch contraction eta\n")
print(f"{'eta':>5s} {'p(deph)':>8s} {'p(depol)':>9s} | " + " ".join(f"{'N=%d' % N:>21s}" for N in (2, 4, 6)))
print(f"{'':>5s} {'':>8s} {'':>9s} | " + " ".join(f"{'deph':>7s}{'depol':>8s}{'ratio':>6s}" for _ in (2, 4, 6)))
for eta in (0.98, 0.90, 0.80):
    p_deph, p_depol = (1 - eta) / 2, 3 * (1 - eta) / 4
    cells = []
    for N in (2, 4, 6):
        a = float(qfi_noisy(apply_local_channel(to_dm(ghz_state(N)), kraus_dephasing, p_deph), (0, 0, 1), N)) / N ** 2
        b = float(qfi_noisy(apply_local_channel(to_dm(ghz_state(N)), kraus_depolarizing, p_depol), (0, 0, 1), N)) / N ** 2
        cells.append(f"{a:7.4f}{b:8.4f}{b / a:6.2f}")
        assert abs(a - (1 - 2 * p_deph) ** (2 * N)) < 1e4 * TOL      # the dephasing column is still Eq. (23)
        assert b > a - 1e-12                                          # depolarising is never the harsher of the two
        # amplitude damping, Eq. (23a), at its own parameter p_ad = 1 - eta^2 (same transverse contraction eta)
        p_ad = 1 - eta ** 2
        c_ad = float(qfi_noisy(apply_local_channel(to_dm(ghz_state(N)), kraus_amplitude_damping, p_ad), (0, 0, 1), N)) / N ** 2
        assert abs(c_ad - 2 * (1 - p_ad) ** N / (1 + p_ad ** N + (1 - p_ad) ** N)) < 1e4 * TOL
    print(f"{eta:5.2f} {p_deph:8.3f} {p_depol:9.3f} | " + " ".join(cells))

# %% [markdown]
# The dephasing panel confirms Eq. (23) exactly: the numerical points lie on the dashed analytic curves, and the
# checkpoint table prints relative deviations below $10^{-14}$. All three channels behave the same way qualitatively: at fixed
# per-qubit error probability the ratio $F_Q/N^2$ falls with $N$, so the curves for larger $N$ lie *below* the smaller ones.
# Concretely, at $p=0.02$ (a very good gate) the dephased $N=6$ GHZ retains $F_Q/N^2=0.61$, while at $p=0.10$ it is down to
# $0.069$. The right-hand figure shows what this means in absolute terms: at $p=0.10$ the *dephased* GHZ state sits below
# the standard quantum limit at every $N$ shown, and its $F_Q$ stops growing altogether around $N\approx4$ (indeed
# $N^2(1-2p)^{2N}$ is maximised at $N=-1/\ln(1-2p)\approx4.5$ and decreases beyond it); the depolarised state still grows,
# but far more slowly than $N^2$.
#
# At equal nominal $p$ the measured ordering is the same at every $N$ and every $p$ we tried: **amplitude damping is the
# gentlest, depolarising is intermediate, dephasing is the harshest** (at $N=6$, $p=0.05$: $0.85$, $0.54$, $0.28$ of the
# ideal value). Part of that ordering is bookkeeping and part of it is physics, and the two must be separated.
#
# **The bookkeeping.** Equal $p$ does not mean equal physical strength. A single qubit under our dephasing channel keeps a
# fraction $1-2p$ of its transverse Bloch components; under our depolarising channel it keeps $1-4p/3$, because the $X$ and
# $Y$ errors destroy transverse coherence just as the $Z$ error does; the convention factor is therefore $3/2$ (counting only
# "$Z$ with probability $p/3$" would suggest $3$). Matching the two contractions, $p_{\text{depol}}=\tfrac32 p_{\text{deph}}$,
# is what STEP 11b does.
#
# **The physics that is left.** Even at matched contraction the depolarising channel is the milder of the two, and the gap
# *grows* with $N$ and with noise strength: the measured ratio $F_Q^{\text{depol}}/F_Q^{\text{deph}}$ runs from $1.02$ at
# $\eta=0.98$, $N=2$ to $1.88$ at $\eta=0.80$, $N=6$. The reason is visible in the structure of the two states. Local
# dephasing keeps all the weight in the single two-dimensional block spanned by $\vert0\cdots0\rangle$ and
# $\vert1\cdots1\rangle$ and merely shrinks its coherence, which is why Eq. (23) is exact. Local depolarising also flips
# bits, and a flip on qubit $q$ maps the pair $(\vert0\cdots0\rangle,\vert1\cdots1\rangle)$ onto another *complementary*
# pair $(\vert s\rangle,\vert\bar s\rangle)$. The state therefore stays block diagonal in complementary pairs, and each
# block is again a small cat, with $J_z$ eigenvalues $\pm(N-2w_s)/2$ instead of $\pm N/2$, where $w_s$ is the number of
# ones in $s$. Those blocks contribute less than the top one but not zero, and their contributions are what the
# depolarising channel retains and the dephasing channel does not.
#
# Amplitude damping is milder again, and the reason can be computed exactly. Its Kraus operators
# $K_0=\mathrm{diag}(1,\sqrt{1-p})$ and $K_1=\sqrt p\,\vert0\rangle\langle1\vert$ multiply the GHZ coherence by
# $(1-p)^{N/2}$ — a transverse contraction $\sqrt{1-p}\approx1-p/2$ per qubit, a much weaker channel than dephasing at the
# same label $p$ (bookkeeping again). On top of that, the decay keeps the surviving cat *pure*. Within the block spanned by
# $\vert0\cdots0\rangle$ and $\vert1\cdots1\rangle$ the populations are $a=\tfrac12(1+p^N)$ and $b=\tfrac12(1-p)^N$ and the
# coherence is $c=\tfrac12(1-p)^{N/2}$, so $c^2=ab$ up to the tiny $p^N$ term: the block is an unbalanced cat
# $\sqrt a\,\vert0\cdots0\rangle+\sqrt b\,\vert1\cdots1\rangle$ that is unbalanced but pure. The weight that has decayed sits on
# strings that are eigenstates of $J_z$ with no coherence between them, and contributes nothing. The two-level formula
# $F_Q=4N^2c^2/(a+b)$ for this block gives
#
# $$\frac{F_Q^{\text{AD}}}{N^2}=\frac{2(1-p)^N}{1+p^N+(1-p)^N}, \tag{23a}$$
#
# which reproduces the amplitude-damping rows of the STEP 11 table (e.g. $0.8473$ at $N=6$, $p=0.05$) and is asserted in
# STEP 11b at the three matched contractions. At matched transverse contraction,
# $(1-p_{\text{AD}})^N=(1-2p_{\text{deph}})^{2N}$, amplitude damping beats dephasing by exactly the factor
# $2/(1+p^N+(1-p)^N)>1$, the price of renormalising a block that has kept its coherence.
# What is convention-independent — and what matters for an experiment — is the exponential-in-$N$ decay shared by all three.
#
# > **Physics insight.** This is the central practical tension of quantum metrology. The GHZ state has the maximum possible
# > $F_Q$ and the minimum possible robustness: a coherence spread over $N$ qubits decays $N$ times faster than a single-qubit
# > coherence. Notebook 32 turns this into a statement about the *optimal* interrogation time; notebook 33 introduces
# > spin-squeezed states, and notebook 34 compares Ramsey, squeezed, GHZ and one-axis-twisting interferometry under noise.

# %% [markdown]
# ## 14. Losing particles: the QFI of a subsystem, and Schmidt compression
#
# A different and equally common imperfection: the phase is imprinted on all $N$ qubits, but only $K$ of them reach the
# detector — atoms escape the trap, photons are lost, a detector goes dark. What survives is the reduced state
#
# $$\rho_A(\theta)=\mathrm{Tr}_B\left[U(\theta)\vert\psi\rangle\langle\psi\vert U^\dagger(\theta)\right],$$
#
# a *mixed* state of the $K$ kept qubits. Its QFI is what an experiment can still achieve.
#
# ### 14.1 The reduced derivative without ever building $\rho_A(\theta)$
#
# Because the partial trace is linear and $\partial_\theta\rho=-i[G,\rho]$,
#
# $$\partial_\theta\rho_A=\mathrm{Tr}_B\left(-i\left[G,\rho\right]\right)
#  =-i\,\mathrm{Tr}_B\Big(G\vert\psi\rangle\langle\psi\vert-\vert\psi\rangle\langle\psi\vert G\Big).$$
#
# Now use the matrix view of a bipartite state (the Schmidt trick of
# [06 — states, observables, entanglement](../ch03_matrix_free_engine/06_states_observables_entanglement.ipynb)):
# reshape the tensor $\psi$ into a matrix $\Psi$ of shape $(d_A,d_B)$ with $d_A=2^K$, $d_B=2^{N-K}$, and reshape
# $\vert\phi\rangle=G\vert\psi\rangle$ into $\Phi$ the same way. Partial traces become matrix products:
#
# $$\rho_A=\Psi\Psi^\dagger,\qquad \mathrm{Tr}_B\left(\vert\phi\rangle\langle\psi\vert\right)=\Phi\Psi^\dagger,$$
#
# so, with $T=\Phi\Psi^\dagger$,
#
# $$\rho_A=\Psi\Psi^\dagger,\qquad \partial_\theta\rho_A=-i\left(T-T^\dagger\right). \tag{24}$$
#
# Feed the pair into the SLD formula in its general form (the one derived in notebook 30),
#
# $$F_Q=2\sum_{m,n\,:\,\lambda_m+\lambda_n>0}\frac{\left\vert\langle m\vert\partial_\theta\rho_A\vert n\rangle\right\vert^2}{\lambda_m+\lambda_n}, \tag{25}$$
#
# and we are done. (Equation (22) is Eq. (25) specialised to a *full* state with unitary encoding; for a subsystem the
# generator no longer acts inside $A$ alone, so we must use the derivative itself.)
#
# ### 14.2 Schmidt compression: making $K=N-1$ as cheap as $K=1$
#
# Equation (25) needs an eigendecomposition of $\rho_A$, costing $O(d_A^3)=O(8^K)$. Losing *one* qubit would then be the
# most expensive case, even though losing one qubit barely mixes the state. The cure is the **Schmidt rank bound**:
# for a pure $\vert\psi\rangle$, $\;\mathrm{rank}(\rho_A)\le\min(d_A,d_B)$. Moreover both $\rho_A$ and $\partial_\theta\rho_A$
# from Eq. (24) have all their rows and columns inside the span of the columns of $\Psi$ and $\Phi$ — a subspace of
# dimension at most $2d_B$. So:
#
# 1. stack $B=\left[\;\Psi\;\;\Phi\;\right]$, of shape $(d_A,2d_B)$;
# 2. compute a thin QR decomposition $B=QR$; the columns of $Q$ (at most $2d_B$ of them) span the **active subspace**;
# 3. project, $\tilde\Psi=Q^\dagger\Psi$, $\tilde\Phi=Q^\dagger\Phi$, and rebuild $\rho_A$, $\partial_\theta\rho_A$ there;
# 4. diagonalise the small matrices.
#
# The projection is an isometry, so the non-zero eigenvalues of $\rho_A$ and all the matrix elements in Eq. (25) are
# unchanged, and the eigendecomposition drops from $O(8^{K})$ to $O(8^{N-K})$. Combining the two branches, the
# diagonalisation costs $O\!\left(8^{\min(K,\,N-K)}\right)$: symmetric around the middle cut, as it should be. Two cheaper
# terms remain in every case and set the floor of the measured timings in Section 15: applying the generator, $O(N2^N)$,
# and the QR plus the projections, $O(2^K4^{N-K})$.

# %%
# ==============================================================================
# STEP 12: QFI of a subsystem, with and without Schmidt (QR) compression
# ==============================================================================
def qfi_from_derivative(rho, drho, tol=1e-12):
    """General SLD quantum Fisher information from a state and its derivative, Eq. (25).

    MATH   F_Q = 2 sum_{m,n} |<m|drho|n>|^2 / (lam_m + lam_n),  summed over lam_m + lam_n > tol,
           with rho = sum_m lam_m |m><m|.   (Derived in notebook 30.)
    COST   one Hermitian eigendecomposition, O(d^3).
    """
    lam, v = jnp.linalg.eigh(rho)
    D = v.conj().T @ drho @ v
    den = lam[:, None] + lam[None, :]
    ok = den > tol
    return 2 * jnp.sum(jnp.where(ok, jnp.abs(D) ** 2 / jnp.where(ok, den, 1.0), 0.0))


def subsystem_qfi(psi, n_dir, K, compress=True, tol=1e-12):
    """QFI of the reduced state of the FIRST K qubits after encoding with G = n.J on all N qubits.

    MATH   Psi  = psi reshaped to (2^K, 2^{N-K}),   Phi = (G psi) reshaped the same way;
           rho_A = Psi Psi^dag,   d rho_A/d theta = -i (T - T^dag) with T = Phi Psi^dag     [Eq. (24)]
    IMPLEMENTATION  when 2^K > 2^{N-K} the two matrices span a subspace of dimension <= 2^{N-K+1};
           a thin QR of [Psi | Phi] gives an isometry Q onto it, and everything is done inside.
    COST   O(8^min(K, N-K)) for the eigendecomposition instead of O(8^K).
    """
    P = pauli_direction(n_dir)
    phi = 0.5 * apply_collective(psi, P)                      # |phi> = (n.J)|psi>, matrix-free
    Psi = psi.reshape(2 ** K, -1)
    Phi = phi.reshape(2 ** K, -1)
    if compress and Psi.shape[0] > Psi.shape[1]:
        Q, _ = jnp.linalg.qr(jnp.concatenate([Psi, Phi], axis=1))     # thin QR: active subspace
        Psi, Phi = Q.conj().T @ Psi, Q.conj().T @ Phi
    rho_A = Psi @ Psi.conj().T
    T = Phi @ Psi.conj().T
    return qfi_from_derivative(rho_A, -1j * (T - T.conj().T), tol)


# --- CHECKPOINT 1: no loss (K = N) must reproduce 4 Var(G) --------------------------
for name, psi, n in (("GHZ(6)", ghz_state(6), (0, 0, 1)), ("W(6)", w_state(6), (1, 0, 0)),
                     ("Haar(6)", haar_state(jax.random.PRNGKey(2), 6), (0, 0, 1))):
    a = float(subsystem_qfi(psi, n, 6))
    b = float(qfi_pure_direction(psi, n))
    print(f"K=N check {name:9s}: subsystem formula {a:12.8f}   4 Var(G) {b:12.8f}   err {abs(a - b):.2e}")
    assert abs(a - b) < 1e4 * TOL

# --- CHECKPOINT 2: compressed vs uncompressed, and vs an independent brute-force route ------
psi_t = haar_state(jax.random.PRNGKey(4), 7)
print()
for K in (2, 4, 6):
    a = float(subsystem_qfi(psi_t, (0, 0, 1), K, compress=True))
    b = float(subsystem_qfi(psi_t, (0, 0, 1), K, compress=False))
    # brute force: build rho_A(theta) explicitly and differentiate it with jax.jacfwd
    def rho_A_of_theta(th, K=K):
        U = jnp.cos(th / 2) * I2 - 1j * jnp.sin(th / 2) * Z
        p = psi_t
        for q in range(7):
            p = apply_gate(p, U, [q])
        return rdm(p, range(K))
    c = float(qfi_from_derivative(rho_A_of_theta(0.0), jax.jacfwd(rho_A_of_theta)(0.0)))
    print(f"K={K}: compressed {a:.10f}  uncompressed {b:.10f}  jacfwd brute force {c:.10f}  "
          f"max err {max(abs(a - b), abs(a - c)):.2e}")
    assert max(abs(a - b), abs(a - c)) < 1e4 * TOL

# %% [markdown]
# Three independent implementations agree: the compressed and uncompressed versions of Eq. (24)–(25), and a brute-force
# route that applies the encoding $e^{-i\theta Z/2}$ gate by gate with `apply_gate`, traces out the lost qubits with `rdm`,
# and differentiates the resulting $\rho_A(\theta)$ at $\theta=0$ with forward-mode automatic differentiation
# (`jax.jacfwd` propagates one tangent through the whole computation; no finite differences and no step size). The last
# check is the important one: it tests the claim that the derivative of the reduced state is $-i(T-T^\dagger)$, the only
# non-obvious step of the derivation, and it does so without using `apply_collective` at all — so it also validates the
# matrix-free generator that Sections 8–12 relied on.

# %%
# ==============================================================================
# STEP 13: the QFI that survives the loss of k qubits
# ==============================================================================
N_LOSS = 8
loss_states = [("GHZ", ghz_state(N_LOSS), (0, 0, 1)),
               ("W", w_state(N_LOSS), (1, 0, 0)),
               ("Dicke N/2", dicke_state(N_LOSS, N_LOSS // 2), (1, 0, 0)),
               ("|+>^N", product_state("+" * N_LOSS), (0, 1, 0)),
               ("cluster", cluster_state(N_LOSS), (0, 0, 1)),
               ("Haar", haar_state(jax.random.PRNGKey(1), N_LOSS), (0, 0, 1))]
loss_tab = {}
print(f"N = {N_LOSS}:  QFI of the reduced state of the first {N_LOSS}-k qubits\n")
print(f"{'state':>12s} | " + " ".join(f"k={k:<7d}" for k in range(N_LOSS)))
for name, psi, n in loss_states:
    vals = [float(subsystem_qfi(psi, n, N_LOSS - k)) for k in range(N_LOSS)]
    loss_tab[name] = np.array(vals)
    print(f"{name:>12s} | " + " ".join(f"{v:8.4f} " for v in vals))

# %%
# ==============================================================================
# FIGURE: metrological power lost with the particles
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
ks = np.arange(N_LOSS)
for j, (name, psi, n) in enumerate(loss_states):
    axes[0].plot(ks, loss_tab[name], MARKERS[j % 6] + "-", color=PALETTE[j % 6], ms=5, label=LBL[name])
    axes[1].semilogy(ks, np.clip(loss_tab[name] / max(loss_tab[name][0], 1e-12), 1e-6, None),
                     MARKERS[j % 6] + "-", color=PALETTE[j % 6], ms=5, label=LBL[name])
axes[1].axhline(1e-6, color="0.6", ls=":", lw=1)
axes[1].text(0.15, 1.4e-6, "floor: values clipped here are exactly zero", fontsize=7, color="0.4")
axes[0].axhline(N_LOSS, color="0.5", ls="--", lw=1)
axes[0].text(4.2, N_LOSS * 1.1, "SQL for the full state", fontsize=8, color="0.4")
axes[0].set_xlabel("number of lost qubits $k$"); axes[0].set_ylabel(r"$F_Q$ of the kept $N-k$ qubits")
axes[0].set_title(f"$N={N_LOSS}$: QFI under particle loss"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("number of lost qubits $k$"); axes[1].set_ylabel(r"$F_Q(k)/F_Q(0)$")
axes[1].set_title("Fraction retained (log scale)"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The GHZ curve drops discontinuously: losing a **single** qubit takes $F_Q$ from $N^2=64$ to exactly $0$.
# The reason is transparent from Eq. (24). Tracing out one qubit of a GHZ state leaves
# $\rho_A=\tfrac12\left(\vert0\cdots0\rangle\langle0\cdots0\vert+\vert1\cdots1\rangle\langle1\cdots1\vert\right)$ — a
# *classical* mixture of two $J_z$ eigenstates. A mixture of eigenstates of the generator does not move at all when the
# generator acts, so $\partial_\theta\rho_A=0$ and no measurement on the survivors can say anything about $\theta$.
# All of the GHZ phase information is stored in a single global coherence, and a single lost particle carries it away.
#
# Everything else degrades gracefully. The **W** state loses a roughly constant factor of about $0.6$ per lost qubit
# ($22\to14.44\to9.00\to5.31\to\ldots$); the **Dicke** state pays a steep price for the first loss ($40\to15$) and then
# settles into a similar geometric decay; the **product** state loses exactly one unit of QFI per qubit
# ($F_Q=N-k$, as it must, since the survivors are just $N-k$ independent qubits); the **cluster** state drops by $2$ at the
# first loss and then by one unit per qubit ($F_Q=N-k-1$ for $k\ge1$); the **Haar** state follows the product state at
# first and then falls steeply once half of the qubits are gone ($4.00$ at $k=3$, $1.57$ at $k=4$, $0.37$ at $k=5$). For
# a random state the reduced state of fewer than $N/2$ qubits is close to maximally mixed (the Page regime), and a
# maximally mixed state carries no phase information. For a resource that must survive real detectors, a graceful
# decay matters more than the largest ideal value. (In the right-hand panel, curves that touch the floor at $10^{-6}$ are exactly
# zero: GHZ from $k=1$ on, cluster and Dicke at $k=7$.)
#
# > **Physics insight.** This is the quantitative version of the folklore "GHZ states are fragile, Dicke and squeezed states
# > are robust". Notebook 37 revisits it as a systematic study of QFI versus particle loss, and notebook 34 shows where along
# > a one-axis-twisting evolution the trade-off is best.

# %% [markdown]
# ## 15. Performance: the total cost
#
# Three regimes, three scalings:
#
# | quantity | algorithm | time | memory |
# |---|---|---|---|
# | $F_Q$ of a pure state | `qfi_pure`: $N$ einsums plus two inner products | $O(N2^N)$ | $O(2^N)$ |
# | $3\times3$ QFI matrix | three collective applications plus a $3\times3$ `eigh` | $O(N2^N)$ | $O(2^N)$ |
# | $F_Q$ of a mixed state, Eq. (22) | dense eigendecomposition of $\rho$ | $O(8^N)$ | $O(4^N)$ |
# | $F_Q$ of a subsystem, compressed | thin QR plus a small `eigh` | $O\!\left(8^{\min(K,N-K)}\right)$ | $O(2^N)$ |
#
# Let us measure the first and the last, separating compile time from run time as always.

# %%
# ==============================================================================
# STEP 14: measured cost of the matrix-free QFI, and of Schmidt compression
# ==============================================================================
print("Pure-state QFI, F_Q = 4 Var(J_z), matrix-free (jit-compiled)\n")
print(f"{'N':>3s} {'dim 2^N':>9s} | {'compile [ms]':>13s} {'run [ms]':>10s} {'F_Q':>10s}")
qfi_rows = []
for N in (6, 8, 10, 12, 14, 16):
    f = jax.jit(lambda p: qfi_pure(p, Z))
    val, tc, tr = timed(f, ghz_state(N))
    qfi_rows.append((N, tc, tr))
    print(f"{N:3d} {2 ** N:9d} | {tc * 1e3:13.1f} {tr * 1e3:10.3f} {float(val):10.1f}")

print("\nSubsystem QFI at N=12, keeping K qubits: compression on and off\n")
print(f"{'K':>3s} {'d_A':>7s} {'d_B':>6s} | {'compressed run [ms]':>20s} {'naive run [ms]':>16s} {'speed-up':>9s}")
psi12 = haar_state(jax.random.PRNGKey(2), 12)
comp_rows = []
for K in (2, 4, 6, 8, 9, 10, 11):
    fc = jax.jit(partial(subsystem_qfi, n_dir=(0, 0, 1), K=K, compress=True))
    _, _, tr_c = timed(fc, psi12)
    if K <= 9:                                     # the naive route becomes painfully slow beyond this
        fn = jax.jit(partial(subsystem_qfi, n_dir=(0, 0, 1), K=K, compress=False))
        _, _, tr_n = timed(fn, psi12)
    else:
        tr_n = float("nan")
    comp_rows.append((K, tr_c, tr_n))
    print(f"{K:3d} {2 ** K:7d} {2 ** (12 - K):6d} | {tr_c * 1e3:20.3f} {tr_n * 1e3:16.3f} {tr_n / tr_c:9.1f}")

# %%
# ==============================================================================
# FIGURE: measured scaling
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
Nq = np.array([r[0] for r in qfi_rows], dtype=float)
tq = np.array([r[2] for r in qfi_rows]) * 1e3
axes[0].semilogy(Nq, tq, "o-", color=PALETTE[0], label="measured run time")
ref = tq[-1] * (Nq / Nq[-1]) * 2.0 ** (Nq - Nq[-1])
axes[0].semilogy(Nq, ref, "k--", lw=1, label=r"reference $\propto N\,2^N$")
axes[0].set_xlabel("number of qubits $N$"); axes[0].set_ylabel("time per QFI evaluation [ms]")
axes[0].set_title("Matrix-free pure-state QFI"); axes[0].legend(fontsize=9)

Kk = np.array([r[0] for r in comp_rows], dtype=float)
axes[1].semilogy(Kk, np.array([r[1] for r in comp_rows]) * 1e3, "o-", color=PALETTE[0], label="Schmidt-compressed")
mask = ~np.isnan([r[2] for r in comp_rows])
axes[1].semilogy(Kk[mask], np.array([r[2] for r in comp_rows])[mask] * 1e3, "s-", color=PALETTE[1],
                 label=r"naive ($O(8^K)$ eigh)")
axes[1].set_xlabel("kept qubits $K$ (out of $N=12$)"); axes[1].set_ylabel("time per evaluation [ms]")
axes[1].set_title("Subsystem QFI: cost of losing $N-K$ qubits"); axes[1].legend(fontsize=9)
fig.tight_layout(); plt.show()

# %% [markdown]
# The left panel is to be read as a comparison of *slopes*: the dashed reference is only fixed up to a constant, and
# it has been anchored at the last point. Below $N\approx10$ the measured curve is almost flat: the arithmetic there is
# faster than the fixed cost of dispatching a compiled program from Python (a few tens of microseconds), so the
# measurement reports the overhead rather than the algorithm. Above $N\approx10$ it turns over and climbs steadily, but its slope
# stays somewhat *below* the $N2^N$ reference: each additional pair of qubits costs a factor of about $2.5$–$4$ rather than the
# $4\,(N+2)/N\approx4.7$ that counting operations predicts. Two effects push the same way — the constant overhead has not
# fully disappeared even at $N=12$, and the largest einsums are the first ones big enough for XLA to spread across several
# cores. The lesson is the one every benchmark teaches: an $O(\cdot)$ is a statement about arithmetic, a timing is a
# statement about one machine, and they coincide only in the window where neither dispatch nor parallelism dominates.
# At $N=16$ (a $65\,536$-dimensional space) one QFI evaluation takes a few milliseconds or less, so a sweep over hundreds
# of parameters is a matter of seconds. Compare that with the dense alternative: a $2^{16}\times2^{16}$ complex matrix
# would need $69$ GB.
#
# The right panel is the point of Schmidt compression. Without it the cost rises steeply with the number of *kept* qubits and
# becomes unusable near $K=N$ — precisely the physically interesting regime of losing one or two particles. With compression
# the run time is essentially flat in $K$; the two curves coincide for $K\le6$ (where $d_A\le d_B$ and the compression branch
# is not taken at all) and separate sharply afterwards, reaching the speed-up printed above at $K=9$. At $K=11$ the naive
# route would have to diagonalise a $2048\times2048$ matrix that we know in advance has rank at most $2$.
#
# > **Numerical practice.** Whenever a matrix is built as $\Psi\Psi^\dagger$ from a tall-and-thin $\Psi$, its rank is bounded
# > by the thin dimension. Never hand such a matrix to a general eigensolver — project onto the column span first. The same
# > idea appears as "low-rank updates" in optimisation and as "active subspace" methods in uncertainty quantification.

# %% [markdown]
# ## 16. Key takeaways
#
# * **Estimation is a statistics problem with a quantum input.** An estimator has bias and variance; the Fisher information
#   $I(\theta)=\sum_x(\partial_\theta p)^2/p$ of the outcome distribution bounds the variance of any *unbiased* estimator by
#   $\mathrm{Var}\ge1/I$ (Cauchy–Schwarz, Section 5.1), the Fisher information of $M$ independent repetitions is $MI$, and
#   maximum likelihood reaches the bound asymptotically — which we measured from sampled data with error bars and confirmed by an exact binomial sum: the bias falls like
#   $-\cot\theta/(2M)$ (and vanishes identically at the symmetric point $\theta=\pi/2$), and the excess of
#   $M\,\mathrm{Var}$ over $1/I$ is $1\%$ at $M=100$ for $\theta=\pi/2$.
# * **The quantum Fisher information is the best measurement's Fisher information**, $F_Q=\max_{\{E_x\}}I(\theta;\{E_x\})$
#   (Braunstein–Caves), hence $\Delta\theta\ge1/\sqrt{MF_Q}$. It is a property of the state and of the generator only.
# * **For pure states with unitary encoding, $F_Q=4\,\mathrm{Var}(G)$** — derived twice, from the SLD and from the overlap
#   expansion, and independent of $\theta$. Matrix-free it costs $O(N2^N)$: apply $G$ once, take two inner products.
# * **Two limits, both proved.** Product states obey $F_Q\le N$ (variances of independent qubits add, each at most $1/4$);
#   every state obeys $F_Q\le N^2$ (the spectrum of $J_{\mathbf n}$ has width $N$), and GHZ saturates it.
# * **The state alone does not decide.** $F_Q$ is a property of the pair (state, generator). The $3\times3$ matrix
#   $\mathcal F_{ab}=4C_{ab}$ makes the direction dependence explicit, $F_Q(\mathbf n)=\mathbf n^{\mathsf T}\mathcal F\mathbf n$,
#   and the optimum over the sphere is the largest eigenvalue of a $3\times3$ matrix — one `eigh`, no optimisation.
# * **Entanglement is necessary but not sufficient.** $F_Q>\lfloor N/k\rfloor k^2+(N\bmod k)^2$ certifies entanglement
#   depth $k+1$; yet Haar-random states (near-maximal entropy) sit at $F_Q\approx N d/(d+1)$, just below the SQL, and the cluster state
#   beats $N$ only by a boundary term $2$. Metrology rewards coherent superpositions of very different generator
#   eigenvalues, which the entanglement entropy does not measure.
# * **Noise and loss destroy the Heisenberg limit.** Dephased GHZ obeys $F_Q=N^2(1-2p)^{2N}$ exactly — the advantage
#   dies exponentially in $N$ — and losing one single qubit of a GHZ state sets $F_Q$ to exactly zero. States that degrade
#   gracefully under loss, such as the Dicke state (which keeps $15$ of its $40$ after one lost qubit at $N=8$), are the
#   better candidates for real detectors.
# * **Implementation.** Validate every new quantity against at least two independent routes before plotting it (we used
#   matrix-free, dense, SLD, finite differences and `jax.jacfwd`); exploit rank structure (Schmidt compression turned an
#   $O(8^K)$ eigendecomposition into an $O(8^{N-K})$ one, more than two orders of magnitude at $K=9$, $N=12$ in our runs).
#
# ## 17. Exercises
#
# 1. ★ **Read the bound.** A magnetometer uses $N=100$ atoms and $M=10^4$ repetitions. Compute the smallest phase
#    uncertainty $\Delta\theta$ allowed at the standard quantum limit and at the Heisenberg limit, and the factor between
#    them. Then use Eq. (23) to find the per-qubit dephasing probability $p$ at which the GHZ advantage is entirely lost,
#    i.e. at which $N^2(1-2p)^{2N}=N$.
# 2. ★ **Fisher information of a tilted readout.** For the single qubit of Section 7.4, compute analytically and then with
#    `fisher_information_ad` the Fisher information of a measurement along the direction
#    $\mathbf m=(\cos\alpha,\sin\alpha,0)$ in the equatorial plane. Show that $p(\pm)=\left(1\pm\cos(\theta-\alpha)\right)/2$
#    and that $I(\theta)$ is the *same* for every $\alpha$, so that a whole family of readouts saturates the quantum
#    Cramér–Rao bound. Then set $\alpha=\theta$, so that the fringe sits at an extremum: what does
#    `fisher_information_ad` return, what is the limit of $I$ as $\alpha\to\theta$, and which of the two conditions of
#    Section 5.2 has been violated?
# 3. ★★ **Dicke states are a family (physics).** Compute $F_Q$ for $\vert D_N^k\rangle$ with $G=J_x$ for all
#    $k=0,\dots,N$ at $N=10$ and plot it against $k$. Derive the closed form from
#    $\langle J_x^2\rangle=\tfrac12[j(j+1)-m^2]$ with $j=N/2$, $m=N/2-k$, and check your formula against the numbers.
#    Which $k$ is optimal and why is $k=N/2$ not a coincidence?
# 4. ★★ **Superposition of the useless (extend the code).** Take two Haar-random states $\vert\psi_1\rangle,\vert\psi_2\rangle$
#    at $N=8$ and compute $F_Q$ of $(\vert\psi_1\rangle+e^{i\varphi}\vert\psi_2\rangle)/\mathcal N$ for $\varphi\in[0,2\pi)$.
#    Two metrologically useless states — how useful can their superposition be? Then repeat with the *pair*
#    $\vert0\cdots0\rangle$ and $\vert1\cdots1\rangle$, each of which has $F_Q=0$ for $G=J_z$. Explain both results with
#    Eq. (14): what property of the two summands decides whether the superposition gains anything?
# 5. ★★ **Optimal direction along a twisting evolution (extend the code).** Evolve $\vert+\rangle^{\otimes N}$ with the
#    engine's `oat_evolve` for a range of twisting times, and plot $\lambda_{\max}(\mathcal F)$ together with the three
#    Cartesian values $F_Q(J_x),F_Q(J_y),F_Q(J_z)$. By how much does choosing the optimal direction beat the best Cartesian
#    one? (This is a preview of notebook 34.)
# 6. ★★ **Noise before or after (extend the code).** Section 13 applied the channel *before* the encoding. Redo the
#    dephasing sweep with the channel applied *after* $U(\theta)$ and compare. Then do the same for $G=J_x$ instead of
#    $G=J_z$. Which combinations give identical results, and what property of the channel explains it? (Notebook 30 answers
#    this in detail — try to get there first.)
# 7. ★★★ **The cluster state's boundary (physics).** We found $\lambda_{\max}(\mathcal F)=N+2$ for the open cluster chain.
#    Verify that the excess disappears for the *periodic* cluster state (`cluster_state(N, periodic=True)`), identify the
#    stabiliser responsible for the off-diagonal elements of $\mathcal F$ in the open chain, and confirm your explanation by
#    computing the relevant correlator with `expect_pauli_string`.
# 8. ★★★ **A better lossy resource (extend the code).** Search numerically for the pure symmetric state of $N=8$ qubits that
#    maximises $F_Q$ of the reduced state *after losing one qubit*. Parametrise a state in the symmetric (Dicke) subspace by
#    $N+1$ complex amplitudes, use `jax.grad` on `subsystem_qfi` and Adam, and compare the optimum with GHZ, with
#    $\vert D_N^{N/2}\rangle$, and with the ideal $F_Q=N^2$. How much of the Heisenberg advantage can survive one lost particle?
#    *Hint on the numerics:* `subsystem_qfi` differentiates through `jnp.linalg.eigh`, whose gradient is undefined when two
#    eigenvalues coincide. Starting from a random complex vector of amplitudes the gradient is finite, but a symmetric
#    starting point such as GHZ sits exactly on a degeneracy and returns `nan`; start away from it, and check
#    `jnp.isfinite` on the gradient before feeding it to the optimiser.
#
# ## References
#
# * C. W. Helstrom, *Quantum Detection and Estimation Theory* (Academic Press, New York, 1976) — the book that founded
#   quantum estimation theory; the symmetric logarithmic derivative and the quantum Cramér–Rao bound.
# * S. L. Braunstein and C. M. Caves, *Statistical distance and the geometry of quantum states*,
#   Phys. Rev. Lett. **72**, 3439 (1994) — the theorem of Section 6.1: the quantum Fisher information is the maximum of the
#   classical Fisher information over all measurements, attained in the eigenbasis of the SLD.
# * V. Giovannetti, S. Lloyd and L. Maccone, *Quantum-enhanced measurements: beating the standard quantum limit*,
#   Science **306**, 1330 (2004), and *Quantum metrology*, Phys. Rev. Lett. **96**, 010401 (2006) — the standard quantum
#   limit versus the Heisenberg limit.
# * L. Pezzè and A. Smerzi, *Entanglement, nonlinear dynamics, and the Heisenberg limit*,
#   Phys. Rev. Lett. **102**, 100401 (2009) — Eq. (20): $F_Q>N$ witnesses entanglement.
# * P. Hyllus, W. Laskowski, R. Krischek, C. Schwemmer, W. Wieczorek, H. Weinfurter, L. Pezzè and A. Smerzi,
#   *Fisher information and multiparticle entanglement*, Phys. Rev. A **85**, 022321 (2012), and
#   G. Tóth, *Multipartite entanglement and high-precision metrology*, Phys. Rev. A **85**, 022322 (2012) — Eq. (21),
#   the entanglement-depth bound for $k$-producible states.
# * M. G. A. Paris, *Quantum estimation for quantum technology*, Int. J. Quantum Inf. **7**, 125–137 (2009) — a compact and
#   very readable review of everything in Sections 3–7.
# * L. Pezzè, A. Smerzi, M. K. Oberthaler, R. Schmied and P. Treutlein, *Quantum metrology with nonclassical states of
#   atomic ensembles*, Rev. Mod. Phys. **90**, 035005 (2018) — the standard modern review; collective spins, Dicke states,
#   squeezing, experiments.
# * G. Tóth and I. Apellaniz, *Quantum metrology from a quantum information science perspective*,
#   J. Phys. A **47**, 424006 (2014) — the entanglement-witness side of the QFI, with proofs of the convexity and of the
#   $k$-producibility bounds.
# * H. Cramér, *Mathematical Methods of Statistics* (Princeton University Press, 1946) — the classical Cramér–Rao bound and
#   the asymptotic theory of maximum likelihood.
