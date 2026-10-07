#@title: Many-body Bell correlators
#@part: Chapter 9 — Entanglement and complexity diagnostics
#@description: From CHSH to N parties: the many-body Bell correlator built from local raising operators, its local-realistic and separability bounds derived from scratch, matrix-free evaluation with einsum, optimisation of the measurement directions with jax.grad and SPSA, noise sweeps on the density tensor and on quantum trajectories with a bias-corrected estimator, and the correlator along a one-axis-twisting evolution.

# %% [markdown]
# ## 1. Introduction and motivation
#
# A Bell inequality is a bound on correlations that follows from one assumption only: that the outcomes of the measurements
# already exist before the measurements are made, and that each party's outcome depends on that party's setting and on shared
# information from the past, not on what the other parties chose. Any theory of this kind is called a **local hidden-variable
# (LHV)** model. Quantum mechanics violates such bounds, which is why they matter: a violated Bell inequality is a
# device-independent certificate — it says something about the physics without assuming anything about the apparatus.
#
# [Notebook 19](../ch08_quantum_information_protocols/19_bell_states_and_chsh.ipynb) derived the archetype, the CHSH
# inequality for two parties: the classical bound is $2$ and quantum mechanics reaches $2\sqrt2$. This notebook asks the same
# question for $N$ parties at once. That change is not cosmetic. With two parties there is one way to be entangled; with $N$
# there is a whole hierarchy, and a many-body Bell inequality can distinguish between "somewhere in this system there are two
# correlated particles" and "all $N$ particles participate in one correlated object". The quantities that do this are the
# objects of the last decade of research in quantum metrology and in cold-atom experiments, where the states are produced by
# collective interactions and the individual particles cannot be addressed at all.
#
# The correlator we study is built from one operator per site,
#
# $$\mathcal B(\boldsymbol\theta)=\bigotimes_{k=0}^{N-1}\tilde\sigma^{+}_{k},\qquad
#   \tilde\sigma^{+}_k=U_k\,\sigma^{+}\,U_k^{\dagger},\qquad \sigma^{+}=\vert1\rangle\langle0\vert , \tag{1}$$
#
# with an independent local unitary $U_k$ on every site, and the quantity of interest is
#
# $$\mathcal E=\max_{\{U_k\}}\bigl\vert\langle\mathcal B\rangle\bigr\vert^2 . \tag{2}$$
#
# At first sight Eq. (1) has nothing to do with Bell inequalities: $\sigma^+$ is not even Hermitian, so it is not an
# observable. Section 4 shows that it is a disguise: $\tilde\sigma^+=\tfrac12(A+iB)$ where $A$ and $B$ are two *orthogonal*
# dichotomic observables — exactly the two measurement settings each party has in a Bell test — so Eq. (2) is a specific
# complex combination of the $2^N$ ordinary correlation functions that an $N$-party Bell experiment measures. Two bounds then
# follow by direct calculation:
#
# $$\mathcal E\le2^{-N}\quad\text{(every local hidden-variable model)},\qquad
#   \mathcal E\le4^{-N}\quad\text{(every fully separable quantum state)} ,$$
#
# while the algebraic maximum over *all* states is $\mathcal E=1/4$, reached by the GHZ state. Taking logarithms turns the two
# bounds into a pair of numbers that are positive exactly when the corresponding bound is violated:
#
# $$Q_{\rm B}=\log_2\mathcal E+N ,\qquad Q_{\rm E}=\tfrac12\log_2\mathcal E+N . \tag{3}$$
#
# $Q_{\rm B}>0$ certifies Bell correlations; $Q_{\rm E}>0$ certifies entanglement; and $Q_{\rm B}\le N-2$,
# $Q_{\rm E}\le N-1$ always, with equality for GHZ.
#
# **Road map.** Section 3 recalls what a Bell inequality is and states CHSH. Section 4 builds the correlator and *derives*
# both bounds, together with the bound for states that factorise into $m$ groups (which certifies genuine $N$-partite
# entanglement). Section 5 turns Eq. (2) into two array lookups: because $\sigma^+{}^{\otimes N}=\vert1\cdots1\rangle\langle
# 0\cdots0\vert$, the whole expectation value is one matrix element of the locally rotated state, costing $O(N2^N)$ instead of
# $O(4^N)$. Section 6 derives closed forms for product, GHZ, W and Dicke states and verifies them. Section 7 is the
# optimisation problem: two angles per site, `jax.grad` plus Adam, `vmap` over random restarts, compared with SPSA and with the
# analytic optimum, plus a look at the landscape. Section 8 runs the test states of the scope, including graph states and
# $T$-doped graph states, and explains why some of them must give identical answers. Section 9 puts noise on the density
# tensor and derives the critical noise strength for each channel and each bound. Section 10 replaces the density tensor by
# quantum trajectories, derives the bias of the naive estimator of a *nonlinear* function of a Monte-Carlo average, corrects
# it, and puts bootstrap error bars on the result. Section 11 evaluates the correlator along a one-axis-twisting evolution and
# reports when each bound is crossed.
#
# ### What you will learn
#
# *Physics*
# * what a Bell inequality is for $N$ parties, and why a single number can separate "entangled" from "Bell-correlated";
# * the derivation of the local-realistic bound $2^{-N}$, of the $m$-group separability bound $4^{-m}$ and of the algebraic
#   maximum $1/4$;
# * closed forms: $\mathcal E=4^{-N}$ for $\vert+\rangle^{\otimes N}$ (the separability bound is *saturated*, not violated),
#   $\mathcal E=1/4$ for GHZ, $\mathcal E=\binom{N}{m}^24^{-N}$ for Dicke states (hence $\mathcal E=N^24^{-N}$ for W);
# * why the correlator is invariant under local unitaries, so that a graph state and the same graph state with a $T$ gate on
#   every qubit are indistinguishable by it — until noise is added, which breaks the invariance for some channels but not for
#   others;
# * how fast Bell correlations die under dephasing, depolarising and amplitude damping, and why the Bell bound is much more
#   fragile than the entanglement bound;
# * that one-axis twisting generates Bell correlations, and at what time each bound is crossed.
#
# *Numerical methods*
# * matrix-free evaluation of a product of $N$ non-Hermitian single-site operators;
# * gradient-based maximisation over $2N$ angles with `jax.grad`, Adam and random restarts, against SPSA;
# * the bias of $\vert\bar z\vert^2$ as an estimator of $\vert\mathbb E z\vert^2$, its exact correction, and bootstrap error
#   bars for a nonlinear function of a Monte-Carlo average;
# * density tensor ($O(4^N)$) against quantum trajectories ($O(M2^N)$) for the same noisy physics.
#
# *Implementation practice*
# * writing a cost function that is simultaneously `jit`-able, `grad`-able and `vmap`-able, and batching the restarts;
# * `lax.scan` for the optimiser loop so that the graph is compiled once, not unrolled 200 times;
# * separating "what the optimiser found" from "what the state can give" by always comparing against an analytic optimum.
#
# ### Prerequisites
# * [01 — JAX from scratch](../ch01_computational_toolbox/01_jax_from_scratch.ipynb) and
#   [02 — einsum from scratch](../ch01_computational_toolbox/02_einsum_from_scratch.ipynb);
# * [19 — Bell states and CHSH](../ch08_quantum_information_protocols/19_bell_states_and_chsh.ipynb): local hidden variables,
#   the CHSH bound $2$, the Tsirelson bound $2\sqrt2$ — we build directly on it;
# * [07 — density matrices and quantum channels](../ch03_matrix_free_engine/07_density_matrices_and_quantum_channels.ipynb)
#   and [17 — Monte-Carlo wave function](../ch06_open_quantum_systems/17_monte_carlo_wave_function.ipynb) for Sections 9–10;
# * [22 — GHZ states and decoherence](../ch08_quantum_information_protocols/22_ghz_states_and_decoherence.ipynb): the GHZ
#   coherence and its fragility, which is what Section 9 measures again through a different quantity;
# * [25 — entanglement negativity](25_entanglement_negativity.ipynb) is the companion diagnostic of this chapter: the
#   negativity detects entanglement across one cut, the Bell correlator detects it in all parties at once;
# * [41 — optimizers](../ch11_variational_quantum_circuits/41_optimizers.ipynb) for Adam and SPSA;
# * [33 — spin squeezing and one-axis twisting](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb)
#   and [34 — from OAT to GHZ](../ch10_quantum_metrology_protocols/34_oat_to_ghz_full_metrology_protocol.ipynb) for Section 11.
#
# **Conventions.** Qubit $q$ = tensor axis $q$, counted from $0$; $\vert0\rangle$ is the $+1$ eigenstate of $Z$;
# $\sigma^{+}=\vert1\rangle\langle0\vert$ raises. Logarithms in Eq. (3) are base $2$.

# %% [markdown]
# ## 2. Engine recap and helpers
#
# We need the gate primitives for state vectors and density tensors, the standard states, the noise channels in both their
# density-matrix and trajectory form, the Adam and SPSA routines, and the exact one-axis-twisting evolution.

# %%
#@engine: apply_gate, apply_gate_dm, apply_kraus_dm, apply_kraus_mcwf, rdm, to_dm, dm_matrix, zero_state, product_state, ghz_state, w_state, dicke_state, haar_state, I2, X, Y, Z, H, S, T, CZ, SP, ry, rz, kraus_dephasing, kraus_depolarizing, kraus_amplitude_damping, adam_init, adam_update, spsa_grad, oat_evolve, spin_squeezing, expect_pauli_string

# %%
# ==============================================================================
# PLOT STYLE + notebook-wide constants
# ==============================================================================
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # colour-blind-friendly, fixed order
MARKERS = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=PALETTE), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "legend.frameon": False})


def max_abs(a):
    """Largest absolute entry of an array, as a Python float."""
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


print("sigma^+ = |1><0| :\n", np.asarray(SP))

# %% [markdown]
# ## 3. From CHSH to $N$ parties
#
# ### 3.1 What a Bell inequality is
#
# $N$ separated parties each receive one particle. Party $k$ chooses one of two measurement settings and records an outcome
# $\pm1$. Repeating many times gives the **correlation functions**: for each choice of one setting per party, the average of
# the product of the $N$ outcomes.
#
# A **local hidden-variable model** explains those numbers as follows. A variable $\lambda$ is distributed with some density
# $p(\lambda)$ and shared by all parties (it is whatever was agreed, or physically established, in their common past). Given
# $\lambda$, party $k$'s outcome for setting $1$ is a fixed number $A_k(\lambda)\in\{-1,+1\}$ and for setting $2$ a fixed
# number $B_k(\lambda)\in\{-1,+1\}$. Because the outcomes are fixed functions of $\lambda$ alone, every correlation function
# is an average of a *product of numbers*:
#
# $$\langle A_0B_1A_2\cdots\rangle_{\rm LHV}=\int d\lambda\,p(\lambda)\,A_0(\lambda)B_1(\lambda)A_2(\lambda)\cdots . \tag{4}$$
#
# Eq. (4) is the whole content of "local realism", and every Bell inequality is an inequality that all numbers of the form (4)
# obey and that quantum mechanics can break. The essential point, and the one that makes the derivations in Section 4 short,
# is that in Eq. (4) *all* the settings of a party exist simultaneously as numbers, even the ones not chosen.
#
# ### 3.2 The CHSH inequality
#
# For $N=2$, with $A_k,B_k=\pm1$, the algebraic identity
#
# $$A_0A_1+A_0B_1+B_0A_1-B_0B_1=A_0(A_1+B_1)+B_0(A_1-B_1)$$
#
# has, for every $\lambda$, one bracket equal to $\pm2$ and the other equal to $0$, so the left-hand side is $\pm2$ and its
# average obeys $\vert S\vert\le2$. Quantum mechanics reaches $S=2\sqrt2$ with a Bell state and the right angles — that is the
# Tsirelson bound, derived in [notebook 19](../ch08_quantum_information_protocols/19_bell_states_and_chsh.ipynb).
#
# ### 3.3 Why many parties need a different object
#
# One can write CHSH-type inequalities for $N$ parties (Mermin 1990; Zukowski and Brukner 2002), and they are violated
# exponentially strongly by GHZ states. They are, however, awkward for a many-body physicist: they involve sums over $2^N$
# correlation functions with alternating signs, and the settings that maximise them are different for every state. The
# correlator of Eq. (1) packages the same $2^N$ correlation functions into one product, which turns out to be (i) trivially
# computable on a state vector, (ii) analytically tractable for the standard many-body states, and (iii) directly measurable
# in experiments with collective spins. That is the object we now build.

# %% [markdown]
# ## 4. The many-body Bell correlator and its bounds
#
# ### 4.1 A rotated raising operator is a pair of orthogonal measurement settings
#
# Start from $\sigma^{+}=\vert1\rangle\langle0\vert$. In terms of Pauli matrices
#
# $$\sigma^{+}=\frac{X-iY}{2} .$$
#
# (Check: $X=\vert0\rangle\langle1\vert+\vert1\rangle\langle0\vert$ and $Y=i\vert1\rangle\langle0\vert-i\vert0\rangle\langle1\vert$,
# so $iY=\vert0\rangle\langle1\vert-\vert1\rangle\langle0\vert$ and $X-iY=2\vert1\rangle\langle0\vert$. The opposite sign,
# $\sigma^{+}=(X+iY)/2$, belongs to the spin convention in which $\vert0\rangle$ is the *lower* state; here
# $\vert0\rangle$ is the $+1$ eigenstate of $Z$ and $\sigma^{+}$ raises the *bit*.) Conjugating with a local unitary $U$ gives
#
# $$\tilde\sigma^{+}=U\sigma^{+}U^{\dagger}=\frac{A+iB}{2},\qquad A=UXU^{\dagger},\quad B=-\,UYU^{\dagger} . \tag{5}$$
#
# $A$ and $B$ are Hermitian, traceless, and $A^2=UX^2U^\dagger=\mathbb 1$, $B^2=\mathbb 1$: each is a **dichotomic
# observable** with outcomes $\pm1$ (the minus sign in $B$ only swaps the labels of its two outcomes). They also
# anticommute, $AB+BA=-U(XY+YX)U^\dagger=0$, which means that as Bloch vectors
# $A=\hat n_A\cdot\vec\sigma$ and $B=\hat n_B\cdot\vec\sigma$ they point along **orthogonal** directions. Conversely, any pair
# of orthogonal directions on the Bloch sphere can be reached by some $U$. So Eq. (5) says:
#
# > choosing the local unitary $U_k$ *is* choosing the two measurement settings of party $k$, constrained to be orthogonal.
#
# Substituting Eq. (5) into Eq. (1),
#
# $$\mathcal B=\bigotimes_k\tilde\sigma^{+}_k=\frac{1}{2^N}\bigotimes_k\bigl(A_k+iB_k\bigr)
#  =\frac{1}{2^N}\sum_{S\subseteq\{0..N-1\}}i^{\vert S\vert}\prod_{k\in S}B_k\prod_{k\notin S}A_k . \tag{6}$$
#
# The right-hand side is a sum over the $2^N$ ways of picking one setting per party, each term a product of commuting
# single-party observables — that is, exactly one of the $2^N$ correlation functions of an $N$-party Bell experiment, with the
# real part collecting the terms with $\vert S\vert$ even and the imaginary part those with $\vert S\vert$ odd. Therefore
#
# $$\mathcal E=\vert\langle\mathcal B\rangle\vert^2
#  =\frac{1}{4^N}\left\vert\Bigl\langle\prod_k(A_k+iB_k)\Bigr\rangle\right\vert^2 \tag{7}$$
#
# is a function of measurable correlation functions only, and a bound on it is a Bell inequality.
#
# ### 4.2 The local hidden-variable bound
#
# In an LHV model every correlation function is of the form (4), so by linearity
#
# $$\Bigl\langle\prod_k(A_k+iB_k)\Bigr\rangle_{\rm LHV}
#  =\int d\lambda\,p(\lambda)\prod_k\bigl(A_k(\lambda)+iB_k(\lambda)\bigr) .$$
#
# For each $\lambda$ and each $k$ the factor $A_k(\lambda)+iB_k(\lambda)$ is one of the four complex numbers $\pm1\pm i$, all
# of modulus $\sqrt2$. Hence the product has modulus exactly $2^{N/2}$, and by the triangle inequality
#
# $$\left\vert\Bigl\langle\prod_k(A_k+iB_k)\Bigr\rangle_{\rm LHV}\right\vert
#  \le\int d\lambda\,p(\lambda)\,2^{N/2}=2^{N/2} .$$
#
# With Eq. (7),
#
# $$\boxed{\ \mathcal E_{\rm LHV}\le\frac{2^{N}}{4^{N}}=2^{-N}\ }\qquad\Longleftrightarrow\qquad Q_{\rm B}=\log_2\mathcal E+N\le0 . \tag{8}$$
#
# Any state and any choice of local unitaries that produce $Q_{\rm B}>0$ have produced correlations that **no** local
# hidden-variable model can reproduce. The bound holds for every $N$ and needs no assumption about the quantum state at all.
#
# ### 4.3 The separability bounds
#
# Now a quantum-mechanical bound. Let the state be separable with respect to a partition of the $N$ qubits into $m$ groups
# $G_1,\dots,G_m$:
#
# $$\rho=\sum_jp_j\,\rho^{(j)}_{G_1}\otimes\cdots\otimes\rho^{(j)}_{G_m} . \tag{9}$$
#
# The operator $\mathcal B$ factorises over the same groups, so
#
# $$\langle\mathcal B\rangle=\sum_jp_j\prod_{g=1}^{m}\mathrm{Tr}\Bigl(\rho^{(j)}_{G_g}\bigotimes_{k\in G_g}\tilde\sigma^{+}_k\Bigr) .$$
#
# Each group factor is bounded by $1/2$:
# $\bigotimes_{k\in G}\tilde\sigma^{+}_k=\vert a_G\rangle\langle b_G\vert$ with
# $\vert a_G\rangle=\bigotimes_{k\in G}U_k\vert1\rangle$ and $\vert b_G\rangle=\bigotimes_{k\in G}U_k\vert0\rangle$, which are
# **orthonormal** because $U\vert0\rangle\perp U\vert1\rangle$ on every site. Then, for any state $\rho$ of that group,
#
# $$\bigl\vert\mathrm{Tr}(\rho\,\vert a\rangle\langle b\vert)\bigr\vert=\bigl\vert\langle b\vert\rho\vert a\rangle\bigr\vert
#  \le\sqrt{\langle a\vert\rho\vert a\rangle\,\langle b\vert\rho\vert b\rangle}
#  \le\frac{\langle a\vert\rho\vert a\rangle+\langle b\vert\rho\vert b\rangle}{2}\le\frac12 ,$$
#
# using the Cauchy–Schwarz inequality for the positive sesquilinear form $(x,y)\mapsto\langle x\vert\rho\vert y\rangle$, then
# the inequality between geometric and arithmetic means, and finally
# $\langle a\vert\rho\vert a\rangle+\langle b\vert\rho\vert b\rangle\le\mathrm{Tr}\,\rho=1$, valid because $\vert a\rangle$ and
# $\vert b\rangle$ are orthonormal. Multiplying the $m$ group factors and averaging over $j$,
#
# $$\vert\langle\mathcal B\rangle\vert\le\sum_jp_j\,2^{-m}=2^{-m}\qquad\Longrightarrow\qquad
#   \boxed{\ \mathcal E\le4^{-m}\ } \tag{10}$$
#
# for any state separable into $m$ groups and any choice of local unitaries. Nothing in the argument used a *fixed*
# partition: a mixture $\rho=\sum_jp_j\rho^{(j)}$ in which different terms factorise over *different* partitions into $m$
# groups obeys the same bound, because the triangle inequality is applied term by term. Three special cases matter.
#
# * $m=N$ (**fully separable**, no entanglement anywhere): $\mathcal E\le4^{-N}$, that is
#   $Q_{\rm E}=\tfrac12\log_2\mathcal E+N\le0$. So $Q_{\rm E}>0$ certifies entanglement.
# * $m=2$ (**biseparable**: a mixture of states each separable across *some* bipartition): $\mathcal E\le4^{-2}=1/16$,
#   that is $Q_{\rm B}\le N-4$ and $Q_{\rm E}\le N-2$. A state with $Q_{\rm E}>N-2$ is not biseparable: it carries
#   **genuine $N$-partite entanglement**.
# * More generally, if every group contains at most $k$ qubits then $m\ge\lceil N/k\rceil$, so
#   $\mathcal E\le4^{-\lceil N/k\rceil}$; violating that bound certifies an entangled group of more than $k$ qubits (an
#   "entanglement depth" larger than $k$). The bound is *tight*: $\lvert\mathrm{GHZ}_k\rangle^{\otimes N/k}$ reaches it
#   exactly, as Exercise 3 asks you to verify.
#
# ### 4.4 The algebraic maximum
#
# Taking $m=1$ in Eq. (10) — no factorisation assumed, i.e. an arbitrary state — gives
#
# $$\mathcal E\le\tfrac14\qquad\Longleftrightarrow\qquad Q_{\rm B}\le N-2,\quad Q_{\rm E}\le N-1 , \tag{11}$$
#
# for every state whatsoever. The inequality chain above is saturated when $\langle a\vert\rho\vert a\rangle=
# \langle b\vert\rho\vert b\rangle=\tfrac12$ and $\rho$ is pure with $\vert\langle b\vert\rho\vert a\rangle\vert$ maximal,
# i.e. for the state $(\vert a\rangle+e^{i\varphi}\vert b\rangle)/\sqrt2$ — a GHZ state in the rotated local basis.
#
# ### 4.5 Summary of the scales
#
# | quantity | definition | fully separable | local realism | any state |
# |---|---|---|---|---|
# | $\mathcal E$ | $\max_{U}\vert\langle\mathcal B\rangle\vert^2$ | $\le4^{-N}$ | $\le2^{-N}$ | $\le 1/4$ |
# | $Q_{\rm E}$ | $\tfrac12\log_2\mathcal E+N$ | $\le0$ | — | $\le N-1$ |
# | $Q_{\rm B}$ | $\log_2\mathcal E+N$ | $\le-N$ | $\le0$ | $\le N-2$ |
#
# The two $Q$ quantities are just two different rulers laid on the same number $\mathcal E$, chosen so that each is positive
# exactly when its own bound is violated. Everything below is measured in these units.
#
# ### 4.6 Relation to the literature
#
# Both bounds and both rulers of Eq. (3) are standard in the many-body Bell literature, under slightly different names.
#
# * The LHV bound $\mathcal E\le2^{-N}$ is the qubit case of the CFRD family of inequalities built from complex
#   combinations of two local observables. Cavalcanti, He, Reid and Wiseman (2011) write it as
#   $\vert\langle\prod_j(X_j+iY_j)\rangle\vert\le2^{(N-T)/2}$, where $T$ counts the parties whose local state is assumed
#   to be quantum: $T=0$ reproduces $\mathcal E\le2^{-N}$ and $T=N$ reproduces $\mathcal E\le4^{-N}$, with the values in
#   between corresponding to multipartite EPR steering. Chwedenczuk (2022) derives the same bound by the Cauchy–Schwarz
#   inequality applied directly to Eq. (4).
# * $Q_{\rm B}=\log_2\mathcal E+N$ is the quantity called $Q_N$ in Plodzien *et al.*, Phys. Rev. Research **6**, 023050
#   (2024), which defines $\mathcal E_N\equiv2^{Q_N-N}$; $Q_{\rm E}=\tfrac12\log_2\mathcal E+N=\log_4(4^N\mathcal E)$ is
#   the quantity called $\mathcal Q$ in Plodzien *et al.*, Phys. Rev. A **110**, 032428 (2024). In that notation Eq. (10)
#   reads $Q_{\rm E}\le N-m$ for $m$-separable states, which is Eq. (16) of the latter paper, and the biseparable case
#   $Q_{\rm E}\le N-2$ is its Eq. (12).
# * The same papers also state **depth ladders** of the form "$Q_{\rm E}>k-2$ certifies entanglement depth $k$" and
#   "$Q_{\rm B}>k-3$ certifies that $k$ qubits are Bell-correlated". Those thresholds are obtained by separating off
#   single qubits one at a time, which is a *different* family of partitions from the one used above: they are the
#   statements for partitions with one large group and $N-k$ singletons, while Eq. (10) with $m=\lceil N/k\rceil$ covers
#   partitions into groups of equal size. The two do not agree — $\lvert\mathrm{GHZ}_3\rangle^{\otimes2}$ at $N=6$ has
#   $Q_{\rm E}=4$ and depth $3$, while the single-qubit ladder would read $Q_{\rm E}>k-2$ with $k=6$ — so in this
#   notebook we quote only the bound we have derived, Eq. (10), which is the one that is saturated by equal GHZ blocks.
#

# %% [markdown]
# ## 5. Matrix-free evaluation
#
# ### 5.1 The whole expectation value is one matrix element
#
# The reason this correlator is cheap is an identity that takes one line. Since $\sigma^{+}=\vert1\rangle\langle0\vert$,
#
# $$\bigotimes_k\sigma^{+}_k=\vert1\cdots1\rangle\langle0\cdots0\vert ,$$
#
# and therefore, pulling the local unitaries out of the tensor product,
#
# $$\mathcal B=\Bigl(\bigotimes_kU_k\Bigr)\,\vert1\cdots1\rangle\langle0\cdots0\vert\,\Bigl(\bigotimes_kU_k^{\dagger}\Bigr) .$$
#
# For a pure state, writing $\vert\psi'\rangle=\bigl(\bigotimes_kU_k^{\dagger}\bigr)\vert\psi\rangle$,
#
# $$\langle\psi\vert\mathcal B\vert\psi\rangle=
#   \underbrace{\langle\psi\vert\Bigl(\bigotimes_kU_k\Bigr)\vert1\cdots1\rangle}_{\overline{\psi'[1,\dots,1]}}\;
#   \underbrace{\langle0\cdots0\vert\Bigl(\bigotimes_kU_k^{\dagger}\Bigr)\vert\psi\rangle}_{\psi'[0,\dots,0]}
#   =\overline{\psi'[1,\dots,1]}\;\psi'[0,\dots,0] . \tag{12}$$
#
# Two entries of an array. The only work is producing $\vert\psi'\rangle$, which is $N$ single-qubit `apply_gate` calls,
# $O(N2^N)$ operations and no extra memory. The $2^N\times2^N$ matrix $\mathcal B$ never exists, and neither does the sum over
# $2^N$ correlation functions in Eq. (6) — they are all inside those two numbers.
#
# For a density tensor the same manipulation gives
#
# $$\mathrm{Tr}(\rho\,\mathcal B)=\langle0\cdots0\vert\rho'\vert1\cdots1\rangle=\rho'[0,\dots,0;1,\dots,1],
#   \qquad \rho'=\Bigl(\bigotimes_kU_k^{\dagger}\Bigr)\rho\Bigl(\bigotimes_kU_k\Bigr) , \tag{13}$$
#
# one entry of the rank-$2N$ array, at cost $O(N4^N)$.
#
# ### 5.2 Parametrising the local unitary
#
# Every $U\in SU(2)$ is $U=R_z(\varphi)R_y(\theta)R_z(\chi)$ (Euler angles). The third angle is free of charge here:
# $R_z(\chi)\sigma^{+}R_z(\chi)^{\dagger}=e^{i\chi}\sigma^{+}$, so it only multiplies $\mathcal B$ by a phase and leaves
# $\vert\langle\mathcal B\rangle\vert$ untouched. **Two angles per site suffice**, and we take
#
# $$U_k(\theta_k,\varphi_k)=R_z(\varphi_k)\,R_y(\theta_k) , \tag{14}$$
#
# so that the pair of measurement directions $(\hat n_A,\hat n_B)$ is the image of $(\hat x,-\hat y)$ under the rotation that
# takes $\hat z$ to the point $(\theta_k,\varphi_k)$ of the Bloch sphere. The parameter vector is an array of shape $(N,2)$.

# %%
# ==============================================================================
# STEP 1: the correlator, matrix-free, for pure states and for density tensors
# ==============================================================================
def local_u(theta, phi):
    """U(theta, phi) = Rz(phi) Ry(theta)  -- Eq. (14).  Traced angles: jit/grad/vmap friendly."""
    return rz(phi) @ ry(theta)


def bell_expectation(psi, angles):
    """<psi| (x)_k U_k sigma^+ U_k^dagger |psi>   for a state TENSOR psi of shape (2,)*N.

    MATH   psi' = (x)_k U_k^dagger psi   =>   <B> = conj(psi'[1,..,1]) * psi'[0,..,0]        [Eq. (12)]
    IMPLEMENTATION  N single-qubit einsums (`apply_gate`), then two scalar lookups.  The operator B is
           never built; the 2^N correlation functions of Eq. (6) are summed implicitly by the rotations.
    COST   O(N 2^N) time, O(2^N) memory.   Returns a COMPLEX scalar.
    JAX    `angles` has shape (N,2) and is traced -> jax.grad differentiates through the whole thing.
    """
    N = psi.ndim
    for q in range(N):
        psi = apply_gate(psi, local_u(angles[q, 0], angles[q, 1]).conj().T, [q])
    return jnp.conj(psi[(1,) * N]) * psi[(0,) * N]


def bell_expectation_dm(rho, angles):
    """Tr(rho B) for a density TENSOR rho of shape (2,)*(2N).

    MATH   rho' = (x)_k U_k^dagger rho (x)_k U_k   =>   Tr(rho B) = rho'[0,..,0 ; 1,..,1]     [Eq. (13)]
    COST   O(N 4^N) time, O(4^N) memory -- the wall that sends us to trajectories in Section 10.
    """
    N = rho.ndim // 2
    for q in range(N):
        rho = apply_gate_dm(rho, local_u(angles[q, 0], angles[q, 1]).conj().T, [q])
    return rho[(0,) * N + (1,) * N]


def correlator(psi, angles):
    """E = |<B>|^2 for a pure state (real, >= 0) -- the quantity of Eq. (2) at FIXED angles."""
    return jnp.abs(bell_expectation(psi, angles)) ** 2


def correlator_dm(rho, angles):
    """E = |Tr(rho B)|^2 for a density tensor at fixed angles."""
    return jnp.abs(bell_expectation_dm(rho, angles)) ** 2


def q_values(E, N):
    """(Q_B, Q_E) of Eq. (3) from the correlator E and the number of qubits N.
    Q_B = log2 E + N  (> 0 violates local realism);  Q_E = (1/2) log2 E + N  (> 0 certifies entanglement)."""
    lg = np.log2(np.maximum(np.asarray(E, dtype=float), 1e-300))
    return lg + N, 0.5 * lg + N

# %% [markdown]
# ### 5.3 Validation against an explicitly constructed operator
#
# Before trusting Eq. (12) we build $\mathcal B$ as a genuine $2^N\times2^N$ matrix with Kronecker products, for $N=3$ and
# random angles, and compare. This is the only place in the notebook where the operator exists.

# %%
# ==============================================================================
# STEP 2: checkpoint -- Eq. (12) and Eq. (13) against a dense Kronecker construction
# ==============================================================================
def bell_operator_dense(angles, N):
    """B = (x)_k U_k sigma^+ U_k^dagger as a dense 2^N x 2^N matrix.  VALIDATION ONLY (cost 4^N)."""
    B = np.ones((1, 1), dtype=complex)
    for q in range(N):
        U = np.asarray(local_u(angles[q, 0], angles[q, 1]))
        B = np.kron(B, U @ np.asarray(SP) @ U.conj().T)
    return B


N_chk = 3
key = jax.random.PRNGKey(0)
k_ang, k_psi = jax.random.split(key)
ang_chk = jax.random.uniform(k_ang, (N_chk, 2)) * 2 * jnp.pi
psi_chk = haar_state(k_psi, N_chk)
B_dense = bell_operator_dense(ang_chk, N_chk)

ref_psi = np.vdot(np.asarray(psi_chk).reshape(-1), B_dense @ np.asarray(psi_chk).reshape(-1))
got_psi = complex(bell_expectation(psi_chk, ang_chk))
rho_chk = to_dm(psi_chk)
ref_dm = np.trace(np.asarray(dm_matrix(rho_chk)) @ B_dense)
got_dm = complex(bell_expectation_dm(rho_chk, ang_chk))

print(f"pure state : matrix-free {got_psi:.12f}")
print(f"             dense       {ref_psi:.12f}      |difference| = {abs(got_psi - ref_psi):.2e}")
print(f"density ten: matrix-free {got_dm:.12f}")
print(f"             dense       {ref_dm:.12f}      |difference| = {abs(got_dm - ref_dm):.2e}")
assert abs(got_psi - ref_psi) < 1e3 * TOL and abs(got_dm - ref_dm) < 1e3 * TOL

# the third Euler angle only changes the phase, never the modulus -- check it explicitly
def bell_expectation_3angle(psi, angles, chi):
    """Same as `bell_expectation` but with the FULL Euler parametrisation U = Rz(phi) Ry(theta) Rz(chi)."""
    N = psi.ndim
    for q in range(N):
        U = local_u(angles[q, 0], angles[q, 1]) @ rz(chi[q])
        psi = apply_gate(psi, U.conj().T, [q])
    return jnp.conj(psi[(1,) * N]) * psi[(0,) * N]


chi = jax.random.uniform(jax.random.PRNGKey(3), (N_chk,)) * 2 * jnp.pi
b3 = complex(bell_expectation_3angle(psi_chk, ang_chk, chi))
print(f"\nwith a third Euler angle: <B> = {b3:.10f},  |<B>| = {abs(b3):.12f}")
print(f"without it              : <B> = {got_psi:.10f},  |<B>| = {abs(got_psi):.12f}")
print(f"difference of the moduli: {abs(abs(b3) - abs(got_psi)):.2e}")
assert abs(abs(b3) - abs(got_psi)) < 1e3 * TOL

# %% [markdown]
# Both identities reproduce the dense construction to $10^{-16}$, and the modulus is unchanged by the extra $R_z$ rotations,
# confirming that the two-angle parametrisation of Eq. (14) loses nothing.
#
# > **JAX practice.** `bell_expectation` is written as a pure function of `(psi, angles)` with the qubit index a static
# > Python integer and the angles traced. That is exactly the shape a function must have to be passed to `jax.grad` and
# > `jax.vmap` later; the two array lookups `psi[(1,)*N]` use static tuples, so they become constant-index gathers at trace
# > time and do not block differentiation.

# %% [markdown]
# ## 6. Closed forms for the standard states
#
# ### 6.1 Product states saturate the separability bound
#
# Take $\vert+\rangle^{\otimes N}$ and $U_k=\mathbb 1$. Then $\psi'=\psi$, and both entries in Eq. (12) equal $2^{-N/2}$, so
# $\langle\mathcal B\rangle=2^{-N}$ and
#
# $$\mathcal E\bigl(\vert+\rangle^{\otimes N}\bigr)=4^{-N},\qquad Q_{\rm E}=0,\qquad Q_{\rm B}=-N .$$
#
# By Eq. (10) with $m=N$ no product state can do better, so $\vert+\rangle^{\otimes N}$ is an *extremal* separable state: it
# sits exactly on the entanglement bound and exactly $N$ units below the local-realism bound. This is the reference point for
# everything else, and it is also a useful test of an optimiser: if the optimiser reports $Q_{\rm E}>0$ for a product state,
# the optimiser is broken.
#
# ### 6.2 GHZ reaches the algebraic maximum
#
# For $\vert\mathrm{GHZ}\rangle=(\vert0\cdots0\rangle+\vert1\cdots1\rangle)/\sqrt2$ and again $U_k=\mathbb 1$, Eq. (12) gives
# $\langle\mathcal B\rangle=\overline{(1/\sqrt2)}\cdot(1/\sqrt2)=1/2$ and
#
# $$\mathcal E(\mathrm{GHZ})=\tfrac14,\qquad Q_{\rm B}=N-2,\qquad Q_{\rm E}=N-1 ,$$
#
# which is the algebraic maximum of Eq. (11). No optimisation is needed: the identity angles are already optimal. GHZ
# therefore violates the local-realism bound by $N-2$ powers of two — an exponentially strong violation, growing with the
# number of particles.
#
# ### 6.3 Dicke and W states
#
# Let $\vert a\rangle=U\vert1\rangle=\alpha\vert0\rangle+\beta\vert1\rangle$ and $\vert b\rangle=U\vert0\rangle$, the same on
# every site. Orthogonality fixes $\vert b\rangle=e^{i\chi}(-\bar\beta\vert0\rangle+\bar\alpha\vert1\rangle)$. For the Dicke
# state $\vert D_N^m\rangle=\binom{N}{m}^{-1/2}\sum_{\vert s\vert=m}\vert s\rangle$ every basis state with $m$ excitations
# contributes the same amplitude, so
#
# $$\langle D_N^m\vert a^{\otimes N}\rangle=\binom{N}{m}^{-1/2}\binom{N}{m}\,\alpha^{N-m}\beta^{m}
#  =\sqrt{\tbinom{N}{m}}\;\alpha^{N-m}\beta^{m} ,$$
#
# and likewise $\langle b^{\otimes N}\vert D_N^m\rangle=\sqrt{\binom{N}{m}}\,(-\beta)^{N-m}\alpha^{m}e^{-iN\chi}$.
# Multiplying the two, as Eq. (12) instructs,
#
# $$\vert\langle\mathcal B\rangle\vert=\binom{N}{m}\,\vert\alpha\vert^{N}\vert\beta\vert^{N}
#  \le\binom{N}{m}2^{-N} ,$$
#
# because $\vert\alpha\beta\vert\le\tfrac12$ with equality at $\theta=\pi/2$. Hence, for the symmetric choice of angles,
#
# $$\mathcal E(D_N^m)=\binom{N}{m}^{2}4^{-N},\qquad Q_{\rm E}=\log_2\binom{N}{m},\qquad
#   Q_{\rm B}=2\log_2\binom{N}{m}-N . \tag{15}$$
#
# The W state is $m=1$, so $\mathcal E(W_N)=N^24^{-N}$, $Q_{\rm E}=\log_2N$ and $Q_{\rm B}=2\log_2N-N$. Two readings follow
# immediately and will be checked numerically: W states are detected as entangled for every $N$ (since $\log_2N>0$), but they
# violate local realism only for $N=3$ ($Q_{\rm B}=2\log_23-3=0.17$), reach exactly $Q_{\rm B}=0$ at $N=4$ and fall below the
# bound from $N=5$ on. Eq. (15) assumes the same $U$ on every site; the numerics below optimise over all $2N$ angles freely
# and can therefore test whether the symmetric choice is really optimal.
#
# ### 6.4 Invariance under local unitaries
#
# Because the maximum in Eq. (2) runs over *all* local unitaries, replacing $\vert\psi\rangle$ by
# $\bigl(\bigotimes_kV_k\bigr)\vert\psi\rangle$ changes nothing: the optimiser simply absorbs $V_k$ into $U_k$. Hence
#
# $$\mathcal E\bigl((\textstyle\bigotimes_kV_k)\psi\bigr)=\mathcal E(\psi)\qquad\text{for any local unitaries }V_k . \tag{16}$$
#
# Three consequences that the scope of this notebook asks us to test:
#
# * a GHZ state written in randomly rotated local bases has exactly the same $\mathcal E$ as the GHZ state itself;
# * the **star graph state** (a central qubit connected by $CZ$ to every other) is a GHZ state up to Hadamards on the leaves,
#   so $\mathcal E=1/4$;
# * applying a $T$ gate to every qubit of any state cannot change $\mathcal E$, because $T$ is a local unitary. The
#   "$T$-doped" star graph, which is *not* a stabilizer state and has non-zero magic (the subject of notebook 27), is
#   indistinguishable from the star graph by this correlator.
#
# The third point has a consequence: $\mathcal E$ is a local-unitary invariant, so it measures a property of the entanglement
# structure and is completely blind to single-qubit non-Clifford resources. Section 9 shows that noise breaks the tie — but
# only for channels that are not unitarily covariant.

# %%
# ==============================================================================
# STEP 3: analytic values at the identity angles (no optimisation yet)
# ==============================================================================
from math import comb

zero_angles = lambda N: jnp.zeros((N, 2))
sym_angles = lambda N: jnp.tile(jnp.array([jnp.pi / 2, 0.0]), (N, 1))     # theta = pi/2 on every site

print(f"{'N':>2s} | {'|+>^N':>28s} | {'GHZ':>28s}")
print(f"{'':>2s} | {'E':>10s} {'Q_B':>8s} {'Q_E':>8s} | {'E':>10s} {'Q_B':>8s} {'Q_E':>8s}")
for N in (3, 4, 5, 6, 8):
    E_p = float(correlator(product_state("+" * N), zero_angles(N)))
    E_g = float(correlator(ghz_state(N), zero_angles(N)))
    qb_p, qe_p = q_values(E_p, N); qb_g, qe_g = q_values(E_g, N)
    print(f"{N:2d} | {E_p:10.3e} {qb_p:8.4f} {qe_p:8.4f} | {E_g:10.3e} {qb_g:8.4f} {qe_g:8.4f}")
    assert abs(E_p - 4.0 ** (-N)) < 1e3 * TOL and abs(E_g - 0.25) < 1e3 * TOL

print("\nDicke states at theta = pi/2 on every site, against Eq. (15)")
print(f"{'N':>2s} {'m':>2s} {'binom':>6s} {'E measured':>12s} {'E Eq. (15)':>12s} {'Q_B':>8s} {'Q_E':>8s}")
for N in (3, 4, 5, 6, 7):
    for m in range(1, N // 2 + 1):
        E = float(correlator(dicke_state(N, m), sym_angles(N)))
        E_ana = comb(N, m) ** 2 * 4.0 ** (-N)
        qb, qe = q_values(E, N)
        print(f"{N:2d} {m:2d} {comb(N, m):6d} {E:12.6e} {E_ana:12.6e} {qb:8.4f} {qe:8.4f}")
        assert abs(E - E_ana) < 1e3 * TOL

# %% [markdown]
# Every entry matches the closed forms to machine precision at the *stated* angles. The W-state row ($m=1$) shows the
# behaviour predicted in Section 6.3: $Q_{\rm B}=+0.1699$ at $N=3$, exactly $0$ at $N=4$, and negative from $N=5$ on, while
# $Q_{\rm E}=\log_2N$ stays positive throughout. What these numbers do *not* yet prove is that the chosen angles are optimal —
# for that we need the optimiser.

# %% [markdown]
# ## 7. Optimising the measurement directions
#
# ### 7.1 The problem
#
# Equation (2) is a maximisation of a smooth, bounded, non-convex function of $2N$ real angles. It is smooth because
# `bell_expectation` is a composition of matrix products and array lookups; it is bounded by $1/4$; and it is non-convex
# because the state can have several inequivalent "best bases" — a fact we will see directly in the landscape plot.
#
# Three ingredients make this cheap in JAX:
#
# * `jax.value_and_grad` differentiates straight through the $N$ einsums. Reverse-mode AD costs a small constant factor over
#   one evaluation, **independent of the number of angles** — compare that with $4N$ evaluations for a central finite
#   difference.
# * `lax.scan` runs the optimiser loop inside one compiled program instead of unrolling $200$ copies of the graph.
# * `jax.vmap` runs many random restarts simultaneously, which is the practical answer to non-convexity.
#
# We use Adam (`adam_init`/`adam_update` from the engine, derived in
# [notebook 41](../ch11_variational_quantum_circuits/41_optimizers.ipynb)) and *maximise* by feeding it $-\nabla\mathcal E$.

# %%
# ==============================================================================
# STEP 4: gradient ascent on the angles, vmapped over random restarts
# ==============================================================================
@partial(jax.jit, static_argnums=(2, 3))
def optimise_pure(psi, angles0, n_steps=200, lr=0.1):
    """Maximise E(psi, angles) over the (N,2) angles, starting from every row of `angles0`.

    ALGORITHM   Adam ascent:  g = grad E ;  angles <- adam_update(angles, -g)     (minus = ascend)
    JAX   lax.scan over the steps (compiled once), vmap over the restarts (all in parallel).
          `angles0` has shape (R, N, 2); the return is (best E, best angles, history of shape (R, n_steps)).
    COST  R * n_steps * O(N 2^N) forward + the same order backward.
    """
    def run(a0):
        def step(carry, _):
            a, st = carry
            val, g = jax.value_and_grad(correlator, argnums=1)(psi, a)
            a, st = adam_update(a, -g, st, lr=lr)
            return (a, st), val
        (a, _), hist = lax.scan(step, (a0, adam_init(a0)), None, length=n_steps)
        return correlator(psi, a), a, hist

    vals, angs, hist = jax.vmap(run)(angles0)
    best = jnp.argmax(vals)
    return vals[best], angs[best], hist


def random_angles(key, n_restarts, N):
    """Uniform starting angles on [0, 2 pi)^(R x N x 2)."""
    return jax.random.uniform(key, (n_restarts, N, 2)) * 2 * jnp.pi


# --- GHZ: the optimiser must find the analytic maximum 1/4 --------------------
N_opt, R_opt = 6, 16
E_best, ang_best, hist = optimise_pure(ghz_state(N_opt), random_angles(jax.random.PRNGKey(1), R_opt, N_opt), 200, 0.1)
E_best = float(E_best); hist = np.asarray(hist)
qb, qe = q_values(E_best, N_opt)
print(f"GHZ, N = {N_opt}, {R_opt} restarts x 200 Adam steps")
print(f"  best E found = {E_best:.12f}   (analytic maximum 0.25)   difference {abs(E_best - 0.25):.2e}")
print(f"  Q_B = {qb:.6f}  (maximum N-2 = {N_opt-2}),   Q_E = {qe:.6f}  (maximum N-1 = {N_opt-1})")
final = hist[:, -1]
print(f"  restarts reaching E > 0.249: {int(np.sum(final > 0.249))}/{R_opt}; "
      f"worst restart ended at E = {final.min():.4f}")
assert abs(E_best - 0.25) < 1e-6

# %% [markdown]
# ### 7.2 Convergence and the value of restarts
#
# The history array holds $\mathcal E$ after every Adam step for every restart, so we can see how many restarts get stuck and
# where. We run two states with very different structure at $N=4$: the $T$-doped star graph, which is a GHZ state dressed
# with local unitaries, and a Haar-random state, which has no structure at all. For each we then scan the two angles of
# qubit $0$ with every other angle frozen at the optimum, which is a two-dimensional slice through a $2N$-dimensional
# landscape.

# %%
# ==============================================================================
# STEP 5: convergence of the restarts, and the landscape around the optimum
# ==============================================================================
# a state with a less trivial landscape: the T-doped star graph on 4 qubits
def star_graph_state(N):
    """Star graph state: |+>^N followed by CZ between qubit 0 (the hub) and every other qubit."""
    psi = product_state("+" * N)
    for q in range(1, N):
        psi = apply_gate(psi, CZ, [0, q])
    return psi


def apply_T_all(psi):
    """A T gate on every qubit -- a LOCAL unitary, so it cannot change the correlator (Eq. (16))."""
    for q in range(psi.ndim):
        psi = apply_gate(psi, T, [q])
    return psi


N_ls = 4
psi_ls = apply_T_all(star_graph_state(N_ls))                 # local-unitary image of a GHZ state
psi_hr = haar_state(jax.random.PRNGKey(17), N_ls)            # no special structure at all
a0_ls = random_angles(jax.random.PRNGKey(4), 16, N_ls)

E_ls, ang_ls, hist_ls = optimise_pure(psi_ls, a0_ls, 200, 0.1)
E_hr, ang_hr, hist_hr = optimise_pure(psi_hr, a0_ls, 200, 0.1)
hist_ls, hist_hr = np.asarray(hist_ls), np.asarray(hist_hr)
for nm, E, h in [("T-doped star graph", float(E_ls), hist_ls), ("Haar random", float(E_hr), hist_hr)]:
    print(f"{nm:>20s}: best E = {E:.8f}, Q_B = {q_values(E, N_ls)[0]:+.4f}; "
          f"final E of the 16 restarts in [{h[:, -1].min():.5f}, {h[:, -1].max():.5f}], "
          f"{int(np.sum(h[:, -1] > 0.999 * E))}/16 within 0.1% of the best")

# landscapes: scan the two angles of qubit 0 with all the other angles held at the optimum
n_grid = 61
th = np.linspace(-np.pi, np.pi, n_grid)
ph = np.linspace(0, 2 * np.pi, n_grid)
TH, PH = np.meshgrid(th, ph, indexing="ij")


def landscape(psi, ang):
    def one(tt, pp):
        return correlator(psi, ang.at[0, 0].set(tt).at[0, 1].set(pp))
    return np.asarray(jax.jit(jax.vmap(jax.vmap(one)))(jnp.asarray(TH), jnp.asarray(PH)))


land_ls, land_hr = landscape(psi_ls, ang_ls), landscape(psi_hr, ang_hr)


def wrap(x, period):
    """Map an angle into [-period/2, period/2) so it can be placed on the scanned grid."""
    return (float(x) + period / 2) % period - period / 2


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for r in range(hist_ls.shape[0]):
    axes[0].plot(hist_ls[r], lw=1.0, color=PALETTE[0], alpha=0.6)
    axes[0].plot(hist_hr[r], lw=1.0, color=PALETTE[1], alpha=0.6)
axes[0].plot([], [], color=PALETTE[0], label=r"$T$-doped star graph")
axes[0].plot([], [], color=PALETTE[1], label="Haar random")
axes[0].axhline(0.25, color="k", ls="--", lw=1, label=r"algebraic maximum $1/4$")
axes[0].axhline(2.0 ** (-N_ls), color="0.45", ls=":", lw=1.2, label=r"local-realism bound $2^{-N}$")
axes[0].set_xlabel("Adam step"); axes[0].set_ylabel(r"$\mathcal{E}$")
axes[0].set_title(rf"16 random restarts, $N={N_ls}$"); axes[0].legend(fontsize=7.5)

for ax, land, ang, ttl in [(axes[1], land_ls, ang_ls, r"$T$-doped star graph"),
                           (axes[2], land_hr, ang_hr, "Haar random")]:
    im = ax.pcolormesh(PH, TH, land, shading="auto", cmap="viridis")
    ax.plot([wrap(ang[0, 1], 2 * np.pi) % (2 * np.pi)], [wrap(ang[0, 0], 2 * np.pi)], "w*", ms=13)
    fig.colorbar(im, ax=ax, label=r"$\mathcal{E}$")
    ax.set_xlabel(r"$\varphi_0$"); ax.set_ylabel(r"$\theta_0$")
    ax.set_title(ttl + "\n(other angles at the optimum, star = optimum)")
fig.tight_layout(); plt.show()

# %% [markdown]
# ### 7.3 SPSA: the same maximisation without a gradient
#
# On real hardware the correlator is estimated from counts, the exact gradient is unavailable, and one uses a stochastic
# method. SPSA (Spall 1992, implemented as `spsa_grad` in the engine) perturbs *all* angles at once along a random $\pm1$
# direction $\Delta$ and estimates
#
# $$\widehat{\partial_k\mathcal E}=\frac{\mathcal E(\boldsymbol\theta+c\Delta)-\mathcal E(\boldsymbol\theta-c\Delta)}{2c}\,\Delta_k ,$$
#
# which costs **two** evaluations regardless of $2N$. It is a biased-but-consistent estimate (the bias is $O(c^2)$), extremely
# noisy per step, and works because Adam's momentum averages the noise. We run both methods from the same starting points and
# compare what they reach and what they cost.

# %%
# ==============================================================================
# STEP 6: gradient ascent vs SPSA from identical starting points
# ==============================================================================
@partial(jax.jit, static_argnums=(2, 3, 4))
def optimise_spsa(psi, angles0, n_steps=200, lr=0.1, c=0.05):
    """The same Adam ascent, but with the gradient replaced by an SPSA estimate (2 evaluations per step)."""
    f = lambda a: correlator(psi, a)

    def run(a0, key):
        def step(carry, k):
            a, st = carry
            g = spsa_grad(k, f, a, c=c)
            a, st = adam_update(a, -g, st, lr=lr)
            return (a, st), f(a)
        (a, _), hist = lax.scan(step, (a0, adam_init(a0)), jax.random.split(key, n_steps))
        return f(a), hist

    keys = jax.random.split(jax.random.PRNGKey(99), angles0.shape[0])
    vals, hist = jax.vmap(run)(angles0, keys)
    return vals, hist


states_cmp = {"GHZ": ghz_state(6), "W": w_state(6), "Dicke m=3": dicke_state(6, 3),
              "star graph": star_graph_state(6), "star + T": apply_T_all(star_graph_state(6))}
a0_cmp = random_angles(jax.random.PRNGKey(7), 16, 6)

# warm-up so that the timings below measure execution, not XLA compilation
optimise_pure(ghz_state(6), a0_cmp, 200, 0.1)[0].block_until_ready()
optimise_spsa(ghz_state(6), a0_cmp, 200, 0.1, 0.05)[0].block_until_ready()

print(f"{'state':>14s} {'E (grad+Adam)':>15s} {'E SPSA best':>13s} {'E SPSA med.':>13s} {'shortfall':>10s} "
      f"{'t grad [s]':>11s} {'t SPSA [s]':>11s}")
cmp_rows = []
for name, psi in states_cmp.items():
    t0 = time.time(); Eg = float(optimise_pure(psi, a0_cmp, 200, 0.1)[0].block_until_ready()); tg = time.time() - t0
    t0 = time.time(); vs, _ = optimise_spsa(psi, a0_cmp, 200, 0.1, 0.05)
    vs = np.asarray(jax.block_until_ready(vs)); tsp = time.time() - t0
    Es, Emed = float(vs.max()), float(np.median(vs))
    cmp_rows.append((name, Eg, Es, Emed))
    print(f"{name:>14s} {Eg:15.8f} {Es:13.8f} {Emed:13.8f} {(Eg-Es)/Eg:9.2%} "
          f"{tg:11.3f} {tsp:11.3f}")

# %% [markdown]
# ### 7.4 Reading the optimisation results
#
# For both states all $16$ restarts land on
# the same value to five decimals — $\mathcal E=0.25000000$ for the $T$-doped star graph and $\mathcal E=0.11847063$ for the
# Haar-random state — after about $50$ Adam steps, although they take visibly different routes and some spend $25$ steps on a
# plateau near $\mathcal E=0$ first. Restarts are cheap insurance here rather than a necessity; that changes as soon as noise
# is added (Section 9), where a second, competing family of optima appears and six random restarts all end up in the wrong
# one.
#
# The two landscape panels explain the difference between the states. For the $T$-doped star graph, $\mathcal E$ does not
# depend on $\varphi_0$ **at all**: the optimum is a horizontal line $\theta_0=0$, not a point. That is a property of GHZ-like
# states, and it follows from Eq. (12) — at $\theta_k=0$ the two array entries are
# $\psi'[0\ldots0]=\tfrac{1}{\sqrt2}\prod_ke^{i\varphi_k/2}$ and $\psi'[1\ldots1]=\tfrac{1}{\sqrt2}\prod_ke^{-i\varphi_k/2}$,
# so every $\varphi_k$ cancels in the modulus. The optimum is a continuous $N$-dimensional manifold, the Hessian is singular
# along it, and any gradient method slides freely down the ridge until it stops at an arbitrary point of it. The Haar-random
# state has no such symmetry: its slice has two genuine, isolated maxima (related by $(\theta_0,\varphi_0)\to
# (-\theta_0,\varphi_0+\pi)$, which is the same pair of measurement directions with the labels exchanged) separated by
# regions where $\mathcal E$ falls to zero.
#
# In the method comparison, gradient ascent reaches the analytic optimum for every state: $0.25000000$ for GHZ, the star
# graph and the $T$-doped star graph, $0.09765625=\binom{6}{3}^24^{-6}$ for the Dicke state, $0.00878906=6^2\cdot4^{-6}$ for
# W. SPSA reproduces GHZ to six digits but falls short by $2$–$3.4\,\%$ on all four remaining states, W included. The
# shortfall is not a property of a particular landscape — all of these states have a continuous manifold of optima, and the
# only state on which SPSA is exact is one of them. It is the signature of a stochastic gradient used with a *constant*
# step size: with two function evaluations per step and $2N=12$ parameters the estimate $\widehat{\nabla\mathcal E}$ is
# dominated by the component along the random direction $\Delta$, Adam's momentum averages it only partially, and the
# iterate ends up performing a random walk in a small ball around the optimum instead of settling in it. The median over
# the $16$ restarts lies a further $3$–$7\,\%$ below the best of them (and only $0.3\,\%$ below for GHZ), which measures the
# radius of that ball and shows how much of the apparent success of the "best restart" is just the luckiest sample of the
# same random walk. Exercise 4 asks
# you to shrink it with a decaying learning rate. In this simulator the exact gradient is
# essentially free: both methods take a few tens of milliseconds for $16$ restarts of $200$ steps (see the last two columns),
# so `jax.grad` wins on both counts.
# On hardware, where each evaluation costs thousands of shots and no gradient is available, the ranking reverses; that is the
# regime [notebook 41](../ch11_variational_quantum_circuits/41_optimizers.ipynb) analyses.


# %% [markdown]
# ## 8. The test states
#
# Now the full list: the product state, GHZ in the computational basis, GHZ written in randomly rotated local bases, the star
# graph state, the star graph with a $T$ gate on every qubit, the W state, the Dicke state at half filling, the linear cluster
# state and a Haar-random state. Every value is obtained by free optimisation over all $2N$ angles, so the analytic formulas
# of Section 6 are now genuine predictions and not just evaluations at a guessed point.

# %%
# ==============================================================================
# STEP 7: the zoo, optimised over all 2N angles
# ==============================================================================
def ghz_random_bases(key, N):
    """GHZ with an independent random single-qubit Clifford-like rotation (H, S H or 1) on every qubit.
    By Eq. (16) this MUST give the same correlator as the plain GHZ state."""
    psi = ghz_state(N)
    codes = jax.random.randint(key, (N,), 0, 3)
    for q in range(N):
        c = int(codes[q])
        if c == 1:
            psi = apply_gate(psi, H, [q])
        elif c == 2:
            psi = apply_gate(psi, H @ S, [q])
    return psi, "".join("ZXY"[int(c)] for c in codes)


def linear_cluster(N):
    """1D cluster state: |+>^N with CZ on every nearest-neighbour bond."""
    psi = product_state("+" * N)
    for q in range(N - 1):
        psi = apply_gate(psi, CZ, [q, q + 1])
    return psi


rows = []
for N in (4, 6):
    psi_gr, bases = ghz_random_bases(jax.random.PRNGKey(20 + N), N)
    zoo = {"product |+>^N": product_state("+" * N),
           "GHZ (Z basis)": ghz_state(N),
           f"GHZ (bases {bases})": psi_gr,
           "star graph": star_graph_state(N),
           "star graph + T": apply_T_all(star_graph_state(N)),
           "W": w_state(N),
           "Dicke m = N/2": dicke_state(N, N // 2),
           "linear cluster": linear_cluster(N),
           "Haar random": haar_state(jax.random.PRNGKey(5), N)}
    a0 = random_angles(jax.random.PRNGKey(31 + N), 24, N)
    print(f"\n=== N = {N} ===  (24 restarts x 250 Adam steps; bounds: Q_B <= {N-2}, Q_E <= {N-1})")
    print(f"{'state':>26s} {'E':>12s} {'Q_B':>9s} {'Q_E':>9s}  verdict")
    for name, psi in zoo.items():
        E = float(optimise_pure(psi, a0, 250, 0.1)[0])
        qb, qe = q_values(E, N)
        MARGIN = 1e-4          # require a strict violation, not a value sitting exactly on the bound
        verdict = ("Bell-correlated" if qb > MARGIN else ("entangled" if qe > MARGIN else "no certificate"))
        if qb > N - 4 + MARGIN:
            verdict += ", genuinely N-partite"
        rows.append((N, name, E, qb, qe))
        print(f"{name:>26s} {E:12.6e} {qb:+9.4f} {qe:+9.4f}  {verdict}")

# %% [markdown]
# ### 8.1 Two entries that need more than one size and more than one sample
#
# The two rows that do not follow from Sections 4 and 6 — the linear cluster state and the Haar-random state — are exactly
# the two for which a single number at a single $N$ proves nothing. The cluster state lands on $\mathcal E=2^{-N}$ at
# $N=4$ and $N=6$; whether that is a rule or a coincidence of two sizes can only be decided by scanning $N$. The
# Haar-random entry is one draw from a distribution; whether the sign of its $Q_{\rm B}$ is typical can only be decided by
# drawing more. We do both.

# %%
# ==============================================================================
# STEP 7b: the linear cluster state as a function of N, and the Haar ENSEMBLE
# ==============================================================================
@partial(jax.jit, static_argnums=(2, 3))
def optimise_many(psis, angles0, n_steps=200, lr=0.1):
    """Optimise a BATCH of pure states at once.  psis: (S,2,..,2), angles0: (S,R,N,2) -> best E per state.
    JAX: vmap over states (outer) and over restarts (inner); one compilation for the whole ensemble."""
    def one(psi, a_restarts):
        def run(a0):
            def step(carry, _):
                a, st = carry
                val, g = jax.value_and_grad(correlator, argnums=1)(psi, a)
                a, st = adam_update(a, -g, st, lr=lr)
                return (a, st), val
            (a, _), _ = lax.scan(step, (a0, adam_init(a0)), None, length=n_steps)
            return correlator(psi, a)
        return jnp.max(jax.vmap(run)(a_restarts))
    return jax.vmap(one)(psis, angles0)


print("linear cluster state, free optimisation over all 2N angles")
print(f"{'N':>2s} {'E':>13s} {'2^-N':>13s} {'Q_B':>8s} {'gamma = -log4(E)-1':>20s}")
for N in range(3, 9):
    E = float(optimise_pure(linear_cluster(N), random_angles(jax.random.PRNGKey(41), 24, N), 250, 0.1)[0])
    gam = -np.log(E) / np.log(4.0) - 1.0
    print(f"{N:2d} {E:13.6e} {2.0**(-N):13.6e} {q_values(E, N)[0]:+8.4f} {gam:20.4f}")

print("\nHaar-random states: 16 independent draws at each N (8 restarts x 200 Adam steps each)")
print(f"{'N':>2s} {'mean Q_B':>10s} {'std':>8s} {'median':>9s} {'min':>9s} {'max':>9s} {'fraction Q_B > 0':>18s}")
haar_stats = {}
for N in (4, 6, 8):
    psis = jnp.stack([haar_state(jax.random.PRNGKey(2000 + s), N) for s in range(16)])
    a0_h = random_angles(jax.random.PRNGKey(9), 16 * 8, N).reshape(16, 8, N, 2)
    qb_h = q_values(np.asarray(optimise_many(psis, a0_h, 200, 0.1)), N)[0]
    haar_stats[N] = qb_h
    print(f"{N:2d} {qb_h.mean():+10.4f} {qb_h.std():8.4f} {np.median(qb_h):+9.4f} {qb_h.min():+9.4f} "
          f"{qb_h.max():+9.4f} {np.mean(qb_h > 0):18.2f}")

# %% [markdown]
# Neither entry survives the extra evidence in the form in which the table presents it.
#
# The cluster state does **not** obey $\mathcal E=2^{-N}$ in general. Writing $\mathcal E=4^{-(1+\gamma)}$, the measured
# exponent is $\gamma=0,1,2,2,3,4$ for $N=3,\dots,8$, i.e. $\gamma=\lfloor 2(N-2)/3\rfloor$ over this range, so
# $Q_{\rm B}=N-2-2\lfloor2(N-2)/3\rfloor$ takes the values $+1,0,-1,0,-1,-2$. The three-qubit chain is a GHZ state in
# disguise and violates the bound; $N=4$ and $N=6$ are the only two sizes in the range that land exactly *on* it; from
# there $Q_{\rm B}$ drifts down by roughly one unit per three qubits. A one-dimensional cluster state is therefore a
# textbook example of a state that is maximally useful for measurement-based computation and useless for this correlator,
# and the agreement with $2^{-N}$ at $N=4,6$ is a coincidence of those two sizes, not a law. (No proof of the $\lfloor
# 2(N-2)/3\rfloor$ pattern is offered here; it is an observation over $3\le N\le8$.)
#
# The Haar-random entry is even more misleading as a single number. Over $16$ draws the *typical* $Q_{\rm B}$ **falls**
# with $N$: the mean is $+0.17$ at $N=4$, $-0.08$ at $N=6$ and $-0.83$ at $N=8$, and the fraction of draws that violate
# the local-realism bound goes $0.62\to0.38\to0.00$. The single seed quoted in the table above happens to be in the lower
# tail at $N=4$ and near the middle at $N=6$, which produces exactly the opposite impression — that random states become
# *more* nonlocal with $N$. They do not: as $N$ grows, a Haar-random state spreads its weight over all $2^N$ basis states
# and the single coherence $\langle0\cdots0\vert\rho\vert1\cdots1\rangle$ that this correlator measures shrinks faster
# than the bound it has to beat.
#
# > **Numerical practice.** One state at one size is an anecdote. Before writing "the correlator is positive for random
# > states", sample the ensemble and quote a distribution; before writing "$\mathcal E=2^{-N}$ for cluster states", scan
# > $N$. Both of these took a few seconds here and both changed the conclusion.

# %% [markdown]
# ### 8.2 What the table says
#
# Read it against Sections 4 and 6.
#
# * $\vert+\rangle^{\otimes N}$ lands on $Q_{\rm E}=0$ to six decimals: the optimiser cannot push a product state above the
#   separability bound, which is the strongest available check that the code and the bound are consistent.
# * GHZ in the computational basis, GHZ in randomly rotated bases, the star graph and the $T$-doped star graph all give the
#   **same** $\mathcal E=1/4$ and $Q_{\rm B}=N-2$. That is Eq. (16) at work: all four are related by local unitaries, and the
#   correlator cannot see the difference. In particular the $T$ gates, which make the state non-stabilizer and give it
#   non-zero magic, are invisible here.
# * All four exceed $N-4$, so each is certified genuinely $N$-partite entangled by the $m=2$ case of Eq. (10).
# * W and Dicke reproduce Eq. (15) exactly, so the symmetric single-angle ansatz of Section 6.3 *is* the global optimum: free
#   optimisation over all $2N$ angles finds nothing better.
# * The linear cluster state and the Haar-random state are the interesting cases. Both are strongly entangled — the cluster
#   state is the universal resource for measurement-based computation, the random state has volume-law entropy — yet the
#   correlator barely notices. The linear cluster state lands on $\mathcal E=2^{-N}$ at both of these sizes
#   ($0.0625=2^{-4}$ and $0.015625=2^{-6}$), i.e. $Q_{\rm B}=0.0000$: it sits precisely on the local-realism bound and
#   violates nothing — and Section 8.1 showed that this is an accident of $N=4,6$, with $Q_{\rm B}$ falling steadily at
#   larger $N$. The Haar-random draws give $Q_{\rm B}=-0.2376$ at $N=4$ and $Q_{\rm B}=+0.0024$ at $N=6$ — a violation
#   so marginal that it would be invisible in any experiment, and, as the ensemble of Section 8.1 shows, not
#   representative: the typical $Q_{\rm B}$ of a Haar state decreases with $N$. This correlator is built around a single $N$-body coherence
#   $\langle0\cdots0\vert\rho\vert1\cdots1\rangle$ and is therefore tuned to GHZ-like states; entanglement spread over many
#   patterns does not show up in it. A non-positive $Q_{\rm B}$ is **not** evidence of a separable state: Bell inequalities
#   are one-sided certificates.
#
# > **Physics insight.** A diagnostic that answers "yes" loudly for GHZ and "no" for a Haar-random state is not a defect. It
# > is the statement that *Bell correlations of this type* are a specific structure, not a synonym for "lots of
# > entanglement". Notebook 25 measures a quantity (the negativity) that ranks the random state far above GHZ; the two
# > diagnostics see genuinely different things.

# %% [markdown]
# ## 9. Noise on the density tensor
#
# ### 9.1 Closed forms at the identity angles
#
# Apply a single-qubit channel to every qubit of the GHZ state. At the identity angles, Eq. (13) says that
# $\langle\mathcal B\rangle$ is simply the corner element $\rho_{0\cdots0,\,1\cdots1}$ — the $N$-body coherence whose decay
# [notebook 22](../ch08_quantum_information_protocols/22_ghz_states_and_decoherence.ipynb) studies through parity
# oscillations. Each channel multiplies it by a fixed factor per qubit:
#
# * **dephasing** with probability $p$ maps $\rho_{01}\to(1-2p)\rho_{01}$ on each qubit, so the corner element picks up
#   $(1-2p)^N$ and
#
#   $$\mathcal E_{\rm deph}(p)=\tfrac14(1-2p)^{2N} ;$$
#
# * **depolarising** with probability $p$ shrinks the Bloch vector by $\eta=1-\tfrac43p$, hence also every off-diagonal
#   single-qubit matrix element, and
#
#   $$\mathcal E_{\rm depol}(p)=\tfrac14\,\eta^{2N},\qquad\eta=1-\tfrac43p ;$$
#
# * **amplitude damping** with probability $\gamma$ has $K_0=\mathrm{diag}(1,\sqrt{1-\gamma})$ and
#   $K_1=\sqrt\gamma\vert0\rangle\langle1\vert$. In $\langle0^N\vert K_m\rho K_m^{\dagger}\vert1^N\rangle$ only the branch
#   $m=0$ survives on every qubit ($K_1^{\dagger}\vert1\rangle=0$), contributing $1\cdot\sqrt{1-\gamma}$ per site, so
#
#   $$\mathcal E_{\rm AD}(\gamma)=\tfrac14(1-\gamma)^{N} .$$
#
# These are values at *fixed* angles, hence lower bounds on $\mathcal E$. Whether they are the maximum is a question for the
# optimiser, and Section 9.3 shows that they stop being the maximum once the noise is strong.
#
# ### 9.2 Critical noise
#
# Setting $Q_{\rm B}=0$ (i.e. $\mathcal E=2^{-N}$) and $Q_{\rm E}=0$ (i.e. $\mathcal E=4^{-N}$) in the dephasing formula gives
#
# $$1-2p_{\rm B}=2^{\frac{2-N}{2N}},\qquad 1-2p_{\rm E}=2^{\frac{1}{N}-1} , \tag{17}$$
#
# and the same two equations with $1-2p$ replaced by $\eta=1-\tfrac43p$ for depolarising, or by $(1-\gamma)^{1/2}$ for
# amplitude damping. Both exponents have finite limits: $\tfrac{2-N}{2N}=\tfrac1N-\tfrac12\to-\tfrac12$ and
# $\tfrac1N-1\to-1$, so
#
# $$p_{\rm B}\longrightarrow\tfrac12\bigl(1-2^{-1/2}\bigr)=0.1464,\qquad
#   p_{\rm E}\longrightarrow\tfrac12\bigl(1-2^{-1}\bigr)=0.25 .$$
#
# Two facts follow. First, the Bell threshold is always *below* the entanglement threshold — at every $N$ there is a window of noise in which
# the state is still certified entangled but no longer certified nonlocal. Second, the per-qubit noise tolerance does **not**
# shrink with $N$: it *grows*, from $p_{\rm B}=0.0546$ at $N=3$ towards the asymptote $0.1464$. The reason is that the
# local-realism bound $2^{-N}$ shrinks exponentially too, and for this particular correlator the two exponentials nearly
# cancel. What does get harder with $N$ is the *total* amount of noise $Np$ that the experiment must keep under control, and
# the preparation of the state itself.
#
# One limitation of Eq. (17) has to be stated now, because it is invisible in the algebra: the equation is solved for the
# *identity-angle* value of $\mathcal E$, so it locates the crossing of the closed form, not of the maximised correlator.
# Section 9.3 shows that for dephasing the two part company at $p\approx0.19$. The Bell thresholds $p_{\rm B}$ all lie
# below that crossover and are therefore genuine. The entanglement thresholds $p_{\rm E}$ do not: for even $N$ the
# maximised correlator under dephasing approaches $4^{-N}$ from *above* and never crosses it, so $Q_{\rm E}$ stays
# non-negative at every $p$ and the entanglement certificate is not lost at all. Read the $p_{\rm E}$ column as "the
# noise at which the *coherence branch* falls to the separability bound", not as the death of the certificate.

# %%
# ==============================================================================
# STEP 8: the three channels on the GHZ state, at the identity angles, N = 6
# ==============================================================================
def noisy_ghz_dm(N, kraus):
    """rho = channel^{(x)N} ( |GHZ><GHZ| ) as a density tensor: one Kraus channel per qubit."""
    rho = to_dm(ghz_state(N))
    for q in range(N):
        rho = apply_kraus_dm(rho, kraus, [q])
    return rho


N_dm = 6
print(f"N = {N_dm}, identity angles; 'analytic' from Section 9.1")
print(f"{'p':>5s} | {'dephasing':>23s} | {'depolarising':>23s} | {'amp. damping':>23s}")
print(f"{'':>5s} | {'E':>11s} {'analytic':>11s} | {'E':>11s} {'analytic':>11s} | {'E':>11s} {'analytic':>11s}")
worst = 0.0
for p in (0.0, 0.05, 0.10, 0.15, 0.20):
    e_de = float(correlator_dm(noisy_ghz_dm(N_dm, kraus_dephasing(p)), zero_angles(N_dm)))
    e_dp = float(correlator_dm(noisy_ghz_dm(N_dm, kraus_depolarizing(p)), zero_angles(N_dm)))
    e_ad = float(correlator_dm(noisy_ghz_dm(N_dm, kraus_amplitude_damping(p)), zero_angles(N_dm)))
    a_de = 0.25 * (1 - 2 * p) ** (2 * N_dm)
    a_dp = 0.25 * (1 - 4 * p / 3) ** (2 * N_dm)
    a_ad = 0.25 * (1 - p) ** N_dm
    worst = max(worst, abs(e_de - a_de), abs(e_dp - a_dp), abs(e_ad - a_ad))
    print(f"{p:5.2f} | {e_de:11.4e} {a_de:11.4e} | {e_dp:11.4e} {a_dp:11.4e} | {e_ad:11.4e} {a_ad:11.4e}")
print(f"\nlargest deviation from the closed forms: {worst:.2e}")
assert worst < 1e3 * TOL

# %% [markdown]
# ### 9.3 Full optimisation under noise
#
# Now let the optimiser choose the angles at every noise strength. Two things can happen: it confirms the closed form, or it
# finds something better. A density tensor is $4^N$ complex numbers and every one of the thousands of optimiser evaluations
# applies $2N$ gates to all of them, so the cost grows as $4^N$ and we run the full sweep at $N=4$, batching the whole noise
# grid and all restarts into a single `vmap` (the closed forms themselves were already verified at $N=6$ in Step 8).
#
# Two of the restarts are not random. Whenever an achievable value of the objective is known in closed form, the angles
# that achieve it belong in the starting batch: here that is the identity angles of Section 9.1 and the symmetric choice
# $\theta_k=\pi/2$ of Section 6.3. With those two included the returned maximum can never be *smaller* than the closed
# form, which turns the comparison "optimised versus analytic" into a one-sided test with a known sign — and lets us
# `assert` it. With six purely random restarts and $150$ steps the sweep does fail that test near $p=0.15$: every restart
# is then attracted to the $\theta=\pi/2$ branch discussed below, which at that noise strength is a *local* maximum lying
# a factor of $2.4$ below the global one.

# %%
# ==============================================================================
# STEP 9: optimised correlator vs noise, density tensor
# ==============================================================================
@partial(jax.jit, static_argnums=(2,))
def optimise_dm_batch(rhos, angles0, n_steps=150, lr=0.1):
    """Optimise the angles for a BATCH of density tensors at once.
    rhos: (P, 2,...,2) of rank 1+2N ; angles0: (P, R, N, 2).  Returns the best E for each of the P states.
    JAX: vmap over the noise grid (outer) and over the restarts (inner); one compilation for the whole sweep."""
    def one(rho, a_restarts):
        def run(a0):
            def step(carry, _):
                a, st = carry
                val, g = jax.value_and_grad(correlator_dm, argnums=1)(rho, a)
                a, st = adam_update(a, -g, st, lr=lr)
                return (a, st), val
            (a, _), _ = lax.scan(step, (a0, adam_init(a0)), None, length=n_steps)
            return correlator_dm(rho, a)
        return jnp.max(jax.vmap(run)(a_restarts))
    return jax.vmap(one)(rhos, angles0)


N_ns = 4                      # small enough that the whole sweep fits in one batched optimisation
p_grid = np.linspace(0.0, 0.45, 10)
channels = {"dephasing": (kraus_dephasing, lambda p: 0.25 * (1 - 2 * p) ** (2 * N_ns)),
            "depolarising": (kraus_depolarizing, lambda p: 0.25 * (1 - 4 * p / 3) ** (2 * N_ns)),
            "amplitude damping": (kraus_amplitude_damping, lambda p: 0.25 * (1 - p) ** N_ns)}

def seeded_restarts(key, n_random, N, n_points):
    """Restart batch of shape (n_points, 2 + n_random, N, 2): the two ANALYTIC candidates of Section 6
    (all angles zero, and theta = pi/2 everywhere) followed by `n_random` uniform starts.
    Seeding the batch with angles whose value we already know guarantees that the optimiser can never
    return less than the closed form -- see the checkpoint below."""
    fixed = jnp.stack([zero_angles(N), sym_angles(N)])                       # (2, N, 2)
    fixed = jnp.tile(fixed[None], (n_points, 1, 1, 1))                       # (P, 2, N, 2)
    rnd = random_angles(key, n_random * n_points, N).reshape(n_points, n_random, N, 2)
    return jnp.concatenate([fixed, rnd], axis=1)


t0 = time.time()
results_ns = {}
for ci, (cname, (kr, ana)) in enumerate(channels.items()):
    rhos = jnp.stack([noisy_ghz_dm(N_ns, kr(float(p))) for p in p_grid])
    a0 = seeded_restarts(jax.random.PRNGKey(300 + ci), 6, N_ns, len(p_grid))
    E_opt = np.asarray(optimise_dm_batch(rhos, a0, 150, 0.1))
    E_id = np.array([ana(p) for p in p_grid])
    # CHECKPOINT: a maximum can never be smaller than a value we can write down
    assert np.all(E_opt >= E_id - 1e3 * TOL), f"{cname}: optimiser fell below the closed form"
    results_ns[cname] = (E_opt, E_id)
print(f"three noise sweeps, N = {N_ns}, 10 points x (2 seeded + 6 random) restarts x 150 steps: "
      f"{time.time()-t0:.1f} s\n")

print(f"{'p':>5s} | " + " | ".join(f"{c:>27s}" for c in channels))
print(f"{'':>5s} | " + " | ".join(f"{'E optimised':>13s} {'E identity':>13s}" for _ in channels))
for i, p in enumerate(p_grid):
    line = f"{p:5.2f} | "
    line += " | ".join(f"{results_ns[c][0][i]:13.5e} {results_ns[c][1][i]:13.5e}" for c in channels)
    print(line)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for j, c in enumerate(channels):
    E_opt, E_id = results_ns[c]
    axes[0].semilogy(p_grid, E_opt, MARKERS[j] + "-", ms=5, color=PALETTE[j], label=c + " (optimised)")
    axes[0].semilogy(p_grid, E_id, "--", lw=1, color=PALETTE[j])
axes[0].axhline(2.0 ** (-N_ns), color="k", ls=":", label=r"local-realism bound $2^{-N}$")
axes[0].axhline(4.0 ** (-N_ns), color="0.5", ls="-.", label=r"separability bound $4^{-N}$")
axes[0].set_xlabel(r"noise strength $p$"); axes[0].set_ylabel(r"$\mathcal{E}$")
axes[0].set_title(rf"GHZ under local noise, $N={N_ns}$"); axes[0].legend(fontsize=7.5)

for j, c in enumerate(channels):
    qb, qe = q_values(results_ns[c][0], N_ns)
    axes[1].plot(p_grid, qb, MARKERS[j] + "-", ms=5, color=PALETTE[j], label=c)
    axes[1].plot(p_grid, qe, MARKERS[j] + ":", ms=4, color=PALETTE[j], alpha=0.6)
axes[1].axhline(0, color="k", lw=0.9)
axes[1].set_xlabel(r"noise strength $p$"); axes[1].set_ylabel(r"$Q_{\rm B}$ (solid), $Q_{\rm E}$ (dotted)")
axes[1].set_title("the two certificates die at different noise levels"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# For **depolarising** noise and **amplitude damping** the optimiser reproduces the closed form of Section 9.1 at every point
# of the grid, to all printed digits: the identity angles stay optimal however strong the noise. For **dephasing** they do
# not. From $p=0.20$ onward the optimised value overtakes the closed form and settles on a plateau just above
# $\mathcal E=3.907\cdot10^{-3}$, while the coherence formula keeps falling and reaches $2.5\cdot10^{-9}$ at $p=0.45$.
#
# The second branch can be written down exactly. Local dephasing leaves the GHZ state in the form
#
# $$\rho(p)=\tfrac12\bigl(\vert0^N\rangle\langle0^N\vert+\vert1^N\rangle\langle1^N\vert\bigr)
#   +\tfrac{c^{N}}{2}\bigl(\vert0^N\rangle\langle1^N\vert+\vert1^N\rangle\langle0^N\vert\bigr),\qquad c=1-2p ,$$
#
# and evaluating Eq. (13) on it with $U_k=R_z(\varphi_k)R_y(\theta_k)$, writing $c_k=\cos\tfrac{\theta_k}{2}$ and
# $s_k=\sin\tfrac{\theta_k}{2}$, gives
#
# $$\rho'[0\ldots0;1\ldots1]=\tfrac{1+(-1)^N}{2}\prod_k c_ks_k
#   +\tfrac{c^{N}}{2}\Bigl[\prod_kc_k^2\,e^{i\Phi}+(-1)^N\prod_ks_k^2\,e^{-i\Phi}\Bigr],\qquad\Phi=\sum_k\varphi_k .$$
#
# The first term is the classical mixture $\tfrac12(\vert0^N\rangle\langle0^N\vert+\vert1^N\rangle\langle1^N\vert)$ that
# survives at $p=\tfrac12$; the second is the surviving coherence. For **even** $N$, $\theta_k=\pi/2$ and $\Phi=0$ make
# both terms real and positive, and
#
# $$\mathcal E_{\theta=\pi/2}(p)=4^{-N}\bigl(1+c^{N}\bigr)^2 .$$
#
# At $p=\tfrac12$ this is exactly $4^{-N}$ — the separability bound, saturated but never exceeded by the (separable)
# classical mixture, as Eq. (10) requires; at smaller $p$ it sits slightly above it. (For odd $N$ the first term cancels
# and the classical mixture gives $\mathcal E=0$; Exercise 7 asks you to check this.) So for even $N$ the optimum is the
# larger of two branches,
#
# $$\mathcal E_{\rm opt}(p)=\max\Bigl[\tfrac14c^{2N},\;4^{-N}(1+c^{N})^2\Bigr] ,$$
#
# which reproduces the measured column to all printed digits. The two branches cross where
# $c^{N}=1/(2^{N-1}-1)$, i.e. at $p=0.1926$ for $N=4$ (and $0.2179$ at $N=6$, $0.2271$ at $N=8$): *that* is where the
# optimal measurement directions jump from $\theta=0$ to $\theta=\pi/2$, not at $p=0.15$ where the two curves in the
# figure already look close. Depolarising and amplitude damping have no such second branch because they drive the state
# towards $\mathbb 1/2^N$ and $\vert0\cdots0\rangle$ respectively, and both of those have $\mathcal E=0$.
#
# > **Common pitfall.** A closed form derived at one fixed choice of settings is a *lower* bound on a maximised quantity,
# > never the answer — and a numerical maximum is a lower bound too. The two failure modes point in opposite directions
# > and you need both guards: let the optimiser check the formula (here it is exact over the whole range for two of the
# > three channels and wrong by five orders of magnitude for the third), and let the formula check the optimiser (put the
# > known angles into the restart batch and `assert` that the returned maximum is at least the closed form).

# %% [markdown]
# ### 9.4 Critical noise as a function of $N$
#
# Solving Eq. (17) gives the noise strength at which each certificate is lost. We compare those closed forms with the noise
# grid just measured, and then evaluate them for larger $N$ where the density tensor would be too expensive.

# %%
# ==============================================================================
# STEP 10: critical noise from Eq. (17), against the measured curves
# ==============================================================================
def p_crit_dephasing(N, which):
    """p at which Q_B (which='B') or Q_E (which='E') crosses zero for a locally dephased GHZ state."""
    expo = (2 - N) / (2 * N) if which == "B" else 1.0 / N - 1.0
    return 0.5 * (1 - 2.0 ** expo)


def p_crit_depolarising(N, which):
    expo = (2 - N) / (2 * N) if which == "B" else 1.0 / N - 1.0
    return 0.75 * (1 - 2.0 ** expo)


def g_crit_damping(N, which):
    expo = (2 - N) / N if which == "B" else 2.0 / N - 2.0
    return 1 - 2.0 ** expo


print(f"{'N':>3s} | {'dephasing':>17s} | {'depolarising':>17s} | {'amp. damping':>17s}")
print(f"{'':>3s} | {'p_B':>8s} {'p_E':>8s} | {'p_B':>8s} {'p_E':>8s} | {'g_B':>8s} {'g_E':>8s}")
Ns_crit = [3, 4, 6, 8, 10, 14, 20]
crit = {}
for N in Ns_crit:
    row = (p_crit_dephasing(N, "B"), p_crit_dephasing(N, "E"),
           p_crit_depolarising(N, "B"), p_crit_depolarising(N, "E"),
           g_crit_damping(N, "B"), g_crit_damping(N, "E"))
    crit[N] = row
    print(f"{N:3d} | {row[0]:8.4f} {row[1]:8.4f} | {row[2]:8.4f} {row[3]:8.4f} | {row[4]:8.4f} {row[5]:8.4f}")

# cross-check against the measured N = 4 sweep: interpolate where Q_B and Q_E cross zero
print(f"\ncheck at N = {N_ns} against the measured (optimised) sweep of Step 9:")
for cname, key_fn in [("dephasing", p_crit_dephasing), ("depolarising", p_crit_depolarising),
                      ("amplitude damping", g_crit_damping)]:
    qb, qe = q_values(results_ns[cname][0], N_ns)
    p_meas_B = np.interp(0.0, qb[::-1], p_grid[::-1])
    print(f"  {cname:>18s}: measured p(Q_B=0) = {p_meas_B:.4f},  Eq. (17) = {key_fn(N_ns, 'B'):.4f}")

fig, ax = plt.subplots(figsize=(6.2, 4.3))
Ns_plot = np.arange(3, 21)
for j, (lbl, fn) in enumerate([("dephasing", p_crit_dephasing), ("depolarising", p_crit_depolarising),
                               ("amplitude damping", g_crit_damping)]):
    ax.plot(Ns_plot, [fn(int(n), "B") for n in Ns_plot], MARKERS[j] + "-", ms=4, color=PALETTE[j],
            label=lbl + r": $Q_{\rm B}=0$")
    ax.plot(Ns_plot, [fn(int(n), "E") for n in Ns_plot], MARKERS[j] + "--", ms=4, color=PALETTE[j], alpha=0.55,
            label=lbl + r": $Q_{\rm E}=0$")
ax.set_xlabel("number of qubits $N$"); ax.set_ylabel("critical noise strength")
ax.set_title("Bell certificates die long before entanglement certificates")
ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout(); plt.show()

# %% [markdown]
# The closed forms of Eq. (17) agree with the measured sweep at $N=4$ to better than $1\,\%$ in every channel: the crossings
# interpolated from the optimised data are $p=0.0788$, $0.1189$ and $0.2927$ for dephasing, depolarising and amplitude
# damping, against $0.0796$, $0.1193$ and $0.2929$ from Eq. (17). (The measured values are slightly *lower* because the
# interpolation runs through a grid of spacing $0.05$.)
#
# The table and the figure then say three things. First, for every channel and every $N$ the Bell threshold is well below the
# entanglement threshold — at $N=6$, dephasing destroys the Bell certificate at $p=0.103$ while the coherence branch reaches
# the separability bound only at $p=0.219$, a factor of $2.1$ (and, by the caveat of Section 9.2, for even $N$ under
# dephasing the entanglement certificate is in fact never lost, because the second branch keeps $\mathcal E$ at or above
# $4^{-N}$; the dotted $Q_{\rm E}$ curves of the previous figure flatten onto zero instead of crossing it). Second, the
# thresholds **grow** with $N$ rather than shrinking, from $p_{\rm B}=0.055$
# at $N=3$ to $0.134$ at $N=20$, approaching the asymptote $0.1464$ derived in Section 9.2: per qubit, a larger GHZ state
# tolerates *more* dephasing, because the local-realism bound it has to beat shrinks exponentially too. Third, the ordering of
# the channels is fixed: dephasing is the most damaging at equal $p$, then depolarising (which is a dephasing plus two more
# error types, but with each occurring at rate $p/3$), then amplitude damping, which needs $\gamma=0.37$ at $N=6$ because it
# costs the coherence only $\sqrt{1-\gamma}$ per qubit instead of $1-2p$.
#
# > **Numerical practice.** Neither the table nor the figure needed a density tensor beyond $N=4$: the closed forms do the
# > extrapolation and the simulation validates them where it can. Doing it the other way round — simulating $N=20$ to read off
# > a number that a two-line calculation gives exactly — is the most common way to waste a week of computer time.

# %% [markdown]
# ## 10. Quantum trajectories and the bias of the naive estimator
#
# ### 10.1 Why trajectories
#
# The density tensor costs $O(4^N)$ in memory and $O(N4^N)$ per evaluation of the correlator, which is what confined Section 9
# to $N\le6$. The Monte-Carlo wave-function unravelling of
# [notebook 17](../ch06_open_quantum_systems/17_monte_carlo_wave_function.ipynb) replaces it by $M$ pure states of $2^N$
# amplitudes each: for every trajectory one Kraus branch is sampled per qubit with probability $\lVert K_m\psi\rVert^2$, and
#
# $$\rho=\mathbb E\bigl[\vert\psi_i\rangle\langle\psi_i\vert\bigr]\qquad\Longrightarrow\qquad
#   \mathrm{Tr}(\rho\,\mathcal B)=\mathbb E\bigl[z_i\bigr],\qquad z_i=\langle\psi_i\vert\mathcal B\vert\psi_i\rangle . \tag{18}$$
#
# The trajectory average estimates $\langle\mathcal B\rangle$, which is **linear** in $\rho$, without bias. The correlator we
# want, however, is $\mathcal E=\vert\langle\mathcal B\rangle\vert^2$, which is **not** linear — and that is where the trouble
# starts.
#
# ### 10.2 The bias, and its exact correction
#
# Let $z_1,\dots,z_M$ be independent complex random variables with mean $\mu=\mathrm{Tr}(\rho\mathcal B)$ and
# $\sigma^2=\mathbb E\vert z-\mu\vert^2$, and let $\bar z=\tfrac1M\sum_iz_i$. Write $\bar z=\mu+\delta$ with
# $\mathbb E\delta=0$ and $\mathbb E\vert\delta\vert^2=\sigma^2/M$. Then
#
# $$\mathbb E\vert\bar z\vert^2=\mathbb E\bigl[(\mu+\delta)\overline{(\mu+\delta)}\bigr]
#  =\vert\mu\vert^2+2\,\mathrm{Re}\bigl(\bar\mu\,\mathbb E\delta\bigr)+\mathbb E\vert\delta\vert^2
#  =\vert\mu\vert^2+\frac{\sigma^2}{M} . \tag{19}$$
#
# So the naive estimator $\widehat{\mathcal E}_{\rm naive}=\vert\bar z\vert^2$ is biased **upward** by $\sigma^2/M$. The bias
# is positive, it decays only as $1/M$, and it is largest exactly where it does most damage: when $\vert\mu\vert^2$ is small,
# i.e. in the strong-noise region where one is trying to decide whether a bound is still violated. A naive analysis would
# report a Bell violation that is not there.
#
# The cure is exact. The sample variance
#
# $$S^2=\frac{1}{M-1}\sum_{i=1}^{M}\vert z_i-\bar z\vert^2$$
#
# is an unbiased estimator of $\sigma^2$ (the usual argument: $\sum_i\vert z_i-\bar z\vert^2=\sum_i\vert z_i-\mu\vert^2
# -M\vert\bar z-\mu\vert^2$, whose expectation is $M\sigma^2-\sigma^2$). Subtracting Eq. (19)'s bias term,
#
# $$\boxed{\ \widehat{\mathcal E}=\vert\bar z\vert^2-\frac{S^2}{M}\ } \tag{20}$$
#
# is an **exactly unbiased** estimator of $\mathcal E$ for every $M>1$. It can come out slightly negative when the true
# $\mathcal E$ is near zero — that is the price of unbiasedness, and it is honest: clipping it at zero would reintroduce a
# positive bias.
#
# Error bars come from the **bootstrap** (Efron 1979): resample the $M$ trajectory values with replacement, recompute
# Eq. (20) on each resample, and take the standard deviation of the resulting distribution. The bootstrap needs no formula
# for the variance of a nonlinear statistic, which is precisely why it is the right tool here.

# %%
# ==============================================================================
# STEP 11: trajectories, the biased estimator, and Eq. (20)
# ==============================================================================
def traj_values(key, psi0, kraus, angles, n_traj):
    """z_i = <psi_i| B |psi_i> for n_traj Monte-Carlo wave-function trajectories.

    MATH   each trajectory applies one sampled Kraus branch per qubit (`apply_kraus_mcwf`), so
           E[|psi_i><psi_i|] = (channel)^{(x)N}(|psi0><psi0|)   exactly.
    JAX    one trajectory is a pure function of its key -> `vmap` over n_traj keys runs them all at once.
    COST   O(n_traj N 2^N) time and O(n_traj 2^N) memory for the batch.
    """
    N = psi0.ndim

    def one(k):
        ks = jax.random.split(k, N)
        psi = psi0
        for q in range(N):
            psi = apply_kraus_mcwf(ks[q], psi, kraus, [q])
        return bell_expectation(psi, angles)

    return jax.vmap(one)(jax.random.split(key, n_traj))


def estimators(z):
    """(naive, unbiased, sample variance) from a 1D complex array of trajectory values -- Eqs. (19)-(20)."""
    M = z.shape[0]
    zbar = jnp.mean(z)
    S2 = jnp.sum(jnp.abs(z - zbar) ** 2) / (M - 1)
    naive = jnp.abs(zbar) ** 2
    return naive, naive - S2 / M, S2


# --- the bias, measured: many independent repetitions at several M -----------
N_b, p_b = 6, 0.08
psi0_b = ghz_state(N_b)
rho_b = noisy_ghz_dm(N_b, kraus_dephasing(p_b))
E_exact_b = float(correlator_dm(rho_b, zero_angles(N_b)))

R_rep, M_max = 200, 512
t0 = time.time()
z_rep = np.asarray(jax.jit(jax.vmap(lambda k: traj_values(k, psi0_b, kraus_dephasing(p_b), zero_angles(N_b), M_max)))(
    jax.random.split(jax.random.PRNGKey(2), R_rep)))   # NOTE: not named Z -- that is the Pauli matrix
print(f"{R_rep} x {M_max} = {R_rep*M_max} trajectories at N = {N_b} in {time.time()-t0:.1f} s")

Ms = [8, 16, 32, 64, 128, 256, 512]
print(f"\nexact E = {E_exact_b:.6e}   (dephasing p = {p_b}, N = {N_b}, identity angles)")
print(f"{'M':>5s} {'<naive>':>12s} {'bias':>11s} {'sigma^2/M':>11s} {'<corrected>':>13s} {'residual':>11s} "
      f"{'SE of mean':>11s}")
bias_meas, bias_pred, corr_meas, sem_list = [], [], [], []
for M in Ms:
    zb = z_rep[:, :M].mean(axis=1)
    S2 = np.sum(np.abs(z_rep[:, :M] - zb[:, None]) ** 2, axis=1) / (M - 1)
    naive = np.abs(zb) ** 2
    corrected = naive - S2 / M
    sem = naive.std(ddof=1) / np.sqrt(R_rep)          # error of the AVERAGE over the R repetitions
    bias_meas.append(naive.mean() - E_exact_b)
    bias_pred.append(S2.mean() / M)
    corr_meas.append(corrected.mean())
    sem_list.append(sem)
    print(f"{M:5d} {naive.mean():12.5e} {naive.mean()-E_exact_b:+11.3e} {S2.mean()/M:11.3e} "
          f"{corrected.mean():13.5e} {corrected.mean()-E_exact_b:+11.3e} {sem:11.3e}")

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.loglog(Ms, np.abs(bias_meas), "o-", label=r"measured bias of $\vert\bar z\vert^2$")
ax.loglog(Ms, bias_pred, "s--", label=r"predicted $\sigma^2/M$, Eq. (19)")
ax.loglog(Ms, np.abs(np.array(corr_meas) - E_exact_b), "^-", label="residual of Eq. (20)")
ax.loglog(Ms, sem_list, "k:", lw=1.2, label=rf"standard error of the mean over {R_rep} repetitions")
ax.set_xlabel("trajectories per estimate $M$"); ax.set_ylabel("deviation from the exact value")
ax.set_title(rf"bias of the naive estimator, averaged over {R_rep} repetitions")
ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The naive estimator is badly biased at small $M$: at $M=8$ it returns $5.63\cdot10^{-2}$ against the exact
# $3.09\cdot10^{-2}$ — **an 82 % overestimate** — and the predicted bias $2.77\cdot10^{-2}$ accounts for essentially all of
# it. The bias then falls roughly as $1/M$, to $6.6\cdot10^{-3}$ at $M=32$ and $1.1\cdot10^{-4}$ at $M=256$.
#
# "Roughly" has to be taken literally. The measured bias is itself an average over only $R=200$ repetitions, and the last
# column of the table gives its standard error; at $M=16$ the measured $8.6\cdot10^{-3}$ sits $2.0$ standard errors below
# the predicted $1.4\cdot10^{-2}$ and at $M=64$ the measured $1.3\cdot10^{-3}$ sits $1.5$ standard errors below the
# predicted $3.5\cdot10^{-3}$, which is why the blue and orange curves in the figure separate by up to a factor of $2.6$
# without either of them being wrong. Resolving the $1/M$ law point by point at this $M$ would need a few thousand
# repetitions, not two hundred.
#
# The residual of Eq. (20) — the last-but-one column, equal by construction to (measured bias) $-$ (predicted bias) — runs
# from $-3\cdot10^{-4}$ to $-5.4\cdot10^{-3}$ and stays within $2.0$ standard errors of zero at every $M$. It does not
# shrink with $M$, and all seven entries have the same sign, because the seven columns of the table are nested subsets of
# the same $200\times512$ trajectory pool: they are seven views of one common fluctuation, not seven independent tests.
# The honest reading of the figure is that the residual of Eq. (20) is at the noise floor of this experiment everywhere,
# while the bias of the naive estimator is an order of magnitude above that floor at small $M$.
#
# > **Numerical practice.** Any nonlinear function of a Monte-Carlo average is biased at order $1/M$, with a coefficient set
# > by the curvature of the function and the variance of the sample. Here the function is $\vert\cdot\vert^2$, the curvature
# > is positive, and the bias is positive — which is the dangerous direction for a certificate. Whenever you take a modulus,
# > a square, a logarithm or a ratio of Monte-Carlo estimates, ask for the $1/M$ term before you quote the number.

# %% [markdown]
# ### 10.3 A noise sweep with trajectories and bootstrap error bars
#
# With the estimator fixed we can go where the density tensor cannot. At $N=8$ the density tensor is already $65\,536$ complex
# numbers and every optimiser evaluation touches all of them; a trajectory ensemble of $M$ states costs $M\cdot256$ numbers.
# We fix the measurement angles at the GHZ optimum (Section 6.2 proved it is optimal for the noiseless state, and Section 9.1
# showed that dephasing does not move it while the coherence branch dominates), sample trajectories, and compare the corrected
# estimator with the exact density-tensor value.

# %%
# ==============================================================================
# STEP 12: trajectory estimate of E(p) with bootstrap error bars, N = 8
# ==============================================================================
def bootstrap_error(key, z, n_boot=200):
    """Bootstrap standard error of the estimator (20).

    ALGORITHM  resample the M trajectory values WITH REPLACEMENT n_boot times, apply Eq. (20) to each
    resample, return the standard deviation of the resulting values.  No formula for the variance of a
    nonlinear statistic is needed -- that is the point of the bootstrap (Efron 1979).
    JAX  `jax.random.randint` draws the indices; vmap over the bootstrap replicas.
    """
    M = z.shape[0]

    def one(k):
        idx = jax.random.randint(k, (M,), 0, M)
        return estimators(z[idx])[1]

    samples = jax.vmap(one)(jax.random.split(key, n_boot))
    return jnp.std(samples)


N_tr, M_tr = 8, 1500
p_tr = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.20])
ang_tr = zero_angles(N_tr)
psi0_tr = ghz_state(N_tr)

traj_jit = jax.jit(traj_values, static_argnums=(4,))       # compile ONCE, outside the loop
boot_jit = jax.jit(bootstrap_error, static_argnums=(2,))

t0 = time.time()
rows_tr = []
for i, p in enumerate(p_tr):
    kr = kraus_dephasing(float(p))
    z = traj_jit(jax.random.fold_in(jax.random.PRNGKey(77), i), psi0_tr, kr, ang_tr, M_tr)
    naive, corrected, S2 = estimators(z)
    err = float(boot_jit(jax.random.fold_in(jax.random.PRNGKey(78), i), z, 200))
    exact = float(correlator_dm(noisy_ghz_dm(N_tr, kr), ang_tr))
    rows_tr.append((p, float(naive), float(corrected), err, exact))
print(f"{len(p_tr)} noise points x {M_tr} trajectories at N = {N_tr}: {time.time()-t0:.1f} s\n")

print(f"{'p':>5s} {'naive':>12s} {'corrected':>12s} {'bootstrap SE':>13s} {'exact (DM)':>12s} "
      f"{'(corr-exact)/SE':>16s}")
for p, nv, co, er, ex in rows_tr:
    ratio = f"{(co-ex)/er:+16.2f}" if er > 1e-12 else f"{'n/a (no noise)':>16s}"
    print(f"{p:5.2f} {nv:12.5e} {co:12.5e} {er:13.3e} {ex:12.5e} {ratio}")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
p_dense = np.linspace(0, 0.22, 100)
axes[0].semilogy(p_dense, 0.25 * (1 - 2 * p_dense) ** (2 * N_tr), "k-", lw=1.2, label="exact, Section 9.1")
axes[0].errorbar(p_tr, [r[2] for r in rows_tr], yerr=[r[3] for r in rows_tr], fmt="o", ms=5, capsize=3,
                 color=PALETTE[0], label=rf"trajectories, Eq. (20), $M={M_tr}$")
axes[0].semilogy(p_tr, [r[1] for r in rows_tr], "s--", ms=5, color=PALETTE[1], alpha=0.8,
                 label="naive $\\vert\\bar z\\vert^2$")
axes[0].axhline(2.0 ** (-N_tr), color="0.4", ls=":", label=r"$2^{-N}$")
axes[0].set_xlabel(r"dephasing $p$"); axes[0].set_ylabel(r"$\mathcal{E}$")
axes[0].set_title(rf"GHZ, $N={N_tr}$: trajectories vs the exact density tensor"); axes[0].legend(fontsize=7.5)

qb_traj = q_values(np.array([max(r[2], 1e-300) for r in rows_tr]), N_tr)[0]
qb_exact = q_values(np.array([r[4] for r in rows_tr]), N_tr)[0]
err_q = np.array([r[3] for r in rows_tr]) / (np.array([max(r[2], 1e-300) for r in rows_tr]) * np.log(2))
axes[1].errorbar(p_tr, qb_traj, yerr=err_q, fmt="o", ms=5, capsize=3, color=PALETTE[0], label="trajectories")
axes[1].plot(p_tr, qb_exact, "k-", lw=1.2, label="exact")
axes[1].axhline(0, color="0.4", lw=0.9)
axes[1].set_xlabel(r"dephasing $p$"); axes[1].set_ylabel(r"$Q_{\rm B}$")
axes[1].set_title(r"$Q_{\rm B}$ with propagated bootstrap errors"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# With $M=1500$ trajectories the corrected estimator agrees with the exact density-tensor value within $1.7$ bootstrap
# standard errors at every noise strength; the largest pull is $+1.67$ at $p=0.03$ and the rest are below $1.7$ in absolute
# value. Unlike the previous figure, this one is a *single* realisation at each $p$, so the bias shows up only where it is
# larger than the fluctuation: at $p=0.12$ and $p=0.15$ the naive points happen to fall below the exact curve, while at
# $p=0.20$ the naive estimator returns $3.0\cdot10^{-4}$ where the exact answer is $7.1\cdot10^{-5}$ — a factor of four —
# because there the true signal has fallen below the $\sigma^2/M$ floor and the estimator is measuring its own variance.
# The corrected value, $1.3\cdot10^{-4}\pm5.4\cdot10^{-4}$, is consistent with the exact one and
# honestly reports that the measurement cannot resolve it.
#
# The right panel shows what that means for the certificate. The error bars on $Q_{\rm B}$ blow up exactly where
# $\mathcal E$ approaches zero, because $\sigma_Q=\sigma_{\mathcal E}/(\mathcal E\ln2)$ diverges — and at the last two points
# even that is optimistic, since the linearisation behind it assumes $\sigma_{\mathcal E}\ll\mathcal E$ whereas here
# $\sigma_{\mathcal E}>\mathcal E$, so the plotted bar is a placeholder for "undefined" rather than a confidence interval. At
# $p=0.15$ and $p=0.20$ the
# trajectory estimate says nothing at all about whether the bound is violated. That is the correct answer: with $1500$
# trajectories one cannot certify anything beyond $p\approx0.12$, and a naive analysis that reported $Q_{\rm B}$ without error
# bars would have claimed a measurement where there is none.

# %% [markdown]
# ## 11. One-axis twisting as a source of Bell correlations
#
# One-axis twisting is the standard way of producing metrologically useful states in cold-atom and trapped-ion experiments:
# starting from the coherent spin state $\vert+\rangle^{\otimes N}$, the Hamiltonian $\chi J_z^2$ shears the uncertainty
# region on the Bloch sphere, first producing spin squeezing and eventually, at $\chi t=\pi/2$, a GHZ-like cat state
# ([notebooks 33 and 34](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb)). The evolution is
# diagonal in the computational basis, so `oat_evolve` applies it exactly with no Trotter error.
#
# Plodzien, Lewenstein, Witkowska and Chwedenczuk (2022) showed that this evolution is also a source of many-body Bell
# correlations, and computed the time at which they appear: for large $N$ they find
#
# $$\chi t_{\rm crit}\simeq\frac{2}{N}\sqrt{\frac{\pi^2}{8\ln2}-1}\approx\frac{1.77}{N} ,$$
#
# whereas the optimal squeezing time of one-axis twisting scales as $\chi t_{\rm s}\sim N^{-2/3}$
# ([notebook 33](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb)). Since
# $N^{-1}\ll N^{-2/3}$, Bell correlations are predicted to appear *before* the squeezing optimum, by a factor
# $t_{\rm crit}/t_{\rm s}\sim N^{-1/3}$ that closes only slowly. We can now check that directly at one size: evaluate
# $\mathcal E$ along the whole trajectory and report when each bound is crossed, alongside the Wineland squeezing
# parameter $\xi^2$ so that the two resources can be compared on one axis.
#
# One difference from the published calculation should be stated before the numbers appear. That work evaluates the
# correlator in a *fixed* measurement plane (the twisting axis and one axis orthogonal to it); here every run maximises
# over all $2N$ angles, as Eq. (2) prescribes. The two agree at late times — both reach $\mathcal E=1/4$ at the cat time —
# but the free maximisation necessarily crosses the bound earlier, and at $N=8$ the difference is large enough to matter:
# the fixed-plane correlator crosses at $\chi t\approx0.37$, *after* the squeezing optimum, the maximised one at
# $\chi t\approx0.15$, before it. Both onset times scale as $1/N$, so the ordering "Bell first" becomes unambiguous at
# larger $N$; at $N=8$ it depends on which of the two quantities is being reported.

# %%
# ==============================================================================
# STEP 13: the correlator along the one-axis-twisting evolution
# ==============================================================================
N_oat = 8
ts_oat = np.linspace(0.0, np.pi / 2, 17)
t_dense = np.linspace(1e-4, np.pi / 2, 400)
psi_cs = product_state("+" * N_oat)
a0_oat = random_angles(jax.random.PRNGKey(123), 8, N_oat)

t0 = time.time()
oat_rows = []
for tt in ts_oat:
    psi_t = oat_evolve(psi_cs, float(tt))
    E = float(optimise_pure(psi_t, a0_oat, 200, 0.1)[0])
    xi2 = float(spin_squeezing(psi_t))
    oat_rows.append((tt, E, xi2))
print(f"{len(ts_oat)} OAT times at N = {N_oat}: {time.time()-t0:.1f} s\n")

print(f"{'chi t':>7s} {'E':>12s} {'Q_B':>9s} {'Q_E':>9s} {'xi^2 (Wineland)':>16s}")
for tt, E, xi2 in oat_rows:
    qb, qe = q_values(E, N_oat)
    print(f"{tt:7.4f} {E:12.5e} {qb:+9.4f} {qe:+9.4f} {xi2:16.4f}")

tt_arr = np.array([r[0] for r in oat_rows])
E_arr = np.array([r[1] for r in oat_rows])
xi_arr = np.array([r[2] for r in oat_rows])
qb_arr, qe_arr = q_values(E_arr, N_oat)
first_B = tt_arr[np.argmax(qb_arr > 0)] if np.any(qb_arr > 0) else np.nan
first_E = tt_arr[np.argmax(qe_arr > 0)] if np.any(qe_arr > 0) else np.nan
best_sq = tt_arr[int(np.argmin(xi_arr))]
# locate the Q_B = 0 crossing by bisection on the optimised correlator
lo = tt_arr[np.max(np.flatnonzero(qb_arr <= 0)[np.flatnonzero(qb_arr <= 0) < np.argmax(qb_arr > 0)])]
hi = first_B
for _ in range(14):
    mid = 0.5 * (lo + hi)
    Emid = float(optimise_pure(oat_evolve(psi_cs, float(mid)), a0_oat, 200, 0.1)[0])
    if q_values(Emid, N_oat)[0] > 0:
        hi = mid
    else:
        lo = mid
t_bell = 0.5 * (lo + hi)

# the squeezing minimum on a dense grid (spin_squeezing is cheap -- no optimisation involved)
xi_dense = np.array([float(spin_squeezing(oat_evolve(psi_cs, float(x)))) for x in t_dense])
t_sq = t_dense[int(np.argmin(xi_dense))]

print(f"\nfirst grid time with Q_E > 0 (entanglement)   : chi t = {first_E:.4f}")
print(f"first grid time with Q_B > 0 (Bell correlated): chi t = {first_B:.4f}")
print(f"Q_B = 0 located by bisection                 : chi t = {t_bell:.4f}")
print(f"best squeezing xi^2 = {xi_dense.min():.4f} at chi t = {t_sq:.4f} (dense grid of {len(t_dense)} points)")
print(f"  -> Bell correlations appear {'before' if t_bell < t_sq else 'after'} the squeezing optimum "
      f"(chi t = {t_bell:.4f} vs {t_sq:.4f})")
print(f"at chi t = pi/2: Q_B = {qb_arr[-1]:.4f} (maximum {N_oat-2}), Q_E = {qe_arr[-1]:.4f} (maximum {N_oat-1})")

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(tt_arr, qb_arr, "o-", ms=4, color=PALETTE[0], label=r"$Q_{\rm B}$ (Bell)")
ax.plot(tt_arr, qe_arr, "s-", ms=4, color=PALETTE[2], label=r"$Q_{\rm E}$ (entanglement)")
ax.axhline(0, color="k", lw=0.9)
ax.axhline(N_oat - 2, color=PALETTE[0], ls=":", lw=1)
ax.axhline(N_oat - 1, color=PALETTE[2], ls=":", lw=1)
ax.set_xlabel(r"$\chi t$"); ax.set_ylabel(r"$Q$")
ax2 = ax.twinx()
l_xi, = ax2.semilogy(t_dense, xi_dense, "--", lw=1.4, color=PALETTE[1], label=r"$\xi^2$ (Wineland, right axis)")
ax2.set_ylabel(r"$\xi^2$ (diverges after the squeezing window)", color=PALETTE[1]); ax2.grid(False)
ax2.set_ylim(0.2, 1e3)
ax2.axhline(1.0, color=PALETTE[1], ls=":", lw=0.8)
l_bell = ax.axvline(t_bell, color="0.45", ls="-.", lw=1, label=rf"$Q_{{\rm B}}=0$ at $\chi t={t_bell:.3f}$")
ax.set_title(rf"one-axis twisting, $N={N_oat}$: both certificates along one evolution")
h, lab = ax.get_legend_handles_labels()
ax.legend(h + [l_xi], lab + [l_xi.get_label()], fontsize=8, loc="lower right")
fig.tight_layout(); plt.show()

# %% [markdown]
# The evolution passes three landmarks. **Entanglement** is certified immediately: $Q_{\rm E}$ is $-0.0000$ at $t=0$ (the
# coherent spin state is a product state and sits exactly on the bound, as Section 6.1 requires) and already $+2.955$ at the
# first grid point $\chi t=0.098$. **Bell correlations** appear at $\chi t=0.1526$, located by bisecting the optimised
# $Q_{\rm B}$. **Optimal squeezing** comes later: on a dense grid of $400$ points the Wineland parameter reaches its minimum
# $\xi^2=0.3541$ at $\chi t=0.2284$. So the Bell certificate switches on *before* the state is optimally squeezed — the two
# resources are produced by the same interaction but peak at different times, and the squeezing optimum is not the moment at
# which the state becomes most nonlocal. At $N=8$ the two times differ by only $50\,\%$, which is why the caveat above
# matters: the ordering is robust because of the *scaling* $\chi t_{\rm crit}\sim1.77/N$ against
# $\chi t_{\rm s}\sim N^{-2/3}$, not because of this one pair of numbers. Exercise 9 asks you to measure both exponents.
#
# After that, $Q_{\rm B}$ rises to a local maximum of $4.276$ near $\chi t=0.69$, dips to $3.59$ around $\chi t=1.0$, and then
# climbs to exactly $N-2=6$ at the cat time $\chi t=\pi/2$, where the state is a GHZ state in a rotated frame and the
# algebraic maximum $\mathcal E=1/4$ is reached. The squeezing parameter, in contrast, has long since diverged: past
# $\chi t\approx0.4$ the mean spin length $\vert\langle\mathbf J\rangle\vert$ collapses and $\xi^2$ runs away
# (it is $10^{33}$ at the cat time, which is why the right-hand axis is clipped). A quantity that measures the shape of a
# small uncertainty patch on the Bloch sphere simply stops being defined once the state is a macroscopic superposition,
# whereas $\mathcal E$ keeps working and reports the maximum possible value.
#
# > **Physics insight.** The same one-axis-twisting evolution that [notebook 33](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb)
# > uses for metrology is, at later times, a generator of many-body Bell correlations — and the certificate reaches its
# > algebraic maximum exactly at the cat time that
# > [notebook 34](../ch10_quantum_metrology_protocols/34_oat_to_ghz_full_metrology_protocol.ipynb) uses for Heisenberg-limited
# > interferometry. One interaction, three different resources, three different optimal times.

# %% [markdown]
# ## 12. Key takeaways
#
# * A **Bell inequality** bounds the correlations of any local hidden-variable model, in which every party's outcome for
#   every setting is a fixed function of a shared variable. For $N$ parties the correlator
#   $\mathcal E=\max_U\vert\langle\bigotimes_kU_k\sigma^{+}U_k^{\dagger}\rangle\vert^2$ packages the $2^N$ correlation
#   functions into one number, because $\tilde\sigma^{+}=\tfrac12(A+iB)$ with $A\perp B$ dichotomic.
# * **Three bounds, all derived here**: $\mathcal E\le2^{-N}$ for every local hidden-variable model (the factors
#   $\pm1\pm i$ have modulus $\sqrt2$); $\mathcal E\le4^{-m}$ for every state separable into $m$ groups (Cauchy–Schwarz plus
#   the orthogonality of $\bigotimes U\vert0\rangle$ and $\bigotimes U\vert1\rangle$); and $\mathcal E\le1/4$ for every state
#   whatsoever. In the logarithmic rulers of Eq. (3): $Q_{\rm B}\le0$, $Q_{\rm E}\le N-m$ (so $\le0$ for full
#   separability), $Q_{\rm B}\le N-2$. Those two rulers are the $Q_N$ of Phys. Rev. Research **6**, 023050 and the
#   $\mathcal Q$ of Phys. Rev. A **110**, 032428.
# * **Evaluation is two array lookups.** $\sigma^{+\otimes N}=\vert1\cdots1\rangle\langle0\cdots0\vert$, so
#   $\langle\mathcal B\rangle=\overline{\psi'[1\ldots1]}\,\psi'[0\ldots0]$ after $N$ single-qubit rotations: $O(N2^N)$ instead
#   of $O(4^N)$, and $\rho'[0\ldots0;1\ldots1]$ for a density tensor.
# * **Closed forms verified**: $\mathcal E=4^{-N}$ for $\vert+\rangle^{\otimes N}$ (the separability bound is saturated),
#   $\mathcal E=1/4$ for GHZ (the algebraic maximum, $Q_{\rm B}=N-2$), $\mathcal E=\binom{N}{m}^24^{-N}$ for Dicke states,
#   hence $Q_{\rm E}(W_N)=\log_2N$ and $Q_{\rm B}(W_N)=2\log_2N-N$, which is positive only for $N=3$.
# * **Local-unitary invariance.** GHZ in rotated bases, the star graph state and the $T$-doped star graph all give exactly
#   $\mathcal E=1/4$: the correlator is blind to local unitaries, hence to single-qubit magic. The linear cluster state lands
#   on $\mathcal E=2^{-N}$ ($Q_{\rm B}=0$) at $N=4$ and $N=6$ but not in general — the measured exponent in
#   $\mathcal E=4^{-(1+\gamma)}$ is $\gamma=\lfloor2(N-2)/3\rfloor$ for $3\le N\le8$, so $Q_{\rm B}$ drifts down by about
#   one unit per three qubits — and over $16$ Haar draws the mean $Q_{\rm B}$ falls from $+0.17$ at $N=4$ to $-0.83$ at
#   $N=8$, with the violating fraction going $0.62\to0.38\to0.00$. Both families are strongly entangled: this is a
#   GHZ-shaped detector, and a non-positive $Q_{\rm B}$ certifies nothing.
# * **Noise.** At the identity angles a locally noisy GHZ state gives $\mathcal E=\tfrac14(1-2p)^{2N}$ (dephasing),
#   $\tfrac14(1-\tfrac43p)^{2N}$ (depolarising) and $\tfrac14(1-\gamma)^{N}$ (amplitude damping), all verified to $10^{-16}$.
#   Free optimisation confirms the closed form for depolarising and amplitude damping at every noise strength, but **not** for
#   dephasing: past $c^N=1/(2^{N-1}-1)$ ($p=0.1926$ at $N=4$) the optimum leaves the identity angles for the branch
#   $\mathcal E=4^{-N}(1+c^N)^2$, $c=1-2p$, which decays onto the separability bound saturated by the classical mixture of
#   $\vert0^N\rangle$ and $\vert1^N\rangle$. Seeding the restart batch with the two analytic angle choices is what makes
#   the comparison one-sided; six purely random restarts land in the wrong branch near $p=0.15$.
# * **Critical noise.** Eq. (17) gives it in closed form for the coherence branch; the measured $Q_{\rm B}$ crossings at
#   $N=4$ agree to better than $1\,\%$. The Bell certificate is always lost first — at $N=6$, $p_{\rm B}=0.103$ against
#   $p_{\rm E}=0.219$ for dephasing — and the per-qubit thresholds *increase* with $N$, towards $p_{\rm B}\to0.1464$ and
#   $p_{\rm E}\to0.25$. The $p_{\rm E}$ column is a property of that branch only: for even $N$ the maximised $Q_{\rm E}$
#   never becomes negative under dephasing.
# * **Trajectories need a corrected estimator.** $\mathcal E$ is a *nonlinear* function of a Monte-Carlo average, so
#   $\mathbb E\vert\bar z\vert^2=\vert\mu\vert^2+\sigma^2/M$: the naive estimator over-reports — by $82\,\%$ at $M=8$ in the
#   measurement above, and by a factor of four at $N=8$, $p=0.20$, $M=1500$ — most severely exactly where a violation is
#   marginal. $\widehat{\mathcal E}=\vert\bar z\vert^2-S^2/M$ is exactly unbiased, and the bootstrap gives error bars without
#   any variance formula; those error bars then show that $1500$ trajectories certify nothing beyond $p\approx0.12$.
# * **One-axis twisting** turns a coherent spin state into a Bell-correlated one. At $N=8$: $Q_{\rm E}>0$ immediately,
#   $Q_{\rm B}=0$ at $\chi t=0.1526$ (bisection), best squeezing $\xi^2=0.3541$ only at $\chi t=0.2284$, and the algebraic
#   maximum $Q_{\rm B}=N-2=6$ at the cat time $\chi t=\pi/2$, where $\xi^2$ has long since diverged.
#
# ## 13. Exercises
#
# 1. ★ **The other GHZ phase.** Compute $\mathcal E$ for $(\vert0\cdots0\rangle+e^{i\phi}\vert1\cdots1\rangle)/\sqrt2$ as a
#    function of $\phi$ at the identity angles, and then after optimisation. Explain the difference in one sentence using
#    Section 5.2.
# 2. ★ **Where the W state loses.** Using Eq. (15), find the largest $N$ for which the W state violates the local-realism
#    bound, and confirm it with the optimiser for $N=3,4,5$.
# 3. ★★ **Entanglement depth (extend the code).** Build the state $\vert\mathrm{GHZ}_k\rangle^{\otimes N/k}$ (several
#    independent GHZ blocks of $k$ qubits) for $N=6$ and $k=2,3,6$, optimise $\mathcal E$, and check the measured values
#    against the bound $\mathcal E\le4^{-N/k}$ of Section 4.3. Which $k$ does the correlator certify?
# 4. ★★ **A better optimiser schedule.** The runs above use a fixed Adam learning rate. Add a cosine decay
#    ($\mathrm{lr}_t=\mathrm{lr}_0\cos(\pi t/2T)$) and measure how many of the $16$ restarts reach the analytic optimum with
#    and without it, on the $T$-doped star graph at $N=6$.
# 5. ★★ **Dephasing does not commute with everything.** Section 6.4 says the star graph and the $T$-doped star graph are
#    indistinguishable. Apply local dephasing to both and compare $\mathcal E(p)$; then apply depolarising noise and compare
#    again. Explain the two different answers using the fact that $T$ commutes with $Z$ and that the depolarising channel is
#    covariant under all single-qubit unitaries.
# 6. ★★ **Shot noise on top of trajectory noise.** Each $z_i$ in Section 10 was computed exactly. On hardware it is itself an
#    average over shots. Model this by adding Gaussian noise of standard deviation $s$ to the real and imaginary parts of each
#    $z_i$ and redo the bias analysis: show that Eq. (20) still removes the bias, and find how the bootstrap error bar grows
#    with $s$.
# 7. ★★★ **The strong-dephasing plateau, and its parity.** (a) For $N=3,4,5$ build the dephased GHZ density tensor, optimise
#    the angles at every $p$, and locate $p$ where $Q_{\rm B}=0$ by bisection; compare with Eq. (17). (b) Evaluate
#    $\mathcal E$ for the classical mixture $\tfrac12(\vert0^N\rangle\langle0^N\vert+\vert1^N\rangle\langle1^N\vert)$ by free
#    optimisation for $N=3,\dots,6$ and check the derivation of Section 9.3: the plateau is $4^{-N}$ for even $N$ and
#    $0$ for odd $N$. Does the $p\to\tfrac12$ limit of the optimised dephasing curve reproduce that parity?
# 8. ★★★ **Graph states beyond the star.** Compute $\mathcal E$ for the linear cluster state, the ring cluster state and the
#    complete-graph state for $N=4,\dots,7$. Which graphs reach $\mathcal E=1/4$, and can you relate the answer to whether the
#    graph is local-Clifford equivalent to a star? (Answer for the complete graph: it is, so $\mathcal E=1/4$ at every $N$.)
# 9. ★★★ **The two time scales of one-axis twisting.** Repeat Section 11 for $N=8,12,16$ and locate, for each, the time at
#    which $Q_{\rm B}$ crosses zero and the time of minimal $\xi^2$. Fit both to power laws in $N$ and compare with
#    $\chi t_{\rm crit}\propto N^{-1}$ and $\chi t_{\rm s}\propto N^{-2/3}$. Then repeat the first fit with the angles
#    *frozen* in a single plane instead of optimised, and explain why the two conventions give the same exponent but
#    different prefactors.
#
# ## 14. References
#
# * J. S. Bell, *On the Einstein Podolsky Rosen paradox*, Physics Physique Fizika **1**, 195 (1964) — the original theorem.
# * J. F. Clauser, M. A. Horne, A. Shimony and R. A. Holt, *Proposed experiment to test local hidden-variable theories*,
#   Phys. Rev. Lett. **23**, 880 (1969) — the CHSH inequality of Section 3.2.
# * B. S. Cirel'son, *Quantum generalizations of Bell's inequality*, Lett. Math. Phys. **4**, 93 (1980) — the bound
#   $2\sqrt2$.
# * N. D. Mermin, *Extreme quantum entanglement in a superposition of macroscopically distinct states*,
#   Phys. Rev. Lett. **65**, 1838 (1990) — $N$-party Bell inequalities violated exponentially by GHZ states.
# * M. Zukowski and C. Brukner, *Bell's theorem for general N-qubit states*, Phys. Rev. Lett. **88**, 210401 (2002) — the
#   general family of two-setting $N$-party inequalities mentioned in Section 3.3.
# * E. G. Cavalcanti, C. J. Foster, M. D. Reid and P. D. Drummond, *Bell inequalities for continuous-variable correlations*,
#   Phys. Rev. Lett. **99**, 210405 (2007) — the CFRD inequalities built from complex combinations of two local observables,
#   of which Eq. (8) is the qubit case.
# * E. G. Cavalcanti, Q. Y. He, M. D. Reid and H. M. Wiseman, *Unified criteria for multipartite quantum nonlocality*,
#   Phys. Rev. A **84**, 032115 (2011), arXiv:1008.5014 — the family
#   $\vert\langle\prod_j(X_j+iY_j)\rangle\vert\le2^{(N-T)/2}$ that contains both bounds of Section 4 as the cases $T=0$
#   and $T=N$, and the steering criteria in between.
# * J. Chwedenczuk, *Many-body Bell inequalities for bosonic qubits*, SciPost Phys. Core **5**, 025 (2022),
#   arXiv:2109.15156 — the correlator of Eq. (1), the Cauchy–Schwarz derivation of the bound $2^{-N}$, and the version for
#   indistinguishable particles.
# * A. Niezgoda and J. Chwedenczuk, *Many-body nonlocality as a resource for quantum-enhanced metrology*,
#   Phys. Rev. Lett. **126**, 210506 (2021), arXiv:2011.06612 — the same correlator, and the bound
#   $F_Q\ge N^2/2^{m+1}$ linking it to the quantum Fisher information.
# * M. Plodzien, M. Lewenstein, E. Witkowska and J. Chwedenczuk, *One-axis twisting as a method of generating many-body Bell
#   correlations*, Phys. Rev. Lett. **129**, 250402 (2022), arXiv:2206.10542 — the physics of Section 11, including the
#   onset time $\chi t_{\rm crit}\approx1.77/N$ and the value $\mathcal E=1/4$ at the cat time.
# * M. Plodzien, T. Wasak, E. Witkowska, M. Lewenstein and J. Chwedenczuk, *Generation of scalable many-body Bell
#   correlations in spin chains with short-range two-body interactions*, Phys. Rev. Research **6**, 023050 (2024),
#   arXiv:2306.06173 — the same correlator for short-range spin chains; its Eq. (3) defines the quantity written
#   $Q_{\rm B}$ here through $\mathcal E_N\equiv2^{Q_N-N}$.
# * M. Plodzien, J. Chwedenczuk, M. Lewenstein and G. Rajchel-Mieldzioc, *Entanglement classification and non-k-separability
#   certification via Greenberger-Horne-Zeilinger-class fidelity*, Phys. Rev. A **110**, 032428 (2024), arXiv:2406.10662 —
#   the quantity written $Q_{\rm E}$ here (its $\mathcal Q=\log_4(4^N\mathcal E)$) and the $k$-separability bound
#   $\mathcal Q\le N-k$, which is Eq. (10) of Section 4.3.
# * M. Plodzien, M. Lewenstein and J. Chwedenczuk, *Many-body quantum resources of graph states*,
#   Rep. Prog. Phys. **88**, 077601 (2025), arXiv:2410.12487 — graph states as in Section 8 and a classification of their
#   Bell correlations; there the maximisation in Eq. (2) is restricted to the three Pauli frames per site.
# * J. C. Spall, *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation*,
#   IEEE Trans. Autom. Control **37**, 332 (1992) — SPSA.
# * D. P. Kingma and J. Ba, *Adam: a method for stochastic optimization*, arXiv:1412.6980, ICLR 2015 — the optimiser.
# * B. Efron, *Bootstrap methods: another look at the jackknife*, Ann. Statist. **7**, 1 (1979) — the error bars of
#   Section 10.
# * M. Kitagawa and M. Ueda, *Squeezed spin states*, Phys. Rev. A **47**, 5138 (1993) — one-axis twisting.
# * D. J. Wineland, J. J. Bollinger, W. M. Itano and D. J. Heinzen, *Squeezed atomic states and projection noise in
#   spectroscopy*, Phys. Rev. A **50**, 67 (1994) — the squeezing parameter $\xi^2$ plotted in Section 11.
