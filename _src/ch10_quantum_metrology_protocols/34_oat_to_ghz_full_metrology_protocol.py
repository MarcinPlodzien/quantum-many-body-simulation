#@title: From one-axis twisting to GHZ — the full metrology protocol
#@part: Chapter 10 — Quantum metrology protocols
#@description: The complete pipeline prepare-twist-encode-read out: the exact derivation that one-axis twisting turns a coherent spin state into a GHZ-like cat at chi t = pi/2 (for even and for odd N), the cat axis tracked numerically with the 3x3 QFI matrix along the whole evolution, the N^2/2 plateau of the multi-component cats, parity readout reaching the Heisenberg limit from sampled data, the SLD quantum Fisher information under dephasing, amplitude damping and depolarising noise, the exponential fragility of the cat derived and verified, the optimal stopping time under noise, and a final comparison of the sensitivity of Ramsey, squeezed, GHZ and one-axis-twisting interferometry.

# %% [markdown]
# ## 1. Introduction and motivation
#
# [33 — spin squeezing by one-axis twisting](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb)
# stopped where the squeezing parameter stops being useful: at a twisting angle of order $N^{-2/3}$, where the
# uncertainty disc has been sheared into a thin ellipse and the Wineland parameter reaches $\xi_R^2\sim N^{-2/3}$. But the
# quantum Fisher information of the same evolution keeps growing long after that, and at $\chi t=\pi/2$ it reaches
# $F_Q=N^2$ — the Heisenberg limit, the largest value any state of $N$ qubits can have for a collective generator.
#
# At that point the state is a **GHZ-like cat**, a coherent superposition of two coherent spin states pointing in
# opposite directions. The same interaction that produces modest, robust squeezing at short times produces, when it is
# left on until $\chi t=\pi/2$, the most fragile and most sensitive state of this chapter.
#
# This notebook follows the protocol from beginning to end:
#
# $$\vert+x\rangle^{\otimes N}
# \;\xrightarrow[\text{twist}]{\;e^{-i\mu J_z^2}\;}\;\vert\psi(\mu)\rangle
# \;\xrightarrow[\text{encode}]{\;e^{-i\theta\,\mathbf n\cdot\mathbf J}\;}\;\vert\psi_\theta\rangle
# \;\xrightarrow[\text{read out}]{\;\text{parity}\;}\;\hat\theta ,$$
#
# computes what each stage is worth, and then asks the question that decides whether any of it is usable: what does noise
# do?
#
# **Road map.**
#
# * **Section 4.** The cat state, derived exactly. $e^{-i\frac{\pi}{2}J_z^2}$ multiplies the amplitude of every
#   $J_z$ eigenvalue $m$ by $e^{-i\frac{\pi}{2}m^2}$. For even $N$, $m$ is an integer and $e^{-i\frac{\pi}{2}m^2}$ takes
#   only two values, which recombine into the identity and the parity operator — producing a cat along $\hat x$. For odd
#   $N$, $m$ is a half-integer, the phase has period $4$ in $m$, and the cat comes out along $\hat y$. Both results are
#   verified by fidelity with the predicted state.
# * **Section 5.** The evolution implemented twice — diagonal phases and a circuit of commuting $ZZ$ gates — and shown
#   identical, as in notebook 33.
# * **Section 6.** The $3\times3$ QFI matrix evaluated at every twisting angle: its largest eigenvalue is $F_Q^{\max}(\mu)$
#   and its eigenvector is the axis of the cat. We identify four features — $F_Q=N$ at $\mu=0$, the squeezing growth
#   $\approx N/\xi_R^2$, a plateau near $N^2/2$, and $F_Q=N^2$ at $\mu=\pi/2$ — and measure how each scales with $N$.
# * **Section 7.** At $\mu=\pi/q$ the state is a superposition of $q$ coherent states. Husimi-$Q$ maps for
#   $q=2,3,4,5,6,8$.
# * **Section 8.** The readout. Encoding with the optimal generator and measuring the parity $\prod_qZ_q$ gives a fringe
#   of period $2\pi/N$ whose classical Fisher information is exactly $N^2$ at every phase but the two fringe extrema;
#   we then sample it and estimate $\theta$ by
#   maximum likelihood, reaching $\Delta\theta=1/(N\sqrt M)$.
# * **Sections 9–11.** Noise. The density tensor, Trotterised as (twist step, channel step); the quantum Fisher
#   information from the symmetric logarithmic derivative; an exact derivation of the cat's exponential fragility,
#   $F_Q=N^2(1-2p)^{2N}$ under dephasing in the cat basis; why the axis of the noise relative to the cat matters; and the
#   optimal stopping time, which under the noise strengths studied here lies in the squeezing regime, well before the cat.
# * **Section 12.** A final comparison of the four protocols of this chapter at equal $N$.
#
# ### What you will learn
#
# *Physics*
# * why a quadratic spectrum $\chi m^2$ produces exact revivals, and why the revival at $\chi t=\pi/2$ is a two-component
#   cat for even $N$ and a cat along a different axis for odd $N$;
# * how to read the structure of a state off a $3\times3$ matrix: eigenvalue = metrological usefulness, eigenvector = the
#   direction in which the state is "large";
# * why a multi-component cat gives $F_Q\approx N^2/2$ rather than $N^2$;
# * why the Heisenberg-limited state is exponentially fragile — the decoherence rate of an $N$-body coherence is $N$
#   times the single-qubit rate — and what the optimal compromise looks like.
#
# *Numerical methods*
# * Gauss sums and the finite Fourier analysis of $e^{-i\frac{\pi}{2}m^2}$, checked by state fidelity;
# * a Trotterised open-system evolution on a rank-$2N$ density tensor, with the unitary part exact;
# * the mixed-state QFI matrix as a quadratic form over generator directions, from the SLD spectrum;
# * classical Fisher information of an exact outcome distribution, and maximum-likelihood estimation from sampled data.
#
# *Implementation practice*
# * `jax.jit` on a whole noisy evolution step; static channel/qubit structure and traced parameters;
# * reusing derived functions across notebooks instead of re-deriving them;
# * honest budget management: pure states to $N=16$, density tensors to $N=6$.
#
# ### Prerequisites
# * [29 — quantum Fisher information](../ch10_quantum_metrology_protocols/29_quantum_fisher_information.ipynb):
#   $F_Q=4\,\mathrm{Var}(G)$, the $3\times3$ QFI matrix, the standard quantum and Heisenberg limits;
# * [30 — QFI from the SLD](../ch10_quantum_metrology_protocols/30_qfi_from_the_sld_prepare_encode_estimate.ipynb):
#   the symmetric logarithmic derivative, the mixed-state QFI, the optimal measurement, maximum-likelihood estimation;
# * [31 — Ramsey interferometry](../ch10_quantum_metrology_protocols/31_ramsey_interferometry.ipynb) and
#   [32 — GHZ interferometry and the Heisenberg limit](../ch10_quantum_metrology_protocols/32_ghz_interferometry_heisenberg_limit.ipynb):
#   the two reference protocols; Section 3 recalls the one formula we need from each;
# * [33 — spin squeezing by one-axis twisting](../ch10_quantum_metrology_protocols/33_spin_squeezing_one_axis_twisting.ipynb):
#   the Hamiltonian, the exact propagator, the squeezing parameters;
# * [07 — density matrices and quantum channels](../ch03_matrix_free_engine/07_density_matrices_and_quantum_channels.ipynb):
#   density tensors, Kraus channels.
#
# **Conventions.** $J_a=\tfrac12\sum_q\sigma^a_q$; $\vert0\rangle$ is the $+1$ eigenstate of $Z$; the twisting angle is
# $\mu=\chi t$ and the propagator is $e^{-i\mu J_z^2}$. SQL $=N$, Heisenberg limit $=N^2$.

# %% [markdown]
# ## 2. Engine recap and notebook helpers
#
# From the engine: the state constructors, `apply_gate` and `apply_kraus_dm`, the two-qubit rotation `rzz`, the Kraus
# channels, `spin_moments`, `spin_squeezing`, `collective_dense`, `qfi_mixed`, `oat_evolve` and `sample_bitstrings`.
# The symmetric-logarithmic-derivative routines of notebook 30 and the Husimi-$Q$ machinery of notebook 33 are
# re-created below, so this notebook stands alone.

# %%
#@engine: apply_gate, apply_kraus_dm, to_dm, dm_matrix, product_state, ghz_state, rzz, I2, X, Y, Z, spin_moments, spin_squeezing, collective_dense, qfi_mixed, oat_evolve, sample_bitstrings, kraus_dephasing, kraus_bit_flip, kraus_depolarizing, kraus_amplitude_damping, purity

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


def pauli_direction(n):
    """Single-qubit matrix n.sigma = n_x X + n_y Y + n_z Z for a unit vector n."""
    n = jnp.asarray(n, dtype=CDTYPE)
    return n[0] * X + n[1] * Y + n[2] * Z


def collective_rotation(psi, angle, n):
    """exp(-i angle n.J) with n.J = (1/2) sum_q (n.sigma)_q, applied to a pure state.

    MATH   n.J is a SUM of commuting single-qubit terms, so the exponential factorises EXACTLY:
               exp(-i angle n.J) = prod_q [ cos(angle/2) 1 - i sin(angle/2) (n.sigma) ]_q .
           (Using (n.sigma)^2 = 1 for a unit vector n.)
    COST   N single-qubit einsums, O(N 2^N).   JAX: `n` and `angle` are traced, qubit indices static.
    """
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    U = c * I2 - 1j * s * pauli_direction(n)
    for q in range(psi.ndim):
        psi = apply_gate(psi, U, [q])
    return psi


def qfi_matrix(psi):
    """3x3 QFI matrix of a PURE state over the collective generators (J_x, J_y, J_z)  [notebook 29].

    MATH   Fcal[a,b] = 4 * ( (1/2)<J_a J_b + J_b J_a> - <J_a><J_b> );   F_Q(n) = n^T Fcal n .
    COST   three matrix-free applications of J_a: O(N 2^N).
    """
    _, cov = spin_moments(psi)
    return 4.0 * cov


def optimal_direction(psi):
    """Best generator direction and the QFI it delivers: (F_max, n_opt) = top eigenpair of Fcal."""
    w, v = jnp.linalg.eigh(qfi_matrix(psi))
    return w[-1], v[:, -1]


def fix_sign(n):
    """Fix the arbitrary global sign of an eigenvector: make its largest component positive."""
    n = np.asarray(n)
    return n * np.sign(n[int(np.argmax(np.abs(n)))])


print("helpers ready")

# %% [markdown]
# ## 3. The protocol, and what we already know
#
# ### 3.1 The pipeline
#
# Every experiment in this chapter has the same three stages, and the notebooks differ only in how they fill them in.
#
# | stage | Ramsey (nb 31) | GHZ (nb 32) | this notebook |
# |---|---|---|---|
# | prepare | $\vert+x\rangle^{\otimes N}$ | $\left(\vert0\rangle^{\otimes N}+\vert1\rangle^{\otimes N}\right)/\sqrt2$ | $e^{-i\mu J_z^2}\vert+x\rangle^{\otimes N}$ |
# | encode | $e^{-i\phi J_z}$ | $e^{-i\theta J_z}$ | $e^{-i\theta\,\mathbf n\cdot\mathbf J}$, $\mathbf n$ optimal |
# | read out | population imbalance | parity in the $X$ basis | parity in the $Z$ basis |
# | ideal $F_Q$ | $N$ | $N^2$ | $N$ to $N^2$, depending on $\mu$ |
#
# Three facts are carried over, each stated here in the form we need and derived in the notebook indicated.
#
# **(a) Quantum Cramér–Rao bound** (notebook 29). For $M$ repetitions of a protocol whose prepared state has quantum
# Fisher information $F_Q$ with respect to the encoding generator,
#
# $$\Delta\theta\;\ge\;\frac{1}{\sqrt{M\,F_Q}},$$
#
# with equality for the optimal measurement. For a pure state and a unitary encoding $F_Q=4\,\mathrm{Var}(G)$, and for a
# collective generator $G=\mathbf n\cdot\mathbf J$ this is the quadratic form $F_Q(\mathbf n)=\mathbf n^{\mathsf T}\mathcal F\mathbf n$
# with $\mathcal F_{ab}=4C_{ab}$; the best direction is the top eigenvector of $\mathcal F$.
#
# **(b) Ramsey and GHZ** (notebooks 31, 32). A coherent spin state reaches $\Delta\phi=1/\sqrt{NM}$ — the standard
# quantum limit; a GHZ state encoded with $J_z$ and read out with the parity $\prod_qX_q$ reaches $\Delta\theta=1/(N\sqrt M)$
# — the Heisenberg limit. The GHZ fringe has period $2\pi/N$, so the phase is only determined modulo $2\pi/N$ unless one
# already knows it that well.
#
# **(c) One-axis twisting** (notebook 33). $H=\chi J_z^2$ is diagonal in the computational basis: with
# $m(s)=\sum_q(\tfrac12-s_q)$,
#
# $$e^{-i\mu J_z^2}\,\psi[s]=e^{-i\mu\,m(s)^2}\,\psi[s],$$
#
# which is the engine's `oat_evolve` — exact, $O(2^N)$, no Trotter error. Equivalently it is a circuit of $N(N-1)/2$
# commuting $R_{ZZ}(\mu)$ gates times a global phase, since $\sum_{i<j}Z_iZ_j=2J_z^2-N/2$. Starting from
# $\vert+x\rangle^{\otimes N}$ the state squeezes, with the best Wineland parameter
# $\xi_R^2$ at $\mu_{\rm opt}\sim N^{-2/3}$, and then oversqueezes.
#
# ### 3.2 Why something special must happen at $\mu=\pi/2$
#
# The spectrum of $J_z^2$ is $\{m^2\}$ with $m$ integer ($N$ even) or half-integer ($N$ odd). For integer $m$, $m^2$ is an
# integer, so $e^{-i\mu m^2}$ is $2\pi$-periodic in $\mu$: the evolution has an exact revival at $\mu=2\pi$, and the
# dynamics never dephases permanently. At the *half* period, $\mu=\pi$, one has $e^{-i\pi m^2}=(-1)^{m^2}=(-1)^m$, a phase
# that depends on $m$ only through its parity — this is a *unitary that acts like a single collective rotation*, so the
# state at $\mu=\pi$ is again a coherent spin state. At the quarter period, $\mu=\pi/2$, the phase $e^{-i\frac{\pi}{2}m^2}$
# takes exactly **two** values, and a two-valued function of $m$ is a superposition of exactly two "rotation-like"
# operators. That is the origin of the cat.
#
# Section 4 turns this counting argument into an exact identity.

# %% [markdown]
# ## 4. The cat state at $\mu=\pi/2$, derived
#
# ### 4.1 Even $N$: a cat along $\hat x$
#
# For $N$ even, $m=\tfrac12(N-2k)=\tfrac{N}{2}-k$ is an **integer**. For integer $m$,
#
# $$m^2\equiv\begin{cases}0\pmod 4,& m\text{ even}\\ 1\pmod 4,& m\text{ odd}\end{cases}
# \qquad\Longrightarrow\qquad
# e^{-i\frac{\pi}{2}m^2}=\begin{cases}1,& m\text{ even}\\ -i,& m\text{ odd.}\end{cases}$$
#
# A function of $m$ taking the value $u$ on even $m$ and $v$ on odd $m$ is $\tfrac{u+v}{2}+\tfrac{u-v}{2}(-1)^m$. With
# $u=1$, $v=-i$,
#
# $$e^{-i\frac{\pi}{2}m^2}=\frac{1-i}{2}+\frac{1+i}{2}\,(-1)^m
#  =\frac{e^{-i\pi/4}}{\sqrt2}+\frac{e^{+i\pi/4}}{\sqrt2}\,(-1)^m . \tag{1}$$
#
# Now identify the operator $(-1)^{J_z}$. With $m=\tfrac{N}{2}-k$ and $k$ = number of qubits in $\vert1\rangle$,
#
# $$(-1)^m=(-1)^{N/2}(-1)^{-k}=(-1)^{N/2}(-1)^{k}=(-1)^{N/2}\prod_{q}Z_q,$$
#
# because $\prod_qZ_q$ has eigenvalue $(-1)^k$ on the basis state with $k$ ones. Substituting into Eq. (1),
#
# $$e^{-i\frac{\pi}{2}J_z^2}
#  =\frac{e^{-i\pi/4}}{\sqrt2}\,\mathbb 1+\frac{e^{+i\pi/4}}{\sqrt2}\,(-1)^{N/2}\prod_qZ_q . \tag{2}$$
#
# Equation (2) is an operator identity for even $N$: the twisting propagator at $\mu=\pi/2$ is a superposition of the
# identity and a single collective spin flip. Applying it to $\vert+x\rangle^{\otimes N}$ and using
# $\prod_qZ_q\vert+x\rangle^{\otimes N}=\vert-x\rangle^{\otimes N}$ (each $Z$ maps $\vert+x\rangle$ to $\vert-x\rangle$):
#
# $$\boxed{\;e^{-i\frac{\pi}{2}J_z^2}\vert+x\rangle^{\otimes N}
#  =\frac{1}{\sqrt2}\left(e^{-i\pi/4}\vert+x\rangle^{\otimes N}
#   +(-1)^{N/2}e^{+i\pi/4}\vert-x\rangle^{\otimes N}\right)\;}\qquad(N\text{ even}). \tag{3}$$
#
# This is a **GHZ state in the $x$ basis**: two coherent spin states pointing at opposite poles of the $x$ axis, in a
# coherent superposition with a relative phase $\pm i$. Up to the local basis change that maps $\hat x$ to $\hat z$
# (a Hadamard on every qubit), it is exactly the state of notebook 32.
#
# ### 4.2 Odd $N$: a cat along $\hat y$
#
# For $N$ odd, $m$ is a **half-integer**: write $m=\ell+\tfrac12$ with $\ell$ an integer. Then
#
# $$m^2=\ell^2+\ell+\tfrac14=\ell(\ell+1)+\tfrac14,$$
#
# and $\ell(\ell+1)$ is the product of two consecutive integers, hence always even. So $m^2-\tfrac14$ is an even integer
# and $e^{-i\frac{\pi}{2}m^2}=e^{-i\pi/8}(-1)^{\ell(\ell+1)/2}$ again takes only two values, $\pm e^{-i\pi/8}$. But which
# sign occurs follows the pattern $+,-,-,+$ for $m=\tfrac12,\tfrac32,\tfrac52,\tfrac72$, which is not the parity of $m$
# (and $(-1)^m=e^{i\pi m}=\pm i$ is not even a sign for half-integer $m$), so the decomposition of Eq. (1) does not
# carry over. Instead, examine the periodicity. For half-integer $m$,
#
# $$\frac{e^{-i\frac{\pi}{2}(m+2)^2}}{e^{-i\frac{\pi}{2}m^2}}=e^{-i\frac{\pi}{2}(4m+4)}=e^{-2\pi i m}e^{-2\pi i}=-1,$$
#
# using $e^{-2\pi i m}=-1$ for half-integer $m$. So the phase has period $4$ in $m$ and changes sign under $m\to m+2$. A
# function with these two properties can be written with only the two "odd" Fourier modes of period $4$,
#
# $$e^{-i\frac{\pi}{2}m^2}=c_+e^{+i\frac{\pi}{2}m}+c_-e^{-i\frac{\pi}{2}m}.$$
#
# Fixing $c_\pm$ from $m=\pm\tfrac12$: both give $e^{-i\pi/8}$, and the two equations
# $c_+e^{i\pi/4}+c_-e^{-i\pi/4}=e^{-i\pi/8}$ and $c_+e^{-i\pi/4}+c_-e^{i\pi/4}=e^{-i\pi/8}$ force $c_+=c_-=c$ and then
# $2c\cos(\pi/4)=e^{-i\pi/8}$, i.e. $c=e^{-i\pi/8}/\sqrt2$. (Check at $m=\tfrac32$: the left side is
# $e^{-i9\pi/8}=e^{+i7\pi/8}$ and the right side is $2c\cos(3\pi/4)=-e^{-i\pi/8}=e^{i7\pi/8}$.) Therefore
#
# $$e^{-i\frac{\pi}{2}J_z^2}=\frac{e^{-i\pi/8}}{\sqrt2}\left(e^{+i\frac{\pi}{2}J_z}+e^{-i\frac{\pi}{2}J_z}\right)
# \qquad(N\text{ odd}). \tag{4}$$
#
# The two terms are ordinary rotations about $\hat z$ by $\mp\pi/2$, which carry $\hat x$ to $\mp\hat y$:
#
# $$e^{-i\frac{\pi}{2}J_z}\vert+x\rangle^{\otimes N}=e^{-iN\pi/4}\vert+y\rangle^{\otimes N},\qquad
#   e^{+i\frac{\pi}{2}J_z}\vert+x\rangle^{\otimes N}=e^{+iN\pi/4}\vert-y\rangle^{\otimes N},$$
#
# since $e^{-i\frac{\pi}{4}\sigma^z}\vert+x\rangle=e^{-i\pi/4}(\vert0\rangle+i\vert1\rangle)/\sqrt2=e^{-i\pi/4}\vert+y\rangle$.
# Hence
#
# $$\boxed{\;e^{-i\frac{\pi}{2}J_z^2}\vert+x\rangle^{\otimes N}
#  =\frac{e^{-i\pi/8}}{\sqrt2}\left(e^{+iN\pi/4}\vert-y\rangle^{\otimes N}+e^{-iN\pi/4}\vert+y\rangle^{\otimes N}\right)\;}
#  \qquad(N\text{ odd}). \tag{5}$$
#
# Same structure, different axis: for odd $N$ the cat lies along $\hat y$, rotated by $90^\circ$ from the even case. The
# reason is the half-integer spectrum — the same reason a spin-$1/2$ needs a $4\pi$ rotation to return to itself.
#
# ### 4.3 Verification
#
# Equations (3) and (5) are exact statements about a specific state vector. We check them by fidelity,
# $F=\vert\langle\psi_{\rm predicted}\vert\psi_{\rm simulated}\rangle\vert^2$, which must be $1$ to machine precision —
# and we also check the overlap itself, which must be exactly $1$, i.e. even the *global phase* of Eqs. (3) and (5) is
# right.

# %%
# ==============================================================================
# STEP 1: the predicted cat state, and its fidelity with the simulated one
# ==============================================================================
def predicted_cat(N):
    """The right-hand side of Eq. (3) (N even) or Eq. (5) (N odd), built as a product-state superposition.

    MATH  N even:  (e^{-i pi/4} |+x>^N + (-1)^{N/2} e^{+i pi/4} |-x>^N)/sqrt(2)
          N odd :  (e^{-i pi/8}/sqrt2)( e^{+i N pi/4} |-y>^N + e^{-i N pi/4} |+y>^N )
    The engine's product_state uses 'r' for the +1 eigenstate of Y and 'l' for the -1 one.
    """
    if N % 2 == 0:
        return (jnp.exp(-1j * jnp.pi / 4) * product_state("+" * N)
                + (-1) ** (N // 2) * jnp.exp(1j * jnp.pi / 4) * product_state("-" * N)) / jnp.sqrt(2.0)
    return jnp.exp(-1j * jnp.pi / 8) / jnp.sqrt(2.0) * (
        jnp.exp(1j * N * jnp.pi / 4) * product_state("l" * N)
        + jnp.exp(-1j * N * jnp.pi / 4) * product_state("r" * N))


print(f"{'N':>4s} {'parity':>7s} | {'fidelity with Eq. (3)/(5)':>26s} {'overlap (with phase)':>26s} "
      f"{'F_Q max':>9s} {'N^2':>6s} {'cat axis':>22s}")
for N in range(4, 12):
    psi_cat = oat_evolve(product_state("+" * N), np.pi / 2)
    pred = predicted_cat(N)
    ov = complex(jnp.vdot(pred, psi_cat))
    fmax, nopt = optimal_direction(psi_cat)
    print(f"{N:4d} {'even' if N % 2 == 0 else 'odd':>7s} | {abs(ov) ** 2:26.14f} "
          f"{f'{ov.real:+.10f}{ov.imag:+.1e}i':>26s} {float(fmax):9.4f} {N ** 2:6d} "
          f"{np.array2string(fix_sign(nopt), precision=3, floatmode='fixed'):>22s}")
    assert abs(abs(ov) ** 2 - 1.0) < 1e4 * TOL
    assert abs(ov - 1.0) < 1e4 * TOL                     # the global phase too: a fidelity check alone would miss it
    assert abs(float(fmax) - N ** 2) < 1e-6 * N ** 2
    axis = np.array([1.0, 0.0, 0.0]) if N % 2 == 0 else np.array([0.0, 1.0, 0.0])
    assert abs(abs(float(np.dot(np.array(nopt), axis))) - 1.0) < 1e-6   # cat axis: x for even N, y for odd N

# %% [markdown]
# Every fidelity is $1$ to fourteen digits, and every overlap is $1+O(10^{-16})i$: Eqs. (3) and (5) reproduce the simulated state
# *including* its global phase, and the cell asserts the overlap itself as well as its modulus. The quantum Fisher
# information in the optimal direction is exactly $N^2$ — the Heisenberg limit — and the optimal direction read off the
# $3\times3$ QFI matrix is $\hat x$ for even $N$ and $\hat y$ for odd $N$ (also asserted), exactly the axes the
# derivation predicts.
#
# > **Physics insight.** Equation (2) says that at $\mu=\pi/2$ the many-body propagator collapses to
# > $\alpha\mathbb 1+\beta\prod_qZ_q$ — a superposition of "do nothing" and "flip every spin". This is why the resulting
# > entanglement involves all $N$ qubits but is *of the simplest possible kind*: the two branches are orthogonal product
# > states on both sides of any cut, so the state has Schmidt rank $2$ with equal weights across every bipartition, i.e.
# > exactly one bit of entanglement entropy — far below the $\min(N_A,N_B)$ bits a cut allows — while carrying $F_Q=N^2$. Metrological usefulness is not
# > entanglement entropy (notebook 29, Section 12).

# %% [markdown]
# ## 5. The same state from a circuit of commuting $ZZ$ gates
#
# Nothing above depends on how the propagator is implemented, but the experiment does. Using
# $J_z^2=\tfrac{N}{4}+\tfrac12\sum_{i<j}Z_iZ_j$,
#
# $$e^{-i\mu J_z^2}=e^{-i\mu N/4}\prod_{i<j}e^{-i\frac{\mu}{2}Z_iZ_j}=e^{-i\mu N/4}\prod_{i<j}R_{ZZ}(\mu)_{ij},$$
#
# with $R_{ZZ}(\theta)=e^{-i\theta Z\otimes Z/2}$ the engine's `rzz`. All the $Z_iZ_j$ are diagonal, hence mutually
# commuting, so the factorisation is an identity: the Trotter error is exactly zero and the gate order is irrelevant.
# At $\mu=\pi/2$ each gate is $R_{ZZ}(\pi/2)=\left(\mathbb 1-iZ\otimes Z\right)/\sqrt2$, a maximally entangling
# two-qubit gate — the cat is built from $N(N-1)/2$ of them.

# %%
# ==============================================================================
# STEP 2: diagonal evolution versus the ZZ-gate circuit
# ==============================================================================
def oat_circuit(psi, mu):
    """exp(-i mu J_z^2) as a circuit of N(N-1)/2 commuting ZZ gates plus a global phase.

    MATH   exp(-i mu J_z^2) = exp(-i mu N/4) prod_{i<j} rzz(mu)_{ij},   rzz(th) = exp(-i th ZZ/2).
    COST   O(N^2) gates of O(2^N) each;  the diagonal version costs O(2^N) in total.
    ERROR  exactly zero -- the terms commute, so the factorisation is an identity, not a Trotter step.
    """
    N = psi.ndim
    gate = rzz(mu)
    for i in range(N):
        for j in range(i + 1, N):
            psi = apply_gate(psi, gate, [i, j])
    return jnp.exp(-1j * mu * N / 4) * psi


print(f"{'N':>4s} {'mu':>8s} | {'max |diagonal - circuit|':>25s} {'fidelity circuit vs Eq. (3)/(5)':>32s}")
for N in (5, 8, 9):
    for mu in (0.3, np.pi / 4, np.pi / 2):
        psi0 = product_state("+" * N)
        e = max_abs(oat_evolve(psi0, mu) - oat_circuit(psi0, mu))
        is_cat = abs(mu - np.pi / 2) < 1e-12
        f = (f"{abs(complex(jnp.vdot(predicted_cat(N), oat_circuit(psi0, mu)))) ** 2:.14f}"
             if is_cat else "-  (not the cat time)")
        print(f"{N:4d} {mu:8.5f} | {e:25.3e} {f:>32s}")
        assert e < 1e4 * TOL

# %% [markdown]
# The two implementations agree to $10^{-15}$, and the circuit version reaches the predicted cat with fidelity $1$. The
# circuit is $O(N^2)$ times more expensive to simulate and is the only one a real device can run.

# %% [markdown]
# ## 6. Tracking the cat: the QFI matrix along the whole evolution
#
# ### 6.1 What the $3\times3$ matrix tells us
#
# For a collective generator $G=\mathbf n\cdot\mathbf J$ and a pure state, notebook 29 derived
#
# $$F_Q(\mathbf n)=\mathbf n^{\mathsf T}\mathcal F\,\mathbf n,\qquad
#   \mathcal F_{ab}=4\left[\tfrac12\langle J_aJ_b+J_bJ_a\rangle-\langle J_a\rangle\langle J_b\rangle\right],$$
#
# so maximising over the sphere of directions is a $3\times3$ eigenvalue problem. Evaluated at every twisting angle this
# gives two curves: $F_Q^{\max}(\mu)=\lambda_{\max}(\mathcal F)$, the best metrological performance the state can offer,
# and $\mathbf n_{\rm opt}(\mu)$, the direction that offers it. For a cat state $\mathbf n_{\rm opt}$ *is* the axis joining
# the two lobes, because that is the direction in which the two branches are maximally distinguishable.
#
# ### 6.2 Four features to look for
#
# 1. $\mu=0$: a coherent spin state, $F_Q^{\max}=N$ (the standard quantum limit), $\mathbf n_{\rm opt}$ anywhere
#    perpendicular to $\hat x$.
# 2. The squeezing regime, $\mu\lesssim N^{-2/3}$: $F_Q^{\max}$ at the squeezing optimum grows like $N^{5/3}$ —
#    faster than the SQL, slower than Heisenberg — with slowly decaying corrections (derived below).
# 3. A broad **plateau** over most of the evolution, where the state is a superposition of several coherent states. We
#    will measure its height relative to $N^2$.
# 4. $\mu=\pi/2$: $F_Q^{\max}=N^2$ exactly, by Section 4.
#
# **Feature 2 in detail.** Notebook 33 derived the bound $F_Q^{\max}\ge N/\xi_R^2$ and the asymptotic optimum
# $\xi_R^2\simeq\tfrac12(3/N)^{2/3}$ at $\mu_{\rm opt}\simeq3^{1/6}N^{-2/3}$, so
# $N/\xi_R^2\simeq2\cdot3^{-2/3}N^{5/3}\approx0.96\,N^{5/3}$. A lower bound alone does not fix the exponent of $F_Q$;
# the anti-squeezed variance does. For a pure state $F_Q^{\max}=4V_+$, with $V_+$ the larger eigenvalue of the
# transverse covariance (notebook 33, Section 9.5): $V_+=\tfrac N4\big[1+\tfrac{N-1}{4}\big(A+\sqrt{A^2+B^2}\big)\big]$
# with $A=1-\cos^{N-2}2\mu\simeq2N\mu^2$ and $B=4\sin\mu\cos^{N-2}\mu\simeq4\mu$. At $\mu_{\rm opt}$ one has
# $B/A\sim N^{-1/3}\to0$, hence $V_+\simeq\tfrac N4(1+N^2\mu^2)$ and
#
# $$F_Q^{\max}(\mu_{\rm opt})\simeq N+3^{1/3}N^{5/3},\qquad
#   \frac{F_Q^{\max}}{N/\xi_R^2}\;\longrightarrow\;\frac{3^{1/3}}{2\cdot3^{-2/3}}=\frac32 .$$
#
# Both quantities grow like $N^{5/3}$ asymptotically, but the corrections are of relative order $N^{-1/3}$ and decay very
# slowly. Step 4 below measures both.

# %%
# ==============================================================================
# STEP 3: F_Q^max and the optimal direction at every twisting angle
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
N_TRACK = (8, 12, 16)                         # sizes for the QFI-versus-time scan
MU_TRACK = np.linspace(0.0, np.pi / 2, 121)   # the whole evolution
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def track_step(N, mu):
    """(F_Q^max, n_opt, xi_R^2) of exp(-i mu J_z^2)|+x>^N -- one spin_moments call, O(N 2^N)."""
    psi = oat_evolve(product_state("+" * N), mu)
    mean, cov = spin_moments(psi)
    w, v = jnp.linalg.eigh(4.0 * cov)
    return w[-1], v[:, -1], spin_squeezing(psi)


track, t0 = {}, time.time()
for N in N_TRACK:
    out = [track_step(N, float(m)) for m in MU_TRACK]
    track[N] = (np.array([float(o[0]) for o in out]),
                np.array([np.array(o[1]) for o in out]),
                np.array([float(o[2]) for o in out]))
print(f"(scanned {len(N_TRACK)} sizes x {len(MU_TRACK)} twisting angles in {time.time() - t0:.1f} s)\n")

print(f"{'N':>4s} | {'F(0)':>8s} {'F(pi/2)':>10s} {'N^2':>7s} | {'plateau F/N^2':>14s} {'1/2+1/(2N)':>11s} "
      f"{'max F/N^2 in (0.1,0.4)pi':>25s} | {'n_opt at pi/2':>22s}")
plateau = {}
for N in N_TRACK:
    F, nvec, _ = track[N]
    sel = (MU_TRACK > 0.20 * np.pi) & (MU_TRACK < 0.40 * np.pi)
    plateau[N] = float(np.median(F[sel] / N ** 2))
    sel2 = (MU_TRACK > 0.10 * np.pi) & (MU_TRACK < 0.40 * np.pi)
    print(f"{N:4d} | {F[0]:8.4f} {F[-1]:10.4f} {N ** 2:7d} | {plateau[N]:14.4f} {0.5 + 0.5 / N:11.4f} "
          f"{float(np.max(F[sel2] / N ** 2)):25.4f} | "
          f"{np.array2string(fix_sign(nvec[-1]), precision=3, floatmode='fixed'):>22s}")
    assert abs(F[0] - N) < 1e-6 * N and abs(F[-1] - N ** 2) < 1e-6 * N ** 2

# %%
# ==============================================================================
# STEP 4: the squeezing-regime growth of F_Q with N  (feature 2)
# ==============================================================================
def refine_argmin(x, y):
    """Parabolic refinement of a discrete minimum on an EQUALLY SPACED abscissa (notebook 33).

    MATH   x* = x_i - (h/2)(y_{i+1}-y_{i-1}) / (y_{i+1} - 2 y_i + y_{i-1}),  i = argmin(y).
    A grid of 21 points locates a minimum only to half a spacing, which is not enough for a
    power-law fit; the vertex of the local parabola removes most of that discretisation error.
    """
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    h = float(x[1] - x[0])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    curv = c - 2 * b + a
    return float(x[i]) + (-0.5 * (c - a) / curv) * h if curv > 0 else float(x[i])


N_GROW = (6, 8, 10, 12, 14, 16)
rows_grow = []
for N in N_GROW:
    grid = np.linspace(0.45, 2.0, 21) * 3 ** (1 / 6) * N ** (-2 / 3)
    vals = np.array([[float(x) for x in track_step(N, float(m))[::2]] for m in grid])   # (F_Q, xi_R^2)
    mu_star = refine_argmin(grid, vals[:, 1])            # best Wineland squeezing, refined
    F_star, xi_star = (float(x) for x in track_step(N, mu_star)[::2])
    rows_grow.append((N, mu_star, xi_star, F_star, N / xi_star))
print(f"{'N':>4s} | {'mu_opt':>8s} {'xi_R^2':>9s} {'F_Q at mu_opt':>14s} {'N/xi_R^2':>10s} "
      f"{'F/N':>8s} {'F/N^2':>8s}")
for N, m, x, F, b in rows_grow:
    print(f"{N:4d} | {m:8.5f} {x:9.5f} {F:14.4f} {b:10.4f} {F / N:8.4f} {F / N ** 2:8.4f}")
    assert F >= b * (1 - 1e-8)                      # F_Q >= N/xi_R^2 (quantum Cramer-Rao bound, notebook 33)
Ng = np.array([r[0] for r in rows_grow], dtype=float)
sl_F = np.polyfit(np.log(Ng), np.log([r[3] for r in rows_grow]), 1)[0]
sl_b = np.polyfit(np.log(Ng), np.log([r[4] for r in rows_grow]), 1)[0]
print(f"\npower-law fits over N = {int(Ng[0])}..{int(Ng[-1])}:")
print(f"   F_Q at the squeezing optimum ~ N^({sl_F:+.4f})      (asymptotic exponent 5/3 = {5 / 3:.4f})")
print(f"   N/xi_R^2 at the same point   ~ N^({sl_b:+.4f})")
# a single fit hides a drifting exponent: print the LOCAL slopes between consecutive sizes
lnN = np.log(Ng)
loc_F = np.diff(np.log([r[3] for r in rows_grow])) / np.diff(lnN)
loc_b = np.diff(np.log([r[4] for r in rows_grow])) / np.diff(lnN)
print("local slopes d ln(.)/d ln N between consecutive sizes:")
print("   F_Q      : " + "  ".join(f"{s:.3f}" for s in loc_F))
print("   N/xi_R^2 : " + "  ".join(f"{s:.3f}" for s in loc_b))

# %%
# ==============================================================================
# FIGURE: the metrological gain along the whole one-axis-twisting evolution
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))

for k, N in enumerate(N_TRACK):
    F, nvec, xiR2 = track[N]
    axes[0].plot(MU_TRACK / np.pi, F / N ** 2, "-", color=PALETTE[k], lw=1.9, label=f"$N={N}$")
    axes[0].axhline(1.0 / N, color=PALETTE[k], ls=":", lw=1.0)
axes[0].axhline(1.0, color="k", ls="--", lw=1.2)
axes[0].axhline(0.5, color="0.5", ls="-.", lw=1.1)
axes[0].text(0.03, 1.03, "Heisenberg limit $N^2$", fontsize=8, color="0.3")
axes[0].text(0.03, 0.52, "$N^2/2$", fontsize=8, color="0.3")
axes[0].text(0.30, 1.0 / N_TRACK[0] + 0.02, "SQL ($1/N$, dotted)", fontsize=8, color="0.3")
axes[0].set_xlabel(r"$\mu/\pi$"); axes[0].set_ylabel(r"$F_Q^{\max}/N^2$")
axes[0].set_ylim(0, 1.15); axes[0].set_title("Quantum Fisher information along the evolution")
axes[0].legend(fontsize=9, loc="lower right")

Nb = 16
F, nvec, xiR2 = track[Nb]
for k, (lbl, comp) in enumerate([(r"$n_x$", 0), (r"$n_y$", 1), (r"$n_z$", 2)]):
    axes[1].plot(MU_TRACK / np.pi, [abs(fix_sign(v)[comp]) for v in nvec], "-", color=PALETTE[k], lw=1.8, label=lbl)
axes[1].set_xlabel(r"$\mu/\pi$"); axes[1].set_ylabel(r"$\vert$component of $\mathbf{n}_{\rm opt}\vert$")
axes[1].set_title(f"Direction of the optimal generator ($N={Nb}$)"); axes[1].legend(fontsize=9)

axes[2].loglog(Ng, [r[3] for r in rows_grow], "o-", color=PALETTE[0], ms=7,
               label=r"$F_Q$ at the squeezing optimum")
axes[2].loglog(Ng, [r[4] for r in rows_grow], "s--", color=PALETTE[1], ms=6, label=r"$N/\xi_R^2$ there")
axes[2].loglog(Ng, Ng, "k:", lw=1.2, label=r"SQL $N$")
axes[2].loglog(Ng, Ng ** 2, "k--", lw=1.2, label=r"Heisenberg $N^2$")
axes[2].loglog(Ng, Ng[0] ** (-5 / 3) * float(rows_grow[0][3]) * Ng ** (5 / 3), "-", color="0.55", lw=1.1,
               label=r"$\propto N^{5/3}$")
axes[2].set_xlabel("$N$"); axes[2].set_ylabel(r"$F_Q$")
axes[2].set_title("Growth in the squeezing regime"); axes[2].legend(fontsize=7.5)
fig.tight_layout(); plt.show()

# %% [markdown]
# All four features are present.
#
# * $F_Q^{\max}(0)=N$ exactly, and $F_Q^{\max}(\pi/2)=N^2$ exactly, for all three sizes.
# * Between them the curve rises fast, overshoots, and settles on a **plateau**. Measured as the median of $F_Q/N^2$ over
#   $0.20\pi<\mu<0.40\pi$, the plateau is $0.577$ at $N=8$, $0.543$ at $N=12$ and $0.531$ at $N=16$ — drifting towards
#   $1/2$ as $N$ grows. So the multi-component states of the middle of the evolution carry $F_Q\approx N^2/2$: half the
#   Heisenberg value, which is still $N/2$ times the standard quantum limit. The reason is geometric. Model the state
#   as an equal-weight superposition of $q\ge3$ non-overlapping coherent spin states pointing along equatorial unit
#   vectors $\mathbf e_p$ at equally spaced azimuths. For an equatorial generator $\mathbf n\cdot\mathbf J$, lobe $p$
#   contributes $\langle(\mathbf n\cdot\mathbf J)^2\rangle_p=\tfrac{N^2}{4}(\mathbf n\cdot\mathbf e_p)^2
#   +\tfrac N4\big[1-(\mathbf n\cdot\mathbf e_p)^2\big]$ (its mean squared plus its own coherent-state variance), the
#   mean $\langle\mathbf n\cdot\mathbf J\rangle$ vanishes, and the average of $(\mathbf n\cdot\mathbf e_p)^2=\cos^2$
#   over $q\ge3$ equally spaced angles is exactly $\tfrac12$. Hence $F_Q=4\mathrm{Var}=\tfrac{N^2}{2}+\tfrac N2$, i.e.
#   $F_Q/N^2=\tfrac12+\tfrac1{2N}$ — the column printed next to the plateau: $0.5625$, $0.5417$, $0.5312$ against the
#   measured $0.577$, $0.543$, $0.531$. The model is close at $N=12$ and $16$; at $N=8$ the lobes still overlap.
# * The middle panel tracks $\mathbf n_{\rm opt}$ at $N=16$. It starts in the $y$–$z$ plane (for a CSS every direction
#   there is equally good, and the numerical eigenvector picks one), rotates towards $\hat y$ as the ellipse shears, and
#   then **jumps discontinuously to $\hat x$ at $\mu\approx0.30\pi$**. The jump is not a numerical artefact: it is the
#   moment at which two eigenvalues of $\mathcal F$ cross, and the *largest* eigenvalue changes branch. From there on the
#   optimal generator is $J_x$, the cat axis of Eq. (3), all the way to $\mu=\pi/2$.
# * In the squeezing regime the single power-law fit over $6\le N\le16$ gives $F_Q\sim N^{1.668}$, numerically within
#   $0.1\%$ of the asymptotic $N^{5/3}=N^{1.667}$. This agreement is a coincidence of the fitting range, and the local
#   slopes printed below the fit show it: $d\ln F_Q/d\ln N$ rises from $1.63$ ($N=6\to8$) to $1.70$ ($N=14\to16$), and
#   the straight-line fit simply averages them. The analytic moments of notebook 33 (Section 9) continue the trend: the
#   local slope keeps rising to about $1.75$ near $N\sim100$ and comes back down to $5/3$ only for $N\gtrsim10^5$, the
#   slow $N^{-1/3}$ corrections of Section 6.2. The quantity $N/\xi_R^2$ at the same twisting angle has the smaller
#   fitted exponent $1.607$, with local slopes rising from $1.56$ to $1.65$. The bound $F_Q\ge N/\xi_R^2$ is respected
#   at every size (the cell asserts it) and is not tight even at the Wineland optimum: the two columns differ by $19\%$
#   at $N=6$ and by $26\%$ at $N=16$, on their way to the asymptotic ratio $3/2$ derived in Section 6.2.
#
# > **Numerical practice.** "Read the eigenvector as well as the eigenvalue." The $3\times3$ QFI matrix costs three
# > applications of a collective operator, and it answers two questions at once: *how good* the state is and *what to do
# > with it*. An optimisation over the sphere would have given only the first, more slowly, and with no certificate of
# > global optimality.

# %% [markdown]
# ## 7. Intermediate times: $q$-component cats
#
# At $\mu=\pi/q$ with integer $q$ the phase $e^{-i\mu m^2}=e^{-i\pi m^2/q}$ is a periodic function of $m$ and, exactly as
# in Section 4, a periodic function can be expanded in a *finite* number of Fourier modes $e^{i2\pi mp/q}$. Each mode is a
# rotation about $\hat z$ by a fixed angle, so the state is a superposition of a few coherent spin states arranged
# around the equator. The number of components is $q$ (for $q$ even) or $q$ with a different arrangement (for $q$ odd);
# the coefficients are quadratic Gauss sums. We do not need the coefficients — we can look.
#
# The Husimi-$Q$ function, $Q(\theta,\varphi)=\vert\langle\theta,\varphi\vert\psi\rangle\vert^2$, is the overlap of the
# state with the coherent spin state pointing in each direction; it is non-negative and it is what "the distribution of
# the spin direction" means. Since $\vert\theta,\varphi\rangle$ is a product state, $Q$ is a contraction of $\psi$ with
# the same $2$-vector on every axis, and the whole grid of directions is one `vmap`.

# %%
# ==============================================================================
# STEP 5: Husimi-Q maps of the q-component cats
# ==============================================================================
def css_overlap(psi, theta, phi):
    """<theta,phi|psi> for the coherent spin state |theta,phi> = (cos(th/2)|0> + e^{i ph} sin(th/2)|1>)^N.

    EINSUM  N=3:  "a,b,c,abc->"  with operands conj(v), conj(v), conj(v), psi.
            Implemented as N successive contractions of the leading axis: O(2^{N+1}) in total.
    """
    v = jnp.conj(jnp.stack([jnp.cos(theta / 2), jnp.exp(1j * phi) * jnp.sin(theta / 2)]).astype(CDTYPE))
    out = psi
    for _ in range(psi.ndim):
        out = jnp.tensordot(v, out, axes=([0], [0]))
    return out


def husimi_q(psi, theta_grid, phi_grid):
    """Q(theta, phi) on a 2D grid; nested vmap -> one fused program, no Python loop over grid points."""
    single = lambda th, ph: jnp.abs(css_overlap(psi, th, ph)) ** 2
    return jax.vmap(jax.vmap(single, in_axes=(0, 0)), in_axes=(0, 0))(theta_grid, phi_grid)


# PARAMETERS ------------------------------------------------------------------
N_Q = 12                     # size for the Bloch-sphere pictures
N_PHI, N_COS = 161, 81       # grid resolution (equal-area projection: azimuth x cos(theta))
# -----------------------------------------------------------------------------
phi_ax = np.linspace(-np.pi, np.pi, N_PHI)
cos_ax = np.linspace(-1.0, 1.0, N_COS)
PHI_M, COS_M = np.meshgrid(phi_ax, cos_ax)
THETA_M = jnp.asarray(np.arccos(np.clip(COS_M, -1.0, 1.0)))
PHI_J = jnp.asarray(PHI_M)
psi_q0 = product_state("+" * N_Q)

def count_equatorial_lobes(Qmap, rel=0.25):
    """Number of local maxima of Q along the equator, with periodic boundaries.

    A 'lobe' is a grid point on the cos(theta) = 0 row that is larger than both neighbours
    (cyclically) and at least `rel` times the row maximum -- the threshold suppresses the tiny
    interference ripples between well-separated components.
    """
    row = Qmap[Qmap.shape[0] // 2][:-1]              # drop the duplicated phi = +pi column
    left, right = np.roll(row, 1), np.roll(row, -1)
    return int(np.sum((row > left) & (row >= right) & (row > rel * row.max())))


fig, axes = plt.subplots(2, 3, figsize=(13.2, 5.6))
qs = [2, 3, 4, 5, 6, 8]
print(f"{'q':>3s} {'mu = pi/q':>10s} | {'lobes on the equator':>21s} {'F_Q^max/N^2':>13s} "
      f"{'lobe separation 2pi/q':>22s} {'CSS width 1/sqrt(N)':>20s}")
for ax, q in zip(axes.ravel(), qs):
    mu = np.pi / q
    psi_mu = oat_evolve(psi_q0, mu)
    Qmap = np.array(husimi_q(psi_mu, THETA_M, PHI_J))
    n_lobes = count_equatorial_lobes(Qmap)
    fq = float(optimal_direction(psi_mu)[0])
    print(f"{q:3d} {mu:10.5f} | {n_lobes:21d} {fq / N_Q ** 2:13.4f} {2 * np.pi / q:22.4f} "
          f"{1 / np.sqrt(N_Q):20.4f}")
    if q <= 5:
        assert n_lobes == q                          # resolved components: exactly q lobes (Fourier argument)
    ax.pcolormesh(phi_ax, cos_ax, Qmap, shading="auto", cmap="magma", vmin=0.0)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_title(rf"$\mu=\pi/{q}$:  {n_lobes} lobe" + ("s" if n_lobes != 1 else "")
                 + rf",   $F_Q^{{\max}}/N^2={fq / N_Q ** 2:.3f}$", fontsize=9)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-1, 0, 1])
for ax in axes[1]:
    ax.set_xlabel(r"azimuth $\varphi$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\cos\theta$")
fig.suptitle(f"Husimi $Q$ at $\\mu=\\pi/q$: $q$-component cat states, $N={N_Q}$", fontsize=11)
fig.tight_layout(); plt.show()

# %% [markdown]
# The lobe counts printed above come from the equatorial cut of each map, not from the eye: a grid point on the
# $\cos\theta=0$ row counts as a lobe if it exceeds both cyclic neighbours and a quarter of the row maximum. The
# azimuth is periodic, so the columns $\varphi=-\pi$ and $\varphi=+\pi$ of each map are the same meridian and the
# counter drops the duplicate: the $q=2$ panel shows three bright blobs but only two distinct lobes. At $N=12$
# the counter finds $2,3,4,5$ lobes for $q=2,3,4,5$ — exactly $q$, as the Fourier argument says — and then $4$ for
# $q=6$ and $1$ for $q=8$.
#
# The last two columns explain the breakdown. The components are separated by $2\pi/q$ in azimuth, while each of them is
# a rotated copy of the original coherent state and therefore has an angular width $\sim1/\sqrt N=0.289$ rad, further
# stretched along the shear direction. At $q=6$ the separation is $1.047$ rad, about $3.6$ lobe widths before the shear
# is taken into account, and the maps at $\mu=\pi/6$ and $\mu=\pi/8$ show a sheared band with interference holes instead
# of resolved components. The Fourier argument bounds the number of components; *resolving* them needs a large $N$, and
# $N=12$ is enough only up to $q=5$.
#
# The metrological price of splitting the state is in the titles. $F_Q^{\max}/N^2$ is exactly $1.000$ for the
# two-component cat and sits between $0.54$ and $0.58$ for every $q\ge3$: two lobes at antipodes of the measurement axis
# give the maximal spread of $\mathbf n\cdot\mathbf J$, $\pm N/2$, while lobes distributed around a circle give an
# average of $\cos^2$, which is $1/2$. This is the origin of the plateau seen in Section 6, and it explains why the
# plateau drifts *down* towards $1/2$ as $N$ grows: the coherent-state variance of each lobe adds the correction
# $1/(2N)$ derived there, $0.5417$ at $N=12$, against the measured $0.5421$ ($q=3$) and $0.5426$ ($q=4$).

# %% [markdown]
# ## 8. Encoding and parity readout: reaching the Heisenberg limit from data
#
# ### 8.1 The fringe
#
# Take the cat $\vert C\rangle=\left(e^{-i\pi/4}\vert{+}\mathbf n\rangle^{\otimes N}+\epsilon\,e^{+i\pi/4}\vert{-}\mathbf n\rangle^{\otimes N}\right)/\sqrt2$
# with $\mathbf n$ the cat axis ($\hat x$ for even $N$, $\hat y$ for odd $N$, as Section 4 proved and Section 6 confirmed
# numerically) and $\epsilon=\pm1$: Eq. (3) has exactly this form with $\epsilon=(-1)^{N/2}$, and Eq. (5) has it up to a
# global phase with $\epsilon=(-1)^{(N-1)/2}$. Encode the phase with the optimal generator $G=\mathbf n\cdot\mathbf J$. Since
# $\vert\pm\mathbf n\rangle^{\otimes N}$ are eigenstates of $G$ with eigenvalues $\pm N/2$,
#
# $$e^{-i\theta G}\vert C\rangle=\frac{1}{\sqrt2}\left(e^{-i\theta N/2}e^{-i\pi/4}\vert{+}\mathbf n\rangle^{\otimes N}
#  +e^{+i\theta N/2}\epsilon\,e^{+i\pi/4}\vert{-}\mathbf n\rangle^{\otimes N}\right). \tag{6}$$
#
# The two branches acquire a **relative phase $N\theta$**: the whole point of the Heisenberg limit is this factor $N$,
# which comes from $N$ particles each collecting $\theta$ *coherently*.
#
# ### 8.2 Why parity reads it out
#
# A relative phase between two orthogonal branches is invisible in any measurement that distinguishes them. It becomes
# visible in a measurement of an observable that *exchanges* them. The **parity**
#
# $$\Pi=\prod_{q=0}^{N-1}Z_q$$
#
# does exactly that whenever $\mathbf n\perp\hat z$: $Z$ anticommutes with both $\sigma^x$ and $\sigma^y$, so
# $Z_q\vert\pm\mathbf n\rangle_q=\vert\mp\mathbf n\rangle_q$ and $\Pi\vert\pm\mathbf n\rangle^{\otimes N}=\vert\mp\mathbf n\rangle^{\otimes N}$.
# The expectation value of $\Pi$ in the state of Eq. (6) is therefore a pure interference term, an oscillation of period
# $2\pi/N$ in $\theta$. And $\Pi$ is *diagonal in the computational basis*: measuring it costs nothing beyond the
# standard readout, because $\Pi=(-1)^{k}$ with $k$ the number of qubits found in $\vert1\rangle$.
#
# The measurement has two outcomes, $\Pi=\pm1$, so the model is a two-outcome model, and its classical Fisher information
# follows from the formula of notebook 29: for $p_\pm(\theta)=\left(1\pm\mathcal{A}\sin(N\theta+\varphi_0)\right)/2$ with
# visibility $\mathcal{A}$,
#
# $$I(\theta)=\frac{\left(\partial_\theta p_+\right)^2}{p_+}+\frac{\left(\partial_\theta p_-\right)^2}{p_-}
#  =\frac{\mathcal{A}^2N^2\cos^2(N\theta+\varphi_0)}{1-\mathcal{A}^2\sin^2(N\theta+\varphi_0)} . \tag{7}$$
#
# At a zero crossing of the fringe and with unit visibility, $I=N^2=F_Q$: the parity readout is optimal there.

# %%
# ==============================================================================
# STEP 6: the parity fringe, its Fisher information, and the sampled experiment
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
N_RO      = 10                                  # qubits in the readout experiment
THETA_TRUE = 0.0                                # working point (a zero crossing of the fringe)
M_SHOTS   = (25, 50, 100, 200, 400, 800)        # shots per experiment
R_EXP     = 600                                 # independent experiments per point
# -----------------------------------------------------------------------------
_PARITY = jnp.asarray(np.array([(-1) ** bin(i).count("1") for i in range(2 ** N_RO)]))

psi_cat_ro = oat_evolve(product_state("+" * N_RO), np.pi / 2)
F_Q_ro, n_cat = optimal_direction(psi_cat_ro)
n_cat = jnp.asarray(fix_sign(n_cat))
print(f"N = {N_RO}:  cat axis n = {np.array2string(np.array(n_cat), precision=4, floatmode='fixed')},"
      f"  F_Q = {float(F_Q_ro):.6f}  (N^2 = {N_RO ** 2})")


def parity_probs(theta):
    """p(Pi = +1), p(Pi = -1) after encoding the phase theta with the optimal generator.

    MATH   |psi_theta> = exp(-i theta n.J)|C> ; Pi = prod_q Z_q is DIAGONAL in the computational
           basis with eigenvalue (-1)^{number of ones}, so p(+) = sum_{s : even} |psi_theta[s]|^2 .
    """
    psi = collective_rotation(psi_cat_ro, theta, n_cat)
    pr = jnp.abs(psi.reshape(-1)) ** 2
    p_even = jnp.sum(jnp.where(_PARITY > 0, pr, 0.0))
    return jnp.stack([p_even, 1.0 - p_even])


def classical_fisher(prob_fn, theta, tol=1e-14):
    """I(theta) = sum_x (d p_x/d theta)^2 / p_x with an EXACT derivative from jax.jacfwd.

    JAX   a double `jnp.where` masks zero-probability outcomes so they contribute exactly 0 (no NaN).
    """
    p = prob_fn(theta)
    dp = jax.jacfwd(prob_fn)(theta)
    ok = p > tol
    return jnp.sum(jnp.where(ok, dp ** 2 / jnp.where(ok, p, 1.0), 0.0))


print(f"\n{'theta':>10s} | {'p(Pi=+1)':>10s} {'<Pi>':>9s} {'I(theta)':>11s} {'I/N^2':>8s}")
for th in (0.0, 0.25 * np.pi / (2 * N_RO), 0.5 * np.pi / (2 * N_RO), np.pi / (2 * N_RO)):
    p = parity_probs(th)
    print(f"{th:10.6f} | {float(p[0]):10.6f} {float(p[0] - p[1]):9.5f} "
          f"{float(classical_fisher(parity_probs, th)):11.6f} "
          f"{float(classical_fisher(parity_probs, th)) / N_RO ** 2:8.5f}")
I_parity = float(classical_fisher(parity_probs, THETA_TRUE))
print(f"\nparity readout at theta = 0:  I = {I_parity:.8f}   F_Q = {float(F_Q_ro):.8f}   "
      f"I/F_Q = {I_parity / float(F_Q_ro):.8f}")
assert abs(I_parity - float(F_Q_ro)) < 1e-6 * N_RO ** 2

# %%
# ==============================================================================
# STEP 7: from clicks to an error bar -- maximum likelihood on sampled parities
# ==============================================================================
HALF_FRINGE = np.pi / (2 * N_RO)                      # half a fringe: the unambiguous search window
THETA_GRID = jnp.linspace(THETA_TRUE - HALF_FRINGE, THETA_TRUE + HALF_FRINGE, 601)
logp_grid = jnp.log(jnp.clip(jax.vmap(parity_probs)(THETA_GRID), 1e-300, None))
p_true = jnp.clip(parity_probs(THETA_TRUE), 1e-300, None)
psi_out_true = collective_rotation(psi_cat_ro, THETA_TRUE, n_cat)      # the state the detector sees

# --- what one shot looks like -------------------------------------------------------------
bits_demo = np.array(sample_bitstrings(jax.random.PRNGKey(7), psi_out_true, 8))
print("eight shots of the cat interferometer (one row = a projective measurement of all N qubits):")
for b in bits_demo:
    print("   " + "".join(str(int(x)) for x in b) + f"    parity Pi = {+1 if b.sum() % 2 == 0 else -1:+d}")

# --- CHECKPOINT: sampled bit strings reproduce the exact parity distribution ---------------
# At the working point theta = 0 the exact p(+1) is 1/2, which ANY source of random parities would
# reproduce. The checkpoint is therefore run at a phase where p(+1) is far from 1/2, and a wrong
# control (the un-encoded cat, as if the encoding step had been skipped) must FAIL the same test.
N_DEMO_SHOTS = 20_000
THETA_CHK = np.pi / (4 * N_RO)                         # quarter of a fringe: p(+1) = (1 + sin(pi/4))/2
p_chk = float(parity_probs(THETA_CHK)[0])
sigma_chk = np.sqrt(p_chk * (1 - p_chk) / N_DEMO_SHOTS)


def sampled_even_fraction(key, psi):
    """Fraction of sampled bit strings with an even number of ones (Pi = +1)."""
    bits = np.array(sample_bitstrings(key, psi, N_DEMO_SHOTS))
    return float(np.mean(bits.sum(axis=1) % 2 == 0))


freq_even = sampled_even_fraction(jax.random.PRNGKey(8), collective_rotation(psi_cat_ro, THETA_CHK, n_cat))
freq_ctrl = sampled_even_fraction(jax.random.PRNGKey(9), psi_cat_ro)          # wrong control: no encoding
print(f"\n{N_DEMO_SHOTS} sampled bit strings at theta = pi/(4N): frequency of Pi = +1 is {freq_even:.5f},"
      f" exact p(+1) = {p_chk:.5f}   (deviation {abs(freq_even - p_chk) / sigma_chk:.2f} sigma,"
      f" sigma = {sigma_chk:.2e})")
print(f"control without the encoding step:      frequency {freq_ctrl:.5f}"
      f"   (deviation {abs(freq_ctrl - p_chk) / sigma_chk:.0f} sigma -> must fail)")
assert abs(freq_even - p_chk) < 5 * sigma_chk
assert abs(freq_ctrl - p_chk) > 5 * sigma_chk


def ml_experiments(key, shots, n_exp):
    """`n_exp` experiments of `shots` shots each; return the maximum-likelihood estimates of theta.

    PROTOCOL  one shot = measure all N qubits in the computational basis, then reduce the bit string
        to its PARITY, (number of ones) mod 2: outcome 0 means Pi = +1.  No extra hardware is needed --
        the parity is a function of the standard readout.  The cell above draws the bit strings
        explicitly with `sample_bitstrings`; here the reduced binary outcome is sampled directly from
        the same distribution, which is statistically identical and avoids materialising the 2^N
        Gumbel variates `jax.random.categorical` would need for each of the shots x n_exp strings.
    MATH  log L(theta) = n_+ log p_+(theta) + n_- log p_-(theta)  ->  one matrix-vector product.
    JAX   vmap over PRNG keys turns the repetitions into a single batched program.
    """
    lp = jnp.log(p_true)

    def one(k):
        idx = jax.random.categorical(k, lp, shape=(shots,))
        counts = jnp.bincount(idx, length=2)                           # [n(Pi=+1), n(Pi=-1)]
        return THETA_GRID[jnp.argmax(logp_grid @ counts)]

    return jax.vmap(one)(jax.random.split(key, n_exp))


t0 = time.time()
rows_ml = []
for M in M_SHOTS:
    # an independent key per M (fold_in), so that the six rows are statistically independent
    est = jax.jit(partial(ml_experiments, shots=M, n_exp=R_EXP))(jax.random.fold_in(jax.random.PRNGKey(2026), M))
    var = float(jnp.var(est))
    rows_ml.append((M, float(jnp.mean(est)) - THETA_TRUE, var, var * np.sqrt(2 / (R_EXP - 1))))
print(f"(simulated {len(M_SHOTS) * R_EXP} experiments in {time.time() - t0:.1f} s)\n")
print(f"{'M':>6s} {'bias':>11s} {'Delta theta':>13s} {'1/(N sqrt M)':>14s} {'SQL 1/sqrt(NM)':>16s} "
      f"{'M F_Q Var':>11s}")
for M, bias, var, err in rows_ml:
    print(f"{M:6d} {bias:+11.6f} {np.sqrt(var):13.6f} {1 / (N_RO * np.sqrt(M)):14.6f} "
          f"{1 / np.sqrt(N_RO * M):16.6f} {var * M * float(F_Q_ro):11.4f}")
assert abs(rows_ml[-1][2] * M_SHOTS[-1] * float(F_Q_ro) - 1.0) < 0.2

# %%
# ==============================================================================
# FIGURE: the parity fringe and the achieved sensitivity
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

th_ax = np.linspace(-np.pi / N_RO, np.pi / N_RO, 401)
par = np.array([float(parity_probs(float(t))[0] - parity_probs(float(t))[1]) for t in th_ax])
I_ax = np.array([float(classical_fisher(parity_probs, float(t))) for t in th_ax])
axes[0].plot(th_ax, par, "-", color=PALETTE[0], lw=1.9, label=r"$\langle\Pi\rangle$")
ax0b = axes[0].twinx()
ax0b.plot(th_ax, I_ax / N_RO ** 2, "--", color=PALETTE[1], lw=1.5, label=r"$I(\theta)/N^2$")
ax0b.set_ylabel(r"$I(\theta)/N^2$", color=PALETTE[1]); ax0b.grid(False); ax0b.set_ylim(-0.05, 1.15)
axes[0].axvline(THETA_TRUE, color="0.6", ls=":", lw=1.2)
axes[0].set_xlabel(r"phase $\theta$"); axes[0].set_ylabel(r"parity $\langle\Pi\rangle$", color=PALETTE[0])
axes[0].set_title(f"Parity fringe of the OAT cat, period $2\\pi/N$ ($N={N_RO}$)")
axes[0].legend(fontsize=9, loc="upper left")

Ms = np.array(M_SHOTS, dtype=float)
v = np.array([r[2] for r in rows_ml]); e = np.array([r[3] for r in rows_ml])
axes[1].errorbar(Ms, np.sqrt(v), yerr=0.5 * e / np.sqrt(v), fmt="o", color=PALETTE[0], ms=7, capsize=3,
                 label="measured (maximum likelihood)")
axes[1].plot(Ms, 1 / np.sqrt(Ms * float(F_Q_ro)), "-", color=PALETTE[0], lw=1.3,
             label=r"$1/\sqrt{M F_Q}=1/(N\sqrt{M})$")
axes[1].plot(Ms, 1 / np.sqrt(N_RO * Ms), "k--", lw=1.4, label=r"SQL $1/\sqrt{NM}$")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("shots per experiment $M$"); axes[1].set_ylabel(r"$\Delta\hat\theta$")
axes[1].set_title("Heisenberg-limited estimation from sampled parities"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The parity oscillates with period $2\pi/N$ and with **full visibility**, $\mathcal{A}=1$, so the denominator of
# Eq. (7) is $1-\sin^2=\cos^2$ and cancels the numerator exactly: $I(\theta)=N^2$ at *every* phase except the two
# extrema of the fringe, where $\cos^2(N\theta)=0$ and the readout carries no information at all (the dashed curve on
# the right axis is flat at $1$ and drops vertically to $0$ at $\theta=\pm\pi/(2N)$). The drop occurs at a single point,
# where one outcome has probability exactly $0$, the score is undefined, and $I$ need not equal the limit of its
# neighbours — the irregular model of notebook 29, Section 5.2. The table confirms it: $I=100.000$
# at $\theta=0$, $0.0393$ and $0.0785$, and $0$ at $\theta=\pi/(2N)$. At the working point $I/F_Q=1.00000000$ — the
# parity measurement is *exactly* optimal, with no need to build the SLD basis.
#
# The eight printed shots show what the apparatus records: strings of ten bits, of which only the parity is kept.
# The sampler checkpoint is run a quarter fringe away from the working point, because at $\theta=0$ the exact
# $p(\Pi=+1)=\tfrac12$ would be reproduced by any source of random parities. There, over $20\,000$ shots, the frequency
# of $\Pi=+1$ is $0.8560$ against the exact $0.8536$, a deviation of $0.98$ standard deviations; the control in which the
# encoding step is skipped gives $0.5004$, $141$ standard deviations off, and fails the same test as it must. Nothing
# about this readout is special hardware — it is the ordinary population measurement, read modulo $2$.
#
# The right panel closes the loop. The maximum-likelihood estimate from $M$ sampled parities shows no detectable bias:
# the six values scatter around zero with no trend, all below $3\times10^{-4}$ in magnitude, i.e. within about one
# Monte-Carlo error $\sqrt{\mathrm{Var}/R}$ of a mean over $R=600$ experiments ($8.3\times10^{-4}$ at $M=25$,
# $1.4\times10^{-4}$ at $M=800$). Its standard deviation follows $1/(N\sqrt M)$ across the whole range, a factor
# $\sqrt N=3.16$ below the standard quantum limit. The scaled variance $MF_Q\mathrm{Var}$ ranges over $0.945$–$1.033$,
# within the $5.8\%$ relative error of a variance measured from $600$ experiments. (Each $M$ uses its own PRNG key,
# so the six rows are statistically independent.)
#
# > **Common pitfall.** The gain is real but the *window* is narrow: the search had to be restricted to
# > $\vert\theta\vert<\pi/(2N)$, half a fringe. Outside it the likelihood has equally high peaks on neighbouring fringes
# > and the estimator jumps. Heisenberg scaling in a real device therefore needs a hierarchical protocol — a coarse,
# > SQL-limited estimate first, then the cat.

# %% [markdown]
# ## 9. Noise: the machinery
#
# Everything so far was unitary. A cat state is the most fragile object in this chapter, so the interesting question is
# what survives.
#
# ### 9.1 The model
#
# We let a single-qubit channel act on every qubit, continuously, *during* the twisting. Splitting the evolution into
# $n_{\rm step}$ intervals of length $\delta\mu=\mu/n_{\rm step}$, each step is
#
# $$\rho\;\longrightarrow\;\mathcal{E}^{\otimes N}_{p}\!\left(U_{\delta\mu}\,\rho\,U_{\delta\mu}^\dagger\right),
# \qquad U_{\delta\mu}=e^{-i\delta\mu J_z^2},\qquad p=\gamma\,\delta\mu, \tag{8}$$
#
# with $\gamma$ the dimensionless noise rate (noise probability per unit twisting angle). The unitary part is exact — it
# is the same diagonal phase multiplication, now applied to both index groups of the density tensor. Only the *splitting*
# between unitary and dissipative parts is first order in $\delta\mu$, and we check convergence in $n_{\rm step}$.
#
# The density tensor has $4^N$ entries ($4096$ complex numbers at $N=6$, a rank-$12$ tensor), and every QFI evaluation
# adds an $O(8^N)$ eigendecomposition of the $2^N\times2^N$ matrix. Neither is large at $N=6$–$8$; what sets the working
# range $N\le6$ is the number of evaluations — $12$ sweeps of $32$ steps, the convergence check and the comparison of
# Section 12 — within the time budget of the notebook.
#
# ### 9.2 The unitary step on a density tensor
#
# $U=e^{-i\mu J_z^2}$ is diagonal: $U\vert s\rangle=e^{-i\mu m(s)^2}\vert s\rangle$. Therefore
#
# $$\left(U\rho U^\dagger\right)[s;s']=e^{-i\mu\left(m(s)^2-m(s')^2\right)}\,\rho[s;s'],$$
#
# one elementwise multiplication by an outer product of the phase tensor with its conjugate. No gates, no Trotter error
# in the unitary part.
#
# ### 9.3 The QFI of a mixed state
#
# For a mixed state the pure-state formula $4\mathrm{Var}(G)$ is wrong, and the correct object is the symmetric
# logarithmic derivative $L$, defined by $\partial_\theta\rho=\tfrac12(L\rho+\rho L)$ and derived in
# [30](../ch10_quantum_metrology_protocols/30_qfi_from_the_sld_prepare_encode_estimate.ipynb). In the eigenbasis
# $\rho=\sum_m\lambda_m\vert m\rangle\langle m\vert$ and for unitary encoding $\partial_\theta\rho=-i[G,\rho]$, it gives
#
# $$F_Q[\rho,G]=2\sum_{\substack{m,n\\ \lambda_m+\lambda_n>0}}\frac{(\lambda_m-\lambda_n)^2}{\lambda_m+\lambda_n}
#  \left\vert\langle m\vert G\vert n\rangle\right\vert^2 . \tag{9}$$
#
# Two things about Eq. (9) matter here. First, it is still a **quadratic form** in the generator direction: substituting
# $G=\sum_an_aJ_a$,
#
# $$F_Q(\mathbf n)=\sum_{a,b}n_an_b\,\Gamma_{ab},\qquad
# \Gamma_{ab}=2\sum_{m,n}\frac{(\lambda_m-\lambda_n)^2}{\lambda_m+\lambda_n}\,
#   \mathrm{Re}\left[\langle m\vert J_a\vert n\rangle\overline{\langle m\vert J_b\vert n\rangle}\right], \tag{10}$$
#
# so the optimal direction is again the top eigenvector of a $3\times3$ real symmetric matrix — for mixed states as for
# pure ones. Second, the kernel of $\rho$ must be excluded, which is what the guarded division below does.
#
# We implement Eq. (9) twice: once through $L$ itself (`sld` / `qfi_from_sld`, the routines of notebook 30) and once
# through the engine's `qfi_mixed`, and check that they agree.

# %%
# ==============================================================================
# STEP 8: noisy one-axis twisting on a density tensor, and the mixed-state QFI matrix
# ==============================================================================
def jz_tensor(N):
    """m(s) = sum_q (1/2 - s_q) as a rank-N array, assembled by BROADCASTING N tiny arrays.

    Each qubit contributes jnp.array([0.5, -0.5]) reshaped to (1,..,2,..,1); their sum broadcasts
    to the full (2,)*N tensor of J_z eigenvalues without any loop over the 2^N basis states.
    """
    return sum(jnp.array([0.5, -0.5]).reshape([2 if a == q else 1 for a in range(N)]) for q in range(N))


def oat_evolve_dm(rho, mu):
    """exp(-i mu J_z^2) acting on a density TENSOR (rank 2N: ket axes 0..N-1, bra axes N..2N-1).

    MATH   (U rho U^dag)[s; s'] = exp(-i mu (m(s)^2 - m(s')^2)) rho[s; s'] .
    COST   O(4^N), exact, no Trotter error.
    """
    N = rho.ndim // 2
    ph = jnp.exp(-1j * mu * jz_tensor(N) ** 2)
    return rho * ph.reshape(ph.shape + (1,) * N) * jnp.conj(ph).reshape((1,) * N + ph.shape)


def local_channel(rho, kraus):
    """Apply the SAME single-qubit channel to every qubit of a density tensor: O(N 4^N)."""
    for q in range(rho.ndim // 2):
        rho = apply_kraus_dm(rho, kraus, [q])
    return rho


def sld(rho_mat, drho_mat, tol=1e-12):
    """Symmetric logarithmic derivative L, from notebook 30.

    MATH   rho = sum_m lam_m |m><m| ;  L_mn = 2 <m|drho|n>/(lam_m + lam_n) where lam_m + lam_n > tol,
           and L_mn = 0 inside the kernel (a free choice that does not affect F_Q).
    JAX    a DOUBLE jnp.where so the unsafe branch is never evaluated (no inf/NaN under jit).
    """
    lam, V = jnp.linalg.eigh(rho_mat)
    D = V.conj().T @ drho_mat @ V
    den = lam[:, None] + lam[None, :]
    ok = den > tol
    L_eig = jnp.where(ok, 2.0 * D / jnp.where(ok, den, 1.0), 0.0)
    return V @ L_eig @ V.conj().T, lam, V


def qfi_from_sld(rho_mat, drho_mat, tol=1e-12):
    """F_Q = Tr(rho L^2), from notebook 30 -- valid for ANY smooth family rho_theta."""
    L, _, _ = sld(rho_mat, drho_mat, tol)
    return jnp.real(jnp.trace(rho_mat @ L @ L))


def qfi_matrix_mixed(rho_mat, Js, tol=1e-12):
    """The 3x3 QFI matrix Gamma of Eq. (10) for a MIXED state and collective generators J_x, J_y, J_z.

    MATH   Gamma[a,b] = 2 sum_{m,n} (lam_m-lam_n)^2/(lam_m+lam_n) Re[ <m|J_a|n> conj(<m|J_b|n>) ]
    COST   ONE eigendecomposition O(8^N) shared by all nine entries, plus 3 basis rotations.
    """
    lam, v = jnp.linalg.eigh(rho_mat)
    num = (lam[:, None] - lam[None, :]) ** 2
    den = lam[:, None] + lam[None, :]
    w = jnp.where(den > tol, num / jnp.where(den > tol, den, 1.0), 0.0)
    Jt = [v.conj().T @ J @ v for J in Js]
    return jnp.stack([jnp.stack([2 * jnp.real(jnp.sum(w * Jt[a] * jnp.conj(Jt[b]))) for b in range(3)])
                      for a in range(3)])


# --- CHECKPOINT 1: mixed-state machinery reduces to the pure-state answers -----------------
N_CHK = 4
Js_chk = [collective_dense(P, N_CHK) for P in (X, Y, Z)]
print(f"{'state':>26s} | {'max |Gamma - 4C|':>17s} {'|qfi_from_sld - qfi_mixed|':>28s} {'purity':>8s}")
for name, mu in (("CSS (mu=0)", 0.0), ("squeezed (mu=0.2)", 0.2), ("cat (mu=pi/2)", np.pi / 2)):
    psi = oat_evolve(product_state("+" * N_CHK), mu)
    rho = to_dm(psi)
    G = collective_dense(Z, N_CHK)
    drho = -1j * (G @ dm_matrix(rho) - dm_matrix(rho) @ G)
    e1 = max_abs(qfi_matrix_mixed(dm_matrix(rho), Js_chk) - qfi_matrix(psi))
    e2 = abs(float(qfi_from_sld(dm_matrix(rho), drho)) - float(qfi_mixed(dm_matrix(rho), G)))
    print(f"{name:>26s} | {e1:17.2e} {e2:28.2e} {float(purity(dm_matrix(rho))):8.5f}")
    assert e1 < 1e4 * TOL and e2 < 1e4 * TOL

# --- CHECKPOINT 2: the same on genuinely mixed states, and against the SLD route ------------
print()
for name, kraus in (("dephasing p=0.08", kraus_dephasing(0.08)), ("depolarising p=0.08", kraus_depolarizing(0.08)),
                    ("amp. damping g=0.08", kraus_amplitude_damping(0.08))):
    rho = local_channel(to_dm(oat_evolve(product_state("+" * N_CHK), np.pi / 2)), kraus)
    G = collective_dense(X, N_CHK)
    drho = -1j * (G @ dm_matrix(rho) - dm_matrix(rho) @ G)
    f_sld = float(qfi_from_sld(dm_matrix(rho), drho))
    f_eng = float(qfi_mixed(dm_matrix(rho), G))
    f_mat = float(np.array([1.0, 0.0, 0.0]) @ np.array(qfi_matrix_mixed(dm_matrix(rho), Js_chk))
                  @ np.array([1.0, 0.0, 0.0]))
    print(f"cat + {name:>22s} | F_Q(J_x): SLD {f_sld:10.6f}   engine {f_eng:10.6f}   "
          f"quadratic form {f_mat:10.6f}   purity {float(purity(dm_matrix(rho))):.5f}")
    assert abs(f_sld - f_eng) < 1e3 * TOL and abs(f_sld - f_mat) < 1e3 * TOL

# %% [markdown]
# Three independent code paths — the SLD construction of notebook 30, the engine's closed-form sum, and the quadratic
# form of Eq. (10) — give the same quantum Fisher information on mixed states, and the mixed-state QFI matrix reduces to
# $4C_{ab}$ on pure ones. From here on we use `qfi_matrix_mixed`, whose single eigendecomposition serves all three
# directions at once.

# %% [markdown]
# ## 10. Fragility of the cat
#
# ### 10.1 Dephasing in the cat basis: an exact law
#
# Take the GHZ state $\vert G\rangle=\left(\vert0\rangle^{\otimes N}+\vert1\rangle^{\otimes N}\right)/\sqrt2$ and let each
# qubit dephase independently, $\mathcal{E}_p(\rho)=(1-p)\rho+pZ\rho Z$, which multiplies every single-qubit coherence
# by $1-2p$. The state has exactly one coherence, $\vert0\cdots0\rangle\langle1\cdots1\vert$, and it involves *all $N$
# qubits*, so it is multiplied by $(1-2p)$ once per qubit:
#
# $$\rho_p=\frac12\Big(\vert0\rangle\langle0\vert^{\otimes N}+\vert1\rangle\langle1\vert^{\otimes N}\Big)
#   +\frac{c}{2}\Big(\vert0\rangle\langle1\vert^{\otimes N}+\vert1\rangle\langle0\vert^{\otimes N}\Big),\qquad
#   c=(1-2p)^N . \tag{11}$$
#
# This $\rho_p$ has rank $2$: its eigenvectors are $\vert\pm\rangle=\left(\vert0\rangle^{\otimes N}\pm\vert1\rangle^{\otimes N}\right)/\sqrt2$
# with eigenvalues $\lambda_\pm=(1\pm c)/2$. With $G=J_z$ we have $J_z\vert+\rangle=\tfrac{N}{2}\vert-\rangle$, so the only
# non-zero matrix element is $\vert\langle+\vert J_z\vert-\rangle\vert^2=N^2/4$. Equation (9) then gives, counting the
# pairs $(+,-)$ and $(-,+)$,
#
# $$F_Q=2\cdot2\cdot\frac{(\lambda_+-\lambda_-)^2}{\lambda_++\lambda_-}\cdot\frac{N^2}{4}
#  =N^2c^2=N^2(1-2p)^{2N}. \tag{12}$$
#
# If the GHZ state is instead exposed to the channel of Eq. (8) for a duration $\mu$ — $n_{\rm step}$ steps of
# probability $\gamma\,\delta\mu$ each — every single-qubit coherence is multiplied by
# $(1-2\gamma\,\delta\mu)^{n_{\rm step}}=(1-2\gamma\mu/n_{\rm step})^{n_{\rm step}}\to e^{-2\gamma\mu}$, so
# $c=e^{-2N\gamma\mu}$ in the continuum limit and
#
# $$\boxed{\;F_Q\;=\;N^2e^{-4N\gamma\mu}\;}$$
#
# The decay rate is proportional to $N$. This is the precise statement of "cat states are exponentially fragile": the
# Heisenberg *gain* is a factor $N$, and the extra decoherence *costs* a factor $e^{-4N\gamma\mu}$, so the advantage
# survives only while $N\gamma\mu\lesssim\tfrac14\ln N$.
#
# ### 10.2 The OAT cat lies along $\hat x$, which changes the answer
#
# Equation (12) assumed that the noise dephases in the *same* basis in which the cat is written. The OAT cat of Eq. (3) is
# a GHZ state in the $x$ basis. Local $Z$ noise therefore acts on it as a **bit flip** in its own basis — and a bit flip
# does not destroy the coherence between the two branches, it moves the pair $\{\vert b\rangle,\vert\bar b\rangle\}$ to
# another pair. What destroys the $x$-cat exponentially is $X$-type noise: $X\vert\pm x\rangle=\pm\vert\pm x\rangle$, so an
# $X$ error flips the relative sign of the two branches, exactly as $Z$ does for the $z$-cat.
#
# The $Z$-dephasing case can be computed exactly. A $Z$ error on a set $S$ of $k$ qubits maps the cat onto a cat on the
# pair of $x$-basis strings $\{s,\bar s\}$ in which the qubits of $S$ are flipped; on that pair $J_x=\pm(N/2-k)$. The
# complementary set $S^c$ leads to the *same* pair with the opposite relative sign, so the two mix into a cat with
# coherence $(a_k-b_k)/(a_k+b_k)$, where $a_k=p^k(1-p)^{N-k}$ and $b_k=p^{N-k}(1-p)^k$. Since $J_x$ is diagonal in the
# $x$ basis it does not connect different pairs, and the QFI is the weighted sum over pairs:
#
# $$F_Q(J_x)=\sum_{k=0}^{N}\frac{1}{2}\binom Nk\frac{(a_k-b_k)^2}{a_k+b_k}\,(N-2k)^2
#  \;\approx\;\big(N(1-2p)\big)^2+4Np(1-p). \tag{13}$$
#
# The approximation drops $b_k$ (of order $p^{N/2}$ or smaller where it matters) and evaluates the binomial average of
# $(N-2k)^2$. The loss is polynomial in $p$: the cat survives, with its two branches shortened from $\pm N/2$ to
# $\pm N(1-2p)/2$ on average. We verify all three statements: the bit-flip channel applied to the OAT cat must reproduce
# Eq. (12), $Z$-dephasing must reproduce Eq. (13), and generators perpendicular to $\hat x$ must keep $F_Q=N$ under
# bit flips, because $X$ leaves each branch $\vert\pm x\rangle^{\otimes N}$ unchanged and each branch on its own is a
# coherent spin state.

# %%
# ==============================================================================
# STEP 9: the exponential fragility law, Eq. (12), and the role of the noise axis
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
N_FRAG = (4, 5, 6)
P_FRAG = (0.0, 0.02, 0.05, 0.10, 0.20)
# -----------------------------------------------------------------------------
print("GHZ along z + dephasing about z    vs    Eq. (12)  F_Q = N^2 (1-2p)^{2N}")
print(f"{'N':>4s} {'p':>6s} | {'F_Q (SLD)':>12s} {'Eq. (12)':>12s} {'error':>10s}")
for N in N_FRAG:
    Js = [collective_dense(P, N) for P in (X, Y, Z)]
    for p in P_FRAG:
        rho = local_channel(to_dm(ghz_state(N)), kraus_dephasing(p))
        F = float(np.array([0, 0, 1.0]) @ np.array(qfi_matrix_mixed(dm_matrix(rho), Js)) @ np.array([0, 0, 1.0]))
        pred = N ** 2 * (1 - 2 * p) ** (2 * N)
        print(f"{N:4d} {p:6.2f} | {F:12.6f} {pred:12.6f} {abs(F - pred):10.2e}")
        assert abs(F - pred) < 1e3 * TOL

import math


def zdeph_xcat_fq(N, p):
    """F_Q(J_x) of the x-cat after Z-dephasing of every qubit -- the closed form of Section 10.2.

    MATH  a Z error on the set S of qubits (|S| = k) maps the x-cat onto a cat on the pair {s, s-bar},
          on which J_x = +-(N/2 - k).  S and its complement land on the SAME pair, with opposite relative
          sign, so they mix into a cat of coherence (a-b)/(a+b), a = p^k(1-p)^(N-k), b = p^(N-k)(1-p)^k:
              F = sum_k binom(N,k)/2 * (a-b)^2/(a+b) * (N-2k)^2 .
    """
    k = np.arange(N + 1)
    a, b = p ** k * (1 - p) ** (N - k), p ** (N - k) * (1 - p) ** k
    binom = np.array([math.comb(N, int(j)) for j in k])
    tot = a + b
    w = np.where(tot > 0, (a - b) ** 2 / np.where(tot > 0, tot, 1.0), 0.0)   # at p = 0 only k = 0, N survive
    return float(np.sum(binom / 2 * w * (N - 2 * k) ** 2))


print("\nOAT cat along x: X-type noise (bit flip) follows the same law; Z-dephasing does not")
print(f"{'N':>4s} {'p':>6s} | {'bit flip F_Q':>13s} {'Eq. (12)':>11s} {'F_Q perp':>9s} | {'dephasing F_Q':>14s} "
      f"{'ratio to Eq. (12)':>18s} {'Eq. (13)':>10s} {'(N(1-2p))^2+4Np(1-p)':>22s}")
for N in (4, 6):
    Js = [collective_dense(P, N) for P in (X, Y, Z)]
    cat = to_dm(oat_evolve(product_state("+" * N), np.pi / 2))
    for p in (0.0, 0.02, 0.05, 0.10):
        wb = np.array(jnp.linalg.eigvalsh(qfi_matrix_mixed(dm_matrix(local_channel(cat, kraus_bit_flip(p))), Js)))
        Fb, Fb_perp = float(wb[-1]), float(wb[-2])
        Fd = float(jnp.linalg.eigvalsh(qfi_matrix_mixed(dm_matrix(local_channel(cat, kraus_dephasing(p))), Js))[-1])
        pred = N ** 2 * (1 - 2 * p) ** (2 * N)
        f13 = zdeph_xcat_fq(N, p)
        approx = (N * (1 - 2 * p)) ** 2 + 4 * N * p * (1 - p)
        print(f"{N:4d} {p:6.2f} | {Fb:13.6f} {pred:11.6f} {Fb_perp:9.4f} | {Fd:14.6f} {Fd / pred:18.3f} "
              f"{f13:10.6f} {approx:22.4f}")
        assert abs(Fb - max(pred, N)) < 1e3 * TOL       # Eq. (12), or the floor N once Eq. (12) drops below it
        assert abs(Fb_perp - N) < 1e3 * TOL              # perpendicular generators: F_Q = N at every p
        assert abs(Fd - f13) < 1e3 * TOL                 # Eq. (13), the Z-dephasing closed form

# %% [markdown]
# The first table confirms Eq. (12) to $3\times10^{-14}$ at every $N$ and every $p$: $F_Q=N^2(1-2p)^{2N}$ holds exactly,
# for small and large $p$ alike. The decay is fast — at $N=6$ and only $5\%$ dephasing per qubit, the quantum Fisher
# information for the generator $J_z$ has fallen from $36$ to $10.17$, barely above the standard quantum limit $N=6$; at
# $p=0.2$ it is $0.078$, thirteen times *below* the $F_Q=1$ of a single unentangled qubit. This is the information about
# a rotation about the cat axis. The same dephased state keeps $F_Q=N$ for a generator perpendicular to that axis
# (the bit-flip column of the second table shows the corresponding floor for the OAT cat).
#
# The second table shows why the *axis* of the noise matters. The bit-flip channel applied to the $x$-oriented OAT cat
# reproduces Eq. (12) exactly at small $p$ — it is dephasing in the cat's own basis, by the same algebra — and departs
# from it only once the formula falls below the floor $F_Q=N$ of the perpendicular generators (column `F_Q perp`,
# exactly $N$ at every $p$): $6.0$ against the predicted $2.47$ at $N=6$, $p=0.1$. Plain $Z$-dephasing on the same
# state is far gentler: at $N=6$, $p=0.1$ it leaves $F_Q=25.18$, ten times the prediction of Eq. (12), and equal to
# Eq. (13) to machine precision; the simple form $(N(1-2p))^2+4Np(1-p)=25.20$ is already within $0.1\%$. Which noise is deadly depends entirely on how the cat is oriented relative to it — and one-axis twisting orients
# it perpendicular to the $J_z^2$ interaction axis, which is often the very axis along which laboratory dephasing acts.
#
# > **Physics insight.** This is the practical argument behind decoherence-free encodings: a cat is fragile with respect
# > to noise that distinguishes its two branches, and robust with respect to noise that moves them together. The same
# > $N$-fold enhancement that makes the phase accumulate $N$ times faster makes the *distinguishing* noise act $N$ times
# > faster; nothing can change that, but choosing the axis can change which noise is "distinguishing".

# %% [markdown]
# ## 11. Noise during the twisting, and when to stop
#
# ### 11.1 The sweep
#
# We now run the Trotterised evolution of Eq. (8) with three channels — dephasing, depolarising and amplitude damping —
# at several rates $\gamma$, and record $F_Q^{\max}(\mu)$, the largest eigenvalue of the mixed-state QFI matrix, at every
# step. The question is where the maximum of that curve lies: at the cat ($\mu=\pi/2$), or earlier.

# %%
# ==============================================================================
# STEP 10: noisy OAT sweeps -- F_Q^max along the evolution for three channels
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
N_NOISE  = 6                                    # density tensor: 4^6 = 4096 entries
N_STEP   = 32                                   # Trotter steps from mu = 0 to mu = pi/2
GAMMAS   = (0.0, 0.05, 0.20, 0.50)              # noise probability per unit twisting angle
CHANNELS = (("dephasing", kraus_dephasing), ("depolarising", kraus_depolarizing),
            ("amplitude damping", kraus_amplitude_damping))
# -----------------------------------------------------------------------------
Js_noise = [collective_dense(P, N_NOISE) for P in (X, Y, Z)]
rho_init = to_dm(product_state("+" * N_NOISE))
MU_NOISE = np.linspace(0.0, np.pi / 2, N_STEP + 1)
d_mu = float(np.pi / 2 / N_STEP)


@jax.jit
def fq_max_dm(rho):
    """Largest eigenvalue of the mixed-state QFI matrix: the best collective generator for this rho."""
    return jnp.linalg.eigvalsh(qfi_matrix_mixed(dm_matrix(rho), Js_noise))[-1]


def noisy_sweep(kraus_fn, gamma):
    """F_Q^max at every step of the Trotterised noisy evolution of Eq. (8)."""
    K = kraus_fn(gamma * d_mu)
    step = jax.jit(lambda r: local_channel(oat_evolve_dm(r, d_mu), K))
    rho, out = rho_init, [float(fq_max_dm(rho_init))]
    for _ in range(N_STEP):
        rho = step(rho)
        out.append(float(fq_max_dm(rho)))
    return np.array(out)


t0 = time.time()
sweeps = {(name, g): noisy_sweep(fn, g) for name, fn in CHANNELS for g in GAMMAS}
print(f"({len(sweeps)} sweeps of {N_STEP} steps at N = {N_NOISE} in {time.time() - t0:.1f} s)\n")

# --- CHECKPOINT: at gamma = 0 the density-tensor sweep must reproduce the pure-state QFI ------
# (independent code path: oat_evolve on a state vector + the 3x3 covariance matrix of Section 6)
F_pure = np.array([float(optimal_direction(oat_evolve(product_state("+" * N_NOISE), float(m)))[0]) for m in MU_NOISE])
err_pure = max(float(np.max(np.abs(sweeps[(name, 0.0)] - F_pure))) for name, _ in CHANNELS)
print(f"gamma = 0: max |F_Q(density tensor) - F_Q(pure state)| over the sweep = {err_pure:.2e}\n")
assert err_pure < 1e4 * TOL

print(f"{'channel':>18s} {'gamma':>6s} | {'max F_Q':>9s} {'at mu':>7s} {'mu/(pi/2)':>10s} "
      f"{'F_Q(pi/2)':>10s} {'F_Q(0)':>8s} {'gain over SQL':>14s} {'xi_R^2 at mu*, no noise':>24s}")
for (name, g), v in sweeps.items():
    i = int(np.argmax(v))
    # Wineland parameter of the NOISELESS state at the optimal stopping angle (undefined near the cat: <J> -> 0)
    xi_txt = f"{float(track_step(N_NOISE, float(MU_NOISE[i]))[2]):.3f}" if MU_NOISE[i] < 1.0 else "- (cat region)"
    print(f"{name:>18s} {g:6.2f} | {v[i]:9.4f} {MU_NOISE[i]:7.4f} {MU_NOISE[i] / (np.pi / 2):10.3f} "
          f"{v[-1]:10.4f} {v[0]:8.4f} {v[i] / N_NOISE:14.3f} {xi_txt:>24s}")

# --- CHECKPOINT: Trotter convergence of the unitary/dissipative splitting ------------------
# The splitting of Eq. (8) is FIRST order in d_mu, so the error should halve when the number of
# steps doubles.  Richardson: with F(n) = F_inf - C/n, F_inf ~ 2 F(2n) - F(n).
ref = sweeps[("depolarising", 0.20)][-1]
conv = {N_STEP: ref}
for n_fine in (64, 128):
    dm_fine = float(np.pi / 2 / n_fine)
    K_fine = kraus_depolarizing(0.2 * dm_fine)
    step = jax.jit(lambda r: local_channel(oat_evolve_dm(r, dm_fine), K_fine))
    rho = rho_init
    for _ in range(n_fine):
        rho = step(rho)
    conv[n_fine] = float(fq_max_dm(rho))
print("\nTrotter convergence of the splitting (depolarising, gamma = 0.2, F_Q at mu = pi/2):")
for n, val in conv.items():
    print(f"   {n:4d} steps -> F_Q = {val:.6f}")
rich = 2 * conv[128] - conv[64]
print(f"   Richardson extrapolation (first order): F_Q(d_mu -> 0) = {rich:.6f}")
print(f"   relative error of the {N_STEP}-step value used in the sweeps: "
      f"{abs(conv[N_STEP] - rich) / rich:.1%}")
ratio_conv = (conv[64] - conv[N_STEP]) / (conv[128] - conv[64])
print(f"   ratio of successive differences (2 for a first-order error): {ratio_conv:.3f}")
assert 1.7 < ratio_conv < 2.3

# %%
# ==============================================================================
# FIGURE: noisy one-axis twisting -- where to stop
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharey=True)
for ax, (name, _) in zip(axes, CHANNELS):
    for k, g in enumerate(GAMMAS):
        v = sweeps[(name, g)]
        ax.plot(MU_NOISE / np.pi, v / N_NOISE ** 2, "-", color=PALETTE[k], lw=1.8,
                label=rf"$\gamma={g}$")
        i = int(np.argmax(v))
        ax.plot(MU_NOISE[i] / np.pi, v[i] / N_NOISE ** 2, MARKERS[k], color=PALETTE[k], ms=8)
    ax.axhline(1.0 / N_NOISE, color="0.4", ls=":", lw=1.2)
    ax.set_xlabel(r"$\mu/\pi$"); ax.set_title(name)
axes[0].set_ylabel(r"$F_Q^{\max}/N^2$")
axes[0].text(0.40, 1.0 / N_NOISE + 0.015, "SQL", fontsize=8, color="0.3")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"Quantum Fisher information under noise during the twisting ($N={N_NOISE}$); "
             f"markers = optimal stopping point", fontsize=11)
fig.tight_layout(); plt.show()

# %% [markdown]
# The noiseless curves ($\gamma=0$) climb to $F_Q/N^2=1$ at $\mu=\pi/2$, as in Section 6. Adding noise changes the shape
# qualitatively, and differently for each channel.
#
# * **Dephasing** ($Z$ noise) barely moves the optimum: as Section 10.2 explained, the cat is oriented along $\hat x$ and
#   $Z$ noise is not the noise that separates its branches. At $\gamma=0.05$ the best stopping point is $\mu=1.52$ and at
#   $\gamma=0.2$ it is $\mu=1.47$, both within $7\%$ of the cat time. Only at $\gamma=0.5$ does the optimum collapse to
#   $\mu=0.196$, deep in the squeezing regime.
# * **Depolarising** noise contains $X$ and $Y$ components, so it *does* dephase the cat in its own basis. Already at
#   $\gamma=0.2$ the maximum has moved to $\mu=0.393$ — a quarter of the way to the cat — and the value at $\mu=\pi/2$
#   has collapsed to $4.47$, below the standard quantum limit $N=6$. At $\gamma=0.5$ the cat retains $F_Q=0.20$: the
#   state that was best in the ideal world is worse than doing nothing at all.
# * **Amplitude damping** sits in between, with the optimum at $\mu=0.442$ for $\gamma=0.2$ and $\mu=0.344$ for
#   $\gamma=0.5$.
#
# This is the central practical message of the chapter. Once the noise reaches the cat's own basis, the optimal
# preparation is an intermediate, partially squeezed state instead of the Heisenberg-limited cat: it gives up a factor of
# a few in the ideal $F_Q$ and buys back much more in robustness. The marker on each curve is the optimal stopping point;
# for the two channels whose noise reaches the cat's own basis it sits in the squeezing regime as soon as
# $\gamma\ge0.2$. The last column of the table makes "squeezing regime" quantitative: the noiseless state at every such
# stopping angle has a Wineland parameter $\xi_R^2<1$ ($0.455$ to $0.760$).
#
# > **Numerical practice.** Report the splitting error. Splitting Eq. (8) into a unitary and a dissipative half
# > is first order in $\delta\mu$, and the printed convergence shows it (the ratio of successive differences is
# > $1.96$, close to $2$): at $\gamma=0.2$ the depolarising value of
# > $F_Q(\pi/2)$ moves from $4.474$ ($32$ steps) to $4.616$ ($64$) to $4.688$ ($128$), with the Richardson
# > extrapolation at $4.761$ — so the $32$-step value used in the sweeps is about $6\%$ low. That is large enough to
# > report and too small to change any conclusion here, because every curve in the figure is computed with the same
# > $\delta\mu$ and the comparison is between them. (For the dephasing channel the split is exact, because $Z$ errors
# > commute with $e^{-i\delta\mu J_z^2}$; the error there comes only from compounding $n_{\rm step}$ discrete channels.)

# %%
# ==============================================================================
# STEP 11: how the fragility grows with N  --  the cat under depolarising noise
# ==============================================================================
print(f"{'N':>4s} | " + " ".join(f"{'g=' + format(g, '.2f'):>10s}" for g in (0.0, 0.02, 0.05, 0.10))
      + f" | {'N^2':>6s} {'N':>4s} | {'2nd eigenvalue, g=0.10':>23s}")
frag_rows = []
for N in (3, 4, 5, 6):
    Js = [collective_dense(P, N) for P in (X, Y, Z)]
    cat = to_dm(oat_evolve(product_state("+" * N), np.pi / 2))
    vals = []
    for g in (0.0, 0.02, 0.05, 0.10):
        rho = local_channel(cat, kraus_depolarizing(g))
        w = np.array(jnp.linalg.eigvalsh(qfi_matrix_mixed(dm_matrix(rho), Js)))
        vals.append(float(w[-1]))
    frag_rows.append((N, vals))
    # w still holds the g = 0.10 spectrum: its second eigenvalue belongs to a generator PERPENDICULAR to the cat axis
    print(f"{N:4d} | " + " ".join(f"{v:10.4f}" for v in vals) + f" | {N ** 2:6d} {N:4d} | {float(w[-2]):23.4f}")

print("\nratio F_Q(g)/F_Q(0), the exponential fit  F_Q/N^2 = exp(-kappa N),  and kappa from Eq. (14):")
for j, g in enumerate((0.02, 0.05, 0.10)):
    r = np.array([row[1][j + 1] / row[1][0] for row in frag_rows])
    Nv = np.array([row[0] for row in frag_rows], dtype=float)
    kappa = -np.polyfit(Nv, np.log(r), 1)[0]
    kappa_th = -np.log((1 - 4 * g / 3) ** 2 / (1 - 2 * g / 3))
    print(f"   gamma = {g:.2f}:  ratios " + " ".join(f"{x:.4f}" for x in r)
          + f"   ->  decay exp(-{kappa:.4f} N) per qubit;   Eq. (14): kappa = {kappa_th:.4f}")
    assert abs(kappa - kappa_th) < 0.02 * kappa_th

# %% [markdown]
# The ratio $F_Q(\gamma)/F_Q(0)$ at the cat time falls geometrically with $N$: each additional qubit multiplies the
# surviving Fisher information by the same factor. The factor follows from the error bookkeeping of Section 10.2. For the
# $x$-cat, a depolarising $X$ error (probability $\gamma/3$) flips the relative sign of the branches, a $Z$ error
# ($\gamma/3$) moves the cat onto another pair of strings, and a $Y\propto ZX$ error ($\gamma/3$) does both. Once a qubit
# has suffered a $Y$ or a $Z$ error, the two equally likely signs cancel the coherence of that pair completely, so only
# the strings with no $Y$ or $Z$ error, of total probability $(1-\tfrac{2\gamma}{3})^N$, contribute. Among those, each
# qubit carries an $X$ error with conditional probability $\tfrac{\gamma/3}{1-2\gamma/3}$, so the coherence is
# $\big[(1-\tfrac{4\gamma}{3})/(1-\tfrac{2\gamma}{3})\big]^N$, and, up to the complementary-set corrections of Eq. (13),
#
# $$\frac{F_Q(J_{\mathbf n_{\rm cat}})}{N^2}=\left[\frac{(1-\tfrac{4\gamma}{3})^2}{1-\tfrac{2\gamma}{3}}\right]^N
#   \equiv e^{-\kappa N}. \tag{14}$$
#
# This is the $(1-2p)^{2N}$ of Eq. (12) with the depolarising error budget. It gives $\kappa=0.0406$, $0.1041$, $0.2172$
# for $\gamma=0.02$, $0.05$, $0.10$, and the fits to the four sizes reproduce these values (the cell asserts agreement to
# $2\%$). At $5\%$ depolarising noise the cat retains $73\%$ of its ideal $F_Q$ at $N=3$ and $54\%$ at $N=6$; at $10\%$
# the per-qubit factor is $e^{-0.217}=0.80$ and only $27\%$ survives at $N=6$.
#
# The exponential law describes the cat axis only. The last column is the second eigenvalue of the QFI matrix at
# $\gamma=0.10$, which belongs to a generator perpendicular to the cat axis; there the two branches act as two
# depolarised coherent spin states, and the value grows linearly, $4.50$ at $N=6$, close to $N(1-\tfrac{4\gamma}{3})^2=0.751N$
# (the Bloch vector of every qubit shrinks by $1-\tfrac{4\gamma}{3}$). Extrapolating Eq. (14) to $N=30$ at $\gamma=0.1$ leaves
# $F_Q\approx900\,e^{-0.217\times30}\approx1.3$ along the cat axis, while the perpendicular direction offers about
# $0.75\times30\approx22$; the two cross near $N\approx13$. Beyond that size the best this state offers is a degraded
# standard-quantum-limit measurement: the Heisenberg advantage of the cat is gone entirely.

# %% [markdown]
# ## 12. The four protocols side by side
#
# We can now put the whole chapter on one axis. For each $N$ and each preparation we compute $F_Q$ with the optimal
# collective generator and convert it into the phase uncertainty of a single-shot experiment,
#
# $$\Delta\theta\sqrt M=\frac{1}{\sqrt{F_Q}},$$
#
# which is the quantum Cramér–Rao bound and, for the readouts of notebooks 31, 32 and Section 8, also the achieved value.
#
# * **Ramsey** (nb 31): $\vert+x\rangle^{\otimes N}$, $F_Q=N$, $\Delta\phi\sqrt M=1/\sqrt N$.
# * **Squeezed** (nb 33): the OAT state at the optimal $\mu$, $F_Q\ge N/\xi_R^2$ with $\xi_R^2\sim N^{-2/3}$.
# * **Ideal GHZ** (nb 32): $F_Q=N^2$, $\Delta\theta\sqrt M=1/N$.
# * **OAT cat** (this notebook): the same $F_Q=N^2$, prepared by letting the *same* interaction run to $\mu=\pi/2$.
# * **Noisy cat and noisy squeezed**: the same quantities with depolarising noise of rate $\gamma$ acting during the
#   twisting, evaluated for $N\le6$ where the density tensor fits.

# %%
# ==============================================================================
# STEP 12: the comparison table and figure
# ==============================================================================
# PARAMETERS ------------------------------------------------------------------
N_CMP  = (4, 6, 8, 10, 12, 14, 16)     # pure-state comparison
N_CMP_NOISY = (3, 4, 5, 6)             # density-tensor comparison
GAMMA_CMP = 0.10                       # depolarising rate for the noisy curves
# -----------------------------------------------------------------------------
ideal = {"Ramsey (CSS)": [], "squeezed (OAT, optimal $\\mu$)": [], "GHZ / OAT cat": []}
mu_best = {}
for N in N_CMP:
    ideal["Ramsey (CSS)"].append(float(N))
    grid = np.linspace(0.45, 2.0, 21) * 3 ** (1 / 6) * N ** (-2 / 3)
    vals = np.array([[float(x) for x in track_step(N, float(m))[::2]] for m in grid])
    mu_best[N] = refine_argmin(grid, vals[:, 1])
    ideal["squeezed (OAT, optimal $\\mu$)"].append(float(track_step(N, mu_best[N])[0]))
    ideal["GHZ / OAT cat"].append(float(N ** 2))

mu_stop = {}
noisy = {"noisy cat ($\\mu=\\pi/2$)": [], "noisy squeezed ($\\mu_{\\rm opt}$)": [], "best stopping point": []}
for N in N_CMP_NOISY:
    Js = [collective_dense(P, N) for P in (X, Y, Z)]
    fq = jax.jit(lambda r: jnp.linalg.eigvalsh(qfi_matrix_mixed(dm_matrix(r), Js))[-1])
    n_st, rho = 32, to_dm(product_state("+" * N))
    dmu = float(np.pi / 2 / n_st)
    K = kraus_depolarizing(GAMMA_CMP * dmu)
    step = jax.jit(lambda r: local_channel(oat_evolve_dm(r, dmu), K))
    mus_n, curve = [0.0], [float(fq(rho))]
    for s in range(n_st):
        rho = step(rho)
        mus_n.append((s + 1) * dmu)
        curve.append(float(fq(rho)))
    curve, mus_n = np.array(curve), np.array(mus_n)
    grid = np.linspace(0.45, 2.0, 21) * 3 ** (1 / 6) * N ** (-2 / 3)
    vals = np.array([[float(x) for x in track_step(N, float(m))[::2]] for m in grid])
    mu_sq = refine_argmin(grid, vals[:, 1])
    noisy["noisy cat ($\\mu=\\pi/2$)"].append(curve[-1])
    noisy["noisy squeezed ($\\mu_{\\rm opt}$)"].append(float(np.interp(mu_sq, mus_n, curve)))
    noisy["best stopping point"].append(float(curve.max()))
    mu_stop[N] = (float(mus_n[int(np.argmax(curve))]), mu_sq,
                  float(track_step(N, float(mus_n[int(np.argmax(curve))]))[0]))   # noiseless F_Q there

print(f"quantum Fisher information (ideal, pure states)\n{'N':>4s} | "
      + " ".join(f"{k:>30s}" for k in ideal) + f" {'N':>6s} {'N^2':>6s}")
for i, N in enumerate(N_CMP):
    print(f"{N:4d} | " + " ".join(f"{ideal[k][i]:30.3f}" for k in ideal) + f" {N:6d} {N ** 2:6d}")

print(f"\nwith depolarising noise, gamma = {GAMMA_CMP}, acting during the twisting\n{'N':>4s} | "
      + " ".join(f"{k:>34s}" for k in noisy) + f" {'SQL N':>7s}")
for i, N in enumerate(N_CMP_NOISY):
    print(f"{N:4d} | " + " ".join(f"{noisy[k][i]:34.3f}" for k in noisy) + f" {N:7d}")
print(f"\n{'N':>4s} | {'best stopping mu':>17s} {'Wineland mu_opt':>16s} {'noiseless F_Q at best stopping mu':>34s}")
for N in N_CMP_NOISY:
    print(f"{N:4d} | {mu_stop[N][0]:17.4f} {mu_stop[N][1]:16.4f} {mu_stop[N][2]:34.3f}")

# %%
# ==============================================================================
# FIGURE: sensitivity of every protocol of Chapter 10 versus N
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
Nc = np.array(N_CMP, dtype=float)
Nn = np.array(N_CMP_NOISY, dtype=float)

for k, (name, vals) in enumerate(ideal.items()):
    axes[0].loglog(Nc, np.array(vals), MARKERS[k] + "-", color=PALETTE[k], ms=7, lw=1.6, label=name)
axes[0].loglog(Nc, Nc, "k:", lw=1.2, label=r"SQL $N$")
axes[0].loglog(Nc, Nc ** 2, "k--", lw=1.2, label=r"Heisenberg $N^2$")
axes[0].set_xlabel("$N$"); axes[0].set_ylabel(r"$F_Q$")
axes[0].set_title("Metrological resource, ideal"); axes[0].legend(fontsize=8)

for k, (name, vals) in enumerate(ideal.items()):
    axes[1].loglog(Nc, 1 / np.sqrt(np.array(vals)), MARKERS[k] + "-", color=PALETTE[k], ms=7, lw=1.6, label=name)
for k, (name, vals) in enumerate(noisy.items()):
    axes[1].loglog(Nn, 1 / np.sqrt(np.array(vals)), MARKERS[k] + "--", color=PALETTE[k + 3], ms=7, lw=1.4,
                   label=name + rf" ($\gamma={GAMMA_CMP}$)")
axes[1].loglog(Nc, 1 / np.sqrt(Nc), "k:", lw=1.2)
axes[1].set_xlabel("$N$"); axes[1].set_ylabel(r"$\Delta\theta\,\sqrt{M}=1/\sqrt{F_Q}$")
axes[1].set_title("Achievable phase uncertainty"); axes[1].legend(fontsize=7)
fig.tight_layout(); plt.show()

# %% [markdown]
# The left panel is the chapter in one picture. The Ramsey line has slope $1$ on the log–log axes (the standard quantum
# limit $F_Q=N$), the GHZ and OAT-cat line has slope $2$ (the Heisenberg limit $F_Q=N^2$), and the squeezed line runs in
# between, with the exponent $1.668$ that Section 6 fitted over $6\le N\le16$ (an average of local slopes rising from
# $1.63$ to $1.70$; the asymptotic exponent is $5/3$).
#
# The right panel converts that into the number an experimentalist quotes, $\Delta\theta\sqrt M=1/\sqrt{F_Q}$, and adds
# the noisy points. Under $10\%$ depolarising noise acting throughout, the whole hierarchy compresses: at $N=3$ the
# noisy cat ($F_Q=4.94$) and the noisy squeezed state ($4.88$) are indistinguishable, both about $60\%$ above the
# Ramsey value $3$; by $N=6$ the noisy squeezed state ($14.73$) has overtaken the noisy cat ($12.83$), and the best
# stopping point of the noisy evolution ($17.39$) beats both. The last table locates it: at $N=6$ the noisy curve peaks
# at $\mu=0.442$, well beyond the Wineland optimum $\mu_{\rm opt}=0.2685$ and far before the cat. The noiseless state there
# would have $F_Q=22.91$ (the QFI keeps growing past the squeezing optimum, Section 6); the noise accumulated up to that
# angle removes about a quarter of it. That the remainder ($17.39$) is close to the noiseless value at $\mu_{\rm opt}$
# ($17.09$) is a coincidence of these parameters. The compromise is found automatically by maximising $F_Q$ over $\mu$,
# which is what an experiment does when it calibrates its interaction time.
#
# > **Physics insight.** The gap between the two panels is the central difficulty of quantum metrology. In the ideal
# > world the answer is "make a cat". With noise the answer is "make the largest $F_Q$ that survives your noise", and
# > for every channel that reaches the cat's own basis at the strengths studied here that is a moderately twisted state
# > with $\xi_R^2<1$. This is consistent with the experimental record reviewed by Pezzè *et al.* (2018), in which most
# > entanglement-enhanced atom interferometers use spin-squeezed states.

# %% [markdown]
# ## 13. Key takeaways
#
# * $e^{-i\frac{\pi}{2}J_z^2}$ acting on a coherent spin state produces a **GHZ-like cat**, exactly. For even $N$ the
#   propagator collapses to $\frac{e^{-i\pi/4}}{\sqrt2}\mathbb 1+\frac{e^{i\pi/4}}{\sqrt2}(-1)^{N/2}\prod_qZ_q$ and the cat
#   lies along $\hat x$, Eq. (3); for odd $N$ the half-integer spectrum makes the phase $4$-periodic in $m$, the
#   propagator becomes $\frac{e^{-i\pi/8}}{\sqrt2}\left(e^{i\frac{\pi}{2}J_z}+e^{-i\frac{\pi}{2}J_z}\right)$, and the cat
#   lies along $\hat y$, Eq. (5). Both were verified by fidelity $1$ including the global phase.
# * The diagonal propagator and the circuit of $N(N-1)/2$ commuting $R_{ZZ}(\mu)$ gates agree to $10^{-15}$: no Trotter
#   error, because all the terms are diagonal.
# * The $3\times3$ QFI matrix tracks the state along the evolution. $F_Q^{\max}$ goes $N$ (at $\mu=0$) $\to$
#   $\simeq N+3^{1/3}N^{5/3}$ at the squeezing optimum (fitted exponent $1.668$ over $6\le N\le16$, local slopes
#   $1.63$–$1.70$) $\to$ a plateau at $0.577$ ($N=8$), $0.543$ ($N=12$), $0.531$ ($N=16$) times $N^2$, close to the lobe
#   model $\tfrac12+\tfrac1{2N}$ $\to$ exactly $N^2$ at $\mu=\pi/2$; its eigenvector jumps onto the cat axis $\hat x$ at
#   $\mu\approx0.30\pi$ and stays there.
# * At $\mu=\pi/q$ the state is a $q$-component cat — an automatic lobe count on the Husimi equator finds exactly
#   $2,3,4,5$ lobes for $q=2,3,4,5$ at $N=12$, and fewer beyond that because the components start to overlap. Two
#   antipodal lobes give $F_Q=N^2$; $q\ge3$ lobes around a circle give $F_Q/N^2$ between $0.54$ and $0.58$, the origin
#   of the plateau.
# * Encoding with the cat axis and measuring the **parity** $\prod_qZ_q$ gives a fringe of period $2\pi/N$ with full
#   visibility, whose classical Fisher information equals $F_Q=N^2$ at every phase but the two fringe extrema. Maximum
#   likelihood on sampled parities reaches $\Delta\theta=1/(N\sqrt M)$ over the whole range $M=25$ to $M=800$ — but only
#   inside a search window of half a fringe.
# * Local dephasing in the cat's own basis gives the exact law $F_Q=N^2(1-2p)^{2N}\simeq N^2e^{-4N\gamma\mu}$, verified
#   to $3\times10^{-14}$: the decoherence rate of the $N$-body coherence is $N$ times the single-qubit rate. Which
#   laboratory noise is deadly depends on the cat's orientation — the OAT cat lies along $\hat x$, and at $N=6$ its
#   $F_Q$ under $Z$ dephasing beats the value under $X$-type noise by a factor $1.5$ at $p=0.02$, $3.0$ at $p=0.05$
#   and $4.2$ at $p=0.10$, following the polynomial law of Eq. (13) instead of the exponential law of Eq. (12).
#   Under depolarising noise the cat-axis QFI decays as $e^{-\kappa N}$ with $\kappa$ given by Eq. (14).
# * With depolarising or amplitude-damping noise at $\gamma\ge0.2$ the maximum of $F_Q(\mu)$ moves out of the cat and
#   into the squeezing regime ($\xi_R^2<1$ at the stopping angle). Measured at $N=6$, $\gamma=0.2$: the depolarising optimum sits at $\mu=0.393$ with
#   $F_Q=13.6$, while the cat at $\mu=\pi/2$ has fallen to $4.47$, below the standard quantum limit $N=6$.
# * The unitary/dissipative splitting is first order in $\delta\mu$; the $32$-step value used in the sweeps is about
#   $6\%$ below the Richardson extrapolation, which is reported rather than hidden.
#
# ## 14. Exercises
#
# 1. **(★)** Evaluate $e^{-i\mu J_z^2}$ at $\mu=\pi$ for even $N$ and show, from the argument of Section 3.2, that the
#    state is again a coherent spin state. Which one? Verify numerically with the Husimi map.
# 2. **(★)** For $N=6$ and $N=7$, compute the entanglement entropy of the cat state across the cut $\{0,1,2\}$ versus the
#    rest, and confirm that it is exactly one bit while $F_Q=N^2$.
# 3. **(★★)** For even $N$, derive the coefficients of the $q$-component cat at $\mu=\pi/3$ by the Fourier method of
#    Section 4.2 (show first that $e^{-i\frac{\pi}{3}m^2}$ changes sign under $m\to m+3$, so only the three modes
#    $e^{i\pi(2p+1)m/3}$ appear), and check the resulting superposition of three rotated coherent states against the
#    simulated state by fidelity.
# 4. **(★★)** *Extend the code.* Replace the parity $\prod_qZ_q$ by a population readout: after the encoding apply a
#    $\pi/2$ pulse about $\hat y$, which maps the cat axis $\hat x$ onto the $z$ axis, and record only the number $k$ of
#    qubits found in $\vert1\rangle$. Compute the classical Fisher information of $p(k\vert\theta)$ for the even-$N$ cat.
#    How much of $F_Q=N^2$ does it capture, and why? Then drop the pulse and record $k$ directly: what changes, and how
#    is the answer related to the parity readout?
# 5. **(★★)** Reconstruct the noisy curves of Section 11 with quantum trajectories
#    ([17 — Monte-Carlo wave function](../ch06_open_quantum_systems/17_monte_carlo_wave_function.ipynb)) instead of the
#    density tensor. The QFI is *not* a linear function of $\rho$, so you cannot average it over trajectories — you must
#    average $\rho$ first. How many trajectories are needed at $N=6$ to reproduce $F_Q$ to $1\%$?
# 6. **(★★)** *Physics.* The lobe model of Section 6 gives $F_Q/N^2=\tfrac12+\tfrac1{2N}$ for $q\ge3$ and neglects
#    the overlap between neighbouring lobes. (a) Redo the calculation for $q=2$ and show that the result depends on the
#    angle between $\mathbf n$ and the cat axis, with maximum $N^2$. (b) Compute $F_Q^{\max}/N^2$ at $\mu=\pi/3$ and
#    $\mu=\pi/4$ for $N=8,10,\dots,20$ and plot its deviation from $\tfrac12+\tfrac1{2N}$ against $N$. Compare with the
#    overlap of two neighbouring coherent states, $\vert\langle\mathbf e_p\vert\mathbf e_{p+1}\rangle\vert^2
#    =\cos^{2N}(\pi/q)$, and decide whether the deviation decays at the same rate (the answer differs between $q=3$
#    and $q=4$; at $N=20$ the state has $2^{20}$ amplitudes, still cheap for `oat_evolve` and `spin_moments`).
# 7. **(★★★)** Optimal stopping as an optimisation. For a given channel and $\gamma$, use `jax.grad` through the
#    Trotterised noisy evolution to find $\mu^\star=\arg\max_\mu F_Q(\mu)$ by gradient ascent rather than by scanning.
#    Compare the cost with the scan, and check that the eigenvalue derivative is well behaved when the top eigenvalue of
#    the QFI matrix is nearly degenerate.
# 8. **(★★★)** Interaction-based readout. Instead of the parity, apply the *inverse* twisting $e^{+i\mu J_z^2}$ after the
#    encoding (with the optimal generator $\mathbf n_{\rm opt}(\mu)\cdot\mathbf J$), then a $\pi/2$ pulse about $\hat y$, and count the qubits in $\vert1\rangle$ — i.e. measure $J_x$, the
#    direction of the initial coherent state. Show numerically, for $N=8$ and several $\mu$ between $0.1$ and $\pi/2$,
#    that the classical Fisher information of this echo protocol at small $\theta$ equals $F_Q^{\max}(\mu)$, not only at
#    the cat time. Then flip every detected bit independently with probability $\epsilon=0.01$–$0.05$ and compare, at
#    $\mu=\pi/2$, the best classical Fisher information over $\theta$ of the echo readout with that of the parity readout.
#    (Echo protocols of this kind: Davis, Bentsen and Schleier-Smith 2016; Macrì, Smerzi and Pezzè 2016.)
#
# ## References
#
# * M. Kitagawa and M. Ueda, *Squeezed spin states*, Physical Review A **47**, 5138 (1993).
# * G. S. Agarwal, R. R. Puri and R. P. Singh, *Atomic Schrödinger cat states*, Physical Review A **56**, 2249 (1997).
# * K. Mølmer and A. Sørensen, *Multiparticle entanglement of hot trapped ions*, Physical Review Letters **82**, 1835 (1999).
# * A. Sørensen, L.-M. Duan, J. I. Cirac and P. Zoller, *Many-particle entanglement with Bose–Einstein condensates*,
#   Nature **409**, 63 (2001).
# * S. F. Huelga, C. Macchiavello, T. Pellizzari, A. K. Ekert, M. B. Plenio and J. I. Cirac, *Improvement of frequency
#   standards with quantum entanglement*, Physical Review Letters **79**, 3865 (1997).
# * D. J. Wineland, J. J. Bollinger, W. M. Itano and D. J. Heinzen, *Squeezed atomic states and projection noise in
#   spectroscopy*, Physical Review A **50**, 67 (1994).
# * S. L. Braunstein and C. M. Caves, *Statistical distance and the geometry of quantum states*, Physical Review Letters
#   **72**, 3439 (1994).
# * L. Pezzè and A. Smerzi, *Entanglement, nonlinear dynamics, and the Heisenberg limit*, Physical Review Letters **102**,
#   100401 (2009).
# * J. Ma, X. Wang, C. P. Sun and F. Nori, *Quantum spin squeezing*, Physics Reports **509**, 89 (2011).
# * L. Pezzè, A. Smerzi, M. K. Oberthaler, R. Schmied and P. Treutlein, *Quantum metrology with nonclassical states of
#   atomic ensembles*, Reviews of Modern Physics **90**, 035005 (2018).
# * E. Davis, G. Bentsen and M. Schleier-Smith, *Approaching the Heisenberg limit without single-particle detection*,
#   Physical Review Letters **116**, 053601 (2016).
# * T. Macrì, A. Smerzi and L. Pezzè, *Loschmidt echo for quantum metrology*, Physical Review A **94**, 010102 (2016).
# * W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes: The Art of Scientific
#   Computing*, 3rd ed., Cambridge University Press (2007) — Section 10.3 for the parabolic refinement of a
#   discrete minimum used in Step 4, and Section 17.3 for Richardson's deferred approach to the limit used in the
#   Trotter convergence check of Step 10.
