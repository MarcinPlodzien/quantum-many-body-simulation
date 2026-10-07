#@title: Quantifying entanglement of quantum states
#@part: Chapter 9 — Entanglement and complexity diagnostics
#@description: Pure-state entanglement and why the Schmidt spectrum is the whole story; the zoo classified (GHZ has no pairwise entanglement, W does); why mixed states are harder; the partial transpose as one axis swap; the PPT criterion and negativity, calibrated on Werner states; entanglement death under noise; negativity, mutual information and pair entropy against distance in the XXZ chain; the same two measures from a matrix product state, with mps_rdm2 derived leg by leg and validated against exact diagonalisation.

# %% [markdown]
# ## 1. Introduction and motivation
#
# The many-body chapters of this course computed entanglement entropies and put them to work. The bond dimension a
# matrix product state needs is set by an entropy; the entanglement barrier that stops a time evolution is an entropy
# growing in time; the quantum phase transition of the XXZ chain was located by an entropy at a cut. None of that asked
# what a *measure* of entanglement is, or what else can be measured, or what the entropy fails to see.
#
# This notebook asks, and it has one concrete question to organise it:
#
# > The entropy across a cut told us where the XXZ chain changes phase. Does it tell us whether **two spins** in that
# > chain are entangled with each other?
#
# The answer will turn out to be no, and the reason is worth the whole notebook. Two spins pulled out of a ground
# state are a **mixed** state, and for mixed states the entropy of a subsystem stops measuring entanglement and starts
# measuring ignorance. So the notebook has two halves:
#
# Sections 2 to 4 are pure states. There everything is clean: the Schmidt spectrum of a cut is the complete answer and
# the entanglement entropy is the canonical measure. Sections 5 to 9 are mixed states, where the situation is genuinely
# harder; we say where the difficulty starts rather than papering over it, and use the one criterion that is computable,
# the positive partial transpose. Section 10 maps a whole phase diagram in both measures. Sections 11 to 13 then do the
# same physics twice, once with an exact state vector and once with a matrix product state, and check that the two agree
# before trusting either.
#
# Everything here runs on a laptop CPU in double precision.

# %%
#@engine: haar_state, haar_unitary, rdm, rdm_dm, purity, von_neumann_entropy, entanglement_entropy, schmidt_values, as_dm_tensor, partial_transpose, negativity, to_dm, dm_matrix, ghz_state, w_state, dicke_state, product_state, heisenberg_terms, lanczos_ground_state, xxz_mpo, dmrg, product_mps, mps_entropies, mps_rdm2, kraus_depolarizing, apply_kraus_dm

# %%
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=140)

def S_bits(rho):
    """von Neumann entropy in bits of a density matrix given as a matrix OR a density tensor."""
    w = np.linalg.eigvalsh(np.asarray(dm_matrix(jnp.asarray(rho))))
    w = w[w > 1e-13]
    return float(-(w * np.log2(w)).sum())

def neg(rho):
    """Negativity, clipped at zero: tiny negative values are rounding, not physics."""
    return max(float(negativity(rho, (0,))[0]), 0.0)

print("helpers ready")

# %% [markdown]
# ## 2. Separable and entangled pure states
#
# Fix a bipartition of the register into $A$ and $B$. A pure state is **separable** across that cut when it factorises,
#
# $$ \vert\psi\rangle = \vert\phi\rangle_A \otimes \vert\chi\rangle_B, $$
#
# and **entangled** when it does not. We already have the test. In the Schmidt decomposition
#
# $$ \vert\psi\rangle = \sum_{k=1}^{\chi} \lambda_k \vert u_k\rangle_A \vert v_k\rangle_B, \qquad \sum_k \lambda_k^2 = 1, $$
#
# the state is a product exactly when the Schmidt rank is $\chi = 1$, and any $\chi > 1$ means entanglement. The
# reduced state $\rho_A$ has eigenvalues $p_k = \lambda_k^2$, so the same statement reads:
#
# > A pure global state is entangled across a cut exactly when the reduced state of either side is **mixed**.
#
# That is why the purity of a reduced state is an entanglement test, and it is the cheapest one there is:
# $\mathrm{Tr}\rho_A^2 = \sum_k p_k^2 = \sum_k \lambda_k^4$ needs neither a logarithm nor a diagonalisation.

# %%
for label, psi in (("product  |0000>", product_state("0000")),
                   ("Bell on (0,1)  ", np.array([1,0,0,1])/np.sqrt(2)),
                   ("GHZ_4          ", ghz_state(4))):
    psi = jnp.asarray(psi).reshape((2,) * int(round(np.log2(jnp.asarray(psi).size))))
    rA = rdm(psi, (0,))
    lam = np.asarray(schmidt_values(psi, (0,)))
    print(f"{label}: Schmidt values across the 0|rest cut = {np.round(lam,6)}   "
          f"purity(rho_A) = {float(purity(rA)):.6f}   S = {S_bits(rA):.6f} bit")

# %% [markdown]
# ### The Schmidt rank of a product state
#
# This is the one fact everything else rests on, so let us derive it rather than assert it. There are two ways to see
# it and both are worth having.
#
# **From the amplitude matrix.** Group the indices by side, $\psi_{ab}$ with $a$ labelling configurations of $A$ and
# $b$ those of $B$, and read $\psi$ as a $d_A\times d_B$ **matrix**. A product state has $\psi_{ab} = \phi_a\chi_b$,
# so every row of that matrix is a multiple of the same vector $\chi$, and every column a multiple of the same $\phi$.
# That is an **outer product** $M = \phi\chi^T$, and an outer product has matrix rank **one** by construction: its
# column space is the single line spanned by $\phi$. The Schmidt values are the singular values of $M$, and a rank-one
# matrix has exactly one non-zero singular value.
#
# **From the reduced state.** The same fact in density-matrix language. For $\vert\psi\rangle =
# \vert\phi\rangle_A\otimes\vert\chi\rangle_B$,
#
# $$ \rho_A = \mathrm{Tr}_B\big(\vert\phi\rangle\langle\phi\vert\otimes\vert\chi\rangle\langle\chi\vert\big)
#  = \vert\phi\rangle\langle\phi\vert \cdot \mathrm{Tr}\big(\vert\chi\rangle\langle\chi\vert\big)
#  = \vert\phi\rangle\langle\phi\vert, $$
#
# because the partial trace touches only the $B$ factor and $\langle\chi\vert\chi\rangle = 1$. So $\rho_A$ is a
# **projector onto a single vector**: rank one, hence one non-zero eigenvalue, necessarily $1$ because the trace is $1$.
#
# The two are the same statement, since $\rho_A = MM^\dagger$ and $M$ and $MM^\dagger$ have equal rank. So:
# **entanglement is the failure of the amplitude matrix to be an outer product**, and the Schmidt rank counts how far
# from an outer product it is.

# %%
print("the amplitude matrix, its rank, and the eigenvalues of rho_A   (N=8, central cut)")
for name, psi in (("product", product_state("+"*8)), ("GHZ", ghz_state(8)),
                  ("W", w_state(8)), ("Haar", haar_state(jax.random.PRNGKey(1), 8))):
    M = np.asarray(psi).reshape(16, 16)                 # rows = config of A, cols = config of B
    sv = np.linalg.svd(M, compute_uv=False)
    ev = np.sort(np.linalg.eigvalsh(np.asarray(rdm(psi, tuple(range(4))))))[::-1]
    print(f"  {name:>8s}: rank(psi_ab) = {int((sv>1e-12).sum()):2d}   "
          f"singular values {np.round(sv[:3],4)}...   eigenvalues of rho_A {np.round(ev[:3],4)}...")

print("\nand explicitly, for a 2x2 example: psi_ab = phi_a chi_b")
phi = np.array([0.6, 0.8]); chi = np.array([1, 1])/np.sqrt(2)
M = np.outer(phi, chi)
print(f"  M = outer(phi, chi) =\n{np.round(M,6)}")
print(f"  rank {np.linalg.matrix_rank(M)},  singular values {np.round(np.linalg.svd(M,compute_uv=False),6)}")
print(f"  M M^dag =\n{np.round(M@M.conj().T,6)}\n  |phi><phi| =\n{np.round(np.outer(phi,phi),6)}   (identical)")

# %% [markdown]
# **Why the entropy and not something else.** A unitary acting on $A$ alone rotates the $\vert u_k\rangle$ and leaves
# every $\lambda_k$ untouched, so it cannot change any function of the Schmidt spectrum. Neither can a local
# measurement whose outcome is averaged over, nor classical communication between the two sides. The class of
# operations that cannot create entanglement is called **LOCC** (local operations and classical communication), and a
# *measure* of entanglement is required to be non-increasing under LOCC and to vanish on separable states. The
# entanglement entropy satisfies both, and for pure bipartite states every other measure is a function of the same
# $p_k$.
#
# Let us check the LOCC invariance numerically: a random local unitary on each side must leave the entropy alone.

# %%
key = jax.random.PRNGKey(7)
psi = haar_state(key, 6)
S0 = float(entanglement_entropy(psi, (0, 1, 2)))
k1, k2 = jax.random.split(key)
UA, UB = haar_unitary(k1, 8), haar_unitary(k2, 8)      # 3 spins on each side -> 8x8
# apply U_A to the first three axes and U_B to the last three, by reshaping to a matrix
M = np.asarray(psi).reshape(8, 8)
M_rot = np.asarray(UA) @ M @ np.asarray(UB).T
psi_rot = jnp.asarray(M_rot).reshape((2,) * 6)
S1 = float(entanglement_entropy(psi_rot, (0, 1, 2)))
print(f"S before local unitaries = {S0:.12f} bit")
print(f"S after  local unitaries = {S1:.12f} bit      difference {abs(S0-S1):.2e}")
assert abs(S0 - S1) < 1e-10
print("[ok] local unitaries cannot change entanglement")

# %% [markdown]
# ### The meaning of the entropy
#
# The formula $S_A=-\sum_k p_k\log_2 p_k$ with $p_k=\lambda_k^2$ is not an abstract functional: it is the Shannon
# entropy of an ordinary probability distribution, and $p_k$ is the probability of finding the pair in the $k$th
# Schmidt branch. Two things follow, and both are worth seeing rather than being told.
#
# **Why zero means product.** If the state factorises there is exactly one non-zero Schmidt value, and normalisation
# forces $p_1=1$. Then $S_A = -1\cdot\log_2 1 = 0$, because $\log_2 1 = 0$. The remaining terms are $0\log_2 0$,
# defined as zero (the limit of $p\log_2 p$). So the entropy vanishes *for the plain reason that there is nothing to
# be uncertain about*: we know in advance which branch we are in.
#
# **What a non-zero value means.** If $r$ Schmidt values are equal, $p_k = 1/r$, then $S_A = \log_2 r$, so
# $2^{S_A} = r$ counts the branches. For an uneven spectrum $2^{S_A}$ is smaller than the true rank, and it is then
# the *effective* number of Schmidt values — the branches that carry appreciable weight. This is exactly the quantity
# a matrix product state must supply, which is why $\chi \gtrsim 2^S$.

# %%
print("2^S is the effective number of Schmidt values")
print(f"{'state (N=8)':>14s} {'S [bit]':>9s} {'2^S':>8s} {'true rank':>10s}  comment")
for name, psi in (("product", product_state("+"*8)), ("GHZ", ghz_state(8)), ("W", w_state(8)),
                  ("Dicke k=4", dicke_state(8, 4)), ("Haar random", haar_state(jax.random.PRNGKey(3), 8))):
    A = tuple(range(4))
    S = float(entanglement_entropy(psi, A))
    rk = int((np.linalg.eigvalsh(np.asarray(rdm(psi, A))) > 1e-12).sum())
    flat = "flat spectrum: 2^S = rank" if abs(2**S - rk) < 1e-6 else f"uneven: uses {2**S:.1f} of {rk} branches"
    print(f"{name:>14s} {S:9.4f} {2**S:8.3f} {rk:10d}  {flat}")

print("\nand one bit is one Bell pair:")
bell2 = np.array([1,0,0,1])/np.sqrt(2)
print(f"  S of one Bell pair = {float(entanglement_entropy(jnp.asarray(bell2).reshape(2,2),(0,))):.6f} bit")

# %% [markdown]
# ## 3. The zoo classified
#
# Now the useful exercise: take the standard states and look at them through **three different cuts** — one spin, two
# spins, and half the chain. A single number at a single cut will turn out not to be a classification.

# %%
rows = []
for N in (4, 6, 8):
    for name, psi in (("product", product_state("+" * N)), ("GHZ", ghz_state(N)),
                      ("W", w_state(N)), ("Dicke k=N/2", dicke_state(N, N // 2))):
        r1 = rdm(psi, (0,)); r2 = rdm(psi, (0, 1)); rh = rdm(psi, tuple(range(N // 2)))
        rk = int((np.linalg.eigvalsh(np.asarray(rh)) > 1e-12).sum())
        rows.append((name, N, float(purity(r1)), S_bits(r1), neg(r2), float(purity(rh)), S_bits(rh), rk))

print(f"{'state':>12s} {'N':>2s} | {'purity_1':>9s} {'S_1':>7s} | {'neg(pair)':>9s} | {'purity_A':>9s} {'S_A':>7s} {'rank':>5s}")
for r in rows:
    print(f"{r[0]:>12s} {r[1]:2d} | {r[2]:9.6f} {r[3]:7.4f} | {r[4]:9.6f} | {r[5]:9.6f} {r[6]:7.4f} {r[7]:5d}")

# %% [markdown]
# Read the one-spin columns first. **GHZ and the Dicke state are identical there**: both have $\rho_q = 1/2$ exactly,
# purity $1/2$, one bit. By that measure they are equally and maximally entangled. Read the pair column and they
# separate completely: GHZ has *exactly zero*, the Dicke state does not.
#
# The GHZ pair is worth writing out, because it needs no machinery at all to classify.

# %%
rho_pair = np.asarray(rdm(ghz_state(6), (0, 1))).real
print("two-spin reduced state of GHZ_6:")
print(rho_pair)
print("\nthis is (1/2)|00><00| + (1/2)|11><11|, a mixture of two PRODUCT states,")
print("so it is separable by the definition -- no criterion needed.")
print(f"and indeed the negativity is {neg(rdm(ghz_state(6),(0,1))):.1e}")

# %% [markdown]
# The W state is the opposite case: its pairs *are* entangled, and the amount decreases with $N$ because the same
# single excitation has to be shared among more and more pairs. This is our first sight of **monogamy**.
#
# So GHZ and W differ in *where* their entanglement lives. GHZ holds one bit shared globally, invisible in any pair
# and destroyed by losing one spin; W holds a little in every pair and survives the loss of one. No single number
# orders them: they are inequivalent under LOCC, i.e. different **classes**, not two points on one scale.

# %%
print("W state: pairwise negativity and the closed form for the one-spin purity")
for N in (4, 6, 8, 10):
    psi = w_state(N)
    print(f"  N={N:2d}: neg(pair) = {neg(rdm(psi,(0,1))):.6f}   "
          f"purity_1 = {float(purity(rdm(psi,(0,)))):.6f}  vs  (1-1/N)^2+(1/N)^2 = {(1-1/N)**2+(1/N)**2:.6f}")

# %% [markdown]
# ## 4. The negativity of a pure-state cut
#
# Before we leave pure states: what does the quantity of Section 7 below, the negativity, do when applied to a *cut*
# of a pure state? For a pure state the partial transpose has eigenvalues $\lambda_i\lambda_j$ on the diagonal and
# $\pm\lambda_i\lambda_j$ in pairs off it, so its trace norm is $(\sum_k\lambda_k)^2$ and
#
# $$ E_{\mathcal N} = \log_2\Big(\sum_k \lambda_k\Big)^2 = 2\log_2\sum_k\lambda_k, $$
#
# which is the **Rényi-$\tfrac12$ entropy** of the Schmidt spectrum. So for a pure state the negativity is a different
# function of the same data and carries nothing new. Worth knowing, because it tells us what the negativity is *for*:
# not a competitor to $S_A$ across a cut, but the tool for the case the entropy cannot handle at all.

# %%
print("log-negativity of a half-chain cut vs the Renyi-1/2 entropy, XXZ ground state at (Delta,hx)=(1.5,0.5)")
for N in (8, 10):
    _, psi = lanczos_ground_state(heisenberg_terms(N, 1., 1., 1.5, hx=0.5), N)
    A = tuple(range(N // 2))
    lam = np.asarray(schmidt_values(psi, A))
    E_N = float(negativity(to_dm(psi), A)[1])
    print(f"  N={N:2d}:  E_N = {E_N:.10f}   2*log2(sum lam) = {2*np.log2(lam.sum()):.10f}   "
          f"S_vN = {S_bits(rdm(psi,A)):.10f} bit")

# %% [markdown]
# ## 5. Entanglement of mixed states
#
# Two spins inside a longer chain are not a pure state. Neither is a register that has passed through a noisy device,
# nor a subsystem of anything. The moment the state is mixed, the argument of Section 2 collapses — and it collapses
# in a way worth seeing rather than being told.

# %%
bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
rho_bell = np.outer(bell, bell.conj())
rho_coin = 0.5 * (np.diag([1, 0, 0, 0]) + np.diag([0, 0, 0, 1])).astype(complex)

print(f"{'state':>34s} {'purity':>8s} {'rho_A':>16s} {'S(rho_A)':>9s} {'negativity':>11s}")
for name, r in (("Bell (|00>+|11>)/sqrt2", rho_bell), ("coin toss: |00> or |11>", rho_coin)):
    rA = np.asarray(rdm_dm(jnp.asarray(r).reshape(2, 2, 2, 2), (0,)))
    print(f"{name:>34s} {float(purity(jnp.asarray(r))):8.4f} {'diag'+str(np.round(np.diag(rA).real,3)):>16s} "
          f"{S_bits(rA):9.4f} {neg(r):11.6f}")

# %% [markdown]
# The entropy of the reduced state is **exactly one bit for both**. Used as an entanglement measure it reports
# maximal entanglement for a state that has none. What it actually measures is how mixed $\rho_A$ is, and for a mixed
# global state that has two sources: entanglement with $B$, *and* the classical randomness already in $\rho_{AB}$.
# Only when the global state is pure can the second source be excluded.
#
# The deeper problem is that a density matrix does not remember its ensemble. The natural definition of its
# entanglement has to take the best case over all preparations,
#
# $$ E_F(\rho) = \min_{\{p_i,\psi_i\}} \sum_i p_i\, S\big(\rho_A^{(i)}\big), $$
#
# the **entanglement of formation**. This is a genuine measure and also a minimisation over an unbounded family. It
# has a closed form for two qubits and none in general, and deciding whether a mixed state is entangled at all becomes
# computationally hard as the dimension grows. There is no known efficiently computable measure of mixed-state
# entanglement that is faithful in general. That is the honest situation — which is why we now switch from *measures*
# to one computable *criterion*.

# %% [markdown]
# ## 6. The partial transpose
#
# Write the density matrix with its indices split, $\rho_{(ab),(a'b')}$, and transpose **only** the indices of $A$:
#
# $$ \big(\rho^{T_A}\big)_{(ab),(a'b')} = \rho_{(a'b),(ab')}. $$
#
# In matrix form this is painful index gymnastics. In **tensor** form it is trivial: a density matrix of $N$ spins is
# a tensor with $N$ ket axes followed by $N$ bra axes, and the partial transpose on a set $A$ just swaps axis $q$ with
# axis $N+q$ for every $q \in A$. One `transpose`, no arithmetic.

# %%
rho_t = jnp.asarray(rho_bell).reshape(2, 2, 2, 2)        # [ket_0, ket_1, bra_0, bra_1]
pt = partial_transpose(rho_t, (0,))
print("Bell pair: eigenvalues of rho and of its partial transpose")
print("  rho      :", np.round(np.linalg.eigvalsh(np.asarray(dm_matrix(rho_t))), 6))
print("  rho^{T_A}:", np.round(np.linalg.eigvalsh(np.asarray(dm_matrix(pt))), 6), "  <- one NEGATIVE eigenvalue")
print("\nand doing it by hand with einsum gives the same thing:")
by_hand = np.einsum("abcd->cbad", np.asarray(rho_t))
print("  max|engine - einsum| =", np.abs(np.asarray(pt) - by_hand).max())

# %% [markdown]
# ### The matrix and tensor forms of a density operator
#
# There are two ways to hold a density matrix in this course, and they are not interchangeable. `rdm` returns a
# $2^{|A|}\times 2^{|A|}$ **matrix**; the channels and the partial transpose want the rank-$2N$ **tensor** with every
# axis of length two. Handing the matrix form to a routine expecting the tensor makes it read $N$ from the number of
# axes, find $N=1$, and transpose the *whole* matrix — which leaves every eigenvalue where it was and reports a
# negativity of exactly zero. Zero negativity is a physically meaningful answer, so nothing looks wrong.
#
# The engine therefore accepts either form and converts, and refuses anything that is neither. A $(2^N,2^N)$ matrix is
# unambiguous, because a density *tensor* has all axes of length two.

# %%
print("the same Bell pair, handed in both forms:")
print(f"  as a 4x4 matrix        : negativity = {neg(rho_bell):.6f}")
print(f"  as a (2,2,2,2) tensor  : negativity = {neg(rho_t):.6f}")
print(f"  as_dm_tensor reshapes  : {np.asarray(rho_bell).shape} -> {as_dm_tensor(rho_bell).shape}")
for bad in (np.zeros((3, 3)), np.zeros((2, 2, 2))):
    try:
        negativity(bad, (0,)); print("   NOT REFUSED:", bad.shape)
    except ValueError as e:
        print(f"   shape {bad.shape} refused: {e}")

# %% [markdown]
# ## 7. The PPT criterion and the negativity
#
# Why does a negative eigenvalue prove anything? Take a separable state,
# $\rho=\sum_i p_i\,\sigma^{(i)}_A\otimes\tau^{(i)}_B$. The partial transpose gives
# $\sum_i p_i (\sigma^{(i)}_A)^{T}\otimes\tau^{(i)}_B$, and the transpose of a density matrix is again a density
# matrix, so the result is a sum of positive operators with positive weights: still positive. Hence
#
# $$ \rho \text{ separable} \implies \rho^{T_A} \ge 0, $$
#
# and by contraposition **a negative eigenvalue of the partial transpose proves entanglement** (Peres 1996). It is
# one-directional in general; for two qubits and for qubit–qutrit it is also sufficient (Horodecki 1996), so for the
# spin pairs in this notebook it decides the question completely.
#
# The criterion becomes a number by measuring how negative the partial transpose is:
#
# $$ \mathcal N(\rho) = \frac{\Vert\rho^{T_A}\Vert_1 - 1}{2} = \sum_{\mu_i<0}\vert\mu_i\vert, \qquad
#    E_{\mathcal N}(\rho) = \log_2\Vert\rho^{T_A}\Vert_1 . $$

# %% [markdown]
# ### The meaning of the negativity
#
# The definition looks arbitrary until one asks what the partial transpose does *geometrically*.
#
# Transposing a **whole** density matrix is harmless: $\rho^{T}$ has the same eigenvalues as $\rho$, so it is again a
# legitimate state. Transposing **half** of one need not be. For a separable state it is harmless, because the
# operation acts separately on each factor of each term and each transposed factor is still a density matrix — that is
# the proof above. So the partial transpose maps separable states *into* the set of states, and it can only take $\rho$
# **outside** that set if $\rho$ was entangled. A negative eigenvalue is what leaving the set looks like: an operator
# with a negative eigenvalue is not a density matrix, because it assigns a negative probability to some outcome.
#
# So the negativity is a **distance**: how far outside the set of states the partial transpose lands, measured as the
# total weight of negative eigenvalue. Zero means the partial transpose is still a state — no contradiction produced,
# no entanglement proved.
#
# The zero case matches the entropy exactly. For a pure state $E_{\mathcal N} = 2\log_2\sum_k\lambda_k$, and since
# $\sum_k\lambda_k^2 = 1$ with $\lambda_k \ge 0$, the sum $\sum_k\lambda_k$ is at least 1, with equality exactly when
# a single $\lambda_k = 1$. A product state gives $\log_2 1 = 0$ again.
#
# And $E_{\mathcal N}$ counts the same branches as $S$ whenever the spectrum is flat: for $r$ equal Schmidt values
# $\sum_k\lambda_k = \sqrt r$, so $E_{\mathcal N} = \log_2 r = S_A$.

# %%
print("for a FLAT Schmidt spectrum of r values, E_N = S = log2 r exactly:")
for r in (2, 4, 8, 16):
    lam = np.ones(r)/np.sqrt(r)
    print(f"  r={r:2d}: S = {-(lam**2*np.log2(lam**2)).sum():.6f}   E_N = {2*np.log2(lam.sum()):.6f}   "
          f"log2 r = {np.log2(r):.6f}")

print("\nwhen the spectrum is NOT flat, E_N > S: it weights the small Schmidt values more heavily")
for N in (8, 10, 12):
    _, psi = lanczos_ground_state(heisenberg_terms(N, 1., 1., 1.5, hx=0.5), N)
    A = tuple(range(N//2))
    lam = np.asarray(schmidt_values(psi, A)); lam = lam[lam > 1e-14]
    S = float(entanglement_entropy(psi, A)); EN = 2*np.log2(lam.sum())
    print(f"  XXZ N={N:2d}: S = {S:.6f} bit, E_N = {EN:.6f}, ratio {EN/S:.4f}")

print("\nThis is why E_N and not N is the version quoted in bits: a Bell pair has N = 1/2 but E_N = 1 bit,")
print("on the same scale as the entropy.")
n_b, en_b = (float(x) for x in negativity(rho_bell, (0,)))
print(f"  Bell pair: N = {n_b:.6f},  E_N = {en_b:.6f} bit")

# %% [markdown]
# ### Calibration on Werner states
#
# A new quantity is worth nothing until it has been run where the answer is known. The **Werner state** is a Bell pair
# mixed with white noise,
#
# $$ \rho_W(p) = p\,\vert\Phi^+\rangle\langle\Phi^+\vert + (1-p)\,\frac{\mathbb 1}{4}, $$
#
# separable for $p\le\tfrac13$ and entangled above. Nothing in the code knows that number.

# %%
print("Werner state: the PPT boundary should appear at p = 1/3")
print(f"  {'p':>7s} {'min eig of PT':>14s} {'negativity':>11s}")
for p in (0.20, 0.30, 1/3, 0.34, 0.50, 1.00):
    r = p * rho_bell + (1 - p) * np.eye(4) / 4
    mu = np.linalg.eigvalsh(np.asarray(dm_matrix(partial_transpose(r, (0,)))))
    verdict = "entangled" if neg(r) > 1e-12 else "separable (PPT)"
    print(f"  {p:7.4f} {mu.min():+14.6f} {neg(r):11.6f}   {verdict}")

ps = np.linspace(0, 1, 201)
ns = [neg(p * rho_bell + (1 - p) * np.eye(4) / 4) for p in ps]
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(ps, ns, lw=1.4)
ax.axvline(1/3, ls="--", color="k", lw=0.9)
ax.text(1/3 + 0.02, max(ns) * 0.8, "p = 1/3", fontsize=9)
ax.set_xlabel("p"); ax.set_ylabel("negativity"); ax.set_title("Werner state: entanglement switches on at p = 1/3")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Entanglement death under noise
#
# Now the point of Section 5 made quantitative. Take a Bell pair and apply the depolarising channel independently to
# each spin, as a noisy device would. Watch the reduced-state entropy and the negativity disagree completely.

# %%
print(f"{'p per spin':>10s} {'purity':>8s} {'S(rho_A)':>9s} {'negativity':>11s}")
pp = np.sort(np.concatenate([np.linspace(0, 0.45, 46), [0.75 * (1 - 1 / np.sqrt(3))]]))
negs, purs = [], []
for p in pp:
    rho = jnp.asarray(rho_bell).reshape(2, 2, 2, 2)
    for q in (0, 1):
        rho = apply_kraus_dm(rho, kraus_depolarizing(float(p)), [q])
    negs.append(neg(rho)); purs.append(float(purity(dm_matrix(rho))))
    if np.isclose(p, [0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.75 * (1 - 1 / np.sqrt(3)), 0.35], atol=1e-9).any():
        rA = np.asarray(rdm_dm(rho, (0,)))
        print(f"{p:10.2f} {purs[-1]:8.4f} {S_bits(rA):9.4f} {negs[-1]:11.6f}")

p_death = pp[np.argmax(np.array(negs) <= 1e-12)] if (np.array(negs) <= 1e-12).any() else np.nan
p_star = 0.75 * (1 - 1 / np.sqrt(3))
print(f"\nthe pair becomes separable at about p = {p_death:.3f} per spin from this scan,")
print(f"and the exact threshold is p* = (3/4)(1 - 1/sqrt(3)) = {p_star:.6f}")
print("  it is not a fitted number: depolarising BOTH spins of |Phi+> with strength p each gives exactly the")
print(f"  Werner state with w = (1-4p/3)^2 = {(1-4*p_star/3)**2:.6f}, and Werner is separable for w <= 1/3.")
print("while S(rho_A) never moves from 1 bit: the depolarising channel is unital, so it maps 1/2 to itself, the")
print("input is a Bell pair whose reduced state is already 1/2, and the far channel is trace preserving.")

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(pp, negs, lw=1.4, label="negativity")
ax.plot(pp, [1.0] * len(pp), "--", lw=1.2, label=r"$S(\rho_A)$ [bits]")
ax.plot(pp, purs, ":", lw=1.2, label="purity")
ax.axvline(p_star, color="k", lw=0.8)
ax.annotate(r"$p^*=\frac{3}{4}(1-1/\sqrt{3})$", (p_star, 0.25), fontsize=7,
            xytext=(p_star + 0.03, 0.30), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlabel("depolarising strength p per spin"); ax.legend(fontsize=8)
ax.set_title("entanglement dies; the subsystem entropy notices nothing")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9. A spin chain: three quantities against distance
#
# Now the question that opened the notebook. Take the ground state of the XXZ chain in a transverse field,
#
# $$ H = \sum_j \big(X_jX_{j+1} + Y_jY_{j+1} + \Delta\, Z_jZ_{j+1}\big) + h_x\sum_j X_j, $$
#
# fix a reference spin near the middle, and follow three quantities as the partner moves away:
#
# * the **negativity** $\mathcal N(r)$ of the pair — their entanglement;
# * the **mutual information** $I(i{:}j) = S(\rho_i) + S(\rho_j) - S(\rho_{ij})$ — their total correlation, classical
#   and quantum together;
# * the **entropy** $S(\rho_{ij})$ of the pair itself.

# %%
N = 14
i0 = N // 2 - 1
RS = list(range(1, N // 2 + 1))
PHASES = [("gapless, D=0.5", 0.5, 0.0), ("Heisenberg, D=1", 1.0, 0.0),
          ("Neel, D=1.5", 1.5, 0.0), ("field, hx=1.5", 0.0, 1.5)]
curves = {}
for name, D, h in PHASES:
    _, psi = lanczos_ground_state(heisenberg_terms(N, 1., 1., D, hx=h), N)
    Si = S_bits(rdm(psi, (i0,)))
    nn, ii, ss = [], [], []
    for r in RS:
        j = i0 + r
        rij = rdm(psi, (i0, j))
        Sij = S_bits(rij); Sj = S_bits(rdm(psi, (j,)))
        nn.append(neg(rij)); ii.append(Si + Sj - Sij); ss.append(Sij)
    curves[name] = (nn, ii, ss)
    print(f"{name}:  S(one spin) = {Si:.4f} bit")
    print("   r    " + "".join(f"{r:8d}" for r in RS))
    print("   N(r) " + "".join(f"{v:8.4f}" for v in nn))
    print("   I(r) " + "".join(f"{v:8.4f}" for v in ii))
    print("   S_ij " + "".join(f"{v:8.4f}" for v in ss))

# %%
fig, ax = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
for name, _, _ in PHASES:
    nn, ii, ss = curves[name]
    ax[0].plot(RS, nn, "-o", ms=3, label=name)
    ax[1].semilogy(RS, np.maximum(ii, 1e-6), "-o", ms=3)
    ax[2].plot(RS, ss, "-o", ms=3)
ax[0].set_ylabel("negativity N(r)"); ax[0].set_title("(a) entanglement of the pair")
ax[1].set_ylabel("I(i:j) [bits]");   ax[1].set_title("(b) total correlation")
ax[2].set_ylabel(r"$S(\rho_{ij})$ [bits]"); ax[2].set_title("(c) mixedness of the pair")
ax[2].axhline(2.0, ls=":", color="k", lw=0.8)
for a in ax:
    a.set_xlabel("distance r")
ax[0].legend(fontsize=7)
plt.show()

# %% [markdown]
# All three panels say something, and none of them says what one might have expected.
#
# **(a) Entanglement between two spins is a nearest-neighbour affair.** The negativity is around $0.2$–$0.3$ at
# $r=1$ and *exactly zero* at every larger separation, in every phase. This is not a small number below a tolerance:
# the partial transpose is positive, and for two qubits that *proves* the pair is separable.
#
# **(b) Correlation is not entanglement.** The mutual information at $r=2$ and $r=3$ is a sizeable fraction of a bit
# and decays slowly, and it distinguishes the phases clearly. All of that correlation is real, and none of it is
# entanglement. (Compare the $\langle Z_iZ_j\rangle$ correlators of the entanglement chapter: large and alternating
# out to many sites.)
#
# **(c) The pair entropy measures the wrong thing.** $S(\rho_{ij})$ *increases* with $r$, towards $2\,S(\rho_i) = 2$
# bits. As the two spins decorrelate, $\rho_{ij}\to\rho_i\otimes\rho_j$ and the entropy becomes the sum of the parts.
# A quantity that grows as two spins are pulled apart is not measuring their mutual entanglement — it measures how
# entangled the *pair* is with the rest of the chain.

# %% [markdown]
# ### Monogamy, measured
#
# The three panels have a single explanation. A spin of this chain is maximally mixed, so it is as entangled with the
# rest of the register as it can be. Sum its pairwise negativities with *all* other spins and compare.

# %%
Nm = 10
_, psi = lanczos_ground_state(heisenberg_terms(Nm, 1., 1., 1., hx=0.0), Nm)
c = Nm // 2 - 1
S_one = S_bits(rdm(psi, (c,)))
pairs = [(j, neg(rdm(psi, (c, j)))) for j in range(Nm) if j != c]
print(f"Heisenberg chain, N={Nm}, reference spin {c}")
print(f"  S(one spin vs the rest)            = {S_one:.6f} bit")
print(f"  sum of its pairwise negativities   = {sum(v for _, v in pairs):.6f}")
print(f"  non-zero pairs: {[(j, round(v,4)) for j, v in pairs if v > 1e-12]}")
print("\nAlmost all of the spin's entanglement is in NO pair at all: it is shared among many spins at once,")
print("and monogamy forbids it from being available bilaterally.  Pairwise entanglement is short-ranged in a")
print("ground state not because the correlations are, but because there is nothing left over.")

# %% [markdown]
# %% [markdown]
# ## 10. The phase diagram in two measures
#
# Sections 1 to 9 compared the entropy and the negativity on states we chose. The chapter's central figure does it
# across a whole phase diagram, the $(\Delta, h_x)$ plane of the XXZ chain in a transverse field, and the point of that
# figure is a negative result: the two quantities take their largest values in *different places*, so they are not two
# estimates of one thing.
#
# The chapter's maps are $N=16$ on a $31\times31$ grid and $N=32$ by DMRG on a $21\times21$ grid, which together take
# hours. The mechanism does not need that resolution, so here we run $N=12$ on an $11\times11$ grid, about a minute,
# and check that the qualitative statement already holds. Anyone wanting the chapter's figure changes three numbers.
#
# Two things to watch for, both of which the chapter discusses and neither of which is obvious from a colour map. The
# negativity falls to zero on the ferromagnetic side ($\Delta \lesssim -1$, $h_x = 0$) while the entropy there is
# almost a full bit: that region is a cat state of two ferromagnets, so the entropy is reporting one *classical* bit of
# which-branch information and the negativity is correctly reporting that a mixture of product states holds no
# entanglement. And the zero is a numerical zero, not an identity.

# %%
Nmap, ng = 12, 11
Ds = np.linspace(-2.0, 2.0, ng)
hs = np.linspace(0.0, 2.0, ng)
cm = Nmap // 2 - 1
S_map = np.zeros((ng, ng)); N_map = np.zeros((ng, ng)); I_map = np.zeros((ng, ng))
t0 = time.perf_counter()
for a, hv in enumerate(hs):
    for b, Dv in enumerate(Ds):
        _, pv = lanczos_ground_state(heisenberg_terms(Nmap, 1., 1., Dv, hx=hv), Nmap)
        S_map[a, b] = float(entanglement_entropy(pv, tuple(range(Nmap // 2))))
        rij = rdm(pv, (cm, cm + 1))
        N_map[a, b] = neg(rij)
        Si = S_bits(rdm(pv, (cm,))); Sj = S_bits(rdm(pv, (cm + 1,)))
        I_map[a, b] = Si + Sj - S_bits(rij)
print(f"{ng}x{ng} grid at N={Nmap} in {time.perf_counter()-t0:.1f} s")

iS = np.unravel_index(S_map.argmax(), S_map.shape)
iN = np.unravel_index(N_map.argmax(), N_map.shape)
print(f"  half-chain entropy: range {S_map.min():.3f} .. {S_map.max():.3f} bits, "
      f"max at (Delta,hx) = ({Ds[iS[1]]:+.2f}, {hs[iS[0]]:.2f})")
print(f"  nn negativity     : range {N_map.min():.3f} .. {N_map.max():.3f},      "
      f"max at (Delta,hx) = ({Ds[iN[1]]:+.2f}, {hs[iN[0]]:.2f})")
print(f"  mutual information: range {I_map.min():.3f} .. {I_map.max():.3f} bits  (never zero anywhere)")
print(f"  the two maxima are at different points: {iS != iN}")

# the ferromagnetic corner, where the entropy and the negativity disagree for a reason
ife = int(np.argmin(np.abs(Ds + 2.0)))
print(f"\nferromagnetic corner (Delta={Ds[ife]:+.2f}, hx=0): S = {S_map[0, ife]:.4f} bits, "
      f"negativity = {N_map[0, ife]:.3e}")
print("  how many of the grid points return EXACTLY 0.0 negativity:",
      int((N_map == 0.0).sum()), "of", N_map.size, "-- the rest are small but non-zero")

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 3.1), constrained_layout=True)
for a, (M, ttl) in enumerate(((S_map, "half-chain entropy [bit]"),
                              (N_map, r"negativity $\mathcal{N}(1)$"),
                              (I_map, "mutual information [bit]"))):
    im = ax[a].imshow(M, origin="lower", aspect="auto", cmap="magma",
                      extent=[Ds[0], Ds[-1], hs[0], hs[-1]])
    ax[a].set_xlabel(r"$\Delta$"); ax[a].set_ylabel(r"$h_x$"); ax[a].set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax[a])
plt.show()

# %% [markdown]
# The two maxima land in different places even on this coarse grid, which is the whole claim. The entropy is largest
# near the ferromagnetic boundary at zero field; the negativity is largest at positive $\Delta$ *and* finite field,
# where the chain is neither ordered nor field-dominated. The mutual information never vanishes, so across a whole
# region the central pair is strongly correlated, provably unentangled, and sitting inside a state with a substantial
# block entropy.
#
# The chapter also maps the same plane at $N=32$ by DMRG, where no state vector exists, and finds that the entropy's
# maximum grows with the chain while the negativity's does not. That is the difference between a quantity attached to a
# cut, which has more block to grow into, and a quantity attached to one pair of neighbours, which has nothing to grow
# into. Reproducing it needs only `dmrg` in place of `lanczos_ground_state` and `mps_rdm2` in place of `rdm`, both of
# which the next sections build and validate.

# ## 11. The same measures from a matrix product state
#
# Everything so far used an exact state vector, which caps us at about twenty spins. Both quantities can be had from a
# matrix product state, and the two divide along a line worth noticing.
#
# **The entropy is free.** An MPS in canonical form *stores* the Schmidt values of every cut, so the entanglement
# entropy is a sum over numbers already in the data structure — that is what `mps_entropies` returns. No contraction.
#
# **The negativity is not free**, because it needs the two-site reduced density matrix, and that is a contraction. Let
# us derive it rather than quote it. For a right-canonical MPS with Schmidt values `lam`:
#
# 1. Everything to the **right** of site $j$ contracts to the identity, by right-canonicity. So it costs nothing.
# 2. Everything to the **left** of site $i$ contracts to the Schmidt weights $\lambda_i^2$ of that cut.
# 3. In between, each site is traced over its physical leg — a transfer matrix.
#
# So: open the physical legs at site $i$, walk the transfer matrix to site $j$, and close with the right bond traced.

# %%
N, chi, D, h = 12, 32, 1.5, 0.5
B, lam, hist = dmrg(xxz_mpo(N, 1., 1., D, hx=h), product_mps(("01" * N)[:N], chi)[0], chi, 6)
i, j = 4, 7

# step 1: open the physical legs at site i, weighted by the Schmidt values of the cut to its left
T = jnp.einsum("a,asb,atc->stbc", lam[i] ** 2, B[i], jnp.conj(B[i]))
print(f"after site {i}: T has shape {T.shape}   (s,t = physical of site i; b,c = bond ket/bra)")

# step 2: walk through the sites in between, tracing their physical legs
for k in range(i + 1, j):
    T = jnp.einsum("stbc,bud,cue->stde", T, B[k], jnp.conj(B[k]))
    print(f"  through site {k}: shape {T.shape}   (physical leg u is summed -> traced out)")

# step 3: close at site j, keeping its physical legs open and tracing the right bond
R = jnp.einsum("stbc,bud,cvd->sutv", T, B[j], jnp.conj(B[j]))
rho_mps = R.reshape(4, 4)
print(f"\nrho has shape {rho_mps.shape}, trace {float(jnp.trace(rho_mps).real):.12f}")
print("and the engine function does exactly this:")
print("  max|by hand - mps_rdm2| =", np.abs(np.asarray(rho_mps) - np.asarray(mps_rdm2(B, lam, i, j))).max())

# %% [markdown]
# ## 12. Agreement of the two implementations
#
# Two pieces of code meant to compute the same thing are worth nothing until they have been made to agree. There is a
# size at which the agreement must be *exact*: a matrix product state with $\chi = 2^{N/2}$ discards nothing, so at
# that bond dimension DMRG and exact diagonalisation are the same calculation in different coordinates.

# %%
print("CHECK 1 -- at chi = 2^(N/2) the MPS is exact, so everything must agree to rounding")
print(f"{'point':>12s} {'N':>2s} {'chi':>4s} | {'|dE|':>9s} {'|dS|':>9s} {'max|d rho_ij| all pairs':>24s}")
for name, D_, h_ in (("gapless", 0.5, 0.0), ("Heisenberg", 1.0, 0.0), ("Neel", 2.0, 0.0),
                     ("Neel+field", 1.5, 0.5), ("field", 0.0, 1.5)):
    for Nv in (4, 6, 8):
        chiv = 2 ** (Nv // 2)
        E, psi_v = lanczos_ground_state(heisenberg_terms(Nv, 1., 1., D_, hx=h_), Nv)
        Bv, lamv, hv = dmrg(xxz_mpo(Nv, 1., 1., D_, hx=h_), product_mps(("01" * Nv)[:Nv], chiv)[0], chiv, 8)
        dS = abs(float(entanglement_entropy(psi_v, tuple(range(Nv // 2))))
                 - float(np.asarray(mps_entropies(lamv))[Nv // 2]))
        drho = max(np.abs(np.asarray(rdm(psi_v, (a, b))) - np.asarray(mps_rdm2(Bv, lamv, a, b))).max()
                   for a in range(Nv - 1) for b in range(a + 1, Nv))
        print(f"{name:>12s} {Nv:2d} {chiv:4d} | {abs(E-hv[-1][3]):9.1e} {dS:9.1e} {drho:24.1e}")

# %% [markdown]
# Away from that limit the MPS is an approximation, and *how* it converges is the second check. The useful thing to
# watch is whether the negativity converges as fast as the energy, and whether the discarded weight that DMRG already
# reports is a usable error bar on it.

# %%
print("CHECK 2 -- convergence in chi at N=12, (Delta,hx)=(1.5,0.5)")
E_ex, psi_ex = lanczos_ground_state(heisenberg_terms(12, 1., 1., 1.5, hx=0.5), 12)
S_ex = float(entanglement_entropy(psi_ex, tuple(range(6))))
n_ex = neg(rdm(psi_ex, (5, 6)))
print(f"  exact: E = {E_ex:.12f}   S = {S_ex:.10f} bit   N(1) = {n_ex:.10f}")
print(f"  {'chi':>4s} {'|dE|':>10s} {'|dS|':>10s} {'|dN(1)|':>10s} {'discarded wt':>13s}")
conv = []
for chiv in (2, 4, 8, 16, 32, 64):
    Bv, lamv, hv = dmrg(xxz_mpo(12, 1., 1., 1.5, hx=0.5), product_mps("01" * 6, chiv)[0], chiv, 8)
    eps = max(x[4] for x in hv if x[0] == hv[-1][0])
    dE = abs(E_ex - hv[-1][3])
    dS = abs(S_ex - float(np.asarray(mps_entropies(lamv))[6]))
    dn = abs(n_ex - neg(mps_rdm2(Bv, lamv, 5, 6)))
    conv.append((chiv, dE, dS, dn, eps))
    print(f"  {chiv:4d} {dE:10.2e} {dS:10.2e} {dn:10.2e} {eps:13.2e}")

# %%
print("CHECK 3 -- is the MPS two-site reduced state even a legitimate density matrix?")
Bv, lamv, hv = dmrg(xxz_mpo(12, 1., 1., 1.5, hx=0.5), product_mps("01" * 6, 64)[0], 64, 8)
wtr = wher = 0.0; weig = 0.0
for a in range(11):
    for b in range(a + 1, 12):
        r = np.asarray(mps_rdm2(Bv, lamv, a, b))
        wtr = max(wtr, abs(np.trace(r).real - 1)); wher = max(wher, np.abs(r - r.conj().T).max())
        weig = min(weig, np.linalg.eigvalsh(r).min())
print(f"  over all 66 pairs: |trace-1| <= {wtr:.1e}, non-hermiticity <= {wher:.1e}, "
      f"most negative eigenvalue {weig:+.1e}")
print("\n  This one matters: a contraction returning a slightly non-positive matrix would hand the partial")
print("  transpose spurious negative eigenvalues and report entanglement where there is none.")

# %%
fig, ax = plt.subplots(figsize=(5, 2.8))
cc = [c[0] for c in conv]
ax.loglog(cc, [c[1] for c in conv], "-o", ms=3, label=r"$|\Delta E|$")
ax.loglog(cc, [c[3] for c in conv], "-s", ms=3, label=r"$|\Delta \mathcal{N}(1)|$")
ax.loglog(cc, [c[4] for c in conv], "--^", ms=3, label="discarded weight")
ax.set_xlabel(r"bond dimension $\chi$"); ax.set_ylabel("error")
ax.legend(fontsize=8); ax.set_title("the negativity converges as fast as the energy")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 13. The cost of each route
#
# Both methods give the same numbers, so the choice between them is cost. Time one ground state at increasing $N$.
# Run this on an otherwise idle machine, or the timings mean nothing. Two further caveats: the DMRG column is six
# sweeps throughout, so it scales linearly in a number that is a choice rather than a property of the method; and the
# first Lanczos call pays for JIT compilation, which at $N=12$ is most of the measured time (about 1.3 s cold against
# 0.26 s warm). The cell below therefore runs each solver once to warm it before timing. What survives the caveats is
# the shape: one cost grows exponentially in $N$ and the other does not.

# %%
print(f"{'N':>3s} | {'Lanczos [s]':>11s} {'DMRG [s]':>9s} | {'|dE|':>9s} {'|dS|':>9s} {'|dN(1)|':>9s}")
cost = []
for Nv in (12, 14, 16, 18, 20):
    cv = Nv // 2 - 1
    lanczos_ground_state(heisenberg_terms(Nv, 1., 1., 1.5, hx=0.5), Nv)          # warm the JIT cache
    t0 = time.perf_counter(); Ev, pv = lanczos_ground_state(heisenberg_terms(Nv, 1., 1., 1.5, hx=0.5), Nv)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    Bv, lamv, hv = dmrg(xxz_mpo(Nv, 1., 1., 1.5, hx=0.5), product_mps(("01" * Nv)[:Nv], 64)[0], 64, 6)
    t3 = time.perf_counter()
    dS = abs(float(entanglement_entropy(pv, tuple(range(Nv // 2)))) - float(np.asarray(mps_entropies(lamv))[Nv // 2]))
    dn = abs(neg(rdm(pv, (cv, cv + 1))) - neg(mps_rdm2(Bv, lamv, cv, cv + 1)))
    cost.append((Nv, t1 - t0, t3 - t2))
    print(f"{Nv:3d} | {t1-t0:11.1f} {t3-t2:9.1f} | {abs(Ev-hv[-1][3]):9.1e} {dS:9.1e} {dn:9.1e}")

print("\nThe state vector is faster while it fits and its cost doubles with every spin; the MPS cost grows")
print("roughly linearly in N.  On the machine used for the chapter the two cross at about twenty spins,")
print("and past that only the MPS is available at all.")

# %% [markdown]
# %% [markdown]
# ## 14. The same plane where no state vector exists
#
# Section 10 mapped the phase diagram with an exact state vector. Now that `dmrg` and `mps_rdm2` have been built and
# validated against it, the same map can be made at a chain length no state vector reaches. $N=32$ needs
# $2^{32}\approx4\times10^9$ amplitudes as a state vector, which is not a laptop calculation; as a matrix product state
# at $\chi=32$ it costs a few seconds per point.
#
# The chapter runs a $21\times21$ grid. Here we run $5\times5$, a few minutes, which is enough for the one qualitative
# claim: as the chain grows the entropy maximum grows with it while the negativity maximum does not. That is the
# difference between a quantity attached to a cut, which has more block to grow into, and a quantity attached to a
# single pair of neighbours, which has nothing to grow into.

# %%
Nb, gb, chib = 32, 5, 32
Db = np.linspace(-2.0, 2.0, gb)
hb = np.linspace(0.0, 2.0, gb)
cb = Nb // 2 - 1
S_b = np.zeros((gb, gb)); N_b = np.zeros((gb, gb))
t0 = time.perf_counter()
for a_, hv in enumerate(hb):
    for b_, Dv in enumerate(Db):
        B_, lam_, hist_ = dmrg(xxz_mpo(Nb, 1., 1., Dv, hx=hv),
                               product_mps(("01" * Nb)[:Nb], chib)[0], chib, 4)
        S_b[a_, b_] = float(np.asarray(mps_entropies(lam_))[Nb // 2])
        N_b[a_, b_] = neg(mps_rdm2(B_, lam_, cb, cb + 1))
el = time.perf_counter() - t0
print(f"{gb}x{gb} DMRG grid at N={Nb}, chi={chib} in {el:.0f} s ({el/gb**2:.1f} s per point)")

iS = np.unravel_index(S_b.argmax(), S_b.shape); iN = np.unravel_index(N_b.argmax(), N_b.shape)
print(f"\n  N={Nb}: entropy    range {S_b.min():.3f} .. {S_b.max():.3f} bits, "
      f"max at (Delta,hx) = ({Db[iS[1]]:+.2f}, {hb[iS[0]]:.2f})")
print(f"  N={Nb}: negativity range {N_b.min():.3f} .. {N_b.max():.3f},      "
      f"max at (Delta,hx) = ({Db[iN[1]]:+.2f}, {hb[iN[0]]:.2f})")
print(f"\n  compare Section 10 at N={Nmap}: entropy max {S_map.max():.3f} bits, negativity max {N_map.max():.3f}")
print("  the entropy maximum grew with the chain; the negativity maximum did not.")

ife_b = int(np.argmin(np.abs(Db + 2.0)))
print(f"\nferromagnetic corner (Delta={Db[ife_b]:+.2f}, hx=0):")
print(f"  N={Nb} by DMRG      : S = {S_b[0, ife_b]:.4f} bits, negativity = {N_b[0, ife_b]:.3e}")
print(f"  N={Nmap} by Lanczos : S = {S_map[0, ife]:.4f} bits, negativity = {N_map[0, ife]:.3e}")
print("  Both are right about the state they found. |0...0> and |1...1> are EXACTLY degenerate at every N, so")
print("  Lanczos returns an arbitrary superposition (a cat, whose entropy is one CLASSICAL bit of which-branch")
print("  information) while DMRG settles on one product configuration (entropy zero). The negativity reads zero")
print("  in both, since neither holds pairwise entanglement. A quantity that changes because the solver made an")
print("  arbitrary choice is measuring the choice.")

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True)
for a_, (M, ttl) in enumerate(((S_b, f"half-chain entropy [bit], N={Nb}"),
                               (N_b, rf"negativity $\mathcal{{N}}(1)$, N={Nb}"))):
    im = ax[a_].imshow(M, origin="lower", aspect="auto", cmap="magma",
                       extent=[Db[0], Db[-1], hb[0], hb[-1]])
    ax[a_].set_xlabel(r"$\Delta$"); ax[a_].set_ylabel(r"$h_x$"); ax[a_].set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax[a_])
plt.show()

# ## 15. Key takeaways
#
# * For a **pure** global state, entanglement across a cut is the Schmidt spectrum. The state is entangled exactly when
#   the reduced state is mixed, and every measure is a function of the $p_k$ -- every Schur-concave function of them is
#   an LOCC monotone, so the entropy is canonical rather than unique. The log-negativity of such a cut is the
#   Rényi-$\tfrac12$ entropy, which is the second such monotone and carries nothing new.
# * A single number at a single cut is **not** a classification. GHZ and the Dicke state are indistinguishable by one
#   spin, yet GHZ has exactly zero pairwise entanglement, provably so from its two-spin state. GHZ and W are
#   inequivalent classes.
# * For a **mixed** state the subsystem entropy is not an entanglement measure: a Bell pair and a classical coin toss
#   have the same reduced state and the same one bit of entropy, and negativity $0.5$ against $0$.
# * Quantifying mixed-state entanglement properly means minimising over ensembles, which has no closed form beyond two
#   qubits. The computable substitute is the **PPT criterion**, and the negativity is how negative the partial
#   transpose is. It reproduces the Werner threshold $p=\tfrac13$ exactly rather than approximately, since the smallest
#   partial-transpose eigenvalue there is $(1-3p)/4$; under two-sided depolarising noise the same threshold appears as
#   $p^* = \tfrac34(1-1/\sqrt3) = 0.316987$ per spin.
# * In a chain ground state, pairwise entanglement is nearest-neighbour in every phase, and provably so, since a more
#   distant pair has a positive partial transpose and two spins lie inside the range where PPT decides. The mutual
#   information is long-ranged and phase-dependent, and the pair entropy grows with separation towards
#   $2S(\hat\rho_i)$. Correlation is not entanglement, and monogamy is the reason. Note that on an open chain the
#   nearest-neighbour negativity alternates strongly from bond to bond, so "the central pair" is a specific strong bond.
# * Both measures come from a state vector or from an MPS. The entropy is free from the Schmidt values; the negativity
#   needs a two-site reduced state, a three-step contraction costing $O(j\chi^3)$ when built from the left boundary.
#   The two agree to $10^{-11}$ and the cost crosses over around twenty spins on the machine used here.
#
# ## 16. Exercises
#
# 1. (★) Write the two-spin reduced state of GHZ at $N=5$ and confirm it is $\tfrac12\mathrm{diag}(1,0,0,1)$. Check that
#    its partial transpose has no negative eigenvalue, and explain why the two facts had to agree.
# 2. (★) Find the PPT boundary of the Werner state by bisection instead of by scanning, and compare with $1/3$. Then
#    replace $\mathbb 1/4$ by $\vert00\rangle\langle00\vert$ and find the new threshold.
# 3. (★) Verify the pure-state identity $E_{\mathcal N} = 2\log_2\sum_k\lambda_k$ on a Haar-random state of eight spins
#    for *every* contiguous cut, and plot it against $S(\ell)$.
# 4. (★★) For the Heisenberg ground state at $N=10$, compute the negativity of every pair containing a fixed spin and
#    compare the sum with that spin's entropy against the rest. Repeat at $\Delta=0$ and $\Delta=2$: does the deficit
#    grow or shrink with the anisotropy?
# 5. (★★) Generalise the contraction of Section 10 to two **blocks** of $L$ adjacent sites. Show that $L=1$ reproduces
#    Section 9(a) and that $L=3$ does *not* vanish beyond $r=1$. Why does block size help?
# 6. (★★) Repeat Section 8 with the dephasing channel instead of the depolarising one. At what strength does the pair
#    become separable, and why does the answer differ?
# 7. (★★★) Reproduce Section 12 on your own machine and locate the crossover. Then repeat the DMRG column at
#    $\chi=32$ and $\chi=128$ and explain which parts of the curve move and which do not.
#
# ## 17. References
#
# * A. Peres, *Separability Criterion for Density Matrices*, Phys. Rev. Lett. **77**, 1413 (1996).
# * M. Horodecki, P. Horodecki, R. Horodecki, *Separability of mixed states: necessary and sufficient conditions*,
#   Phys. Lett. A **223**, 1 (1996).
# * G. Vidal, R. F. Werner, *Computable measure of entanglement*, Phys. Rev. A **65**, 032314 (2002).
# * M. B. Plenio, *Logarithmic Negativity: A Full Entanglement Monotone That is not Convex*, Phys. Rev. Lett. **95**,
#   090503 (2005).
# * W. K. Wootters, *Entanglement of Formation of an Arbitrary State of Two Qubits*, Phys. Rev. Lett. **80**, 2245 (1998).
# * V. Coffman, J. Kundu, W. K. Wootters, *Distributed entanglement*, Phys. Rev. A **61**, 052306 (2000).
# * L. Amico, R. Fazio, A. Osterloh, V. Vedral, *Entanglement in many-body systems*, Rev. Mod. Phys. **80**, 517 (2008).
