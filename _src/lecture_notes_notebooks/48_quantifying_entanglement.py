#@title: Quantifying entanglement of quantum states
#@part: Chapter 9 — Entanglement and complexity diagnostics
#@description: Pure-state entanglement and why the Schmidt spectrum is the whole story; the zoo classified (GHZ has no pairwise entanglement, W does); why mixed states are harder; the partial transpose as one axis swap; the PPT criterion and negativity, calibrated on Werner states; entanglement death under noise; negativity, mutual information and pair entropy against distance in the XXZ chain; the same two measures from a matrix product state, with mps_rdm2 derived leg by leg and validated against exact diagonalisation.

# %% [markdown]
# ## 1. Introduction and motivation
#
# The many-body notebooks of this course computed entanglement entropies and put them to work. The bond dimension a
# matrix product state needs is set by an entropy; the entanglement barrier that stops a time evolution is an entropy
# growing in time; the half-chain entropy of the XXZ chain changes character between its phases. None of that asked
# what a *measure* of entanglement is, what else can be measured, or what the entropy fails to see.
#
# This notebook organises those questions around one concrete question:
#
# > The entropy across a cut distinguishes the phases of the XXZ chain. Does it tell us whether **two spins** in that
# > chain are entangled with each other?
#
# The answer is no. Two spins taken out of a ground state are in a **mixed** state, and for mixed states the entropy of
# a subsystem no longer measures entanglement alone: it also measures classical uncertainty. The notebook therefore has
# two halves.
#
# **Road map.** Sections 2 to 4 treat pure states. There the Schmidt spectrum of a cut is the complete answer and the
# entanglement entropy is the canonical measure; the GHZ and W states show that one number at one cut is still not a
# classification. Sections 5 to 8 treat mixed states: why the problem is harder, the partial transpose as an axis swap,
# the PPT criterion and the negativity, calibrated on Werner states and on a Bell pair under depolarising noise.
# Section 9 follows the negativity, the mutual information and the pair entropy against distance in the XXZ chain, and
# measures monogamy with the concurrence. Section 10 maps the $(\Delta,h_x)$ phase diagram in both measures. Sections
# 11 to 13 compute the same quantities from a matrix product state, check them against the exact state vector, and
# compare costs. Section 14 repeats the map at $N=32$, where no state vector is available.
#
# Everything here runs on a laptop CPU in double precision.
#
# ### What you will learn
#
# *Physics*
# * why, for a pure state, entanglement across a cut is a property of the Schmidt spectrum, and in what sense the
#   entanglement entropy is the canonical measure (asymptotic conversion into Bell pairs);
# * why GHZ and W states belong to different classes, measured by their pairwise negativity and concurrence;
# * why the subsystem entropy fails for mixed states, the entanglement of formation, the PPT criterion and the
#   negativity, and the Werner and noisy-Bell-pair thresholds;
# * that pairwise entanglement in a spin-chain ground state is confined to neighbours while correlations are not, and
#   what the monogamy inequality of Coffman, Kundu and Wootters says about it.
#
# *Numerical methods*
# * the partial transpose as an axis permutation of the density tensor;
# * the two-site reduced density matrix of a matrix product state, derived leg by leg, and its validation against exact
#   diagonalisation;
# * convergence in the bond dimension, and the dependence of DMRG on its starting state.
#
# *Implementation practice*
# * wrong controls: every check is paired with a case that must fail;
# * the matrix and tensor forms of a density operator, and the one confusion between them that no code can detect.
#
# ### Prerequisites
# * [06 — states, observables, entanglement](../ch03_matrix_free_engine/06_states_observables_entanglement.ipynb):
#   reduced density matrices, the Schmidt decomposition, the state zoo;
# * [07 — density matrices and quantum channels](../ch03_matrix_free_engine/07_density_matrices_and_quantum_channels.ipynb):
#   density tensors, partial trace, Kraus channels;
# * [11 — Hamiltonians and ground states](../ch05_ground_states_and_unitary_dynamics/11_hamiltonians_and_ground_states.ipynb):
#   Lanczos and the XXZ chain;
# * [18 — matrix product states, DMRG and TEBD](../ch07_tensor_networks/18_mps_tebd.ipynb): canonical forms and DMRG,
#   for Sections 11 to 14;
# * [25 — entanglement negativity](../ch09_entanglement_and_complexity/25_entanglement_negativity.ipynb) treats the
#   negativity in more depth (bound entanglement, block negativity, dynamics); it is a companion, not a prerequisite.
#
# **Conventions.** Qubit $q$ = tensor axis $q$, counted from $0$; $\vert0\rangle$ is the $+1$ eigenstate of $Z$. A
# density tensor of $N$ qubits has $N$ ket axes followed by $N$ bra axes. Hamiltonians are in Pauli convention. All
# entropies are in bits.

# %%
#@engine: haar_state, haar_unitary, rdm, rdm_dm, purity, von_neumann_entropy, entanglement_entropy, schmidt_values, as_dm_tensor, partial_transpose, negativity, to_dm, dm_matrix, ghz_state, w_state, dicke_state, product_state, heisenberg_terms, lanczos_ground_state, xxz_mpo, dmrg, product_mps, mps_entropies, mps_rdm2, kraus_depolarizing, apply_kraus_dm

# %% [markdown]
# Three helpers. `S_bits` is the von Neumann entropy in bits of a density operator in either form. `neg` clips the
# negativity at zero, since the sum of negative eigenvalues of a positive matrix can come out as $-10^{-17}$. `pt_min`
# returns the smallest eigenvalue of the partial transpose of a two-qubit state, which tells *how far* from the PPT
# boundary a state is, not only on which side. `concurrence` is Wootters' formula for two qubits, defined in Section 3.

# %%
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=140)

def S_bits(rho):
    """von Neumann entropy in bits of a density matrix given as a matrix OR a density tensor.
    MATH  S = -sum_k w_k log2 w_k over the eigenvalues w_k > 1e-13 of rho."""
    w = np.linalg.eigvalsh(np.asarray(dm_matrix(jnp.asarray(rho))))
    w = w[w > 1e-13]
    return float(-(w * np.log2(w)).sum())

def neg(rho):
    """Negativity of a two-qubit state across (0 | 1), clipped at zero: tiny negative values are rounding."""
    return max(float(negativity(rho, (0,))[0]), 0.0)

def pt_min(rho):
    """Smallest eigenvalue of the partial transpose (on qubit 0) of a two-qubit state.
    > 0: PPT with a margin (separable, for two qubits); < 0: entangled, and -pt_min is the negativity."""
    return float(np.linalg.eigvalsh(np.asarray(dm_matrix(partial_transpose(rho, (0,))))).min())

_YY = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))

def concurrence(rho):
    """Wootters concurrence of a two-qubit state (4x4 matrix or (2,2,2,2) tensor).
    MATH  rho_tilde = (Y x Y) rho^* (Y x Y);  l_1 >= ... >= l_4 = square roots of the eigenvalues of rho rho_tilde
          (real and non-negative);  C = max(0, l_1 - l_2 - l_3 - l_4).
    COST  one 4x4 non-Hermitian eigenvalue problem."""
    r = np.asarray(dm_matrix(as_dm_tensor(rho)))
    ev = np.linalg.eigvals(r @ _YY @ r.conj() @ _YY).real
    l = np.sqrt(np.clip(np.sort(ev)[::-1], 0.0, None))
    return float(max(0.0, l[0] - l[1] - l[2] - l[3]))

print("helpers ready")

# %% [markdown]
# ## 2. Separable and entangled pure states
#
# Fix a bipartition of the register into $A$ and $B$. A pure state is **separable** across that cut when it factorises,
#
# $$ \vert\psi\rangle = \vert\phi\rangle_A \otimes \vert\chi\rangle_B, $$
#
# and **entangled** when it does not. In the Schmidt decomposition
#
# $$ \vert\psi\rangle = \sum_{k=1}^{r} \lambda_k \vert u_k\rangle_A \vert v_k\rangle_B, \qquad \lambda_k > 0, \qquad \sum_k \lambda_k^2 = 1, $$
#
# the state is a product exactly when the Schmidt rank is $r = 1$ (derived below), and any $r > 1$ means entanglement. The
# reduced state $\rho_A$ has eigenvalues $p_k = \lambda_k^2$, so the same statement reads:
#
# > A pure global state is entangled across a cut exactly when the reduced state of either side is **mixed**.
#
# The purity of a reduced state is therefore an entanglement test for pure states, and the cheapest one:
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
# Everything else rests on this fact, so we derive it, in two ways.
#
# **From the amplitude matrix.** Group the indices by side, $\psi_{ab}$ with $a$ labelling configurations of $A$ and
# $b$ those of $B$, and read $\psi$ as a $d_A\times d_B$ **matrix** $M$. A product state has $\psi_{ab} = \phi_a\chi_b$,
# so every row of $M$ is a multiple of the same vector $\chi$, and every column a multiple of the same $\phi$.
# That is an **outer product** $M = \phi\chi^T$, which has matrix rank **one**: its column space is the single line
# spanned by $\phi$. The Schmidt values are the singular values of $M$, and a rank-one matrix has exactly one non-zero
# singular value. Conversely, a matrix of rank one is an outer product, so Schmidt rank one implies a product state.
#
# **From the reduced state.** For $\vert\psi\rangle = \vert\phi\rangle_A\otimes\vert\chi\rangle_B$,
#
# $$ \rho_A = \mathrm{Tr}_B\big(\vert\phi\rangle\langle\phi\vert\otimes\vert\chi\rangle\langle\chi\vert\big)
#  = \vert\phi\rangle\langle\phi\vert \cdot \mathrm{Tr}\big(\vert\chi\rangle\langle\chi\vert\big)
#  = \vert\phi\rangle\langle\phi\vert, $$
#
# because the partial trace touches only the $B$ factor and $\langle\chi\vert\chi\rangle = 1$. So $\rho_A$ is a
# **projector onto a single vector**: rank one, hence one non-zero eigenvalue, equal to $1$ because the trace is $1$.
#
# The two are the same statement, since $\rho_A = MM^\dagger$ and $M$ and $MM^\dagger$ have equal rank. Entanglement
# is the failure of the amplitude matrix to be an outer product, and the Schmidt rank counts how many outer products
# are needed.

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
# The table confirms the derivation: the product state has rank one; GHZ and W have rank two across the central cut,
# with the same two singular values $1/\sqrt2$; a Haar-random state has full rank $16$. The squared singular values
# are the eigenvalues of $\rho_A$ in every row.
#
# ### Measures of entanglement and LOCC
#
# A unitary acting on $A$ alone rotates the $\vert u_k\rangle$ and leaves every $\lambda_k$ untouched, so it cannot
# change any function of the Schmidt spectrum. The operations the two parties can perform without exchanging quantum
# systems are **LOCC** (local operations and classical communication): local unitaries, local measurements, and
# classical messages that let one side condition its actions on the other side's outcomes. LOCC cannot create
# entanglement, and a *measure* of entanglement is required to vanish on separable states and not to increase, on
# average, under LOCC.
#
# For pure bipartite states the theory is complete. Nielsen (1999) showed that $\vert\psi\rangle$ can be converted into
# $\vert\phi\rangle$ by LOCC exactly when the Schmidt probabilities of $\vert\phi\rangle$ majorise those of
# $\vert\psi\rangle$, so every pure-state measure is a function of the $p_k$, and every Schur-concave function of them is
# an LOCC monotone (Vidal 2000). There are therefore many pure-state measures; the entropy $S_A$ is singled out by the
# asymptotic limit. From $n$ copies of $\vert\psi\rangle$, LOCC can produce $m \approx nS_A$ Bell pairs with an error
# that vanishes as $n\to\infty$, and $m$ Bell pairs can be converted back into $n \approx m/S_A$ copies; both rates are
# optimal (Bennett, Bernstein, Popescu, Schumacher 1996). The distillable entanglement and the entanglement cost of a
# pure state are therefore both equal to $S_A$. The number of copies made per Bell pair, $n/m = 1/S_A$, exceeds one
# whenever $S_A < 1$ bit.
#
# The cell below checks the local-unitary invariance numerically. As a wrong control, a unitary acting across the cut
# must change the entropy.

# %%
key = jax.random.PRNGKey(7)
psi = haar_state(key, 6)
S0 = float(entanglement_entropy(psi, (0, 1, 2)))
k1, k2, k3 = jax.random.split(key, 3)
UA, UB = haar_unitary(k1, 8), haar_unitary(k2, 8)      # 3 spins on each side -> 8x8
# apply U_A to the first three axes and U_B to the last three, by reshaping to a matrix: M -> U_A M U_B^T
M = np.asarray(psi).reshape(8, 8)
M_rot = np.asarray(UA) @ M @ np.asarray(UB).T
psi_rot = jnp.asarray(M_rot).reshape((2,) * 6)
S1 = float(entanglement_entropy(psi_rot, (0, 1, 2)))
print(f"S before local unitaries = {S0:.12f} bit")
print(f"S after  local unitaries = {S1:.12f} bit      difference {abs(S0-S1):.2e}")
assert abs(S0 - S1) < 1e-10

# wrong control: a Haar unitary on qubits (2,3), i.e. across the cut, is NOT local
U23 = np.asarray(haar_unitary(k3, 4)).reshape(2, 2, 2, 2)
psi_x = jnp.asarray(np.einsum("cdab,xyabzw->xycdzw", U23, np.asarray(psi)))   # psi'[..cd..] = sum U[cd,ab] psi[..ab..]
S2 = float(entanglement_entropy(psi_x, (0, 1, 2)))
print(f"S after a unitary on qubits (2,3) across the cut = {S2:.6f} bit   change {abs(S2-S0):.3f}")
assert abs(S2 - S0) > 1e-3
print("[ok] local unitaries leave S unchanged; a unitary across the cut does not")

# %% [markdown]
# ### The meaning of the entropy
#
# The formula $S_A=-\sum_k p_k\log_2 p_k$ with $p_k=\lambda_k^2$ is the Shannon entropy of an ordinary probability
# distribution: $p_k$ is the probability of finding the pair in the $k$-th Schmidt branch when both sides are measured in
# the Schmidt basis.
#
# **Why zero means product.** If the state factorises there is exactly one non-zero Schmidt value, and normalisation
# forces $p_1=1$. Then $S_A = -1\cdot\log_2 1 = 0$. The remaining terms are $0\log_2 0$, defined as zero (the limit of
# $p\log_2 p$). Conversely $S_A=0$ forces one $p_k=1$. The entropy vanishes because there is nothing to be uncertain
# about: the branch is known in advance.
#
# **What a non-zero value means.** If $r$ Schmidt values are equal, $p_k = 1/r$, then $S_A = \log_2 r$, so
# $2^{S_A} = r$ counts the branches. For an uneven spectrum $2^{S_A}$ is smaller than the true rank and is the
# *effective* number of Schmidt values, those that carry appreciable weight. A matrix product state truncated to bond
# dimension $\chi$ keeps $\chi$ Schmidt values, so it needs at least $\chi \gtrsim 2^{S}$ to describe a cut of entropy
# $S$; this is a rule of thumb, since the truncation error is set by the discarded weight, not by $S$.

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
# GHZ and W have a flat two-value spectrum across the central cut and one bit each. The Dicke state $D_8^{(4)}$ has rank
# five but uses about three branches; the Haar-random state uses about ten of sixteen.
#
# ## 3. The zoo classified
#
# Take the standard states and look at them through **three different cuts**: one spin, two spins, and half the chain.
# A single number at a single cut is not a classification.

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
# Read the one-spin columns first. **GHZ and the Dicke state are identical there**: both have $\rho_q = \mathbb 1/2$
# exactly, purity $1/2$, one bit. Read the pair column and they separate: GHZ has *exactly zero*, the Dicke state does
# not.
#
# The GHZ pair needs no criterion at all to classify.

# %%
rho_pair = np.asarray(rdm(ghz_state(6), (0, 1))).real
print("two-spin reduced state of GHZ_6:")
print(rho_pair)
print("\nthis is (1/2)|00><00| + (1/2)|11><11|, a mixture of two PRODUCT states,")
print("so it is separable by the definition -- no criterion needed.")
print(f"and indeed the negativity is {neg(rdm(ghz_state(6),(0,1))):.1e} and the concurrence {concurrence(rdm(ghz_state(6),(0,1))):.1e}")

# %% [markdown]
# ### The W state, and a second pairwise measure
#
# The W state is the opposite case. Its pair state follows from counting where the single excitation sits: on spin 0,
# on spin 1, or elsewhere with probability $(N-2)/N$,
#
# $$ \rho_{01} = \frac{N-2}{N}\,\vert00\rangle\langle00\vert + \frac{2}{N}\,\vert\Psi^+\rangle\langle\Psi^+\vert,
#    \qquad \vert\Psi^+\rangle = \frac{\vert01\rangle+\vert10\rangle}{\sqrt2}. $$
#
# The partial transpose moves the coherence $\langle01\vert\rho\vert10\rangle = 1/N$ to the corner
# $\langle00\vert\rho^{T_A}\vert11\rangle$, where it meets the diagonal entries $(N-2)/N$ and $0$. The $2\times2$ block
# $\begin{pmatrix}(N-2)/N & 1/N\\ 1/N & 0\end{pmatrix}$ has the negative eigenvalue
# $\big[(N-2) - \sqrt{(N-2)^2+4}\big]/(2N)$, so
#
# $$ \mathcal N_W = \frac{\sqrt{(N-2)^2+4}-(N-2)}{2N} \;>\; 0 \quad\text{for every } N . $$
#
# A second two-qubit measure, the **concurrence** of Wootters (1998), will be needed for monogamy. With
# $\tilde\rho = (Y\otimes Y)\rho^*(Y\otimes Y)$ and $l_1\ge l_2\ge l_3\ge l_4$ the square roots of the eigenvalues of
# $\rho\tilde\rho$,
#
# $$ C(\rho) = \max(0,\; l_1-l_2-l_3-l_4). $$
#
# For a pure two-qubit state $C = 2\vert\psi_{00}\psi_{11}-\psi_{01}\psi_{10}\vert$, and the entanglement of formation of
# Section 5 is a monotone function of $C$. For states whose only coherences are $\rho_{00,11}$ and $\rho_{01,10}$ ("X
# states"; the W pair, the Werner states and the zero-field chain pairs below are of this kind) the formula reduces to
# $C = 2\max\big(0,\ \vert\rho_{01,10}\vert-\sqrt{\rho_{00,00}\rho_{11,11}},\ \vert\rho_{00,11}\vert-\sqrt{\rho_{01,01}\rho_{10,10}}\big)$.
# For the W pair this gives $C = 2/N$.
#
# **Monogamy.** For a pure state of $N$ qubits, Coffman, Kundu and Wootters (2000; three qubits) and Osborne and
# Verstraete (2006; any $N$) proved
#
# $$ \sum_{j\ne i} C_{ij}^2 \;\le\; \tau_i \equiv 4\det\rho_i , $$
#
# where $\tau_i$, the *tangle* of spin $i$ with the rest, equals $C^2$ of the bipartition $i\vert\text{rest}$. A spin
# that is strongly entangled with one partner cannot be strongly entangled with another. For the W state
# $\det\rho_i = (N-1)/N^2$, so $\tau_i = 4(N-1)/N^2$, and the $N-1$ partners contribute $(N-1)(2/N)^2$: the W state
# **saturates** the inequality. The single excitation is shared among more pairs as $N$ grows, and each pair gets less.

# %%
print("W state: pairwise negativity and concurrence against the closed forms, and the CKW balance")
print(f"  {'N':>3s} {'neg(pair)':>10s} {'formula':>10s} {'C(pair)':>9s} {'2/N':>7s} {'sum_j C^2':>10s} {'tau_1':>8s}")
for N in (4, 6, 8, 10):
    psi = w_state(N)
    n01, C01 = neg(rdm(psi, (0, 1))), concurrence(rdm(psi, (0, 1)))
    n_th = (np.sqrt((N - 2) ** 2 + 4) - (N - 2)) / (2 * N)
    tau = 4 * float(np.linalg.det(np.asarray(rdm(psi, (0,)))).real)
    sumC2 = sum(concurrence(rdm(psi, (0, j))) ** 2 for j in range(1, N))
    print(f"  {N:3d} {n01:10.6f} {n_th:10.6f} {C01:9.6f} {2/N:7.4f} {sumC2:10.6f} {tau:8.6f}")
    assert abs(n01 - n_th) < 1e-10 and abs(C01 - 2 / N) < 1e-8 and abs(sumC2 - tau) < 1e-8
print(f"  GHZ_8 for contrast: C(pair) = {concurrence(rdm(ghz_state(8),(0,1))):.1e}, tau_1 = "
      f"{4*float(np.linalg.det(np.asarray(rdm(ghz_state(8),(0,)))).real):.3f}")

# %% [markdown]
# Both closed forms hold to rounding, and the W state saturates the monogamy bound at every $N$. GHZ sits at the other
# extreme: its tangle $\tau_1 = 1$ is maximal, and none of it is in any pair.
#
# GHZ and W therefore differ in *where* their entanglement lives. GHZ holds one bit shared globally, invisible in every
# pair; tracing out one spin leaves $\tfrac12(\vert0\ldots0\rangle\langle0\ldots0\vert+\vert1\ldots1\rangle\langle1\ldots1\vert)$,
# which is separable. W holds a little in every pair and keeps pairwise entanglement after the loss of a spin. No single
# number orders them: for three qubits Dür, Vidal and Cirac (2000) proved that GHZ and W cannot be converted into each
# other even probabilistically by LOCC; they are two different classes, not two points on one scale.
#
# ## 4. The negativity of a pure-state cut
#
# The negativity, defined in Section 7, can also be applied to a *cut* of a pure state. Write
# $\vert\psi\rangle\langle\psi\vert = \sum_{ij}\lambda_i\lambda_j\,\vert u_i\rangle\langle u_j\vert\otimes\vert v_i\rangle\langle v_j\vert$
# and transpose the $A$ factor, $\vert u_i\rangle\langle u_j\vert \to \vert\bar u_j\rangle\langle\bar u_i\vert$ (the bar is
# complex conjugation). The result maps $\vert\bar u_k v_l\rangle \mapsto \lambda_k\lambda_l\vert\bar u_l v_k\rangle$:
# eigenvalue $\lambda_k^2$ for $k=l$, and on each pair $k\ne l$ a swap with eigenvalues $\pm\lambda_k\lambda_l$. The
# trace norm is $\sum_k\lambda_k^2 + 2\sum_{k<l}\lambda_k\lambda_l = (\sum_k\lambda_k)^2$, so
#
# $$ E_{\mathcal N} = \log_2\Big(\sum_k \lambda_k\Big)^2 = 2\log_2\sum_k\lambda_k, $$
#
# which is the **Rényi-$\tfrac12$ entropy** $S_{1/2} = 2\log_2\sum_k p_k^{1/2}$ of the Schmidt spectrum. For a pure
# state the negativity is a different function of the same data and carries nothing new. Its use is the mixed case,
# where the entropy fails.

# %%
print("log-negativity of a half-chain cut vs the Renyi-1/2 entropy, XXZ ground state at (Delta,hx)=(1.5,0.5)")
for N in (8, 10):
    _, psi = lanczos_ground_state(heisenberg_terms(N, 1., 1., 1.5, hx=0.5), N)
    A = tuple(range(N // 2))
    lam = np.asarray(schmidt_values(psi, A))
    E_N = float(negativity(to_dm(psi), A)[1])
    print(f"  N={N:2d}:  E_N = {E_N:.10f}   2*log2(sum lam) = {2*np.log2(lam.sum()):.10f}   "
          f"S_vN = {S_bits(rdm(psi,A)):.10f} bit")
    assert abs(E_N - 2 * np.log2(lam.sum())) < 1e-9

# %% [markdown]
# The brute-force partial transpose of the full $2^N\times2^N$ density matrix and the Schmidt formula agree to all printed
# digits; the von Neumann entropy is smaller, as $S_{1/2}\ge S_1$ requires.
#
# ## 5. Entanglement of mixed states
#
# Two spins inside a longer chain are not in a pure state. Neither is a register that has passed through a noisy
# device. For a mixed state the argument of Section 2 fails, as the following two states show.

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
# maximal entanglement for a state that has none. What it measures is how mixed $\rho_A$ is, and for a mixed global
# state that has two sources: entanglement with $B$, and the classical randomness already present in $\rho_{AB}$. Only
# when the global state is pure is the second source absent.
#
# A density matrix does not determine the ensemble it was prepared from. The natural definition of its entanglement
# takes the most favourable decomposition,
#
# $$ E_F(\rho) = \min_{\{p_i,\psi_i\}} \sum_i p_i\, S\big(\rho_A^{(i)}\big), \qquad \rho = \sum_i p_i\vert\psi_i\rangle\langle\psi_i\vert, $$
#
# the **entanglement of formation**: a minimisation over all pure-state decompositions of $\rho$. It vanishes exactly on
# separable states and is an LOCC monotone. It has a closed form for two qubits (Wootters 1998, through the concurrence)
# and none in general. Deciding whether a given mixed state is entangled at all is NP-hard as the dimension grows
# (Gurvits 2003). Other measures do not remove the difficulty: the distillable entanglement $E_D$ (Bell pairs per copy
# obtainable by LOCC) is zero for the "bound entangled" states of Horodecki (1997), so $E_D$ is **not faithful**, and the
# computable quantity of the next sections, the negativity, misses the same states. No efficiently computable measure
# that is faithful for all mixed states is known. The rest of the notebook uses one computable *criterion*, the positive
# partial transpose, which is complete for the two-qubit states we need.

# %% [markdown]
# ## 6. The partial transpose
#
# Write the density matrix with its indices split, $\rho_{(ab),(a'b')}$, and transpose **only** the indices of $A$:
#
# $$ \big(\rho^{T_A}\big)_{(ab),(a'b')} = \rho_{(a'b),(ab')}. $$
#
# In matrix form this is index bookkeeping. In **tensor** form it is one permutation: a density matrix of $N$ spins is a
# tensor with $N$ ket axes followed by $N$ bra axes, and the partial transpose on a set $A$ swaps axis $q$ with axis
# $N+q$ for every $q \in A$. One `transpose`, no arithmetic. For two qubits, `rho[a,b,c,d]` with kets $(a,b)$ and bras
# $(c,d)$ becomes `rho[c,b,a,d]`, the einsum string `"abcd->cbad"`.

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
# $2^{|A|}\times 2^{|A|}$ **matrix**; the channels and the partial transpose work on the rank-$2N$ **tensor** with every
# axis of length two. A routine that expects the tensor and receives the matrix reads $N$ from the number of axes, finds
# $N=1$, and transposes the *whole* matrix, which leaves every eigenvalue unchanged and reports a negativity of exactly
# zero. Zero negativity is a physically meaningful answer, so nothing looks wrong.
#
# `as_dm_tensor` therefore converts a $(2^N,2^N)$ matrix into the tensor, and refuses shapes that are neither. A
# $(2^N,2^N)$ matrix with $N>1$ is unambiguous, because a density tensor has all axes of length two. One confusion
# cannot be detected by any shape test: a **state vector** of $2N$ qubits has the shape $(2,)^{2N}$ of a density tensor
# of $N$ qubits, and is accepted without complaint.

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

# the undetectable case: a 4-qubit STATE VECTOR has the shape of a 2-qubit density tensor
psi4 = jnp.asarray(np.kron(bell, bell)).reshape(2, 2, 2, 2)     # |Phi+>_{01} |Phi+>_{23}, a pure state of 4 qubits
n_wrong = float(negativity(psi4, (0,))[0])
n_right = float(negativity(to_dm(psi4), (0, 2))[0])
print(f"\n  4-qubit state vector passed as if it were a density tensor: accepted, 'negativity' = {n_wrong:.6f}")
print(f"  the correct call, negativity(to_dm(psi), (0,2)) for the cut 02|13  = {n_right:.6f}")
assert abs(n_wrong - n_right) > 0.1

# %% [markdown]
# The two legitimate forms give the same $\mathcal N = 1/2$, and the two impossible shapes are refused. The last two
# lines are the pitfall: the state vector is accepted and returns a number that has nothing to do with the state. The
# cut $02\vert13$ is crossed by both Bell pairs, so it has four equal Schmidt values $\tfrac12$ and, from Section 4,
# $\mathcal N = \big[(\sum_k\lambda_k)^2-1\big]/2 = 3/2$.
#
# > **Common pitfall.** Pass `to_dm(psi)` (or a matrix from `rdm`) to `negativity` and `partial_transpose`, never the
# > state vector itself. Shape checks catch a matrix of the wrong size but cannot catch a state vector of twice as many
# > qubits.
#
# ## 7. The PPT criterion and the negativity
#
# Why does a negative eigenvalue prove anything? Take a separable state,
# $\rho=\sum_i p_i\,\sigma^{(i)}_A\otimes\tau^{(i)}_B$ with $p_i\ge0$. The partial transpose gives
# $\sum_i p_i (\sigma^{(i)}_A)^{T}\otimes\tau^{(i)}_B$, and the transpose of a density matrix is again a density
# matrix (same eigenvalues, unit trace), so the result is a sum of positive operators with non-negative weights, which is
# positive. Hence
#
# $$ \rho \text{ separable} \implies \rho^{T_A} \ge 0, $$
#
# and by contraposition **a negative eigenvalue of the partial transpose proves entanglement** (Peres 1996). The
# implication is one-directional in general; for two qubits and for a qubit and a qutrit it is also sufficient
# (Horodecki, Horodecki, Horodecki 1996), so for the spin pairs in this notebook it decides the question completely.
#
# The criterion becomes a number by measuring how negative the partial transpose is. Since $\mathrm{Tr}\rho^{T_A} = 1$,
# the eigenvalues $\mu_i$ satisfy $\sum_i\vert\mu_i\vert = 1 + 2\sum_{\mu_i<0}\vert\mu_i\vert$, so
#
# $$ \mathcal N(\rho) = \frac{\Vert\rho^{T_A}\Vert_1 - 1}{2} = \sum_{\mu_i<0}\vert\mu_i\vert, \qquad
#    E_{\mathcal N}(\rho) = \log_2\Vert\rho^{T_A}\Vert_1 = \log_2(1+2\mathcal N). $$
#
# Vidal and Werner (2002) proved that $\mathcal N$ does not increase under LOCC, and Plenio (2005) that
# $E_{\mathcal N}$ is a monotone as well although it is not convex. The negativity is zero on every PPT state, including
# the bound entangled ones of Section 5, so it is **not faithful** beyond $2\times2$ and $2\times3$.
#
# ### The meaning of the negativity
#
# Transposing a **whole** density matrix gives an operator with the same eigenvalues, again a state. Transposing **half**
# of one need not. For a separable state it does, by the proof above. So the partial transpose maps separable states
# *into* the set of states, and can take $\rho$ **outside** that set only if $\rho$ is entangled. A negative eigenvalue
# is what leaving the set looks like: an operator with a negative eigenvalue is not a density matrix, because it assigns
# a negative probability to some outcome.
#
# The negativity is the total weight of negative eigenvalues, a measure of how far outside the set of states the partial
# transpose lands. Zero means the partial transpose is still a state: no contradiction produced, no entanglement proved.
#
# For a pure state the zero case matches the entropy. With $E_{\mathcal N} = 2\log_2\sum_k\lambda_k$ and
# $\sum_k\lambda_k^2 = 1$, $\lambda_k \ge 0$, the sum $\sum_k\lambda_k \ge (\sum_k\lambda_k^2)^{1/2} = 1$, with equality
# exactly when a single $\lambda_k = 1$. A product state gives $\log_2 1 = 0$.
#
# $E_{\mathcal N}$ counts the same branches as $S$ when the spectrum is flat: for $r$ equal Schmidt values
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
# For the XXZ ground states the ratio $E_{\mathcal N}/S$ lies between $1.25$ and $1.8$ and is not monotone in $N$: the
# open chain alternates between strong and weak central bonds as $N$ changes by two, and the shape of the spectrum
# changes with it.
#
# ### Calibration on Werner states
#
# A new quantity has to be run where the answer is known. The **Werner state** is a Bell pair mixed with white noise,
#
# $$ \rho_W(w) = w\,\vert\Phi^+\rangle\langle\Phi^+\vert + (1-w)\,\frac{\mathbb 1}{4}, \qquad 0\le w\le1 . $$
#
# The partial transpose of $\vert\Phi^+\rangle\langle\Phi^+\vert$ is $\tfrac12\,\mathrm{SWAP}$ (check it on the four
# basis products), whose eigenvalues are $+\tfrac12$ three times (symmetric states) and $-\tfrac12$ once (the singlet).
# Hence $\rho_W^{T_A}$ has eigenvalues $(1+w)/4$ three times and
#
# $$ \mu_{\min} = \frac{1-w}{4} - \frac w2 = \frac{1-3w}{4}, $$
#
# so the state is PPT, hence separable, for $w\le\tfrac13$ and entangled above, with $\mathcal N = (3w-1)/4$. The code
# below does not use this formula.

# %%
print("Werner state: the PPT boundary should appear at w = 1/3")
print(f"  {'w':>7s} {'min eig of PT':>14s} {'(1-3w)/4':>10s} {'negativity':>11s}")
for w in (0.20, 0.30, 1/3, 0.34, 0.50, 1.00):
    r = w * rho_bell + (1 - w) * np.eye(4) / 4
    mu = pt_min(r)
    verdict = "entangled" if neg(r) > 1e-12 else "separable (PPT)"
    print(f"  {w:7.4f} {mu:+14.6f} {(1-3*w)/4:+10.6f} {neg(r):11.6f}   {verdict}")
    assert abs(mu - (1 - 3 * w) / 4) < 1e-12

ws = np.linspace(0, 1, 201)
ns = [neg(w * rho_bell + (1 - w) * np.eye(4) / 4) for w in ws]
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(ws, ns, lw=1.4, label="computed")
ax.plot(ws, np.maximum((3 * ws - 1) / 4, 0), "k:", lw=1.0, label=r"$\max(0,(3w-1)/4)$")
ax.axvline(1/3, ls="--", color="k", lw=0.9)
ax.text(1/3 + 0.02, max(ns) * 0.8, "w = 1/3", fontsize=9)
ax.set_xlabel("Werner weight w"); ax.set_ylabel(r"negativity $\mathcal{N}$"); ax.legend(fontsize=8)
ax.set_title("Werner state: entanglement switches on at w = 1/3")
plt.tight_layout(); plt.show()

# %% [markdown]
# The smallest eigenvalue of the partial transpose follows $(1-3w)/4$ to rounding, and the negativity switches on at
# $w=\tfrac13$ and grows linearly to $\tfrac12$ at the pure Bell pair.
#
# ## 8. Entanglement death under noise
#
# Section 5 made quantitative: take a Bell pair and apply the depolarising channel
# $\rho\to(1-p)\rho+\tfrac p3(X\rho X+Y\rho Y+Z\rho Z)$ independently to each spin, as a noisy device would. The reduced
# state entropy and the negativity behave completely differently.
#
# The threshold can be derived. The channel shrinks the Bloch vector by $\eta = 1-\tfrac{4p}{3}$, i.e. it maps
# $X\to\eta X$, $Y\to\eta Y$, $Z\to\eta Z$ and $\mathbb 1\to\mathbb 1$. Since
# $\vert\Phi^+\rangle\langle\Phi^+\vert = \tfrac14(\mathbb 1 + XX - YY + ZZ)$, applying it to both spins gives
# $\tfrac14(\mathbb 1 + \eta^2(XX-YY+ZZ)) = \rho_W(\eta^2)$, a Werner state with $w = (1-4p/3)^2$. It is separable for
# $w\le\tfrac13$, i.e. for
#
# $$ p \;\ge\; p^* = \tfrac34\big(1-1/\sqrt3\big) = 0.316987\ldots \text{ per spin}. $$
#
# The cell scans $p$ on a grid that does not contain $p^*$ and then locates the boundary by bisection on the computed
# negativity, without using the formula.

# %%
def noisy_bell(p):
    """Bell pair with depolarising strength p applied to each of its two spins; returns the density tensor."""
    rho = jnp.asarray(rho_bell).reshape(2, 2, 2, 2)
    for q in (0, 1):
        rho = apply_kraus_dm(rho, kraus_depolarizing(float(p)), [q])
    return rho

print(f"{'p per spin':>10s} {'purity':>8s} {'S(rho_A)':>9s} {'negativity':>11s}")
pp = np.linspace(0, 0.45, 46)
negs, purs = [], []
for p in pp:
    rho = noisy_bell(p)
    negs.append(neg(rho)); purs.append(float(purity(dm_matrix(rho))))
    if np.isclose(p, [0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.31, 0.32, 0.35], atol=1e-9).any():
        rA = np.asarray(rdm_dm(rho, (0,)))
        print(f"{p:10.2f} {purs[-1]:8.4f} {S_bits(rA):9.4f} {negs[-1]:11.6f}")
        assert abs(S_bits(rA) - 1.0) < 1e-10

lo, hi = 0.30, 0.35                     # bracket read off the scan: entangled at 0.30, separable at 0.35
assert neg(noisy_bell(lo)) > 1e-6 and neg(noisy_bell(hi)) < 1e-14
for _ in range(40):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if neg(noisy_bell(mid)) > 1e-14 else (lo, mid)
p_star = 0.75 * (1 - 1 / np.sqrt(3))
print(f"\nbisection on the computed negativity: separable from p = {hi:.6f} per spin")
print(f"exact threshold p* = (3/4)(1 - 1/sqrt(3)) = {p_star:.6f},  difference {abs(hi-p_star):.1e}")
assert abs(hi - p_star) < 1e-6
print("S(rho_A) stays at 1 bit: the depolarising channel is unital, so it maps 1/2 to itself; the input is a Bell")
print("pair whose reduced state is already 1/2; and the channel on the other spin is trace preserving.")

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(pp, negs, lw=1.4, label="negativity")
ax.plot(pp, [1.0] * len(pp), "--", lw=1.2, label=r"$S(\rho_A)$ [bits]")
ax.plot(pp, purs, ":", lw=1.2, label="purity")
ax.axvline(p_star, color="k", lw=0.8)
ax.annotate(r"$p^*=\frac{3}{4}(1-1/\sqrt{3})$", (p_star, 0.25), fontsize=7,
            xytext=(p_star + 0.03, 0.60), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlabel("depolarising strength p per spin"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("entanglement dies; the subsystem entropy does not change")
plt.tight_layout(); plt.show()

# %% [markdown]
# The bisection, which knows nothing of the formula, finds the threshold $p^* = 0.316987$ to six digits. Below it the
# negativity falls almost linearly from $\tfrac12$; above it the pair is separable although the global state is far
# from maximally mixed (purity $1/3$ at $p^*$, against $1/4$). The reduced-state entropy reports one bit at every $p$.
#
# ## 9. A spin chain: three quantities against distance
#
# Now the question that opened the notebook. Take the ground state of the XXZ chain in a transverse field (open chain),
#
# $$ H = \sum_j \big(X_jX_{j+1} + Y_jY_{j+1} + \Delta\, Z_jZ_{j+1}\big) + h_x\sum_j X_j, $$
#
# fix a reference spin $i$ next to the centre, and follow three quantities as the partner $j=i+r$ moves away:
#
# * the **negativity** $\mathcal N(r)$ of the pair, their entanglement (and the smallest eigenvalue of the partial
#   transpose, which shows the margin by which a pair is PPT);
# * the **mutual information** $I(i{:}j) = S(\rho_i) + S(\rho_j) - S(\rho_{ij})$, their total correlation, classical
#   and quantum together;
# * the **entropy** $S(\rho_{ij})$ of the pair itself.
#
# The four parameter points are a gapless point $\Delta=0.5$, the Heisenberg point $\Delta=1$, the Néel side
# $\Delta=1.5$, and the XX chain in a transverse field $h_x=1.5$.

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
    nn, ii, ss, mm = [], [], [], []
    for r in RS:
        j = i0 + r
        rij = rdm(psi, (i0, j))
        Sij = S_bits(rij); Sj = S_bits(rdm(psi, (j,)))
        nn.append(neg(rij)); ii.append(Si + Sj - Sij); ss.append(Sij); mm.append(pt_min(rij))
    curves[name] = (nn, ii, ss)
    print(f"{name}:  S(one spin) = {Si:.4f} bit")
    print("   r       " + "".join(f"{r:8d}" for r in RS))
    print("   N(r)    " + "".join(f"{v:8.4f}" for v in nn))
    print("   min PT  " + "".join(f"{v:+8.4f}" for v in mm))
    print("   I(r)    " + "".join(f"{v:8.4f}" for v in ii))
    print("   S_ij    " + "".join(f"{v:8.4f}" for v in ss))
    assert nn[0] > 0.1 and min(mm[1:]) > 0.01       # r=1 entangled; r>=2 PPT with a finite margin

# %%
fig, ax = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
for name, _, _ in PHASES:
    nn, ii, ss = curves[name]
    ax[0].plot(RS, nn, "-o", ms=3, label=name)
    ax[1].semilogy(RS, np.maximum(ii, 1e-6), "-o", ms=3)
    ax[2].plot(RS, ss, "-o", ms=3)
ax[0].set_ylabel(r"negativity $\mathcal{N}(r)$"); ax[0].set_title("(a) entanglement of the pair")
ax[1].set_ylabel(r"$I(i:j)$ [bits]");   ax[1].set_title("(b) total correlation")
ax[2].set_ylabel(r"$S(\rho_{ij})$ [bits]"); ax[2].set_title("(c) mixedness of the pair")
ax[2].axhline(2.0, ls=":", color="k", lw=0.8)
for a in ax:
    a.set_xlabel("distance r")
ax[0].legend(fontsize=7)
plt.show()

# %% [markdown]
# **(a) Entanglement between two spins is confined to nearest neighbours here.** The negativity is $0.21$–$0.29$ at
# $r=1$ and zero at every larger separation, at all four parameter points. The zero is not a small number below a
# tolerance: the smallest eigenvalue of the partial transpose is *positive*, by at least $0.03$, so the partial transpose
# is a state with a margin, and for two qubits that proves the pair separable.
#
# **(b) Correlation is not entanglement.** The mutual information at $r=2$ and $r=3$ is $0.13$–$0.29$ bit and decays
# slowly, with an even-odd alternation. All of that correlation is real, and none of it is entanglement. The three
# zero-field points give similar curves; the field point differs, with larger mutual information at $r\ge2$.
#
# **(c) The pair entropy measures something else.** $S(\rho_{ij})$ grows with $r$, with the same even-odd alternation,
# towards $S(\rho_i)+S(\rho_j)$, which is $2$ bits at zero field, where each spin is maximally mixed. As the two spins
# decorrelate, $\rho_{ij}\to\rho_i\otimes\rho_j$ and the entropy becomes the sum of the parts; equivalently
# $S(\rho_{ij}) = S(\rho_i)+S(\rho_j)-I(i{:}j)$. In the field each spin is slightly polarised ($S(\rho_i) = 0.975$ bit),
# and at $r=7$ the partner is the end spin of the open chain, whose polarisation is stronger, so the curve drops. A
# quantity that grows as two spins are pulled apart is not measuring their mutual entanglement; for a pure global state
# it measures the entanglement of the *pair* with the rest of the chain.
#
# ### Monogamy of concurrence in the spin chain
#
# A spin of the zero-field chain is maximally mixed, so its tangle with the rest is maximal, $\tau_i = 4\det\rho_i = 1$.
# The monogamy inequality of Section 3 says how much of it pairs can hold: $\sum_j C_{ij}^2 \le \tau_i$. The cell
# computes every pair concurrence of a central spin.

# %%
Nm = 10
c = Nm // 2 - 1
print(f"open chain N={Nm}, reference spin {c}: monogamy  sum_j C_cj^2 <= tau_c = 4 det rho_c")
for D in (0.0, 1.0, 2.0):
    _, psi = lanczos_ground_state(heisenberg_terms(Nm, 1., 1., D, hx=0.0), Nm)
    tau = 4 * float(np.linalg.det(np.asarray(rdm(psi, (c,)))).real)
    Cs = [(j, concurrence(rdm(psi, (c, j))), neg(rdm(psi, (c, j)))) for j in range(Nm) if j != c]
    sumC2 = sum(C ** 2 for _, C, _ in Cs)
    print(f"  Delta={D:3.1f}: tau = {tau:.6f}   sum C^2 = {sumC2:.6f}   residual tau - sum C^2 = {tau-sumC2:.6f}")
    print(f"             non-zero pairs (j, C, N): {[(j, round(C,4), round(n,4)) for j, C, n in Cs if C > 1e-10]}")
    assert sumC2 <= tau + 1e-10
    assert all(abs(C - 2 * n) < 1e-8 for _, C, n in Cs)    # C = 2N for these pair states (see text)

# %% [markdown]
# The bound holds with room to spare: the two neighbours hold $28$–$42\,\%$ of the tangle ($\sum_jC^2 = 0.28$, $0.42$
# and $0.32$ at $\Delta = 0, 1, 2$), and the rest, the *residual tangle*, is in no pair at all; it is entanglement shared
# among three or more spins. The bond to the right, $(4,5)$, is the strong bond of the open chain and carries most of the
# pairwise part.
#
# The cell also checks $C = 2\mathcal N$ for every pair. That is a property of these states, not a general law: at zero
# field the pair state commutes with $Z_iZ_j$ and is real, so it is an X state, and for an X state with a single
# negative partial-transpose eigenvalue $C = 2\mathcal N$ exactly when the relevant diagonal entries are equal, which the
# symmetry of $H$ under flipping all spins ensures here. In general only $C \ge 2\mathcal N$ holds for two qubits.
#
# Monogamy bounds the *sum* of the pairwise entanglement; it does not by itself force the more distant pairs to zero,
# since the bound is not saturated. The zeros at $r\ge2$ come from the positive margin of the partial transpose measured
# in Section 9(a): each distant pair is mixed enough, by its entanglement with the rest of the chain, to be separable.
#
# ## 10. The phase diagram in two measures
#
# The comparison so far used chosen states. Now it is made across the $(\Delta, h_x)$ plane of the XXZ chain in a
# transverse field. The question is whether the half-chain entropy and the nearest-neighbour negativity are two
# estimates of one thing; if they were, they would take their largest values at the same place.
#
# The map below is $N=12$ on a $9\times9$ grid (under a minute), steps $0.5$ in $\Delta$ and $0.25$ in $h_x$, so that
# Section 14 can compare with it point by point. One region needs care before looking at the colours. For $\Delta<-1$ and $h_x=0$ the chain is a
# ferromagnet along $z$ (rotate every second spin by $\pi$ about $z$: $XX+YY\to-(XX+YY)$, and $H$ becomes the
# ferromagnetic XXZ chain with anisotropy $\vert\Delta\vert>1$). There $\vert0\ldots0\rangle$ and $\vert1\ldots1\rangle$
# are exact eigenstates with the same energy $\Delta(N-1)$ (each bond contributes $\Delta$ and the $XX+YY$ terms
# annihilate them), and they are the ground states. The doublet is **exactly** degenerate at every $N$, so the ground
# state is any superposition $a\vert0\ldots0\rangle+b\vert1\ldots1\rangle$, and Lanczos returns whichever superposition
# its random start vector selects. The half-chain entropy of that state is the binary entropy $H_2(\vert a\vert^2)$,
# anything between $0$ and $1$ bit, and it is a property of the solver's start vector, not of the Hamiltonian. The
# nearest-neighbour pair state is $\vert a\vert^2\vert00\rangle\langle00\vert+\vert b\vert^2\vert11\rangle\langle11\vert$
# for every choice, a mixture of products, so its negativity is exactly zero. At $\Delta=-1$, $h_x=0$ the rotated chain
# is the isotropic ferromagnet, whose ground level is the whole spin-$N/2$ multiplet, $(N+1)$-fold degenerate, and the
# same caveat applies with more freedom.

# %%
print("the exactly degenerate ferromagnetic doublet, N=12, Delta=-2, hx=0:")
for k in range(3):
    E, pv = lanczos_ground_state(heisenberg_terms(12, 1., 1., -2.0), 12, key=jax.random.PRNGKey(k))
    amp = np.asarray(pv).reshape(-1)
    print(f"  start key {k}: E = {E:.10f} (Delta(N-1) = -22)   |a|^2 = {abs(amp[0])**2:.4f}  |b|^2 = {abs(amp[-1])**2:.4f}"
          f"   S_half = {float(entanglement_entropy(pv, tuple(range(6)))):.4f} bit   N(5,6) = {neg(rdm(pv,(5,6))):.1e}")

# %% [markdown]
# Three start vectors give three different entropies and the same zero negativity. With that in mind, the map.

# %%
Nmap, ng = 12, 9
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

degen = (hs[:, None] == 0.0) & (Ds[None, :] <= -1.0)       # the exactly degenerate ferromagnetic line (and Delta = -1)
S_ok = np.where(degen, -np.inf, S_map)
iS = np.unravel_index(S_ok.argmax(), S_map.shape)
iN = np.unravel_index(N_map.argmax(), N_map.shape)
print(f"  half-chain entropy: range {S_map.min():.3f} .. {S_map.max():.3f} bits; max outside the degenerate line "
      f"{S_map[iS]:.3f} at (Delta,hx) = ({Ds[iS[1]]:+.2f}, {hs[iS[0]]:.2f})")
print(f"  nn negativity     : range {N_map.min():.3f} .. {N_map.max():.3f},       "
      f"max at (Delta,hx) = ({Ds[iN[1]]:+.2f}, {hs[iN[0]]:.2f})")
print(f"  mutual information: range {I_map.min():.3f} .. {I_map.max():.3f} bits  (never zero anywhere)")
print(f"  the two maxima are at different points: {iS != iN}")
zero = N_map < 1e-10
print(f"\n  points with negativity < 1e-10: {int(zero.sum())} of {N_map.size}, at (Delta,hx) = "
      f"{[(round(float(Ds[b]),2), round(float(hs[a]),2)) for a, b in zip(*np.nonzero(zero))]}")
print(f"  smallest negativity elsewhere: {N_map[~zero].min():.4f}")
ife = int(np.argmin(np.abs(Ds + 2.0)))
print(f"  ferromagnetic corner (Delta={Ds[ife]:+.2f}, hx=0): S = {S_map[0, ife]:.4f} bits (start-vector dependent), "
      f"I = {I_map[0, ife]:.4f} bits, negativity = {N_map[0, ife]:.1e}")

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 3.1), constrained_layout=True)
for a, (M, ttl) in enumerate(((S_map, "half-chain entropy [bit]"),
                              (N_map, r"negativity $\mathcal{N}(1)$"),
                              (I_map, "mutual information [bit]"))):
    im = ax[a].imshow(M, origin="lower", aspect="auto", cmap="magma",
                      extent=[Ds[0] - 0.25, Ds[-1] + 0.25, hs[0] - 0.125, hs[-1] + 0.125])
    ax[a].set_xlabel(r"$\Delta$"); ax[a].set_ylabel(r"$h_x$"); ax[a].set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax[a])
plt.show()

# %% [markdown]
# The two maxima are at different places. The overall entropy maximum, $2.32$ bit, sits at the degenerate point
# $(\Delta,h_x)=(-1,0)$ and is a property of the Lanczos start vector. Outside the degenerate line the entropy is largest
# at $(-1, 0.25)$, next to the isotropic ferromagnetic point; the negativity is largest at positive $\Delta$ *and* finite
# field, $(0.5, 1)$, where the chain is neither ordered nor field-dominated. They are not two estimates of one quantity.
#
# The negativity is zero, to rounding, only on the zero-field line $\Delta\le-1$; everywhere else on the grid the
# central pair is entangled (the smallest value is $8\times10^{-4}$). For $\Delta<-1$ the zero holds for every choice of
# the degenerate ground state. At $\Delta=-1$ it does not: the multiplet contains (up to the local sublattice rotation)
# the Dicke state $D_N^{(N/2)}$, whose pairs are entangled (Section 3), so the zero there is again the solver's choice. On that line the central pair
# still carries mutual information equal to the half-chain entropy, $H_2(\vert a\vert^2)$, which is the classical
# correlation of a mixture of $\vert00\rangle$ and $\vert11\rangle$. The half-chain entropy there is genuine
# entanglement of the pure state Lanczos returned (a GHZ-like superposition carries $H_2(\vert a\vert^2)$ ebits across
# the cut), but the superposition is arbitrary, so this value belongs to the solver and not to the phase.
#
# ## 11. The same measures from a matrix product state
#
# An exact state vector limits us to about twenty spins. Both quantities can be had from a matrix product state, at
# very different cost.
#
# **The entropy is free.** An MPS in canonical form *stores* the Schmidt values of every cut, so the entanglement
# entropy is a sum over numbers already in the data structure; that is what `mps_entropies` returns. This presupposes
# that the stored values belong to the stored tensors. The values DMRG records during its sweeps do not, in general:
# later local steps change the left block of a cut whose values were already recorded. `dmrg` therefore ends with an
# exact canonicalisation pass, `mps_recanonicalise`, that recomputes `lam` from the returned tensors (notebook
# [18](../ch07_tensor_networks/18_mps_tebd.ipynb), Section 6.7), and with that pass the entropies of its output are those
# of the state the tensors encode.
#
# **The negativity is not free**, because it needs the two-site reduced density matrix, and that is a contraction. For a
# right-canonical MPS $B^{[k]}_{a s b}$ (left bond $a$, physical $s$, right bond $b$) and sites $i<j$:
#
# 1. Everything to the **right** of site $j$ contracts to the identity, by right-canonicity
#    $\sum_{s,b} B^{[k]}_{asb}\bar B^{[k]}_{a'sb} = \delta_{aa'}$. It costs nothing.
# 2. Everything to the **left** of site $i$ contracts to a $\chi\times\chi$ matrix $L$, the density matrix of the bond to
#    the left of site $i$. We build it from the tensors themselves, by the transfer-matrix sweep of Eq. (1) below.
# 3. Between $i$ and $j$, each site is traced over its physical leg, a transfer matrix.
#
# The left sweep starts from the dummy boundary bond, $L^{(0)}=e_0e_0^{\rm T}$, and adds one site at a time,
#
# $$ L^{(k+1)}_{bd}=\sum_{a,c,s} L^{(k)}_{ac}\,B^{[k]}_{asb}\,\bar B^{[k]}_{csd}, \tag{1} $$
#
# which is the einsum `"ac,asb,csd->bd"`. When the bond basis is the Schmidt basis, $L^{(i)}$ equals
# $\mathrm{diag}(\lambda^2)$, but using the stored $\lambda$ instead would make the result only as good as those stored
# values; built from $B$, it is exact for any right-canonical MPS.
#
# So: build the left environment, open the physical legs at site $i$, walk the transfer matrix to site $j$, and close
# with the right bond traced. The cost is $O(j\,d\,\chi^3)$ with $d=2$.

# %%
N, chi, D, h = 12, 32, 1.5, 0.5
B, lam, hist = dmrg(xxz_mpo(N, 1., 1., D, hx=h), product_mps(("01" * N)[:N], chi)[0], chi, 6)
i, j = 4, 7

# step 0: the left environment of the cut left of site i, from the tensors alone (no lam)
L = jnp.zeros((chi, chi), dtype=B.dtype).at[0, 0].set(1.0)          # e_0 e_0^T: the dummy left bond
for k in range(i):
    L = jnp.einsum("ac,asb,csd->bd", L, B[k], jnp.conj(B[k]))
print(f"left environment of site {i}: shape {L.shape},  max|L - diag(lam[{i}]^2)| = "
      f"{np.abs(np.asarray(L) - np.diag(np.asarray(lam[i]) ** 2)).max():.1e}")

# step 1: open the physical legs at site i against the left environment
T = jnp.einsum("ax,asb,xtc->stbc", L, B[i], jnp.conj(B[i]))
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
# The left environment built from the tensors equals $\mathrm{diag}(\lambda^2)$ to rounding, so after
# `mps_recanonicalise` the bond basis is the Schmidt basis. The hand-written contraction reproduces `mps_rdm2` exactly.
# That only shows the two codes are the same code; the next section checks them against an independent calculation.
#
# ## 12. Agreement of the two implementations
#
# There is a size at which the agreement must be *exact*: a matrix product state with $\chi = 2^{N/2}$ discards
# nothing, so at that bond dimension DMRG and exact diagonalisation solve the same problem in different coordinates.

# %%
print("CHECK 1 -- at chi = 2^(N/2) the MPS is exact, so everything must agree to rounding")
print(f"{'point':>12s} {'N':>2s} {'chi':>4s} | {'|dE|':>9s} {'|dS|':>9s} {'max|d rho_ij| all pairs':>24s}")
worst1 = 0.0
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
        worst1 = max(worst1, abs(E - hv[-1][3]), dS, drho)
        print(f"{name:>12s} {Nv:2d} {chiv:4d} | {abs(E-hv[-1][3]):9.1e} {dS:9.1e} {drho:24.1e}")
assert worst1 < 1e-10
print(f"largest difference: {worst1:.1e}")

# %% [markdown]
# Energies, entropies and all two-site reduced states agree to $10^{-12}$ or better at every point. Away from that limit
# the MPS is an approximation, and *how* it converges is the second check: does the negativity converge as fast as the
# energy, and is the discarded weight that DMRG reports a usable error bar for it?

# %%
print("CHECK 2 -- convergence in chi at N=12, (Delta,hx)=(1.5,0.5)")
E_ex, psi_ex = lanczos_ground_state(heisenberg_terms(12, 1., 1., 1.5, hx=0.5), 12)
S_ex = float(entanglement_entropy(psi_ex, tuple(range(6))))
n_ex = neg(rdm(psi_ex, (5, 6)))
print(f"  exact: E = {E_ex:.12f}   S = {S_ex:.10f} bit   N(1) = {n_ex:.10f}")
print(f"  {'chi':>4s} {'|dE|':>10s} {'|dS|':>10s} {'|dN(1)|':>10s} {'discarded wt':>13s} {'|dN|/wt':>8s}")
conv = []
for chiv in (2, 4, 8, 16, 32, 64):
    Bv, lamv, hv = dmrg(xxz_mpo(12, 1., 1., 1.5, hx=0.5), product_mps("01" * 6, chiv)[0], chiv, 8)
    eps = max(x[4] for x in hv if x[0] == hv[-1][0])
    dE = abs(E_ex - hv[-1][3])
    dS = abs(S_ex - float(np.asarray(mps_entropies(lamv))[6]))
    dn = abs(n_ex - neg(mps_rdm2(Bv, lamv, 5, 6)))
    conv.append((chiv, dE, dS, dn, eps))
    ratio = f"{dn/eps:8.1f}" if eps > 1e-20 else "     ---"
    print(f"  {chiv:4d} {dE:10.2e} {dS:10.2e} {dn:10.2e} {eps:13.2e} {ratio}")

# %% [markdown]
# All three errors fall together, by ten to eleven orders of magnitude between $\chi=2$ and $\chi=32$, and at $\chi=64$
# (the exact bond dimension for $N=12$) they are at rounding level. The negativity error stays within a factor of seven
# of the energy error at every $\chi$ in this run. The discarded weight (the largest of the last sweep) tracks both, but it is **not** an
# upper bound on the negativity error: the ratio $\vert\delta\mathcal N\vert/\varepsilon$ is between about $7$ and $25$
# in the converging rows. A usable error bar is therefore the change between two bond dimensions, not $\varepsilon$
# itself. In general a local observable can converge like $\sqrt{\varepsilon}$, slower than the energy; that it does not
# here is a measured property of this gapped point.

# %%
print("CHECK 3 -- is the MPS two-site reduced state a legitimate density matrix?  (the chi = 64 state of CHECK 2)")
wtr = wher = 0.0; weig = 0.0
for a in range(11):
    for b in range(a + 1, 12):
        r = np.asarray(mps_rdm2(Bv, lamv, a, b))
        wtr = max(wtr, abs(np.trace(r).real - 1)); wher = max(wher, np.abs(r - r.conj().T).max())
        weig = min(weig, np.linalg.eigvalsh(r).min())
print(f"  over all 66 pairs: |trace-1| <= {wtr:.1e}, non-hermiticity <= {wher:.1e}, "
      f"most negative eigenvalue {weig:+.1e}")
assert wtr < 1e-10 and wher < 1e-10 and weig > -1e-10

# %% [markdown]
# Unit trace, Hermiticity and positivity hold to rounding for all 66 pairs. This check matters for the negativity in
# particular: a contraction returning a slightly non-positive matrix would hand the partial transpose spurious negative
# eigenvalues and report entanglement where there is none.

# %%
fig, ax = plt.subplots(figsize=(5, 2.8))
cc = [c[0] for c in conv[:-1]]                       # chi = 64 is exact: its errors are rounding
ax.loglog(cc, [c[1] for c in conv[:-1]], "-o", ms=3, label=r"$|\Delta E|$")
ax.loglog(cc, [c[3] for c in conv[:-1]], "-s", ms=3, label=r"$|\Delta \mathcal{N}(1)|$")
ax.loglog(cc, [c[4] for c in conv[:-1]], "--^", ms=3, label=r"discarded weight $\varepsilon$")
ax.set_xticks(cc); ax.set_xticklabels([str(c) for c in cc]); ax.minorticks_off()
ax.set_xlabel(r"bond dimension $\chi$"); ax.set_ylabel("error")
ax.legend(fontsize=8); ax.set_title(r"N = 12, $(\Delta, h_x) = (1.5, 0.5)$")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 13. The cost of each route
#
# Both methods give the same numbers, so the choice between them is cost. The cell times one ground state at increasing
# $N$, with the field point of Section 12 throughout. Run it on an otherwise idle machine; the timings below come from
# the build machine and vary by tens of per cent between runs. Two caveats: the DMRG column is six sweeps at $\chi=64$
# throughout, so it scales linearly in a number that is a choice rather than a property of the method; and the first
# Lanczos call at a new $N$ pays a compilation overhead (measured separately on the build machine: $2.0$ s for the first
# call at $N=12$ against $0.4$ s for the next, $4.8$ s against $3.3$ s at $N=16$). The cell therefore warms each $N$
# with a three-step Lanczos call, which costs little, before timing. What survives the caveats is the shape: one cost
# grows exponentially in $N$ and the other does not.

# %%
print(f"{'N':>3s} | {'Lanczos [s]':>11s} {'DMRG [s]':>9s} | {'|dE|':>9s} {'|dS|':>9s} {'|dN(1)|':>9s}")
cost = []
for Nv in (12, 14, 16, 18, 20):
    cv = Nv // 2 - 1
    lanczos_ground_state(heisenberg_terms(Nv, 1., 1., 1.5, hx=0.5), Nv, m=3, restarts=1)   # warm-up at this N (cheap)
    t0 = time.perf_counter(); Ev, pv = lanczos_ground_state(heisenberg_terms(Nv, 1., 1., 1.5, hx=0.5), Nv)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    Bv, lamv, hv = dmrg(xxz_mpo(Nv, 1., 1., 1.5, hx=0.5), product_mps(("01" * Nv)[:Nv], 64)[0], 64, 6)
    t3 = time.perf_counter()
    dS = abs(float(entanglement_entropy(pv, tuple(range(Nv // 2)))) - float(np.asarray(mps_entropies(lamv))[Nv // 2]))
    dn = abs(neg(rdm(pv, (cv, cv + 1))) - neg(mps_rdm2(Bv, lamv, cv, cv + 1)))
    cost.append((Nv, t1 - t0, t3 - t2))
    print(f"{Nv:3d} | {t1-t0:11.1f} {t3-t2:9.1f} | {abs(Ev-hv[-1][3]):9.1e} {dS:9.1e} {dn:9.1e}")
    assert dS < 1e-8 and dn < 1e-8
cross = [Nv for Nv, tl, td in cost if tl > td]
print(f"\nfirst N at which Lanczos is slower than DMRG in this run: {cross[0] if cross else 'none up to 20'}")

# %% [markdown]
# The two routes agree to $10^{-10}$ or better at every $N$, so the comparison is between equally accurate answers. The
# Lanczos time grows by a factor of between about $2.5$ and $5$ per two spins, approaching the factor $4$ of the $2^N$
# vector length (each of the $80$ Krylov vectors is reorthogonalised against all previous ones) once the fixed overheads
# no longer dominate; the DMRG time grows roughly linearly in $N$. The crossover in
# this run is printed above; past $N\approx20$ only the MPS is practical on a laptop, and the state vector itself
# ($2^N$ complex numbers of 16 bytes: $17$ MB at $N=20$, $69$ GB at $N=32$) stops fitting in memory a dozen spins later.
#
# ## 14. The same plane where no state vector exists
#
# Section 10 mapped the phase diagram with an exact state vector. With `dmrg` and `mps_rdm2` validated, the same map
# can be made at $N=32$, where a state vector would need $2^{32}\approx4\times10^9$ amplitudes; as a matrix product
# state at $\chi=32$ it costs a few seconds per point. We run a $5\times5$ grid and compare with $N=12$ at the **same**
# points.
#
# DMRG improves its state by local updates, and a local update started from a state with a conserved quantity can only
# leave that symmetry sector through rounding or through a deliberate random restart of the local solver. The Néel state
# used so far has $\sum_jZ_j = 0$, and at $h_x=0$ the Hamiltonian conserves $\sum_jZ_j$; at $\Delta=1$ it conserves
# $\sum_jX_j$ for any $h_x$. A product state with a generic spin direction overlaps every sector, so the runs below start
# from one, and a control at one point repeats the run from the Néel state.

# %%
Nb, gb, chib = 32, 5, 32
Db = np.linspace(-2.0, 2.0, gb)
hb = np.linspace(0.0, 2.0, gb)
cb = Nb // 2 - 1
v_gen = np.array([np.cos(0.55), np.exp(0.7j) * np.sin(0.55)])         # a generic spin direction

def generic_mps(N, chi):
    """Product MPS with every spin along the generic direction v_gen (right-canonical: one bond index used)."""
    M0 = product_mps("0" * N, chi)[0]
    return M0.at[:, 0, :, 0].set(jnp.asarray(v_gen, dtype=M0.dtype))

S_b = np.zeros((gb, gb)); N_b = np.zeros((gb, gb)); E_b = np.zeros((gb, gb))
S_12 = np.zeros((gb, gb)); N_12 = np.zeros((gb, gb))
t0 = time.perf_counter()
for a_, hv in enumerate(hb):
    for b_, Dv in enumerate(Db):
        B_, lam_, hist_ = dmrg(xxz_mpo(Nb, 1., 1., Dv, hx=hv), generic_mps(Nb, chib), chib, 4)
        E_b[a_, b_] = hist_[-1][3]
        S_b[a_, b_] = float(np.asarray(mps_entropies(lam_))[Nb // 2])
        N_b[a_, b_] = neg(mps_rdm2(B_, lam_, cb, cb + 1))
el = time.perf_counter() - t0
for a_, hv in enumerate(hb):                     # the same points at N = 12, taken from the Section 10 map
    for b_, Dv in enumerate(Db):
        a10, b10 = int(np.argmin(np.abs(hs - hv))), int(np.argmin(np.abs(Ds - Dv)))
        assert abs(hs[a10] - hv) < 1e-12 and abs(Ds[b10] - Dv) < 1e-12
        S_12[a_, b_], N_12[a_, b_] = S_map[a10, b10], N_map[a10, b10]
print(f"{gb}x{gb} DMRG grid at N={Nb}, chi={chib}, 4 sweeps, generic start: {el:.0f} s ({el/gb**2:.1f} s per point)")

# control: the Neel start at one field point
a_c, b_c = 4, 2                                   # (Delta, hx) = (0, 2)
_, _, hN = dmrg(xxz_mpo(Nb, 1., 1., Db[b_c], hx=hb[a_c]), product_mps(("01" * Nb)[:Nb], chib)[0], chib, 4)
print(f"control at (Delta,hx) = ({Db[b_c]:+.0f}, {hb[a_c]:.1f}): E(generic start) = {E_b[a_c, b_c]:.8f}, "
      f"E(Neel start) = {hN[-1][3]:.8f}, difference {hN[-1][3]-E_b[a_c, b_c]:+.1e}")
assert abs(E_b[a_c, b_c] - hN[-1][3]) < 1e-6

print(f"\n{'(Delta,hx)':>12s} | {'S, N=12':>8s} {'S, N=32':>8s} | {'neg, N=12':>9s} {'neg, N=32':>9s}")
for a_, hv in enumerate(hb):
    for b_, Dv in enumerate(Db):
        tag = "  degenerate (h=0, Delta<=-1)" if (hv == 0 and Dv <= -1) else ("  quasi-degenerate" if Dv < -1 else "")
        print(f"  ({Dv:+.0f}, {hv:.1f}) | {S_12[a_, b_]:8.3f} {S_b[a_, b_]:8.3f} | {N_12[a_, b_]:9.4f} {N_b[a_, b_]:9.4f}{tag}")

ok = Db[None, :] > -1.5 + 0 * hb[:, None]                  # leave out the Delta = -2 column
ok &= ~((hb[:, None] == 0) & (Db[None, :] == -1))           # and the SU(2) ferromagnet at (-1, 0)
grow = (S_b > S_12 + 0.01) & ok
print(f"\nover the {int(ok.sum())} non-degenerate points: S grows from N=12 to N=32 at {int(grow.sum())}, "
      f"changes by less than 0.01 bit at {int((ok & (np.abs(S_b - S_12) <= 0.01)).sum())}, decreases at "
      f"{int((ok & (S_b < S_12 - 0.01)).sum())}")
print(f"  max S over these points: N=12 {S_12[ok].max():.3f}, N=32 {S_b[ok].max():.3f};   "
      f"max negativity: N=12 {N_12[ok].max():.3f}, N=32 {N_b[ok].max():.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True)
for a_, (M, ttl) in enumerate(((S_b, f"half-chain entropy [bit], N={Nb}"),
                               (N_b, rf"negativity $\mathcal{{N}}(1)$, N={Nb}"))):
    im = ax[a_].imshow(M, origin="lower", aspect="auto", cmap="magma",
                       extent=[Db[0] - 0.5, Db[-1] + 0.5, hb[0] - 0.25, hb[-1] + 0.25])
    ax[a_].set_xlabel(r"$\Delta$"); ax[a_].set_ylabel(r"$h_x$"); ax[a_].set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax[a_])
plt.show()

# %% [markdown]
# **Starting state.** At the control point the Néel start and the generic start reach the same energy (difference
# printed above), so here the result does not depend on the start. That is a property of the local solver, which
# restarts from a random vector when its Krylov space closes, and it is cheap to verify; a DMRG map is only as good as
# its worst point. Separately measured convergence runs ($\chi=64$, ten sweeps, same generic start) change the entropy
# by at most $1.3\times10^{-4}$ bit and the negativity by at most $2\times10^{-6}$ at every point of the grid except
# two. At the SU(2) ferromagnet $(-1,0)$ the ground state is a degenerate multiplet and the changes are $7\times10^{-4}$
# bit and $3\times10^{-4}$. At $(\Delta,h_x)=(-2,2)$ four sweeps at $\chi=32$ are not converged: the entropy is $0.63$
# bit there and $1.04$ bit at $\chi=64$ after ten sweeps, whose last sweep still lowers the energy by $10^{-6}$.
#
# **Degenerate points.** In the $\Delta=-2$ column the two methods disagree completely about the entropy: about one bit
# at $N=12$, nearly zero at $N=32$ for $h_x\le1.5$. Both are right about the state they found. At $h_x=0$ the doublet
# $\vert0\ldots0\rangle,\vert1\ldots1\rangle$ is exactly degenerate (Section 10); at $h_x>0$ the field mixes the two
# only at order $N$ in perturbation theory, so the splitting is exponentially small. Lanczos returns a superposition of
# the two branches, a GHZ-like state with up to one bit across the cut; DMRG, which works with a limited bond dimension
# and local updates, settles on a single symmetry-broken branch with entropy near zero. The negativity of the central
# pair is nearly the same in both for $h_x\le1.5$ ($0.003$, $0.012$, $0.03$), because a pair state is the same mixture
# of almost-product states whether the global state is the superposition or one branch. At $(\Delta,h_x)=(-1,0)$
# the chain is the isotropic ferromagnet (after the rotation of Section 10), whose ground level is $(N+1)$-fold
# degenerate; any value of the entropy there is a property of the solver.
#
# **Everywhere else** the comparison is meaningful. The half-chain entropy grows from $N=12$ to $N=32$ at most of these
# points, as expected for gapless chains, where it grows logarithmically with the length, and stays unchanged at the
# gapped, field-polarised points such as $(-1, 2)$. The nearest-neighbour negativity does not grow systematically: it
# moves up at some points and down at others, by up to about $0.2$. The largest changes are at $\Delta=1$, where
# $\sum_jX_j$ is conserved and the finite-chain ground state changes magnetisation sector in discrete jumps as $h_x$
# grows, at different fields for different $N$. Its maximum over these points is smaller at $N=32$ than at $N=12$. That is the difference between a quantity attached to a
# cut, which has more block to grow into, and a quantity attached to one pair of neighbours, which is bounded by
# monogamy and converges to its bulk value.

# %% [markdown]
# ## 15. Key takeaways
#
# * For a **pure** global state, entanglement across a cut is the Schmidt spectrum. The state is entangled exactly when
#   the reduced state is mixed, and every pure-state measure is a function of the $p_k$; every Schur-concave function
#   of them is an LOCC monotone. The entropy is singled out by asymptotic conversion: $n$ copies are interconvertible
#   with $nS_A$ Bell pairs. The log-negativity of such a cut is the Rényi-$\tfrac12$ entropy and carries nothing new.
# * A single number at a single cut is **not** a classification. GHZ and the Dicke state are indistinguishable by one
#   spin, yet GHZ has exactly zero pairwise entanglement, from its two-spin state alone. The W state has concurrence
#   $2/N$ in every pair and saturates the monogamy inequality; GHZ puts all of its tangle outside the pairs.
# * For a **mixed** state the subsystem entropy is not an entanglement measure: a Bell pair and a classical coin toss
#   have the same reduced state and the same one bit of entropy, and negativity $0.5$ against $0$.
# * Quantifying mixed-state entanglement properly means minimising over ensembles, with a closed form only for two
#   qubits; deciding separability is NP-hard in general, and $E_D$ and the negativity are not faithful. The PPT
#   criterion is the computable substitute and is complete for two qubits. The negativity reproduces the Werner
#   threshold $w=\tfrac13$, where the smallest partial-transpose eigenvalue $(1-3w)/4$ changes sign; under two-sided
#   depolarising noise the same threshold appears as $p^* = \tfrac34(1-1/\sqrt3) = 0.316987$ per spin.
# * In the XXZ ground states studied here, pairwise entanglement is nearest-neighbour at all four parameter points: the
#   more distant pairs have a partial transpose that is positive with a margin of at least $0.03$, which for two qubits
#   proves separability. The mutual information is long-ranged, and the pair entropy grows with separation. Monogamy
#   bounds the pairwise share of a spin's tangle; the neighbours hold $28$–$42\,\%$ of it.
# * Both measures come from a state vector or from an MPS. The entropy is free from the Schmidt values; the negativity
#   needs a two-site reduced state, a three-step contraction costing $O(j\chi^3)$ when built from the left boundary.
#   The two routes agree to $10^{-10}$ or better. The discarded weight tracks the negativity error but is not a bound on
#   it, and a DMRG map needs a symmetry-breaking start and attention to degenerate points.
#
# ## 16. Exercises
#
# 1. (★) Write the two-spin reduced state of GHZ at $N=5$ and confirm it is $\tfrac12\mathrm{diag}(1,0,0,1)$. Check that
#    its partial transpose has no negative eigenvalue, and explain why the two facts had to agree.
# 2. (★) Find the PPT boundary of the Werner state by bisection instead of by scanning, and compare with $1/3$. Then
#    replace $\mathbb 1/4$ by $\vert00\rangle\langle00\vert$ and find the new threshold.
# 3. (★) Verify the pure-state identity $E_{\mathcal N} = 2\log_2\sum_k\lambda_k$ on a Haar-random state of eight spins
#    for *every* contiguous cut $\{0,\ldots,\ell-1\}$, and plot it against $S(\ell)$.
# 4. (★★) For the ground state at $N=10$, $\Delta=1$, compute the concurrence of every pair containing a fixed spin and
#    the residual tangle $\tau_i-\sum_jC_{ij}^2$, for each reference spin $i=0,\ldots,9$. Where along the chain is the
#    pairwise share of the tangle largest, and why? Repeat at $\Delta=0$ and $\Delta=2$.
# 5. (★★) Generalise the contraction of Section 11 to two **blocks** of $\ell$ adjacent sites. Check that $\ell=1$
#    reproduces Section 9(a), then compute the negativity between two blocks of $\ell=3$ sites separated by a gap of one
#    site in the Heisenberg chain of Section 9. Is it zero? Why can a larger block be entangled when its single spins
#    are not?
# 6. (★★) Repeat Section 8 with the dephasing channel $\rho\to(1-p)\rho+pZ\rho Z$ (`kraus_dephasing`) instead of the
#    depolarising one. At what strength does the pair become separable, and why does the answer differ?
# 7. (★★★) Reproduce Section 13 on your own machine and locate the crossover. Then repeat the DMRG column at
#    $\chi=32$ and $\chi=128$ and explain which parts of the curve move and which do not.
#
# ## 17. References
#
# * A. Peres, *Separability criterion for density matrices*, Phys. Rev. Lett. **77**, 1413 (1996) — the PPT criterion.
# * M. Horodecki, P. Horodecki, R. Horodecki, *Separability of mixed states: necessary and sufficient conditions*,
#   Phys. Lett. A **223**, 1 (1996) — PPT is sufficient for $2\times2$ and $2\times3$.
# * P. Horodecki, *Separability criterion and inseparable mixed states with positive partial transposition*,
#   Phys. Lett. A **232**, 333 (1997) — entangled PPT states.
# * M. A. Nielsen, *Conditions for a class of entanglement transformations*, Phys. Rev. Lett. **83**, 436 (1999) —
#   pure-state LOCC conversion and majorisation.
# * G. Vidal, *Entanglement monotones*, J. Mod. Opt. **47**, 355 (2000) — LOCC monotones of pure states.
# * C. H. Bennett, H. J. Bernstein, S. Popescu, B. Schumacher, *Concentrating partial entanglement by local operations*,
#   Phys. Rev. A **53**, 2046 (1996) — the entropy as the asymptotic conversion rate.
# * W. Dür, G. Vidal, J. I. Cirac, *Three qubits can be entangled in two inequivalent ways*, Phys. Rev. A **62**, 062314
#   (2000) — the GHZ and W classes.
# * G. Vidal, R. F. Werner, *Computable measure of entanglement*, Phys. Rev. A **65**, 032314 (2002).
# * M. B. Plenio, *Logarithmic negativity: a full entanglement monotone that is not convex*, Phys. Rev. Lett. **95**,
#   090503 (2005).
# * W. K. Wootters, *Entanglement of formation of an arbitrary state of two qubits*, Phys. Rev. Lett. **80**, 2245 (1998).
# * V. Coffman, J. Kundu, W. K. Wootters, *Distributed entanglement*, Phys. Rev. A **61**, 052306 (2000).
# * T. J. Osborne, F. Verstraete, *General monogamy inequality for bipartite qubit entanglement*, Phys. Rev. Lett.
#   **96**, 220503 (2006).
# * L. Gurvits, *Classical deterministic complexity of Edmonds' problem and quantum entanglement*, in Proceedings of the
#   35th Annual ACM Symposium on Theory of Computing (STOC '03), 10–19 (2003) — NP-hardness of deciding separability.
# * R. F. Werner, *Quantum states with Einstein-Podolsky-Rosen correlations admitting a hidden-variable model*,
#   Phys. Rev. A **40**, 4277 (1989) — Werner states.
# * L. Amico, R. Fazio, A. Osterloh, V. Vedral, *Entanglement in many-body systems*, Rev. Mod. Phys. **80**, 517 (2008).
