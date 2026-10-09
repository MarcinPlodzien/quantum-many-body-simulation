goal: Simulate how information is sent reliably and kept secret, from Shannon's noisy channels to quantum key distribution with noisy entanglement
---
Communication is the problem of making a distant party learn something. The sender, Alice, and the receiver, Bob, are
connected only by a physical channel such as a cable, an optical fibre or a radio link, and every real channel makes
errors. Cryptography adds a second requirement: the message must stay secret from an eavesdropper, Eve, who may listen
to the channel or even tamper with it. Every bank transfer, every encrypted message and every software update depends on
solving both problems at once. Today secrecy rests mostly on mathematical problems that are believed to be hard, above
all the factoring of large integers and the discrete logarithm, and a large quantum computer would solve both
efficiently (P. W. Shor, *SIAM J. Comput.* **26**, 1484 (1997), doi:10.1137/S0097539795293172). Quantum mechanics also
offers a remedy: a way to create a shared secret key whose security rests on the laws of physics and on clearly stated
assumptions about the devices, instead of on unproven assumptions about computing power.

The literature on communication tells its stories with a fixed cast. Alice and Bob are the trusted parties, who follow
the protocol, the complete public list of instructions saying who prepares, measures and sends what and when. Eve is the
adversary, and a security claim is meaningful only together with a statement of what she is allowed to do, the threat
model. On a classical channel she may listen without leaving a trace; on a quantum channel she cannot learn anything
without disturbing the signal in general, and this is the physics behind quantum key distribution. A third party,
Charlie, appears in some protocols as the source of entangled pairs or as a relay in the middle; whether Alice and Bob
must trust him is part of the threat model.

**Part I, the classical toolbox.** Quantum cryptography solves a classical problem, so the chapter starts with the
classical language in which its results are stated. [Notebook
50](../ch14_quantum_communication_and_cryptography/50_information_entropy_and_noisy_channels.ipynb) introduces Shannon's
entropy as the measure of information, conditional entropy and mutual information, channels and their capacity, and
error correction by parity checks, including the public parity-check search with which Alice and Bob remove the errors
from a shared key. [Notebook
51](../ch14_quantum_communication_and_cryptography/51_secrecy_one_time_pad_and_authentication.ipynb) treats secrecy:
the one-time pad, which is perfectly secure but needs a secret random key as long as the message, the key-distribution
problem that this creates, the idea of authentication (the public discussion between Alice and Bob must be protected
against forgery, which requires a short key shared in advance), and privacy amplification, which removes Eve's partial
knowledge of a shared key.

**Part II, the quantum part.** The second half asks what quantum mechanics changes. An unknown quantum state cannot be
copied, and gaining information about it disturbs it; entangled pairs give correlations that no classical mechanism
reproduces, yet carry no signal. [Notebook
52](../ch14_quantum_communication_and_cryptography/52_quantum_rules_channels_and_capacities.ipynb) derives and
simulates these rules, together with quantum channels and their capacities for classical and quantum information.
[Notebook 53](../ch14_quantum_communication_and_cryptography/53_quantum_key_distribution.ipynb) turns these rules
into protocols: quantum key distribution, prepare-and-measure (BB84) and entanglement-based, in which the error rate
that Alice and Bob measure bounds what Eve can know and fixes how long a secret key they can distil. Real
entanglement is noisy, and [notebook
54](../ch14_quantum_communication_and_cryptography/54_noisy_entanglement_distillation_repeaters_certification.ipynb)
treats noisy pairs as a resource: teleportation through them, twirling, the key rate when Eve holds the noise of the
pairs, distillation, quantum repeaters that carry
them over long distances, and how to certify them before use. Every protocol is run as a
simulation, round by round, with the eavesdropper and the noise included.

The quantum part builds on notebook 07 (density matrices and quantum channels) and on chapter 8, where the Bell states and the CHSH inequality (notebook 19), quantum
teleportation (notebook 20) and entanglement swapping and superdense coding (notebook 21) are derived and simulated.
