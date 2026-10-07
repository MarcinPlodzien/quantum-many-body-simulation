goal: Find ground states with Lanczos and propagate states with Trotter, Chebyshev and Krylov methods
---
Each method of this chapter has its own notebook. Spin Hamiltonians on arbitrary graphs are written as lists of
local terms, and the Lanczos algorithm, derived from the power method, delivers their ground states without
building a matrix. Three propagators follow: Trotter–Suzuki splitting with two-site gates (orders one, two and
four), the Chebyshev expansion of the propagator, and Krylov-subspace exponentiation with an a-posteriori error
bound. Each is derived, implemented with `jit` and `lax.scan`, and validated against exact evolution.

A head-to-head benchmark compares the integrators on accuracy and cost, and the final notebook uses them for the
physics of quantum quenches in spin chains: relaxation of the magnetisation, light cones, domain-wall melting,
entanglement growth and the Loschmidt echo.
