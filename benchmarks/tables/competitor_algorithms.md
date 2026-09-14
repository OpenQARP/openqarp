# Algorithm benchmarks — end-to-end optimization, qarp vs the field

_Generated 2026-09-14 10:59 UTC by `python -m benchmarks.run algorithms` — do not edit by hand._

Host: macOS-26.5.2-arm64-arm-64bit-Mach-O (arm64), Python 3.13.7.
Versions: openqarp 0.1.0, qiskit 2.5.2, qiskit-aer 0.17.2, pennylane 0.45.1, pennylane-lightning 0.45.0, qulacs 0.6.14, cirq-core 1.7.0, qsimcirq 0.22.1, ffsim 0.0.84, scipy 1.18.1, numpy 2.5.3.

**This track measures wall time of a full optimization**, not one circuit: the same classical optimizer (COBYLA — the same scipy code object, same start point, same evaluation budget) drives each SDK's native energy-evaluation pathway.  What differs between columns is exactly what a user buys: how fast the quantum side evaluates.

Protocol: fresh subprocess per measurement, medians of 3, pinned threads. The build phase (circuits, observables, engine setup) is excluded from the run time. Correctness: every stack's energy at the shared starting point must match the exact reference at rtol 1e-05 — a tight, trajectory-independent anchor — and its converged energy must land in the same optimum neighbourhood (±5e-3 Ha; a deterministic optimizer amplifies last-bit summation differences between stacks into slightly different trajectories, so bit-equality of finals is not a property the physics guarantees). VQE rows are additionally variational against the FCI energy stored with the committed molecule; QAOA rows against the brute-force MaxCut optimum (`benchmarks/algorithms/verify.py`).

† **ffsim optimizes a different, fermionic ansatz (LUCJ)** from the Hartree-Fock state — that is its design point and why it is fast; its cell is not trajectory-comparable and is checked against the variational band instead. `exact ref` is the numpy reference driven by the identical optimizer: the trajectory every column reproduces.

Expectation values: every stack contracts ⟨ψ|H|ψ⟩ inside its own engine through the path `python -m benchmarks.statevector.tune --energy` measured fastest on this host (a pulled-out statevector contracted with scipy is not the SDK's number and is not a candidate): lightning observable=sparse; qulacs observable=hermitian; aer observable=save_expectation_value; qsim observable=pauli_sum. Building the observable — lightning's CSR matrix included — is the build phase for every stack. Gate fusion follows `spec.FUSION` as on the statevector track; qulacs's parametric circuit cannot be merged without freezing its angles and runs gate by gate.

A cell marked ‡ was measured in a child whose single-qubit rotation kernels still ran far above their nominal cost (ratio to an H gate above 6) after 8 relaunches — a process-memory-layout artefact of the qulacs macOS wheel (`why_fast.md`); every child times that ratio before measuring and relaunches itself in a different layout when it trips. Read such a cell as indicative.

**Reading the VQE energies**: the qubit-side columns hold at the Hartree-Fock energy by construction — the circuit prepares the HF determinant, and Brillouin's theorem leaves a real single-rotation ansatz with no first-order descent direction, so a budgeted derivative-free optimizer stays on the plateau. That makes the row a pure throughput measurement of identical work (`verify.py` pins E(start) = HF and FCI ≤ final ≤ HF). The ffsim column reaching near-FCI at the same budget is the ansatz-quality contrast, not a simulator-speed one.

### Wall time to the evaluation budget

| workload | n | exact ref | **qarp** | qulacs | Aer | lightning | qsim | ffsim† | check |
|---|---|---|---|---|---|---|---|---|---|
| vqe | 4 | 147.6 ms | 55.8 ms | 52.4 ms | 130.4 ms† | 195.5 ms | 176.2 ms | 71.9 ms | ✓ |
| vqe | 12 | 29.88 s | 744.8 ms | 1.65 s | 4.01 s | 1.42 s | 11.21 s | 876.7 ms | ✓ |
| qaoa | 8 | 70.1 ms | 27.1 ms | 25.7 ms | 82.4 ms | 163.9 ms | 166.3 ms | n/a | ✓ |
| qaoa | 12 | 220.7 ms | 41.3 ms | 51.9 ms | 136.6 ms | 235.2 ms | 255.0 ms | n/a | ✓ |

### Converged energy (Ha for VQE; −⟨cut⟩ for QAOA)

| workload | n | exact ref | **qarp** | qulacs | Aer | lightning | qsim | ffsim† |
|---|---|---|---|---|---|---|---|---|
| vqe | 4 | -1.116999 | -1.116999 | -1.116999 | -1.116999 | -1.116999 | -1.116998 | -1.136336 |
| vqe | 12 | -7.862023 | -7.862023 | -7.862023 | -7.862023 | -7.862023 | -7.862005 | -7.880430 |
| qaoa | 8 | -9.248041 | -9.246009 | -9.246009 | -9.246009 | -9.246009 | -9.246004 | n/a |
| qaoa | 12 | -14.251321 | -14.251321 | -14.251321 | -14.251321 | -14.251321 | -14.251292 | n/a |

### Peak memory (MiB, whole process — imports included)

| workload | n | exact ref | **qarp** | qulacs | Aer | lightning | qsim | ffsim† |
|---|---|---|---|---|---|---|---|---|
| vqe | 4 | 73 | 147 | 74 | 114 | 357 | 192 | 208 |
| vqe | 12 | 75 | 150 | 77 | 124 | 512 | 196 | 209 |
| qaoa | 8 | 72 | 147 | 74 | 112 | 365 | 192 | n/a |
| qaoa | 12 | 72 | 146 | 74 | 114 | 490 | 192 | n/a |

### Capabilities (not timed)

Algorithms outside the rows above are ported only where an SDK ships an
idiomatic implementation — a hand-rolled port dressed up as a competitor
column would measure our porting skill, not the SDK:

| algorithm | qarp | qiskit | pennylane | cirq | qulacs |
|---|---|---|---|---|---|
| AdaptVQE | native (`qarp.algorithms`) | `qiskit-algorithms` (separate package) | `AdaptiveOptimizer` | — | — |
| QPE (composite) | native, structured fast path | tutorial circuits | tutorial circuits | tutorial circuits | — |
| QSE / QMEGS | native | paper ports only | — | — | — |
| SSVQE / VQD / PCE / DOS-QPE | native | partial (`VQD` in qiskit-algorithms) | — | — | — |

`—` = no idiomatic implementation; a cell names the mechanism, not a timing.

### Workloads

- **qaoa** — full QAOA (p = 2) for MaxCut on a deterministic 3-regular graph, same COBYLA budget on every stack; anchored to the brute-force optimum
- **vqe** — full VQE to a fixed COBYLA budget: hardware-efficient ansatz (3 layers), committed molecular Hamiltonian (4 q = H2, 12 q = LiH); ffsim optimizes its native LUCJ ansatz instead
