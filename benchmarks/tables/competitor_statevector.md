# Statevector benchmarks — qarp vs qulacs / Aer / lightning / qsim

_Generated 2026-09-14 10:59 UTC by `python -m benchmarks.run statevector` — do not edit by hand._

Host: macOS-26.5.2-arm64-arm-64bit-Mach-O (arm64), Python 3.13.7.
Versions: openqarp 0.1.0, qiskit 2.5.2, qiskit-aer 0.17.2, pennylane 0.45.1, pennylane-lightning 0.45.0, qulacs 0.6.14, cirq-core 1.7.0, qsimcirq 0.22.1, numpy 2.5.3.

Protocol: fresh subprocess per measurement, medians of 3, pinned threads (`OMP_NUM_THREADS=1` and friends). `n` is the qubit count. Times are the **run** phase only — circuit construction and observable assembly are the build phase and are excluded identically on every stack. Each child first warms its stack on a 1-qubit circuit, so SDK imports and one-time backend initialization land in the baseline rather than in the measurement; what remains is one evaluation of a freshly built circuit or observable, so a stack's per-object first-call cost is included (a warm loop reads lower where that cost is material — about 1 ms on lightning's sparse expectation path). Memory is peak RSS of the whole child, pinned immediately after the run so the check pass cannot inflate it.

Rows marked † had a spread across the 3 repeats wider than 25% of the median — read those as indicative. Sub-millisecond rows are dominated by dispatch overhead in every stack and are not a meaningful ranking.

Correctness: every stack's canonical fingerprint (norm, |ψ₀|², max probability, Hamming-weight moment, and a seeded-reference overlap) must agree with the **numpy reference** — a gate-by-gate statevector implementation in this repo, independent of every SDK under test (§18) — at rtol 1e-09. The `numpy ref` column is that oracle's own timing: the unoptimized baseline. qsim ships float32-only wheels, so it is checked at rtol 1e-05; its speed is not comparable like for like with the double-precision stacks.

Gate fusion: every stack runs the fusion its engine offers, at the setting `python -m benchmarks.statevector.tune` measured fastest on this host (single thread, geometric mean over the state families at n ≥ 16), and the fusion is inside the timed run phase for every stack because its product bakes in the gate parameters — see `spec.FUSION` for the settings and the numbers behind them: aer fusion_enable=True, fusion_max_qubit=5, fusion_threshold=14; qulacs optimizer=light; qsim max_fused_gate_size=3; lightning transform=off. qarp's per-gate kernels are a vendored qulacs v0.6.14; the two columns separate because `QarpSimulator` fuses into dense blocks (`fusion_max_qubits`, qarp_conventions.md §14) where qulacs's optimizer merges single-qubit runs, and again on full-stack rows where the engine takes algorithmic fast paths.

Expectation values: every stack contracts ⟨ψ|H|ψ⟩ inside its own engine through the path `python -m benchmarks.statevector.tune --energy` measured fastest on this host (a pulled-out statevector contracted with scipy is not the SDK's number and is not a candidate): lightning observable=sparse; qulacs observable=hermitian; aer observable=save_expectation_value; qsim observable=pauli_sum. Building the observable — lightning's CSR matrix included — is the build phase for every stack.

A cell marked ‡ was measured in a child whose single-qubit rotation kernels still ran far above their nominal cost (ratio to an H gate above 6) after 8 relaunches — a process-memory-layout artefact of the qulacs macOS wheel (`why_fast.md`); every child times that ratio before measuring and relaunches itself in a different layout when it trips. Read such a cell as indicative.

Three families reuse the `n` column for a different axis: **`vqe_terms`** sweeps the percentage of Hamiltonian terms at fixed n = 12 (the term-count cost, statevector held fixed); **`brickwork_threads`** sweeps the thread count at fixed n = 20 (`OMP_NUM_THREADS` and friends set per row — the core-scaling axis, exempt from the pinned-mode rule); **`vqe_molecular`** selects the molecule by qubit count (4 = H₂, 12 = LiH, 14 = H₂O; STO-3G, committed JSON validated against FCI at generation time).

### Run time

| workload | n | numpy ref | **qarp** | qulacs | Aer | lightning | qsim | check |
|---|---|---|---|---|---|---|---|---|
| brickwork | 8 | 1.2 ms† | 0.1 ms† | 0.1 ms† | 1.2 ms† | 3.8 ms† | 1.3 ms† | ✓ |
| brickwork | 12 | 6.2 ms | 0.3 ms | 0.4 ms | 2.3 ms | 5.4 ms | 1.8 ms | ✓ |
| brickwork | 16 | 75.8 ms | 2.1 ms | 5.8 ms | 7.7 ms | 16.4 ms | 4.7 ms | ✓ |
| brickwork | 20 | 4.95 s | 40.1 ms | 116.8 ms | 125.0 ms | 249.7 ms | 50.7 ms | ✓ |
| qft | 8 | 1.2 ms | 0.1 ms | 0.1 ms | 1.2 ms | 4.4 ms | 1.4 ms | ✓ |
| qft | 12 | 8.0 ms | 0.6 ms | 0.9 ms | 3.2 ms | 8.2 ms | 3.4 ms | ✓ |
| qft | 16 | 125.4 ms | 8.1 ms | 19.8 ms | 13.5 ms | 36.8 ms | 14.2 ms | ✓ |
| qft | 20 | 8.90 s | 191.5 ms | 491.3 ms | 222.5 ms | 779.6 ms | 238.3 ms | ✓ |
| trotter_step | 8 | 1.0 ms | 0.1 ms† | 0.1 ms | 1.2 ms | 4.2 ms | 1.4 ms† | ✓ |
| trotter_step | 12 | 4.5 ms | 0.5 ms† | 0.7 ms† | 2.2 ms | 5.8 ms | 2.1 ms | ✓ |
| trotter_step | 16 | 52.7 ms | 4.7 ms | 11.4 ms | 12.8 ms | 15.4 ms | 8.0 ms | ✓ |
| trotter_step | 20 | 3.29 s | 95.3 ms | 230.5 ms | 217.2 ms | 198.9 ms | 111.7 ms | ✓ |
| vqe_energy | 4 | 0.4 ms† | 0.0 ms† | 0.0 ms | 0.4 ms | 0.9 ms | 0.5 ms | ✓ |
| vqe_energy | 8 | 0.9 ms | 0.1 ms† | 0.1 ms | 0.6 ms | 1.3 ms | 1.1 ms | ✓ |
| vqe_energy | 12 | 4.9 ms | 0.4 ms | 0.7 ms | 1.7 ms | 2.2 ms | 2.8 ms | ✓ |
| qpe_phase | 6 | 0.9 ms | 0.1 ms | 0.1 ms | 0.8 ms | 3.2 ms | 1.1 ms | ✓ |
| qpe_phase | 10 | 2.7 ms | 0.3 ms | 0.4 ms | 2.2 ms | 7.2 ms | 2.8 ms | ✓ |
| qpe_phase | 14 | 26.4 ms | 2.3 ms | 6.6 ms | 9.2 ms† | 14.4 ms | 6.5 ms | ✓ |
| vqe_terms | 25 | 2.1 ms | 0.2 ms† | 0.6 ms | 1.3 ms† | 2.3 ms | 1.3 ms | ✓ |
| vqe_terms | 50 | 2.8 ms | 0.3 ms† | 0.6 ms | 1.2 ms | 2.3 ms | 1.6 ms | ✓ |
| vqe_terms | 100 | 5.0 ms | 0.4 ms† | 0.7 ms† | 1.6 ms | 2.2 ms | 2.8 ms | ✓ |
| brickwork_threads | 1 | 5.00 s | 39.7 ms | 119.0 ms | 125.7 ms | 255.9 ms† | 50.9 ms | ✓ |
| brickwork_threads | 2 | 5.04 s | 21.8 ms | 119.5 ms | 67.0 ms | 254.5 ms | 29.0 ms | ✓ |
| brickwork_threads | 4 | 4.99 s | 12.3 ms | 120.0 ms | 35.7 ms | 253.2 ms | 18.6 ms | ✓ |
| brickwork_threads | 8 | 4.95 s | 8.9 ms | 118.8 ms | 29.0 ms† | 257.7 ms | 14.6 ms | ✓ |
| vqe_molecular | 4 | 0.5 ms | 0.0 ms | 0.0 ms† | 0.7 ms | 0.9 ms | 0.6 ms | ✓ |
| vqe_molecular | 12 | 69.2 ms | 1.1 ms | 3.1 ms | 7.5 ms | 2.3 ms | 32.1 ms | ✓ |
| vqe_molecular | 14 | 343.9 ms | 6.3 ms | 19.6 ms | 43.8 ms | 4.7 ms | 140.6 ms | ✓ |

### Peak memory (MiB, whole process — imports included)

Each SDK's import footprint is a fixed offset here, so compare columns by their growth with `n`, not by their absolute value. The next table subtracts that offset.

| workload | n | numpy ref | **qarp** | qulacs | Aer | lightning | qsim |
|---|---|---|---|---|---|---|---|
| brickwork | 8 | 31 | 127 | 34 | 77 | 357 | 191 |
| brickwork | 12 | 30 | 127 | 32 | 77 | 357 | 192 |
| brickwork | 16 | 38 | 128 | 33 | 78 | 358 | 192 |
| brickwork | 20 | 86 | 161 | 65 | 93 | 389 | 224 |
| qft | 8 | 30 | 128 | 33 | 77 | 357 | 192 |
| qft | 12 | 30 | 129 | 34 | 78 | 357 | 191 |
| qft | 16 | 39 | 128 | 34 | 79 | 358 | 192 |
| qft | 20 | 86 | 161 | 66 | 95 | 388 | 224 |
| trotter_step | 8 | 30 | 128 | 33 | 77 | 357 | 192 |
| trotter_step | 12 | 30 | 128 | 33 | 77 | 358 | 192 |
| trotter_step | 16 | 39 | 129 | 34 | 78 | 356 | 191 |
| trotter_step | 20 | 86 | 159 | 65 | 94 | 387 | 223 |
| vqe_energy | 4 | 201 | 217 | 203 | 236 | 359 | 204 |
| vqe_energy | 8 | 202 | 216 | 204 | 235 | 367 | 203 |
| vqe_energy | 12 | 201 | 217 | 203 | 235 | 491 | 205 |
| qpe_phase | 6 | 30 | 126 | 33 | 77 | 357 | 192 |
| qpe_phase | 10 | 30 | 127 | 33 | 77 | 357 | 192 |
| qpe_phase | 14 | 31 | 127 | 33 | 78 | 356 | 192 |
| vqe_terms | 25 | 202 | 216 | 203 | 235 | 486 | 204 |
| vqe_terms | 50 | 202 | 216 | 203 | 235 | 486 | 205 |
| vqe_terms | 100 | 202 | 217 | 205 | 236 | 489 | 204 |
| brickwork_threads | 1 | 86 | 160 | 65 | 94 | 386 | 224 |
| brickwork_threads | 2 | 86 | 161 | 65 | 94 | 389 | 223 |
| brickwork_threads | 4 | 86 | 161 | 65 | 94 | 387 | 224 |
| brickwork_threads | 8 | 86 | 160 | 66 | 94 | 389 | 224 |
| vqe_molecular | 4 | 30 | 146 | 33 | 77 | 357 | 192 |
| vqe_molecular | 12 | 32 | 147 | 35 | 79 | 514 | 194 |
| vqe_molecular | 14 | 34 | 150 | 36 | 81 | 1051 | 197 |

### Simulation memory (MiB, peak minus post-import baseline)

What the circuit costs on top of a loaded, warmed stack — the baseline is sampled after the warmup, so SDK imports and backend initialization are excluded. The 2ⁿ floor for one double-precision amplitude vector is 16 B × 2ⁿ = 16 MiB at n = 20, 256 MiB at n = 24, 4 GiB at n = 28; a stack sitting well above it is holding work copies.

| workload | n | numpy ref | **qarp** | qulacs | Aer | lightning | qsim |
|---|---|---|---|---|---|---|---|
| brickwork | 8 | 1 | 2 | 2 | 0 | 1 | 0 |
| brickwork | 12 | 0 | 2 | 1 | 1 | 2 | 0 |
| brickwork | 16 | 8 | 3 | 2 | 2 | 1 | 1 |
| brickwork | 20 | 56 | 35 | 34 | 17 | 32 | 32 |
| qft | 8 | 0 | 2 | 1 | 1 | 1 | 0 |
| qft | 12 | 0 | 3 | 2 | 1 | 1 | 0 |
| qft | 16 | 9 | 3 | 3 | 2 | 3 | 1 |
| qft | 20 | 56 | 35 | 35 | 18 | 32 | 33 |
| trotter_step | 8 | 0 | 2 | 2 | 0 | 0 | 0 |
| trotter_step | 12 | 0 | 3 | 1 | 1 | 1 | 1 |
| trotter_step | 16 | 9 | 3 | 2 | 2 | 2 | 1 |
| trotter_step | 20 | 56 | 33 | 33 | 17 | 32 | 32 |
| vqe_energy | 4 | 0 | 1 | 0 | 0 | 2 | 1 |
| vqe_energy | 8 | 0 | 1 | 1 | 1 | 8 | 0 |
| vqe_energy | 12 | 0 | 1 | 1 | 0 | 133 | 1 |
| qpe_phase | 6 | 0 | 1 | 2 | 0 | 2 | 0 |
| qpe_phase | 10 | 0 | 2 | 1 | 1 | 2 | 1 |
| qpe_phase | 14 | 1 | 2 | 1 | 2 | 0 | 0 |
| vqe_terms | 25 | 1 | 2 | 0 | 0 | 128 | 1 |
| vqe_terms | 50 | 0 | 1 | 0 | 0 | 129 | 1 |
| vqe_terms | 100 | 1 | 2 | 2 | 0 | 130 | 0 |
| brickwork_threads | 1 | 56 | 35 | 33 | 17 | 32 | 32 |
| brickwork_threads | 2 | 56 | 35 | 34 | 17 | 33 | 32 |
| brickwork_threads | 4 | 56 | 35 | 33 | 17 | 32 | 33 |
| brickwork_threads | 8 | 56 | 35 | 34 | 17 | 33 | 32 |
| vqe_molecular | 4 | 0 | 2 | 2 | 0 | 1 | 2 |
| vqe_molecular | 12 | 1 | 3 | 3 | 3 | 157 | 3 |
| vqe_molecular | 14 | 4 | 6 | 4 | 5 | 695 | 6 |

### Workloads

- **brickwork** — 4 layers of random two-qubit blocks (2 CX + 6 rotations each) in an even/odd brick pattern; gate count O(n)
- **brickwork_threads** — the brickwork circuit at fixed n = 20; the row's size is the thread count (OMP and friends set in the child environment) — the core-scaling axis
- **qft** — textbook QFT including final bit-reversal swaps; gate count O(n^2)
- **qpe_phase** — QPE-shaped circuit: 4 ancillas, controlled Trotter evolution of a TFIM system register, inverse QFT on the ancillas
- **trotter_step** — 5 first-order Trotter steps of the 1D transverse-field Ising chain (J=1.0, h=0.6, dt=0.1)
- **vqe_energy** — one VQE energy evaluation: 3-layer hardware-efficient ansatz, expectation of the Jordan-Wigner Fermi-Hubbard Hamiltonian (t=1.0, U=4.0)
- **vqe_molecular** — one energy evaluation of a pyscf-generated molecular Hamiltonian (4 q = H2, 12 q = LiH, 14 q = H2O, STO-3G; committed JSON, FCI-validated at generation time), same hardware-efficient ansatz as vqe_energy
- **vqe_terms** — one energy evaluation at fixed n = 12; the row's size is the percentage of Fermi-Hubbard terms kept — the term-count axis of the cost, with the statevector held fixed
