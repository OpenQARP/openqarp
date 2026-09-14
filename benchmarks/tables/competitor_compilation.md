# Compilation benchmarks — qarp vs qiskit O1/O2/O3 vs pytket

_Generated 2026-09-13 14:52 UTC by `python -m benchmarks.run compilation` — do not edit by hand._

Host: macOS-26.5.2-arm64-arm-64bit-Mach-O (arm64), Python 3.13.7.
Versions: openqarp 0.1.0, qiskit 2.5.2, pytket 2.18.1, mqt-bench 2.3.0, numpy 2.5.3.

**Lower is better here** — this track measures compiled-circuit quality, not speed. Compile wall time is reported last and is secondary: a compiler is allowed to be slower if the circuit it emits is materially cheaper to run.

Protocol: fresh subprocess per measurement, medians of 3, pinned threads. Every stack is given the **same edge list** and asked for the **same target basis** {h, rx, ry, rz, cx, swap} — two-qubit counts are not comparable otherwise. Metrics are computed here from the compiled gate list, not read from each SDK's own counters, so `depth` means one thing across the table. qiskit's layout and routing are stochastic above level 0, so `seed_transpiler` is pinned.

The `uncompiled` column is the fixture as written: the pre-routing cost every other column is trying to beat, and the oracle every compiled circuit is checked against. It is not routed, so its coupling and SWAP cells read `—`.

Correctness: each compiled circuit is simulated with the numpy reference, its reported final layout undone, and must reproduce the uncompiled circuit's state fingerprint (§18). `benchmarks/compilation/verify.py` additionally asserts full-unitary equivalence at small width and that every two-qubit gate lands on an architecture edge. The certification matrix below is a separate MQT QCEC exact-equivalence oracle.

### MQT QCEC equivalence certification

A ✓ requires QCEC's exact `equivalent` or global-phase-only `equivalent_up_to_global_phase` verdict from a clean record for this host and source commit. Probabilistic, relative-phase-capable, timed-out, missing and stale results fail closed. Runtime includes compilation and checking; memory is peak RSS of the fresh certification process.

Record: mqt-qcec 3.10.0, commit 01af2da76306.

| workload | n | uncompiled | **qarp** | qiskit O1 | qiskit O2 | qiskit O3 | pytket |
|---|---|---|---|---|---|---|---|
| trotter_line | 8 | ✓ decision_diagram_alternating; 278.8 ms; 86 MiB | ✓ decision_diagram_alternating; 1.12 s; 173 MiB | ✓ decision_diagram_alternating; 331.8 ms; 92 MiB | ✓ decision_diagram_alternating; 297.2 ms; 96 MiB | ✓ decision_diagram_alternating; 340.5 ms; 96 MiB | ✓ decision_diagram_alternating; 904.4 ms; 132 MiB |
| trotter_line | 12 | ✓ decision_diagram_alternating; 289.7 ms; 88 MiB | ✓ decision_diagram_alternating; 1.03 s; 175 MiB | ✓ decision_diagram_alternating; 348.7 ms; 95 MiB | ✓ decision_diagram_alternating; 518.6 ms; 99 MiB | ✓ decision_diagram_alternating; 350.4 ms; 100 MiB | ✓ decision_diagram_alternating; 1.89 s; 136 MiB |
| trotter_line | 14 | ✓ decision_diagram_alternating; 334.0 ms; 89 MiB | ✓ decision_diagram_alternating; 1.46 s; 178 MiB | ✓ decision_diagram_alternating; 514.9 ms; 96 MiB | ✓ decision_diagram_alternating; 331.0 ms; 101 MiB | ✓ decision_diagram_alternating; 335.0 ms; 102 MiB | ✓ decision_diagram_alternating; 1.49 s; 137 MiB |
| trotter_grid | 8 | ✓ decision_diagram_alternating; 285.7 ms; 86 MiB | ✓ decision_diagram_alternating; 1.06 s; 173 MiB | ✓ decision_diagram_alternating; 302.8 ms; 92 MiB | ✓ decision_diagram_alternating; 290.9 ms; 96 MiB | ✓ decision_diagram_alternating; 296.9 ms; 96 MiB | ✓ decision_diagram_alternating; 926.8 ms; 132 MiB |
| trotter_grid | 12 | ✓ decision_diagram_alternating; 301.1 ms; 88 MiB | ✓ decision_diagram_alternating; 1.10 s; 179 MiB | ✓ decision_diagram_alternating; 315.7 ms; 96 MiB | ✓ decision_diagram_alternating; 304.4 ms; 99 MiB | ✓ decision_diagram_alternating; 350.3 ms; 100 MiB | ✓ decision_diagram_alternating; 1.97 s; 138 MiB |
| trotter_grid | 14 | ✓ decision_diagram_alternating; 268.1 ms; 90 MiB | ✓ decision_diagram_alternating; 1.04 s; 179 MiB | ✓ decision_diagram_alternating; 304.2 ms; 97 MiB | ✓ decision_diagram_alternating; 273.0 ms; 102 MiB | ✓ decision_diagram_alternating; 274.4 ms; 101 MiB | ✓ decision_diagram_alternating; 1.69 s; 137 MiB |
| hea_line | 8 | ✓ decision_diagram_alternating; 266.2 ms; 86 MiB | ✓ decision_diagram_alternating; 900.2 ms; 173 MiB | ✓ decision_diagram_alternating; 263.5 ms; 92 MiB | ✓ decision_diagram_alternating; 273.9 ms; 96 MiB | ✓ decision_diagram_alternating; 277.3 ms; 96 MiB | ✓ decision_diagram_alternating; 849.9 ms; 132 MiB |
| hea_line | 12 | ✓ decision_diagram_alternating; 261.3 ms; 89 MiB | ✓ decision_diagram_alternating; 914.3 ms; 179 MiB | ✓ decision_diagram_alternating; 268.6 ms; 95 MiB | ✓ decision_diagram_alternating; 266.5 ms; 99 MiB | ✓ decision_diagram_alternating; 269.1 ms; 98 MiB | ✓ decision_diagram_alternating; 933.1 ms; 134 MiB |
| hea_line | 14 | ✓ decision_diagram_alternating; 278.0 ms; 89 MiB | ✓ decision_diagram_alternating; 1.13 s; 176 MiB | ✓ decision_diagram_alternating; 300.0 ms; 96 MiB | ✓ decision_diagram_alternating; 277.8 ms; 100 MiB | ✓ decision_diagram_alternating; 287.2 ms; 100 MiB | ✓ decision_diagram_alternating; 1.05 s; 137 MiB |
| hea_grid | 8 | ✓ decision_diagram_alternating; 262.2 ms; 86 MiB | ✓ decision_diagram_alternating; 925.1 ms; 173 MiB | ✓ decision_diagram_alternating; 280.2 ms; 90 MiB | ✓ decision_diagram_alternating; 289.3 ms; 95 MiB | ✓ decision_diagram_alternating; 293.4 ms; 95 MiB | ✓ decision_diagram_alternating; 857.6 ms; 132 MiB |
| hea_grid | 12 | ✓ decision_diagram_alternating; 288.8 ms; 88 MiB | ✓ decision_diagram_alternating; 1.06 s; 179 MiB | ✓ decision_diagram_alternating; 327.6 ms; 92 MiB | ✓ decision_diagram_alternating; 298.5 ms; 97 MiB | ✓ decision_diagram_alternating; 297.6 ms; 97 MiB | ✓ decision_diagram_alternating; 976.7 ms; 133 MiB |
| hea_grid | 14 | ✓ decision_diagram_alternating; 324.7 ms; 89 MiB | ✓ decision_diagram_alternating; 1.01 s; 178 MiB | ✓ decision_diagram_alternating; 308.3 ms; 94 MiB | ✓ decision_diagram_alternating; 313.6 ms; 98 MiB | ✓ decision_diagram_alternating; 315.4 ms; 98 MiB | ✓ decision_diagram_alternating; 1.09 s; 135 MiB |
| qv_line | 8 | ✓ decision_diagram_alternating; 300.4 ms; 87 MiB | ✓ decision_diagram_alternating; 1.01 s; 173 MiB | ✓ decision_diagram_alternating; 308.8 ms; 92 MiB | ✓ decision_diagram_alternating; 311.4 ms; 96 MiB | ✓ decision_diagram_alternating; 309.0 ms; 96 MiB | ✓ decision_diagram_alternating; 802.0 ms; 133 MiB |
| qv_line | 12 | ✓ decision_diagram_alternating; 308.0 ms; 89 MiB | ✓ decision_diagram_alternating; 1.02 s; 175 MiB | ✓ decision_diagram_alternating; 311.3 ms; 94 MiB | ✓ decision_diagram_alternating; 319.1 ms; 98 MiB | ✓ decision_diagram_alternating; 316.0 ms; 98 MiB | ✓ decision_diagram_alternating; 1.35 s; 137 MiB |
| qv_line | 14 | ✓ decision_diagram_alternating; 321.3 ms; 89 MiB | ✓ decision_diagram_alternating; 1.09 s; 176 MiB | ✓ decision_diagram_alternating; 312.8 ms; 95 MiB | ✓ decision_diagram_alternating; 312.9 ms; 98 MiB | ✓ decision_diagram_alternating; 311.9 ms; 99 MiB | ✓ decision_diagram_alternating; 1.10 s; 137 MiB |
| qv_grid | 8 | ✓ decision_diagram_alternating; 319.5 ms; 87 MiB | ✓ decision_diagram_alternating; 1.05 s; 173 MiB | ✓ decision_diagram_alternating; 310.0 ms; 92 MiB | ✓ decision_diagram_alternating; 317.0 ms; 96 MiB | ✓ decision_diagram_alternating; 318.7 ms; 96 MiB | ✓ decision_diagram_alternating; 811.1 ms; 132 MiB |
| qv_grid | 12 | ✓ decision_diagram_alternating; 303.5 ms; 89 MiB | ✓ decision_diagram_alternating; 1.05 s; 175 MiB | ✓ decision_diagram_alternating; 309.0 ms; 95 MiB | ✓ decision_diagram_alternating; 332.9 ms; 98 MiB | ✓ decision_diagram_alternating; 325.3 ms; 98 MiB | ✓ decision_diagram_alternating; 934.7 ms; 134 MiB |
| qv_grid | 14 | ✓ decision_diagram_alternating; 324.4 ms; 90 MiB | ✓ decision_diagram_alternating; 1.06 s; 176 MiB | ✓ decision_diagram_alternating; 326.2 ms; 96 MiB | ✓ decision_diagram_alternating; 331.0 ms; 99 MiB | ✓ decision_diagram_alternating; 333.9 ms; 100 MiB | ✓ decision_diagram_alternating; 1.16 s; 136 MiB |
| mqt_line | 8 | ✓ decision_diagram_alternating; 335.6 ms; 89 MiB | ✓ decision_diagram_alternating; 1.13 s; 178 MiB | ✓ decision_diagram_alternating; 359.3 ms; 94 MiB | ✓ decision_diagram_alternating; 447.9 ms; 98 MiB | ✓ decision_diagram_alternating; 298.6 ms; 98 MiB | ✓ decision_diagram_alternating; 926.1 ms; 136 MiB |
| mqt_line | 12 | ✓ decision_diagram_alternating; 295.2 ms; 92 MiB | ✓ decision_diagram_alternating; 1.02 s; 178 MiB | ✓ decision_diagram_alternating; 307.6 ms; 97 MiB | ✓ decision_diagram_alternating; 316.6 ms; 108 MiB | ✓ decision_diagram_alternating; 321.1 ms; 108 MiB | ✓ decision_diagram_alternating; 1.93 s; 148 MiB |
| mqt_line | 14 | ✓ decision_diagram_alternating; 305.1 ms; 93 MiB | ✓ decision_diagram_alternating; 2.37 s; 183 MiB | ✓ decision_diagram_alternating; 767.5 ms; 98 MiB | ✓ decision_diagram_alternating; 581.7 ms; 117 MiB | ✓ decision_diagram_alternating; 1.45 s; 116 MiB | ✓ decision_diagram_alternating; 3.70 s; 169 MiB |
| mqt_grid | 8 | ✓ decision_diagram_alternating; 5.74 s; 90 MiB | ✓ decision_diagram_alternating; 9.95 s; 178 MiB | ✓ decision_diagram_alternating; 336.7 ms; 94 MiB | ✓ decision_diagram_alternating; 338.8 ms; 98 MiB | ✓ decision_diagram_alternating; 328.0 ms; 98 MiB | ✓ decision_diagram_alternating; 1.32 s; 137 MiB |
| mqt_grid | 12 | ✓ decision_diagram_alternating; 278.4 ms; 92 MiB | ✓ decision_diagram_alternating; 1.02 s; 179 MiB | ✓ decision_diagram_alternating; 299.8 ms; 96 MiB | ✓ decision_diagram_alternating; 287.5 ms; 107 MiB | ✓ decision_diagram_alternating; 323.7 ms; 108 MiB | ✓ decision_diagram_alternating; 2.15 s; 148 MiB |
| mqt_grid | 14 | ✓ decision_diagram_alternating; 288.9 ms; 93 MiB | ✓ decision_diagram_alternating; 975.7 ms; 180 MiB | ✓ decision_diagram_alternating; 285.8 ms; 98 MiB | ✓ decision_diagram_alternating; 384.1 ms; 117 MiB | ✓ decision_diagram_alternating; 478.4 ms; 118 MiB | ✓ decision_diagram_alternating; 1.87 s; 170 MiB |

### CX-equivalent two-qubit gates (SWAP counted as 3)

The headline. A SWAP costs three CXs on hardware without a native SWAP, and the compilers disagree about whether to emit SWAP at all, so the raw two-qubit count is not comparable on its own.

| workload | n | uncompiled | **qarp** | qiskit O1 | qiskit O2 | qiskit O3 | pytket | check |
|---|---|---|---|---|---|---|---|---|
| trotter_line | 8 | 56 | 125 | 125 | 125 | 125 | 137 | ✓ |
| trotter_line | 12 | 132 | 312 | 318 | 315 | 315 | 471 | ✓ |
| trotter_line | 14 | 182 | 434 | 443 | 437 | 437 | 491 | ✓ |
| trotter_grid | 8 | 56 | 92 | 92 | 92 | 92 | 101 | ✓ |
| trotter_grid | 12 | 132 | 234 | 240 | 237 | 234 | 276 | ✓ |
| trotter_grid | 14 | 182 | 314 | 326 | 323 | 308 | 422 | ✓ |
| hea_line | 8 | 24 | 87 | 93 | 93 | 93 | 81 | ✓ |
| hea_line | 12 | 36 | 135 | 141 | 141 | 141 | 147 | ✓ |
| hea_line | 14 | 42 | 159 | 165 | 165 | 165 | 153 | ✓ |
| hea_grid | 8 | 24 | 24 | 24 | 24 | 24 | 24 | ✓ |
| hea_grid | 12 | 36 | 36 | 36 | 36 | 36 | 36 | ✓ |
| hea_grid | 14 | 42 | 42 | 42 | 42 | 42 | 42 | ✓ |
| qv_line | 8 | 32 | 56 | 56 | 52 | 52 | 55 | ✓ |
| qv_line | 12 | 48 | 117 | 120 | 117 | 117 | 156 | ✓ |
| qv_line | 14 | 56 | 155 | 155 | 150 | 150 | 201 | ✓ |
| qv_grid | 8 | 32 | 41 | 41 | 37 | 37 | 37 | ✓ |
| qv_grid | 12 | 48 | 66 | 75 | 69 | 66 | 84 | ✓ |
| qv_grid | 14 | 56 | 92 | 92 | 90 | 93 | 108 | ✓ |
| mqt_line | 8 | 68 | 161 | 170 | 170 | 170 | 137 | ✓ |
| mqt_line | 12 | 150 | 411 | 420 | 417 | 420 | 468 | ✓ |
| mqt_line | 14 | 203 | 575 | 584 | 587 | 584 | 491 | ✓ |
| mqt_grid | 8 | 68 | 113 | 113 | 110 | 110 | 92 | ✓ |
| mqt_grid | 12 | 150 | 261 | 285 | 270 | 273 | 264 | ✓ |
| mqt_grid | 14 | 203 | 386 | 407 | 392 | 401 | 413 | ✓ |

### Depth

Longest gate chain, computed from the compiled circuit.

| workload | n | uncompiled | **qarp** | qiskit O1 | qiskit O2 | qiskit O3 | pytket |
|---|---|---|---|---|---|---|---|
| trotter_line | 8 | 40 | 59 | 60 | 60 | 60 | 58 |
| trotter_line | 12 | 64 | 106 | 103 | 99 | 99 | 130 |
| trotter_line | 14 | 76 | 144 | 111 | 119 | 123 | 133 |
| trotter_grid | 8 | 40 | 68 | 48 | 53 | 59 | 62 |
| trotter_grid | 12 | 64 | 127 | 124 | 102 | 120 | 141 |
| trotter_grid | 14 | 76 | 122 | 114 | 110 | 93 | 138 |
| hea_line | 8 | 28 | 50 | 55 | 55 | 55 | 83 |
| hea_line | 12 | 40 | 73 | 76 | 76 | 79 | 114 |
| hea_line | 14 | 46 | 85 | 91 | 91 | 91 | 123 |
| hea_grid | 8 | 28 | 28 | 28 | 28 | 28 | 36 |
| hea_grid | 12 | 40 | 40 | 40 | 40 | 40 | 48 |
| hea_grid | 14 | 46 | 46 | 46 | 46 | 46 | 54 |
| qv_line | 8 | 16 | 23 | 23 | 23 | 23 | 34 |
| qv_line | 12 | 16 | 30 | 30 | 30 | 29 | 55 |
| qv_line | 14 | 16 | 35 | 35 | 32 | 34 | 60 |
| qv_grid | 8 | 16 | 18 | 18 | 18 | 18 | 27 |
| qv_grid | 12 | 16 | 22 | 29 | 35 | 35 | 43 |
| qv_grid | 14 | 16 | 25 | 25 | 25 | 26 | 48 |
| mqt_line | 8 | 58 | 92 | 81 | 79 | 79 | 80 |
| mqt_line | 12 | 90 | 149 | 148 | 134 | 138 | 164 |
| mqt_line | 14 | 106 | 181 | 176 | 163 | 159 | 169 |
| mqt_grid | 8 | 58 | 94 | 76 | 68 | 70 | 74 |
| mqt_grid | 12 | 90 | 169 | 148 | 140 | 148 | 164 |
| mqt_grid | 14 | 106 | 159 | 173 | 134 | 148 | 164 |

### SWAPs inserted

Routing overhead alone; `—` where the stack is not routed.

| workload | n | uncompiled | **qarp** | qiskit O1 | qiskit O2 | qiskit O3 | pytket |
|---|---|---|---|---|---|---|---|
| trotter_line | 8 | — | 23 | 23 | 23 | 23 | 27 |
| trotter_line | 12 | — | 60 | 62 | 61 | 61 | 113 |
| trotter_line | 14 | — | 84 | 87 | 85 | 85 | 103 |
| trotter_grid | 8 | — | 12 | 12 | 12 | 12 | 15 |
| trotter_grid | 12 | — | 34 | 36 | 35 | 34 | 48 |
| trotter_grid | 14 | — | 44 | 48 | 47 | 42 | 80 |
| hea_line | 8 | — | 21 | 23 | 23 | 23 | 6 |
| hea_line | 12 | — | 33 | 35 | 35 | 35 | 22 |
| hea_line | 14 | — | 39 | 41 | 41 | 41 | 23 |
| hea_grid | 8 | — | 0 | 0 | 0 | 0 | 0 |
| hea_grid | 12 | — | 0 | 0 | 0 | 0 | 0 |
| hea_grid | 14 | — | 0 | 0 | 0 | 0 | 0 |
| qv_line | 8 | — | 8 | 8 | 8 | 8 | 9 |
| qv_line | 12 | — | 23 | 24 | 23 | 23 | 36 |
| qv_line | 14 | — | 33 | 33 | 32 | 32 | 49 |
| qv_grid | 8 | — | 3 | 3 | 3 | 3 | 3 |
| qv_grid | 12 | — | 6 | 9 | 7 | 6 | 12 |
| qv_grid | 14 | — | 12 | 12 | 12 | 13 | 18 |
| mqt_line | 8 | — | 31 | 34 | 34 | 34 | 27 |
| mqt_line | 12 | — | 87 | 90 | 89 | 90 | 112 |
| mqt_line | 14 | — | 124 | 127 | 128 | 127 | 103 |
| mqt_grid | 8 | — | 15 | 15 | 14 | 14 | 12 |
| mqt_grid | 12 | — | 37 | 45 | 40 | 41 | 44 |
| mqt_grid | 14 | — | 61 | 68 | 63 | 66 | 77 |

### Compile wall time

Secondary metric, included for completeness. The `uncompiled` column does no work and is a floor, not a competitor.

| workload | n | uncompiled | **qarp** | qiskit O1 | qiskit O2 | qiskit O3 | pytket |
|---|---|---|---|---|---|---|---|
| trotter_line | 8 | 0.0 ms† | 6.5 ms | 2.2 ms | 4.7 ms | 5.2 ms | 267.2 ms |
| trotter_line | 12 | 0.0 ms | 19.3 ms | 3.0 ms | 9.8 ms | 11.0 ms | 1.21 s |
| trotter_line | 14 | 0.0 ms | 28.8 ms | 3.7 ms | 13.4 ms | 14.8 ms | 908.7 ms |
| trotter_grid | 8 | 0.0 ms† | 5.0 ms | 2.3 ms | 4.7 ms | 5.4 ms | 265.3 ms |
| trotter_grid | 12 | 0.0 ms | 15.5 ms | 3.4 ms | 10.9 ms | 12.9 ms | 1.26 s |
| trotter_grid | 14 | 0.0 ms† | 23.0 ms | 4.2 ms | 15.2 ms | 17.0 ms | 1.06 s |
| hea_line | 8 | 0.0 ms† | 4.3 ms | 2.0 ms | 3.7 ms | 4.3 ms | 236.6 ms |
| hea_line | 12 | 0.0 ms† | 8.2 ms | 2.8 ms | 4.9 ms | 5.9 ms | 356.8 ms |
| hea_line | 14 | 0.0 ms† | 10.0 ms | 2.4 ms | 5.2 ms | 6.2 ms | 420.1 ms |
| hea_grid | 8 | 0.0 ms† | 0.0 ms | 1.9 ms | 2.8 ms | 3.1 ms | 234.4 ms |
| hea_grid | 12 | 0.0 ms† | 0.1 ms | 2.1 ms | 3.4 ms | 3.9 ms | 353.1 ms |
| hea_grid | 14 | 0.0 ms | 0.1 ms | 2.2 ms | 3.6 ms | 4.0 ms | 412.4 ms |
| qv_line | 8 | 0.0 ms | 3.4 ms | 2.0 ms | 3.7 ms† | 4.0 ms | 152.1 ms |
| qv_line | 12 | 0.0 ms | 8.8 ms | 2.5 ms | 5.6 ms | 6.7 ms | 668.2 ms |
| qv_line | 14 | 0.0 ms† | 12.7 ms | 2.9 ms | 7.7 ms | 9.1 ms | 396.8 ms |
| qv_grid | 8 | 0.0 ms† | 2.6 ms | 2.1 ms | 3.6 ms | 3.8 ms | 154.4 ms |
| qv_grid | 12 | 0.0 ms | 6.1 ms | 2.6 ms | 5.4 ms | 6.0 ms | 258.9 ms |
| qv_grid | 14 | 0.0 ms† | 8.9 ms | 2.9 ms | 6.6 ms | 7.1 ms | 477.2 ms |
| mqt_line | 8 | 0.0 ms† | 10.6 ms | 2.4 ms | 5.2 ms | 6.2 ms | 286.5 ms |
| mqt_line | 12 | 0.0 ms | 32.0 ms | 3.8 ms | 14.6 ms | 16.9 ms | 1.25 s |
| mqt_line | 14 | 0.0 ms | 46.1 ms | 5.1 ms | 21.4 ms | 25.4 ms | 961.7 ms |
| mqt_grid | 8 | 0.0 ms | 7.2 ms | 2.4 ms | 5.0 ms | 5.9 ms | 287.6 ms |
| mqt_grid | 12 | 0.0 ms | 21.9 ms | 3.8 ms | 12.4 ms | 14.5 ms | 1.43 s |
| mqt_grid | 14 | 0.0 ms | 35.5 ms | 4.6 ms | 18.0 ms | 22.0 ms | 1.12 s |

### Workloads

- **hea_grid** — hardware-efficient ansatz, circular entangler, 3 layers; routed onto a grid coupling map
- **hea_line** — hardware-efficient ansatz, circular entangler, 3 layers; routed onto a line coupling map
- **mqt_grid** — MQT Bench `qft`, flattened to the common basis; routed onto a grid coupling map
- **mqt_line** — MQT Bench `qft`, flattened to the common basis; routed onto a line coupling map
- **qv_grid** — quantum-volume-like random disjoint pairings, 4 layers; routed onto a grid coupling map
- **qv_line** — quantum-volume-like random disjoint pairings, 4 layers; routed onto a line coupling map
- **trotter_grid** — all-pairs ZZ + X layer (every qubit pair interacts, so routing is forced); routed onto a grid coupling map
- **trotter_line** — all-pairs ZZ + X layer (every qubit pair interacts, so routing is forced); routed onto a line coupling map
