# Sampler benchmarks — qarp vs qulacs / Aer / lightning / qsim

_Generated 2026-09-14 10:59 UTC by `python -m benchmarks.run sampler` — do not edit by hand._

Host: macOS-26.5.2-arm64-arm-64bit-Mach-O (arm64), Python 3.13.7.
Versions: openqarp 0.1.0, qiskit 2.5.2, qiskit-aer 0.17.2, pennylane 0.45.1, pennylane-lightning 0.45.0, qulacs 0.6.14, cirq-core 1.7.0, qsimcirq 0.22.1, numpy 2.5.3.

Protocol: fresh subprocess per measurement, medians of 3, pinned threads (`OMP_NUM_THREADS=1` and friends). Default shot count is 8192; circuits are the statevector track's, unchanged, so the two documents compare exact amplitudes against shots on identical work. The timed region is exactly "produce the shots" — device and circuit construction are the build phase, and reducing each SDK's output container to a histogram happens after the clock stops.

**Read the output containers before reading the times.** Each stack is timed through its own sampling call, so the cell includes whatever that call materializes: qulacs returns a raw list of integers, lightning and qsim return a (shots x n) bit array, Aer builds a counts dictionary keyed by bitstring, and qarp's `Sampler` builds a {bitstring-tuple: probability} dictionary. Richer containers cost more to produce and that cost is real for a user calling that API — but it is not raw sampling throughput, and rows where the gap is large are usually measuring container construction, not the kernel.

For `shots_axis` the **n column is the shot count**, at a fixed 12-qubit brickwork circuit; the qubit and shot axes never multiply into one another.

`shots_axis` at 8192 shots and `brickwork_sample` at n = 12 are deliberately the **same measurement** — same seed, same circuit, same shot count, measured in separate subprocesses. The difference between those two rows is this harness's own reproducibility on this host: read it before drawing conclusions from any gap of comparable size elsewhere in the table.

Rows marked † had a spread across the 3 repeats wider than 25% of the median — read those as indicative.

Every stack samples from the circuit its statevector-track adapter prepares, gate fusion included (`benchmarks/statevector/spec.py::FUSION`, inside the timed region as there), so the two documents compare the same tuned engines.

A cell marked ‡ was measured in a child whose single-qubit rotation kernels still ran far above their nominal cost (ratio to an H gate above 6) after 8 relaunches — a process-memory-layout artefact of the qulacs macOS wheel (`why_fast.md`); every child times that ratio before measuring and relaunches itself in a different layout when it trips. Read such a cell as indicative.

Correctness: each stack's Hamming-weight moments (mean, variance, and the all-zeros probability) must match the **exact** Born values within 6 sigma of shot noise — a sampled row is checked against exact physics, never against another sampler's draw (§18). Hamming weight is invariant under qubit relabelling, so MSB stacks need no conversion inside the timed region. The `exact` column computes Born probabilities and does **not** sample: it is the oracle, not a competitor timing.

### Run time

| workload | n / shots | exact (no sampling) | **qarp** | qulacs | Aer | lightning | qsim | check |
|---|---|---|---|---|---|---|---|---|
| brickwork_sample | 4 | 0.5 ms | 0.2 ms† | 0.1 ms | 2.4 ms | 3.6 ms† | 3.9 ms | ✓ |
| brickwork_sample | 8 | 1.4 ms | 0.3 ms | 0.2 ms | 4.1 ms | 4.4 ms | 6.3 ms | ✓ |
| brickwork_sample | 12 | 6.1 ms | 2.7 ms† | 0.6 ms | 11.6 ms | 6.2 ms | 9.9 ms | ✓ |
| brickwork_sample | 16 | 79.2 ms | 7.5 ms | 6.5 ms | 22.6 ms | 19.0 ms | 14.3 ms | ✓ |
| trotter_sample | 4 | 0.6 ms† | 0.2 ms† | 0.2 ms | 2.4 ms† | 3.2 ms† | 4.1 ms | ✓ |
| trotter_sample | 8 | 1.1 ms | 0.4 ms | 0.3 ms | 3.5 ms | 5.7 ms | 6.5 ms | ✓ |
| trotter_sample | 12 | 4.5 ms | 0.9 ms | 1.0 ms | 7.5 ms† | 6.9 ms | 9.3 ms | ✓ |
| trotter_sample | 16 | 52.9 ms | 6.0 ms | 12.3 ms | 16.9 ms | 16.9 ms | 16.4 ms | ✓ |
| qpe_sample | 6 | 1.0 ms | 0.3 ms | 0.2 ms | 3.0 ms | 4.7 ms | 5.5 ms | ✓ |
| qpe_sample | 10 | 2.6 ms | 0.8 ms | 0.6 ms† | 9.5 ms | 7.7 ms | 9.4 ms | ✓ |
| qpe_sample | 14 | 27.0 ms | 5.8 ms | 6.8 ms | 18.1 ms | 15.4 ms | 15.3 ms | ✓ |
| shots_axis | 1024 | 6.0 ms | 0.6 ms | 0.5 ms | 4.1 ms | 6.2 ms | 3.2 ms | ✓ |
| shots_axis | 8192 | 6.0 ms | 2.9 ms | 0.6 ms | 11.5 ms | 6.6 ms† | 9.9 ms | ✓ |
| shots_axis | 65536 | 5.5 ms | 5.7 ms | 2.2 ms | 64.1 ms | 8.3 ms | 60.3 ms | ✓ |

### Peak memory (MiB, whole process — imports included)

| workload | n / shots | exact (no sampling) | **qarp** | qulacs | Aer | lightning | qsim |
|---|---|---|---|---|---|---|---|
| brickwork_sample | 4 | 30 | 146 | 33 | 78 | 357 | 192 |
| brickwork_sample | 8 | 30 | 144 | 33 | 78 | 358 | 193 |
| brickwork_sample | 12 | 30 | 147 | 34 | 79 | 356 | 193 |
| brickwork_sample | 16 | 42 | 150 | 34 | 81 | 358 | 195 |
| trotter_sample | 4 | 30 | 146 | 34 | 78 | 357 | 192 |
| trotter_sample | 8 | 30 | 146 | 33 | 78 | 357 | 193 |
| trotter_sample | 12 | 30 | 147 | 34 | 78 | 356 | 193 |
| trotter_sample | 16 | 39 | 147 | 34 | 80 | 360 | 195 |
| qpe_sample | 6 | 30 | 146 | 33 | 78 | 357 | 192 |
| qpe_sample | 10 | 30 | 147 | 33 | 79 | 357 | 193 |
| qpe_sample | 14 | 32 | 149 | 33 | 80 | 357 | 194 |
| shots_axis | 1024 | 30 | 147 | 34 | 78 | 357 | 192 |
| shots_axis | 8192 | 30 | 147 | 34 | 79 | 357 | 193 |
| shots_axis | 65536 | 30 | 148 | 35 | 85 | 367 | 207 |

### Workloads

- **brickwork_sample** — sample 8192 shots from the brickwork circuit
- **qpe_sample** — sample 8192 shots from the QPE-shaped circuit
- **shots_axis** — shot-count sweep at a fixed 12-qubit brickwork circuit; the ladder value is the shot count
- **trotter_sample** — sample 8192 shots after the TFIM Trotter steps
