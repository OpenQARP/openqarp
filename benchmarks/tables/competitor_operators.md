# Operator benchmarks — qarpx vs openfermion vs qiskit vs pennylane

_Generated 2026-09-13 13:29 UTC by `python -m benchmarks.run operators` — do not edit by hand._

Host: macOS-26.5.2-arm64-arm-64bit-Mach-O (arm64), Python 3.13.7.
Versions: openqarp 0.1.0, openfermion 1.7.1, qiskit 2.5.2, pennylane 0.45.1, numpy 2.5.3, scipy 1.18.1.

Protocol: fresh subprocess per measurement, medians of 3, pinned threads (`OMP_NUM_THREADS=1` and friends — kernel comparison, not a scaling study); memory is peak RSS of the whole child, imports included. `t_run` is the measured kernel; input generation and operator assembly are excluded identically on every stack. check ✓ = every stack's canonical summary (term count, coefficient norms, key fingerprint) agrees with the openfermion oracle at rtol 1e-06 (§18). qiskit cells include the `.simplify()` a SparsePauliOp user needs for the equivalent result; fermionic-mapping cells are n/a for qiskit (qiskit-nature is not a bench dependency). speedup = openfermion / qarpx. Rows marked † had a spread across repeats wider than 25% of the median — read those as indicative.

| workload | size | qarpx | openfermion | speedup | qiskit | pennylane | qarpx peak MiB | of peak MiB | check |
|---|---|---|---|---|---|---|---|---|---|
| construct_string | 20000 | 14.3 ms | 67.4 ms | **4.7×** | 287.4 ms | 78.6 ms | 115 | 215 | ✓ |
| construct_string | 100000 | 22.1 ms | 250.2 ms | **11.3×** | 1.43 s | 156.2 ms | 219 | 267 | ✓ |
| accumulate | 20000 | 4.3 ms | 40.4 ms | **9.4×** | 108.8 ms | 14.8 ms | 95 | 207 | ✓ |
| accumulate | 100000 | 21.7 ms | 211.0 ms | **9.7×** | 579.4 ms | 77.5 ms | 124 | 232 | ✓ |
| op_product | 100 | 0.5 ms | 9.1 ms | **16.8×** | 1.2 ms | 13.4 ms | 93 | 203 | ✓ |
| op_product | 300 | 4.4 ms | 66.6 ms | **15.1×** | 8.3 ms | 114.6 ms | 130 | 215 | ✓ |
| commutator | 100 | 1.5 ms | 51.5 ms | **34.5×** | 1.9 ms | 36.1 ms | 92 | 204 | ✓ |
| commutator | 300 | 12.8 ms | 450.5 ms | **35.2×** | 13.5 ms | 374.3 ms | 117 | 229 | ✓ |
| hermitian_conjugated | 100000 | 5.4 ms | 15.9 ms | **2.9×** | 0.6 ms† | 24.4 ms | 140 | 260 | ✓ |
| jw_molecular | 10 | 0.5 ms | 10.6 ms | **22.4×** | n/a | 14.8 ms† | 89 | 201 | ✓ |
| jw_molecular | 20 | 2.0 ms | 59.5 ms | **29.0×** | n/a | 81.1 ms | 100 | 205 | ✓ |
| jw_molecular | 40 | 9.2 ms | 453.7 ms | **49.6×** | n/a | 495.2 ms | 163 | 224 | ✓ |
| bk_molecular | 10 | 0.5 ms | 12.3 ms | **26.5×** | n/a | 41.2 ms | 90 | 201 | ✓ |
| bk_molecular | 20 | 2.3 ms | 56.1 ms | **24.5×** | n/a | 191.3 ms | 97 | 204 | ✓ |
| parity_encode | 10 | 1.2 ms | 31.1 ms | **26.6×** | n/a | 15.0 ms | 90 | 201 | ✓ |
| parity_encode | 16 | 4.1 ms | 100.3 ms | **24.7×** | n/a | 47.3 ms | 93 | 204 | ✓ |
| sparse_construct | 12 | 31.7 ms | 522.6 ms | **16.5×** | 111.9 ms | 70.5 ms† | 173 | 521 | ✓ |
| sparse_construct | 14 | 147.3 ms | 1.09 s | **7.4×** | 214.3 ms | 241.2 ms | 492 | 1581 | ✓ |
| symbolic_algebra | 500 | 82.1 ms | 30.96 s | **377.0×** | n/a | n/a | 180 | 270 | ✓ |
| memory_hold | 1000000 | 8.59 s | 11.33 s | **1.3×** | 14.36 s | 9.50 s | 146 | 313 | ✓ |

### Workloads

- **accumulate** — H += term accumulation with ~50% key collisions
- **bk_molecular** — Bravyi-Kitaev of the same random FermionOperator family (n modes)
- **commutator** — A·B − B·A for T-term random operators (n=20, weight ≤ 4)
- **construct_string** — QubitOperator(term_string, coeff) constructor loop
- **hermitian_conjugated** — hermitian_conjugated of a T-term operator (×5)
- **jw_molecular** — Jordan-Wigner of a random 1+2-body FermionOperator (n modes)
- **memory_hold** — build + hold a T-term operator (peak RSS)
- **op_product** — A·B for T-term random operators (n=20, weight ≤ 4)
- **parity_encode** — parity mapping of the same random FermionOperator family (n modes)
- **sparse_construct** — sparse-matrix realization of a 1000-term operator (n qubits)
- **symbolic_algebra** — H·H and H−H† with symbolic coefficients
