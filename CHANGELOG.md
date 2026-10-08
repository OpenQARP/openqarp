# Changelog

All notable changes to OpenQARP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Structured execution.  `Block.statevector` and `QarpEngine`'s sampling
  paths read the block tree and run each piece as the cheapest exact kernel:
  a basis-state permutation as one gather, a small block as one dense
  matrix, a ladder of the same controlled block as one controlled-powers
  kernel, and the rest as fused gates.  A circuit with no such structure
  runs exactly as before.  `structured=False` on either turns it off and
  `QARP_STRUCTURED` sets the process default.  A block may declare its
  permutation with `classical_action` (`ModularMultiplicationBlock` does)
  or its composition with `structure()` and `qarp.blocks.Repeat`
  (`QPEBlock` and `DOSQPEBlock` do); `Block.kernels()` lists what
  `statevector` would run.  See §14 of the conventions and the *Structured
  execution* section of the configuration page.
- `optimization_level` on `Block.statevector` and `QarpEngine`: the
  transpiler level applied to the gate slices after planning, or to the
  whole stream on the gate path.
- `qarp.SamplingDistribution`, the read-only result of `Sampler` and
  `PostSelection.apply`: aligned `outcomes` and `probabilities` arrays,
  `n_bits_measured`, `n_shots`, `probability_of`, `to_dict`, `from_dict`,
  `counts`, `standard_errors`, `marginal`, `to_dense`,
  `parity_expectation`, `top`, `most_likely`, `total_variation`,
  `hellinger_fidelity` and `sample`.
- `QARP_BLAS_THREADS`: `limit`, the default, lowers numpy's and scipy's
  OpenBLAS thread count to qarp's at the first simulation; `pool` also runs
  their parallel jobs on a qarpx-owned pool whose workers sleep as soon as a
  call ends; `native` leaves OpenBLAS untouched (see *Changed*).
- `examples/engines/mwe_structured_execution.ipynb`.

### Changed

- `Sampler` and `PostSelection.apply` return a `SamplingDistribution`
  instead of a `dict`, and `SamplingDictionary` is a read-only `Mapping`
  alias.  `isinstance(result, dict)` and mutating a result no longer work;
  `to_dict()` returns a plain dict.  Keys and probabilities are unchanged.
- `QPE` and `DOSQPE` on a `QarpEngine` without a device no longer build
  their controlled-U ladder; the run goes through one controlled-powers
  kernel whatever the ancilla count.
- OpenBLAS's thread count in numpy and scipy is lowered to qarp's
  (`QARP_NUM_THREADS`) at the first simulation, so a BLAS call no longer
  leaves a worker spinning on every logical CPU against the next simulator
  call (a 16-qubit `statevector` right after a norm: 108 ms to under 8 ms).
  A count set in `OPENBLAS_NUM_THREADS` or in code is kept, and
  `QARP_BLAS_THREADS=native` restores the previous behaviour.

### Removed

- `Engine.prepare_structured_qpe`, `StructuredQPEPlan` and the
  `QarpSimulator.simulate_qpe_structured` and `simulate_dosqpe_structured`
  bindings.  Structured execution covers their case for any block.

### Fixed

- `QPEBlock` and `DOSQPEBlock` place a copy of the eigenstate block, so one
  state block can serve two QPE blocks; sharing it corrupted the first.
- The default thread count under OpenMP binding (`OMP_PROC_BIND`,
  `OMP_PLACES`) counts the CPUs of all the places; the bound main thread's
  own place gave a count of 1.

## [0.1.1] - 2026-09-25

### Fixed

- `SynthesizedUnitaryBlock` and the other Quantum Shannon Decomposition
  paths are accurate to ~1e-13 on structured unitaries (DFT, Hadamard,
  phase-decorated targets) up to 7 qubits, where a 6-qubit DFT was far off:
  the cosine–sine decomposition divided by small sines, and the eigenvalue
  and diagonal tolerances merged distinct values.
- Synthesized unitaries are phase-exact.  The global phase is kept down to
  1e-15, so a synthesized block under `ControlledBlock` (for example inside
  QPE) matches the controlled target exactly instead of carrying a spurious
  relative phase.
- The default thread count no longer starts an OpenMP worker on every
  hardware thread, which starved the main thread.  With neither
  `QARP_NUM_THREADS` nor `OMP_NUM_THREADS` set, every layer uses the physical
  cores in the process's CPU affinity, capped one below its logical CPUs, and
  a container CPU limit (any cgroup v1 or v2 quota) lowers it further.  Under
  WSL and some virtual machines the reported topology undercounts physical
  cores; set `QARP_NUM_THREADS` there.

### Changed

- `CITATION.cff` names the arXiv paper as the preferred citation, with the
  Zenodo concept DOI for the software.

## [0.1.0] - 2026-09-14

First public release.  Distributed on PyPI as `openqarp`; the import name
is `qarp`.  Pre-release history from the internal repository is not carried
over; this file records changes from 0.1.0 onward.

[Unreleased]: https://github.com/OpenQARP/openqarp/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/OpenQARP/openqarp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/OpenQARP/openqarp/releases/tag/v0.1.0
