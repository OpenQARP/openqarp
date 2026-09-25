# Changelog

All notable changes to OpenQARP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
