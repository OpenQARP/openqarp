# Report instruction-count and memory regressions on every PR

**Status:** Planned
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-28
**Tier:** Standard
**Branch:** chore/perf-profiling
**Green-lit:**
**Scope:**
- New: `docs/contributions/perf_profiling_plan.md`, `tests/test_perf/__init__.py`,
  `tests/test_perf/test_block_build.py`, `tests/test_perf/test_operator_build.py`,
  `tests/test_perf/test_simulation.py`, `tests/test_perf/test_routing.py`,
  `tests/test_perf/test_algorithms.py`
- Modified: `.github/workflows/ci.yml`, `pyproject.toml`,
  `docs/contributions/README.md`, `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md`

---

## Why

Most features leave room for speed and memory gains, and nothing in CI shows
whether a change made the hot paths faster, slower, larger or smaller.

The `@pytest.mark.bench` tests pin wall-clock ceilings for past speedups
(block lowering, fusion, the expectation kernel, SABRE, per-shot cost, exact
readout).  They are excluded from the default run and no CI job selects them,
so none of them can catch a regression today.  Their thresholds were set on
one laptop and do not transfer to a shared runner.

`benchmarks/` is the competitor harness; its nightly `bench-smoke` job is a
correctness gate by design, never a performance gate.

Wall-clock time on shared GitHub runners varies by 10–30 % between identical
runs, so a stored baseline of seconds cannot tell a regression from noise.

## Design

**Measurement: CodSpeed.**  The `pytest-codspeed` plugin runs benchmarked
tests in two modes, each reported against the default branch (`develop`):

- `simulation`: CPU instructions under valgrind, whole process, so `qarpx`
  is counted with the Python around it.  Reproducible on shared runners.
- `memory`: allocator hooks via eBPF (glibc, libstdc++).  Reports peak
  allocated memory, peak RSS, allocation count and total allocated bytes.
  Peak RSS needs Linux ≥ 6.8, hence `ubuntu-24.04`.

Results live on codspeed.io.  CodSpeed posts one PR comment with the diff per
case and a check run.  Pushes to `develop` become the baseline.  Fork PRs
upload tokenless; same-repo runs authenticate by OIDC (`id-token: write`),
so no secret is stored.

**Flag, never fail.**  The `perf` job succeeds whatever the numbers say.  The
CodSpeed check reports the regression; it is not a required check, and the
Pages deploy does not `need` the `perf` job.  A case is flagged at a 5 %
change in either direction, set in the CodSpeed dashboard.  The CodSpeed PR
comment and check are the whole report: no step of ours writes to the run
summary, since the numbers live on codspeed.io.

**Cases are ordinary tests.**  They live in `tests/test_perf/`, one file per
area, written with pytest-codspeed's `benchmark` fixture so input
construction stays outside the measured call.  Without `--codspeed` the
fixture calls the target once, so the cases also run in the normal `pytest`
gate and cannot rot.  ruff, mypy and the coverage gates apply as to any test.
The folder groups by what it measures rather than mirroring the package: the
CI job collects only this folder, because collecting all of `tests/` under
valgrind would take minutes.

**Determinism.**  The job pins `QARP_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1` and `PYTHONHASHSEED=0`.  Every case seeds its RNG.
Thread scheduling would otherwise change instruction and allocation counts
from run to run.

**Size.**  Valgrind runs 20–50× slower than native.  Cases are sized at
4–20 qubits so the `simulation` leg finishes in under 15 minutes.  The memory
instrument works best below ~2 M allocations per case.

**Versions.**  `pytest-codspeed==5.0.3` is pinned exactly in the `test`
extra, for the same reason as `pytest-cov`: a measurement change shifts every
number, so a bump is deliberate and re-baselines once.  The action is
`CodSpeedHQ/action@v5`, pinned by major tag like the other actions.

**Conventions.**  Fits §15 (top-level trees import in one direction: the
cases import `qarp` only) and §18 (validation rows below have analytic
oracles).  No convention edit.  The module-scope rule from `CLAUDE.md`
applies: cases parametrize over plain specs and build `qarpx` objects inside
the test.

**Not changed.**  The `bench` tests stay as they are, as do `benchmarks/` and
`tests/test_import_time.py`.  Import time is left out of CodSpeed: re-importing
a nanobind extension in-process is not possible, and `test_import_time.py`
already pins which modules an import loads.  No `examples/` notebook: the
change adds no library surface.

## API sketch

```python
# tests/test_perf/test_simulation.py
import numpy as np
import pytest

import qarpx as qx

SEED = 20261028


def _brickwork(n, depth, seed):  # builds and flattens inside the test
    ...


@pytest.mark.parametrize("n", [16], ids=["16q"])
def test_brickwork_statevector(benchmark, n):
    cmds = _brickwork(n, 20, SEED)
    sim = qx.QarpSimulator()
    psi = benchmark(sim.statevector, cmds, n)
    assert np.isclose(np.vdot(psi, psi).real, 1.0)
```

```yaml
# .github/workflows/ci.yml
  perf:
    needs: [wheel]
    if: github.event_name == 'pull_request' || github.ref == 'refs/heads/develop'
    runs-on: ubuntu-24.04
    timeout-minutes: 30
    permissions:
      contents: read
      id-token: write
    strategy:
      fail-fast: false
      matrix:
        mode: [simulation, memory]
    env:
      QARP_NUM_THREADS: "1"
      OMP_NUM_THREADS: "1"
      OPENBLAS_NUM_THREADS: "1"
      PYTHONHASHSEED: "0"
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - uses: actions/download-artifact@v7
        with:
          name: wheel
          path: dist
      - run: python -m pip install "$(ls dist/*.whl)[full-dev]"
      - uses: CodSpeedHQ/action@v5
        with:
          mode: ${{ matrix.mode }}
          run: pytest tests/test_perf --codspeed
```

**Cases:**

| File | Case | Size |
|---|---|---|
| `test_block_build.py` | `SimpleBlock` of random rotations and CX: build + flatten | 16 q, 2000 ops |
| | `HEABlock` build + flatten | 16 q, 4 layers |
| | `ControlledBlock` lowering of a rotation block | 1 control, 500 ops |
| | UCCSD ansatz build | H4 (8 q) |
| `test_operator_build.py` | `FermionOperator` from frozen LiH integrals | 12 q |
| | Jordan–Wigner mapping of that operator | 12 q |
| | Grouping into commuting sets | 12 q |
| `test_simulation.py` | Brickwork statevector, default fusion | 16 q, depth 20 |
| | Hadamard layer statevector (memory oracle row) | 20 q |
| | ⟨ψ\|H\|ψ⟩ on a chemistry-shaped operator | 16 q, ~2000 terms |
| | Shots on a mid-circuit-measurement circuit | 12 q, 1000 shots |
| | Exact readout distribution | 16 q |
| `test_routing.py` | SABRE routing of random CX on a line | 20 q, 500 CX |
| `test_algorithms.py` | Pauli-averaging post-processing | 12 q, 200 terms, 10k shots |
| | VQE, fixed optimizer iterations | H2 (4 q), 20 iterations |

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Memory mode sees `qarpx`'s native allocations | Analytic: a 20-qubit statevector holds 2²⁰ complex128 = 16 MiB, so the case's peak allocated memory is ≥ 16 MiB.  Expected reading ≈ 32 MiB: a malloc-hook probe on macOS measured exactly 2× the state at 20 and 22 qubits (internal state plus the returned copy); another value is investigated, not failed | `tests/test_perf/test_simulation.py`; read once from the Draft PR's CodSpeed report |
| Instruction counts detect a regression | Analytic: a throwaway commit that calls one case's target twice shows ≈ +100 % on that case and ≈ 0 % on the others | Draft PR; commit reverted before ready |
| Instruction counts reproduce | Re-running the job on the same commit reports no change beyond CodSpeed's stated noise | Draft PR, job re-run |
| Cases run in the normal gate | Each case asserts a cheap invariant on its result (norm, term count, qubit count).  These are smoke asserts, not oracles: each operation's correctness is owned by its existing tests | `tests/test_perf/*` in `pytest` |

## Phases

### Phase 0 — Repository setup (owner, outside the repo)

- [ ] Install the CodSpeed GitHub app on the OpenQARP organisation and import
  `openqarp` on codspeed.io
- [ ] Set the regression threshold in the CodSpeed dashboard to 5 %
- [ ] Leave the CodSpeed check out of the required checks

### Phase 1 — Cases

- [ ] Pin `pytest-codspeed==5.0.3` in the `test` extra
- [ ] `tests/test_perf/` with the cases above
- [ ] The cases pass in the default `pytest` run, and
  `pytest tests/test_perf --codspeed` runs locally

### Phase 2 — CI job

- [ ] `perf` job in `ci.yml` as sketched
- [ ] Validation rows on the Draft PR: memory oracle, deliberate regression,
  re-run

### Phase 3 — Docs

- [ ] One line in the Commands section of `CLAUDE.md` and `AGENTS.md`
  (byte-identical)
- [ ] `CONTRIBUTING.md`: the CodSpeed report is read at review, never a
  blocker

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
