# OpenQARP — agent guide

Ground rules for AI assistants (and a two-minute refresher for humans).
The authoritative conventions document is
`docs/contracts/qarp_conventions.md` — **when code and that
doc disagree, the doc wins**. Read §13 (blocks), §14 (engines/devices),
§15 (repo), §17 (symbols), §18 (test oracles), §19 (resources) before
touching those areas.

## Contribution workflow (plan-first)

- Standard/structural work starts as a plan in `docs/contributions/` (its
  README has the tier table and the loop; `_template.md` is the scaffold).
- **One PR per contribution, two gates inside it.**  Open a *Draft* PR whose
  first commit is the plan; the reviewer green-lights the design there, and
  only then does implementation start — on the same branch, in the same PR.
  Mark the PR ready when the implementation is done; that is the second gate.
- The PR declares deviations from the *green-lit* plan — silent drift is the
  violation.  Fold declared drift back into the plan file before merge.
- Trivial fixes (typo, doc fix, bugfix + regression test) need no plan.
- Plan tooling is two plain-markdown files, readable by any agent —
  `plan/SKILL.md` (scaffold a plan) and `plan-review/SKILL.md`
  (conformance-review a PR against its plan).  Read them directly; they carry
  no tool-specific syntax.  They sit under `.claude/skills/` only because that
  is where Claude Code discovers them, which is also what lets it expose them
  as `/plan` and `/plan-review`.

## Commands

- Editable install: `pip install -e ".[full-dev]"` (`[dev]` is ruff, mypy
  and pre-commit only — no pytest, no optional backends; `[full-dev]` is not
  every extra — `docs`, `notebooks`, `integrations`, `bench` and
  `cudaq-runtime` are excluded).  C++ edits then need
  the install re-run.  Run it **inside the repo-root venv** (`python -m venv
  .venv && source .venv/bin/activate`) — a system Python is PEP 668
  externally-managed and pip refuses to install into it.  One `.venv` per
  checkout, worktrees included: a shared venv pins a single checkout, so the
  others silently test the wrong branch.
- Auto-rebuild-on-import (C++ work only) needs **both** lines, never one:
  `pip install scikit-build-core cmake ninja` then `pip install -e
  ".[full-dev]" --no-build-isolation -C editable.rebuild=true`.  The
  import-time `cmake --build` runs after pip exits, so the toolchain must
  persist in the venv; `--no-build-isolation` also skips
  `build-system.requires`, so the backend must be installed by hand first.
  An editable install that opted into `rebuild` with a pip-deleted toolchain
  raises on *every* `import qarp`.
- Cold builds on a < 8 GB box need `export CMAKE_BUILD_PARALLEL_LEVEL=3`, or
  `cc1plus` is OOM-killed compiling SymEngine.
- Python tests: `pytest` (defaults exclude `slow`/`bench` markers; full run
  ≈ 8 min).  C++ tests: `python scripts/run_cpp_tests.py` (CI: the `ctest` job).
- Notebooks: `pytest --nbmake examples/` needs `pip install -e ".[notebooks]"`
  (CI: the `notebooks` job, nightly).  Under a `-C editable.rebuild=true` install,
  export `SKBUILD_EDITABLE_VERBOSE=0` first: the import-time rebuild streams to
  a Jupyter kernel's stdout, which has no `fileno()`, so **every** notebook
  dies at `import qarp` with `UnsupportedOperation`.
- Lint + format, blocking in CI and pre-commit (config in `[tool.ruff]`):
  `ruff check --fix . && ruff format .`
- Types: `mypy qarp/ tests/ --disable-error-code=import-untyped
  --disable-error-code=method-assign`
- Git worktrees: a shared venv resolving `qarpx` to another checkout's build
  now fails fast at `import qarp` (ABI + source-dir stamps, `qarp/_abi.py`).
  The fix is a venv + editable install inside the worktree;
  `QARP_SKIP_ABI_CHECK=1` bypasses for deliberate cross-checkout runs.

## Landmines (each has caused a real bug)

- **Endianness**: everything in qarp is LSB (qubit 0 = least significant
  bit) — statevectors, sampler keys, ONVs *and* operator matrices
  (`op.sparse_matrix()`).  MSB exists only at external boundaries:
  openfermion / cirq / pennylane matrices (`qarp.operators.compat.
  get_sparse_operator` is openfermion's MSB layout, interop only) and
  quimb's kron-ordered `from_dense`.  Contracting across such a boundary
  needs a bit reversal (`qarp.endianness`); inside qarp it never does.
- **Angles are radians**, convention `exp(-iθP/2)` (§1–§12).  Legacy
  half-turn inputs are a recurring source of silent factor-π bugs.
- **Modulo-global-phase is not exact**: a block's global phase becomes a
  *relative* phase under `ControlledBlock` (QPE read shifted eigenphases from
  an uncalibrated synthesis).  Controllable blocks must be phase-exact and
  their unitary oracles compare exact equality (§13, §18).
- **`.symbols` is a canonically sorted tuple** (sorted by `str`) on every
  built block.  Never hand-zip parameter vectors against any other list —
  use `block.parameter_map(values)` / `optimal_parameters` (§17).
- **Block inheritance**: new internal blocks extend `SimpleBlock` (leaf) or
  `CompositeBlockBase` (tree).  The `Block` alias is gone — annotate "any
  block" with `AnyBlock` (= `qx.Block`); it is a type, not a base class.
- **SDK imports are integration-only** — qiskit/pytket/pennylane/qulacs never
  appear in core code paths; the emit/absorb adapters import them lazily
  (§15).  They live only in the test-only `[integrations]` extra (nightly
  pipeline), reached through `pytest.importorskip` in `tests/` — never a
  runtime or `[full-dev]` dependency.  The compiled backend imports as
  `import qarpx as qx`.
- **`zip(..., strict=...)` is mandatory** (ruff B905): prove equal length →
  `strict=True`; intentional truncation → `strict=False`.
- **Never build qarpx-backed objects at module scope in tests** — above all in
  `@pytest.mark.parametrize` lists, whose values are retained by both the
  module global and pytest's `CallSpec2` for the whole session.  They are then
  still alive at interpreter shutdown and nanobind prints `leaked N instances`.
  Parametrize over plain specs (strings/tuples) and construct inside the test.
  The report is accurate, not spurious; and it only surfaces when a later test
  perturbs teardown order, so it appears "intermittent" and unrelated to the
  file that actually causes it.  `pytest --leakdiag` names the holder.
  Deliberate caches are exempt — `@lru_cache` on an expensive SCF keeps
  operators alive on purpose; the rule targets *accidental* module scope.

## Test rules

- Every numerical feature tests against an **independent oracle** (§18) —
  an analytic value, a published number, an openfermion/scipy reference —
  never only against the implementation's own output.  Round-trip tests
  are additional, never the oracle.  Ask: would this test fail if the
  feature were wrong?
- Tests live in `tests/`, mirroring the package; shared fixtures in
  `tests/conftest.py` are seeded.  Mark expensive tests `slow`/`bench`
  (`--strict-markers` is on: an unregistered mark is a collection error).
- **Property tests are not oracles.**  A hypothesis test (`-m property`) proves
  an invariant holds across the input space; it says nothing about whether the
  value is *right* — a units error round-trips perfectly.  §18 still needs an
  independent oracle alongside.  The default `gate` profile is `derandomize`d
  so it cannot flake; explore with `HYPOTHESIS_PROFILE=nightly pytest -m
  property` and pin every find as an `@example`, which is then exercised in
  the gate forever after.  Draw a *seed* and build the object, never the
  object's entries — entry-drawing shrinks toward invalid inputs and reports a
  maximal counterexample.  `st.data()` cannot be combined with `@example`; use
  a composite strategy that draws both together.
- **Generalizing a test never deletes the case that found the bug.**  When a
  property test subsumes an ad-hoc regression test, both stay: the ad-hoc one
  records the incident, the property one searches around it.
- Standard/structural features also ship an example in `examples/` — and
  `examples/` is executed CI surface (nightly `pytest --nbmake`), so breaking
  a notebook is breaking a test.
- **Coverage gates** (the CI `coverage` job): the ratchet floor `COVERAGE_FLOOR`
  only ever rises, and changed lines need ≥ 80 % patch coverage on a PR
  (§15).  Lowering the floor is not a fix.

## Style

- Comments: 1–3 lines, constraints only — state what the code cannot say,
  not what the next line does or why a change is correct.
- Prefer extending existing abstract classes over inventing new hierarchies.
- Branch names: `feature/…`, `bugfix/…`, `docs/…`, `improvement/…`,
  `chore/…` (all lowercase).
