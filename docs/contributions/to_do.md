# To do

Side quests: fixes and small changes found while working on something else
and not central to that work, parked here so they are not lost.  Each entry
says what changes, why, and where to start, so it can be picked up cold.  An
entry leaves when it lands.  Anything that needs a plan gets its own
`<topic>_plan.md` and an index row in [`README.md`](README.md); this file
never stands in for one.

## Before 0.2.0

0.2.0 is tagged after `lazy_circuit_derivation_plan.md`, the compiled-backend
type stubs and the repository layout move land, in that order.  These ride
the same window.

- **`CHANGELOG.md` carries each PR's lines.**  Every PR into `develop`
  before the release adds its entries under `[Unreleased]`; the release PR
  only moves them.
- **Plan text on the plan-only branches.**  `pauli_engine_plan.md` and
  `oqtopus_plan.md` name an ABI bump 8 → 9, and the OQTOPUS plan names
  `prepare_structured_qpe`; the counter is 15 and the hook is gone.  Fix when each branch rebases.

## Breaking changes waiting for a minor release

Each changes a default, a width, a return value or a class shape, so it rides
a `0.x` minor with whatever else breaks then.

- **Optimisation stages inside `compile_for_device`.**  The binding takes
  `commands, n_qubits, device, router` and routes the rebased circuit as is.
  Intended pipeline: rebase to the routable set, optimise, route, lower
  directed SWAPs, rebase to the full set, optimise, with an `opt_level`
  argument defaulting to O1 and O0 reproducing today's output.  Device-path
  circuits change; `QarpEngine` passes its level, `ResourceEstimator`
  mirrors the staging, §14's pipeline sentence gains the two optimise
  stages.
- **`QROMBlock` default construction.**  Unary iteration becomes the
  default, with `k−1` caller-placed work qubits through `target_qubits`;
  `ancilla_free=True` keeps the current construction.  A default `QROMBlock`
  then needs more qubits and fewer gates.
- **`MPSStateBlock` without a bond register.**  Sliding-window construction
  with `n_qubits = n_sites`.  The block's width changes.
- **Coefficient magnitude in single-operator primitives.**  `HadamardTest`,
  `SWAPTest` and `MirrorTest` take one operator block and never read its
  `.coefficient`, so `|c|` is dropped: `HadamardTest` with `coefficient=2.5`
  returns the value for `c = 1`.  Apply `abs(coefficient)` where the
  primitive can express it, reject a non-unit modulus where it cannot; §18
  oracles for both.
- **`HEABlock` as a tree.**  `src/qarp/blocks/_primitives/hea_block.py` builds
  it as a `SimpleBlock` leaf; its layers are a composition that a
  `CompositeBlockBase` tree expresses.  `isinstance` answers change; ships
  with the symbol-contract regression test.
- **`Consumes.OBSERVABLE` on the engine boundary.**
  `pauli_engine_plan.md` adds the member and makes `StateVector.consumes` a
  property, so a third-party engine that dispatches on `consumes` adapts.

- **Shadow error bar from a certified variance.**  `ShadowEstimate.error`
  puts the empirical `V_hat` where HKP Theorem 1 needs the true variance.
  On an observable driven by rare outcomes `V_hat` is usually 0, and so is
  the bar: `X = 100` with probability `1e-4` over 200 settings misses at
  `K = 8` 98 % of the time against a reported `delta` of 0.037.  Bound the
  variance from above instead, by the shadow norm (state-independent) or a
  confidence bound on `V_hat`.  The same change can report the tighter
  failure bound the half-width already earns: Chebyshev gives each batch
  mean `1/34`, so a median of `K` fails at most `P[Bin(K, 1/34) >=
  ceil(K/2)]`, `2.9e-2` at `K = 1` and `4.8e-5` at `K = 8` against HKP's
  1 and 0.037.  Both change the reported values: `error` widens, and
  `delta` stops being HKP's bound, so a public field changes meaning.
  §18 oracles for each.

## Additive, no window needed

- **Installation docs traps, from a clean-clone walk-through.**  "Checking it
  worked" runs `pytest` after a plain `pip install .`, which installs no
  pytest, and sends C++ tests to `bootstrap_qarpx.py --with-tests`, which
  refuses beside a pip install; the route that works is
  `cmake -S src/qarpx -B src/qarpx/build -DQARP_BUILD_TESTS=ON` then
  `scripts/run_cpp_tests.py`, and the script's hint should say so.  On macOS
  every re-run of `pip install -e` needs the `OpenMP_ROOT` define again or
  the backend silently rebuilds serial.  `build_docs.sh` says one intersphinx
  warning remains; the build shows nineteen duplicate-description warnings
  for `SamplingDistribution`, documented twice on `docs/api/qarp.rst`.

- **`QARP_BLAS_THREADS` default on bare metal.**  The `limit` default
  and the pool's standing as opt-in rest on WSL2 timings, where OpenBLAS
  at 12 threads is slow on its own.  Repeat the Threads measurements of
  `openblas_thread_sharing_plan.md` on bare-metal Linux and revisit the
  default from them.
- **`check_untyped_defs` for `tests/`.**  `compiled_backend_stubs_plan.md`
  turns it on for `src/qarp/` only, so unannotated test bodies, which is nearly
  every test, are still unchecked.  It reports 762 errors in 134 test files
  on macOS, 416 of them `arg-type`.  It touches nearly half the test files,
  so it gets its own plan; land it after the layout move, which would
  otherwise conflict with it across them.
- **`qarp` typed for its users.**  `qarp` ships no `py.typed`, so user code
  sees it untyped.  Running mypy over the example notebooks and the docs
  snippets finds four gaps to close before it does: a list of Python blocks
  of different kinds infers as `list[_BlockMixin]`, which a
  `Sequence[QarpBlock]` parameter rejects, so the six classes need one
  shared typing base; `PreparesKnownState`'s members are attached by
  `setattr`, so mypy sees none of them on the state blocks;
  `run()` returns `PrimitiveResult`, `float | complex | SamplingDistribution`,
  so 45 notebook and docs sites must narrow a result before reading it, which
  a return typed by primitive kind would remove; and a `MonteCarlo` walker's
  `state_data` is `np.ndarray | AnyBlock`, so 3 notebook cells must narrow it.
- **`Sampler(postselect=PostSelection(...))`.**  Applying post-selection
  inside `Sampler.run()` waited on a result type that could carry
  `success_rate` without breaking the dict return; `SamplingDistribution`
  carries `n_shots` the same way.  Post-selected qubits must be measured and
  leave the key.
- **`QarpEngine`'s plan cache is bounded by count, not size.**
  `_PLAN_CACHE_SIZE` (`src/qarp/engines/_qarp_engine.py`) keeps 64 programs of
  the built path, whatever their tables weigh: sixteen distinct 22-qubit
  permutation circuits held 517 MB after every primitive was gone, and 64
  such tables at 26 qubits reach 32 GB.  Bound it by the bytes its
  permutation tables and dense matrices hold.
- **`_plan_unbuilt` is not cached.**  Every `Engine.build` collects an
  unbuilt circuit's parts again, flattening its plain parts: `PauliAveraging`
  flattens a shared ket once more per group on every build, and a
  parametric ket is flattened only to be declined.  A cache would need a
  key for an unbuilt composition, which has no stream to digest.
- **A parametric planned program copies its tables on every binding.**
  `Program::substituted` (`src/qarpx/src/simulator/program.cpp`) copies
  every permutation table and dense matrix by value, and `QarpEngine`
  substitutes a parametric program once per `run(params)` and per
  `batch_run` set.  Structural kernels never hold a parameter, so a built
  circuit with one rotation beside a large permutation copies the
  permutation each time: about 0.013 s per 2^22-entry table per set.  Hold
  the tables behind a shared pointer, so substitution copies only the gate
  kernels.
