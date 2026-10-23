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
- **`PrimitiveResult` is a public annotation without an export.**
  `qarp/_types.py` defines it and `PrimitiveAlgorithm.run`, `Engine.run` and
  `Engine.batch_run` return it, but `qarp.__all__` exports only
  `SamplingDictionary`.  Export it next to `SamplingDictionary`, or annotate
  with the public types.
- **`lambda_factor()` is the last compatibility shim.**
  `BlockEncodingBlock.lambda_factor` and the projector blocks' copy return
  `lambda_norm`; `QSVTBlock` is the one caller and `QubitizationBlock` sets
  an attribute of the same name.  Drop it in the release or keep it for
  good.
- **Plan text on the plan-only branch.**  `pauli_engine_plan.md` names an
  ABI bump 8 → 9; the counter is 16.  Fix when the branch rebases.

## Breaking changes waiting for a minor release

Each changes a default, a width, a return value or a class shape, so it rides
a `0.x` minor with whatever else breaks then; none is close enough for 0.2.0.

- **`Engine.batch_run` without an implicit rebuild.**  `rebuild=True` is the
  default, so every sweep transpiles its primitives again.  Default
  `rebuild=False`, reusing the prior `build()`, and a `RuntimeError` naming
  `engine.build` when nothing is built.  Tutorial 03 builds first.
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
- **Shadow confidence from the actual batch count.**  `ShadowEstimate.delta`
  stores the requested value even when `n_batches` overrides the count it
  implies, and at `n_batches=1` the bound `2e^{-K/2}` exceeds 1.  Report
  the achieved confidence `min(1, 2e^{-K/2})` from the actual `K`.  A public
  field changes meaning.
- **`HEABlock` as a tree.**  `qarp/blocks/_primitives/hea_block.py` builds
  it as a `SimpleBlock` leaf; its layers are a composition that a
  `CompositeBlockBase` tree expresses.  `isinstance` answers change; ships
  with the symbol-contract regression test.
- **`Consumes.OBSERVABLE` on the engine boundary.**
  `pauli_engine_plan.md` adds the member and makes `StateVector.consumes` a
  property, so a third-party engine that dispatches on `consumes` adapts.

## Additive, no window needed

- **`OqtopusEngine` checks the device basis at construction.**  The
  transport's `get_device` serves the topology JSON only and
  `device_from_oqtopus` assumes `x, sx, rz, cx`, so a device with another
  basis fails the job at the gateway, after queuing.  The SDK's device
  record carries `basis_gates` beside the topology.  Widen `get_device` to
  carry it, have `LocalOqtopusTransport` serve the `sx_rz` set by default,
  and raise `CapabilityError` when the four are not all present.  Wait for
  the first QPU record (`oqtopus_plan.md`, Phase 3b) to confirm the field's
  spelling.  Start at `qarp/engines/_oqtopus/_transport.py` and
  `qarp/devices/_oqtopus.py`.
- **Refuse readers of the per-shot register on a counts-only engine.**
  `CuttingPrimitive` and the QPD path read `cbit_history`, which
  `OqtopusEngine` leaves empty, and raise `RuntimeError` at `run`, after the
  job has run.  Add `requires_cbit_history` on the two primitives and
  `provides_cbit_history` on engines, checked in `Engine.build` next to the
  `requires_noiseless` check in `qarp/engines/_engine.py`; the engines
  package cannot name the primitives (`test_layering.py`).
- **`QARP_BLAS_THREADS` default on bare metal.**  The `limit` default
  and the pool's standing as opt-in rest on WSL2 timings, where OpenBLAS
  at 12 threads is slow on its own.  Repeat the Threads measurements of
  `openblas_thread_sharing_plan.md` on bare-metal Linux and revisit the
  default from them.
- **`check_untyped_defs` for `tests/`.**  `compiled_backend_stubs_plan.md`
  turns it on for `qarp/` only, so unannotated test bodies, which is nearly
  every test, are still unchecked.  It reports 762 errors in 134 test files
  on macOS, 416 of them `arg-type`.  It touches nearly half the test files,
  so it gets its own plan; land it after the layout move, which would
  otherwise conflict with it across them.
- **`qarp` typed for its users.**  `qarp` ships no `py.typed`, so user code
  sees it untyped.  Running mypy over the example notebooks and the docs
  snippets finds five gaps to close before it does: a list of Python blocks
  of different kinds infers as `list[_BlockMixin]`, which a
  `Sequence[QarpBlock]` parameter rejects, so the six classes need one
  shared typing base; `symbols` is `Optional`, so `len(block.symbols)` fails
  in 22 notebook cells and 3 docs snippets, and making it always a tuple
  changes `symbols is None` checks; `PreparesKnownState`'s members are
  attached by `setattr`, so mypy sees none of them on the state blocks;
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
  `_PLAN_CACHE_SIZE` (`qarp/engines/_qarp_engine.py`) keeps 64 programs of
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
  `Program::substituted` (`cpp/libqarpx/src/simulator/program.cpp`) copies
  every permutation table and dense matrix by value, and `QarpEngine`
  substitutes a parametric program once per `run(params)` and per
  `batch_run` set.  Structural kernels never hold a parameter, so a built
  circuit with one rotation beside a large permutation copies the
  permutation each time: about 0.013 s per 2^22-entry table per set.  Hold
  the tables behind a shared pointer, so substitution copies only the gate
  kernels.
