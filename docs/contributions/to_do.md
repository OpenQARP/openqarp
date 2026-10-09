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
- **Plan text on the plan-only branches.**  `pauli_engine_plan.md` and
  `oqtopus_plan.md` name an ABI bump 8 → 9, and the OQTOPUS plan names
  `prepare_structured_qpe`; the counter is 14 after structured execution and
  the hook is gone.  Fix when each branch rebases.
- **No shim for `prepare_structured`.**  The lazy-circuits plan asks whether
  the hook gets a one-release shim; the release order answers it: the hook
  never reaches a release, so it is a plain removal.  Record that in the
  plan's decisions at the sit-down.

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
- **One `Runnable` change for the engine boundary.**  The lazy-circuits plan
  groups `Runnable` by boundary and moves `Target` below the engine layer;
  `pauli_engine_plan.md` adds `Consumes.OBSERVABLE` and makes
  `StateVector.consumes` a property.  Settle both at one sit-down so a
  third-party engine adapts once.

## Additive, no window needed

- **`QARP_BLAS_THREADS` default on bare metal.**  The `limit` default
  and the pool's standing as opt-in rest on WSL2 timings, where OpenBLAS
  at 12 threads is slow on its own.  Repeat the Threads measurements of
  `openblas_thread_sharing_plan.md` on bare-metal Linux and revisit the
  default from them.
- **`Sampler(postselect=PostSelection(...))`.**  Applying post-selection
  inside `Sampler.run()` waited on a result type that could carry
  `success_rate` without breaking the dict return; `SamplingDistribution`
  carries `n_shots` the same way.  Post-selected qubits must be measured and
  leave the key.
