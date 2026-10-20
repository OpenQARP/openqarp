# Derive primitive circuits lazily and plan declared structure inside `Engine.build`

**Status:** Landed
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-07
**Tier:** Structural
**Branch:** improvement/lazy-circuit-derivation
**Green-lit:** 2026-10-09, plan blob aa0f219c5fb305e2fda24fa0259776cd048dfdc9
**Scope:** `qarp/_types.py`; `qarp/algorithms/_primitives/target.py` (re-export only),
`__init__.py` (docstring), `primitive_algorithm.py`, `sampler.py`, `pauli_averaging.py`,
`basis_rotation_averaging.py`, `hadamard_test.py`, `termwise_hadamard_test.py`,
`grouped_transition_hadamard_test.py`, `swap_test.py`, `termwise_swap_test.py`,
`mirror_test.py`, `interferometric_test.py`, `cutting_primitive.py`, `state_vector.py`,
`shadows/`; `qarp/engines/_runnable.py`, `_engine.py`, `_qarp_engine.py`, `_gradients.py`,
`_cudaq_engine.py`; `qarp/_program.py` (`plan_structure`); `qarp/algorithms/_composite/qpe.py`,
`dos_qpe.py`; `qarp/blocks/_composite_block.py` (children readable before build, if needed);
`docs/contracts/qarp_conventions.md` (§14); `tests/test_engines/test_layering.py`,
`tests/test_engines/test_structured_execution.py`, `tests/test_blocks/test_structure.py`,
`tests/test_algorithms/test_composite/test_qpe.py`, `test_dos_qpe.py`,
`tests/test_algorithms/test_primitives/`; the structured-execution example notebook;
`docs/contributions/README.md`.  Based on `feature/structured-execution`; rebased onto
`develop` once that branch lands.

---

## Why

Three facts about the engine–primitive boundary, found while reviewing structured execution:

- `Target` names the quantity a primitive computes and engines dispatch on it, but it lives in
  `qarp.algorithms._primitives.target`, above the engine layer.  `Consumes` was moved to
  `qarp._types` so engines could read it without importing algorithms; `Target` never was.
  Three engine sites therefore match `prim.target.name == "EXPECTATION_VALUE"` by string
  (`_gradients.py`, `_cudaq_engine.py`, the capability error in `_engine.py`), and the
  structured-execution review added a fourteenth `Runnable` member (`samples_block`) because
  `target is Target.SAMPLING` could not be written in `QarpEngine.prepare_structured`.
- `Runnable` has grown one member per engine need and mixes three kinds of fact: the quantity
  (`target`, `operator`, `supported_targets`), the circuits (`sub_blocks`, `_n_qubits_list`,
  `initial_state`, `accepts_initial_state`, `samples_block`) and the estimator (`consumes`,
  `supports_exact`, `requires_noiseless`, `gradient_kind`, `supports_backprop_gradient`,
  `returns_probability`).  `returns_probability` is read by composites only.
- Every primitive's `build()` builds its ket and derives its circuits as built blocks
  (`PauliAveraging` composes the built ket with a basis change per group, the Hadamard and
  swap tests wrap it with ancilla logic, `Sampler` uses it as is).  The engine sees a circuit
  only after it is built, so a ladder declared by `structure()` is built before it can be
  planned.  Structured execution therefore runs through a side door —
  `Engine.prepare_structured` and `StructuredRun` — with these consequences: `QPE` and
  `DOSQPE` hold `block = None` on that path, `engine.run()` returns `[]` and `batch_run`
  raises after a structured build, the engine-path plan is recomputed on every build
  (233 ms refused plan per build at 17 qubits in the review), and a `PauliAveraging` over a
  declaring block never has its ladder lowered.

## Design

A primitive is a measurement protocol: a state preparation, a set of measurement settings
(the circuits it runs) and an estimator from counts to a number.  The engine is the
apparatus.  `Target` names the quantity and `Consumes` the readout; the settings are the one
part the apparatus cannot inspect before the protocol has built them.  This plan makes them
data the engine reads before anything is built, and moves `Target` to where `Consumes` is.

**Phase 1 — `Target` below the layer line.**  `Target` is defined in `qarp._types` next to
`Consumes`; `qarp.algorithms._primitives.target` re-exports it, so every existing import
keeps working.  Engines compare `prim.target is Target.X`.  The lint gate in
`tests/test_engines/test_layering.py` also fails a `.name ==` comparison against a `Target`
member name.  `samples_block` is retired: `prepare_structured` (until Phase 2 removes it)
serves `target is Target.SAMPLING and consumes is Consumes.COUNTS`, and §14 says so.  The
`Runnable` docstring groups its members by boundary (quantity, circuits, estimator) so the
next fact lands on the right side.  No behaviour change.

**Phase 2 — Lazy circuits.**  `PrimitiveAlgorithm.circuits()` returns the primitive's
circuits as block trees built from its *unbuilt* inputs: `Sampler` returns `[ket]`;
`PauliAveraging` returns one `CompositeBlock([ket, rotation_g])` per group; the Hadamard,
swap, mirror and interferometric tests return their ancilla constructions over the unbuilt
ket and operator blocks; `StateVector` returns `[ket]` (or the ket–operator composite it
contracts).  `build()` sets `sub_blocks = circuits()` and `_n_qubits_list` from the trees
and builds nothing.  `Engine.build()` then, per sub-block, asks `plan_structure` first: it
is generalised to descend an unbuilt composite's constructor list, lowering a declaring part
from its declaration and building a plain part on its own (the rotation layer, the
eigenstate preparation), so a `QPEBlock` under a basis change is planned without its
ladder; when `plan_structure` refuses, the engine builds, flattens, transpiles and plans the
built block exactly as today.  Programs are cached on the engine by the same digest key as
`Block.statevector`.  `Engine.prepare_structured`, `StructuredRun` and the QPE/DOSQPE
special case are removed: `QPE.build()` is `engine.build([Sampler(ket=QPEBlock(...))])` and
`QPE.block` is always the block, unbuilt when it was planned.  Hard break, as for
`prepare_structured_qpe`.

Conventions: §14 *Structured execution* loses the `prepare_structured` paragraph and gains
"a declared structure is planned inside `Engine.build` before the block is built; the
gate path builds it only when planning refuses"; §13 is unchanged; `Runnable`'s member
list is documented by boundary.

The questions left open at drafting are settled under *Decisions* below.

## API sketch

```python
# qarp/_types.py — moved, re-exported unchanged from qarp.algorithms
class Target(Enum):
    SAMPLING = 0
    EXPECTATION_VALUE = 1
    OVERLAP = 2
    TRANSITION_AMPLITUDE = 3

# qarp/engines/_runnable.py — same members, grouped; samples_block gone
class Runnable(Protocol):
    # the quantity
    target: Any
    operator: Any
    supported_targets: frozenset
    # the circuits: block trees after build(), none of them built by build()
    sub_blocks: list
    compiled_circuits: list[list]
    _n_qubits_list: list[int]
    initial_state: Any
    accepts_initial_state: bool
    n_shots: Optional[Union[int, Shots]]
    # the estimator
    consumes: Consumes
    supports_exact: bool
    requires_noiseless: bool
    supports_backprop_gradient: bool
    gradient_kind: str
    def build(self) -> "Runnable": ...
    def run(self, results: list) -> Any: ...
    def run_from_amplitudes(self, compiled_circuits: list, simulator=None) -> Any: ...

# qarp/algorithms/_primitives/primitive_algorithm.py
class PrimitiveAlgorithm:
    def circuits(self) -> list[AnyBlock]: ...          # unbuilt trees; every subclass overrides
    def build(self) -> Self:                           # derives, builds nothing
        self.sub_blocks = self.circuits()
        self._n_qubits_list = [b.n_qubits for b in self.sub_blocks]
        return self

# qarp/_program.py
def plan_structure(block, n_qubits: int, fusion_width: Optional[int] = None) -> Optional[Plan]:
    """... an unbuilt composite is read from its constructor list: a declaring part
    from its declaration, any other part built and flattened on its own ..."""

# qarp/engines/_qarp_engine.py
def _compile_one(self, block, prim) -> _Compiled:
    # plan_structure(block) -> program (cached by digest of the lowered stream);
    # else block.build(), flatten, transpile, _plan_built as today

# removed: Engine.prepare_structured, StructuredRun, QPE._structured_run, QPE.block = None
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| `qarp.algorithms.Target is qarp._types.Target` | identity | `tests/test_engines/test_layering.py` |
| Every primitive's `circuits()` leaves its inputs unbuilt; once the engine builds them the estimates are unchanged | the primitives' existing analytic oracles (expectation values, overlaps, phases) | `tests/test_algorithms/test_primitives/` |
| `plan_structure` on an unbuilt composite with a declaring child lowers the child without building it | `unitary_matrix()` of a separately built copy | `tests/test_blocks/test_structure.py`, `test_unbuilt_planning.py` |
| `engine.build([Sampler(ket=QPEBlock)])` never builds the ladder: `block.is_built` is False and the ladder is one `controlled_powers` kernel at 10, 12 and 14 ancillas | analytic phase 3/8 from `P(2π·3/8)`; `Engine.kernels` listing | `tests/test_algorithms/test_composite/test_qpe.py` |
| `engine.run()` and `batch_run` serve a planned QPE and DOSQPE | analytic phase; the Dicke sector DOS (1/4 and 1/2 at half weight) | `test_qpe.py`, `test_dos_qpe.py` |
| `PauliAveraging` over a declaring block is planned (`controlled_powers` in the program) and its expectation matches `StateVector` | `StateVector` (amplitude path, never planned) | `tests/test_engines/test_structured_execution.py` |
| A second `engine.build()` of the same circuit does not re-plan | planner call-count spy | `tests/test_engines/test_structured_execution.py` |
| `prepare_structured` and `StructuredRun` are gone from the public surface | `AttributeError` | `tests/test_engines/test_structured_execution.py` |

## Phases

### Phase 1 — `Target` below the layer line

- [x] `Target` in `qarp._types`, re-exported from `qarp.algorithms._primitives.target`; engines compare by identity; the two imports tested identical *(2026-10-09)*
- [x] `samples_block` retired for `target is Target.SAMPLING and consumes is Consumes.COUNTS`; §14 wording *(2026-10-09)*
- [x] `Runnable` docstring grouped by boundary *(2026-10-09)*

### Phase 2 — Lazy circuits

- [x] `PrimitiveAlgorithm.circuits()`; every primitive derives from unbuilt inputs; `build()` builds nothing *(2026-10-09)*
- [x] `plan_structure` descends an unbuilt composite's constructor list *(2026-10-14)*
- [x] `Engine.build()` plans before building; programs cached on the engine path *(2026-10-14)*
- [x] `prepare_structured`, `StructuredRun` and the QPE/DOSQPE special case removed; §14 rewritten *(2026-10-14)*
- [x] Tests per the table; the structured-execution notebook shows the pipeline without the hook *(2026-10-14)*

### Phase 3 — Second-round review fixes (agreed 2026-10-19)

Findings of a second review at `af9a1ed`, each checked against the code and, where it is a
claim about cost or behaviour, reproduced or timed in the worktree venv.

- [x] **`_plan_built` keys its cache by the block tree as well as the stream.**  The key is
  `(stream digest, n, fusion width, level)`, but `_program.plan` reads the tree (declarations,
  action digests, child spans) and a None result is cached too.  Reproduced: a `SimpleBlock`
  replaying a declaring composite's stream, built first on the same engine, turns the declaring
  primitive's `engine.kernels()` from `[['gates', 'permutation']]` into `[['dense']]`.  Fix: a
  `tree_signature(block)` in `qarp/_program.py` (type, placement, dagger, action digest, child
  stream lengths, children, read as `_children` and `_declares_below` read them) joins the key.
  Caching on the block instead is not an option: `rebuild=True` derives fresh composites per
  build, so a per-object cache never hits across builds.  Test in
  `tests/test_engines/test_compiled_circuits.py`: each of the two primitives built after the
  other on one engine lists the kernels it lists built alone on a fresh engine.
- [x] **`_collect_parts` declines a part placed outside its node's frame.**  `qmap[t]` raises
  IndexError for a placement index at or past `len(qmap)`.  Reproduced:
  `CompositeBlock([SimpleBlock(2, target_qubits=[0, 3]) with h(0)], n_qubits=2)` under a
  `Sampler` builds and runs on `structured=False` (its one gate lands on q0) and dies in
  `engine.build()` with IndexError on the default `structured=True`.  Fix: return False from
  `_collect_parts` when any placement index of a part (a `Repeat`'s block included) is out of
  range, so the gate path takes the circuit as before.  Test in
  `tests/test_blocks/test_unbuilt_planning.py`: `plan_structure` returns None for that block,
  and the two paths give the same distribution.
- [x] **`_composition` ignores an untrusted declaration instead of refusing the block.**  A
  `CompositeBlock` subclass that overrides `structure()` and inherits `build_vanilla` is built
  eagerly, although `unbuilt_children()` (which itself guards on the vanilla build) is exactly
  what that build wires.  No class in the repo hits this; a user ladder declared over a plain
  blocks list does.  Fix: `return node.structure()` only when trusted, then fall through to the
  `unbuilt_children` branch.  Test in `test_unbuilt_planning.py`: such a subclass over a
  ladder list is planned unbuilt (`is_built` False after `engine.build`, the ladder kernel in
  `engine.kernels()`), with the built copy's distribution as oracle.
- [x] **`ShiftPlan.circuits` goes.**  Every `Occurrence` indexes a rewritten circuit, whose
  record in `self.compiled` carries the same list, so `stencil` reads
  `self.compiled[occ.circuit].commands[occ.command].gate`.  The parameter-shift tests are the
  check; no new test.

All four landed *(2026-10-19)*.

Findings not taken up, with the reason:

- **`Engine.kernels()` returning `[]` for a never-built primitive.**  Settled above
  (*Deviations*, `Engine.kernels`): the engine cannot tell a constant operator from a primitive
  it never built, and neither has a kernel.
- **Sharing `_is_built` with the plotting adapter now.**  `docs/contributions/to_do.md` records
  the fix at the binding (`def_prop_ro`, ABI bump), which removes both helpers; an interim
  shared helper is work that fix deletes.
- **Memoising the gates-wrapper `qx.Program` per record.**  Timed: `_programs_of` costs
  0.08 ms for 3 500 gates on 4 qubits (15 % of a 0.55 ms run) and 0.16 ms for 9 500 gates on
  10 qubits (2 % of a 7 ms run).  The `batch_run` figures above already include it, and
  `to_do.md` holds the program-copy item.
- **Checking `part.symbols` before flattening a plain part.**  Timed on a `PauliAveraging`
  over a 760-gate parametric ansatz with three groups: the three declines cost 0.46 ms with
  the one ket build and 0.15 ms once the ket is built, against 3.3 ms for the rebuild they sit
  in.  The first review listed the same re-collection under *Smaller cleanups*.
- **A shared bind predicate for `_substituted` and `_circuit_results`.**  Two one-line
  expressions; a helper adds a call for no behaviour.

## Decisions (green-light, 2026-10-09)

- `circuits()` is public: every primitive overrides it, a third-party one included, so it is
  the extension point `build()` is today.
- `returns_probability` stays a flag.  Composites alone read it, outside the engine boundary,
  and Phase 1 stays a move of `Target` with no behaviour change.
- A primitive accepts a block built or not.  Its own `build()` builds nothing; the engine
  plans what it is handed and builds only what planning refuses.  A built block is planned
  from its tree with its declaration checked by digest, as today; handing `QPEBlock` unbuilt
  is what avoids the ladder, and building it first costs the build once and still ends in
  the kernel.  `.symbols` exists only on a built block, so every parameter flow builds first.
- `prepare_structured`, `StructuredRun` and `samples_block` are removed outright: 0.2.0 is
  tagged after this plan lands, so none of them reaches a release.

## Deviations log

- **No removal-gate test.**  The test row asserting `prepare_structured` and `StructuredRun`
  raise `AttributeError` is not written: a removal is a one-time change, and a standing
  "must not appear" test is not kept after it.
- **Two primitives build their ket in `circuits()`.**  `CuttingPrimitive` cuts the ket's gate
  stream, which exists only once built, and `PauliShadow` refuses a ket with free symbols,
  which `.symbols` reports only on a built block.  Both say so in their docstrings; every
  other primitive leaves its inputs unbuilt.
- **The unbuilt hook serves every `Consumes.COUNTS` primitive**, not `Target.SAMPLING` alone:
  the engine runs each circuit's program and hands the results to `run()` in order, as on
  the built path, so the restriction the side door needed (it handed the bare block's counts
  to the primitive) no longer applies.
- **Programs of the built path are cached on the engine** by `(stream digest, tree
  signature, width, fusion width, level)`, bounded to 64 entries, the tree signature being
  what `plan` reads from the tree besides the stream (`_program.tree_signature`); the unbuilt path is not cached, having no stream to
  digest and costing milliseconds.
- **`QarpEngine.batch_run` evaluates each parameter set as `run()` does**, set 0 on `run()`'s
  seeds and each later set stepping every circuit's seed by its index.  `QarpEngine` does not
  call the C++ `sim.batch_run` sweep, which loops over the sets with no work shared between
  them.  A counts circuit runs as a `qx.Program`: its planned program, or its gate stream as
  one gates kernel when the stream has no mid-circuit operation and no noise model is
  active, so a parameter set is bound in C++ (`Program.substituted`) rather than through one
  Python command per gate; a noisy or mid-circuit stream runs `sim.run`.  A circuit with no
  free symbol is not substituted at all.  Measured as whole `batch_run` sweeps against the C++
  sweep, this runs 8–10% faster on 2–4 qubits and 3750–7000 gates and level on 10 qubits
  and 270 gates, with counts identical to `sim.run` on every set; looping over
  `substitute_all` and `sim.run` instead ran 32–36% slower on the same circuits here, and
  65–85% slower in review.  A finite-shot
  sweep therefore runs a planned circuit's kernels, built or lowered, and accepts a
  `Sampler`'s `initial_state`.
- **A compiled circuit is one record on its primitive.**  `prim.compiled` holds a
  `CompiledCircuit` per circuit: the transpiled stream (empty for a lowered circuit), its
  planned program or None, the simulator width, the routing layout, whether the stream has
  free symbols or a mid-circuit operation, and the stamp of the engine settings it was
  compiled under (engine class, device, optimisation level, structured execution).
  `build(rebuild=False)` reuses a primitive's records only when every stamp matches this
  engine and compiles it afresh otherwise, so a lowered circuit's program travels with its
  primitive and no engine keeps per-build programs or layouts.  `compiled_circuits` and
  `_n_qubits_list` are read-only views of the records, which the engine alone writes; the
  engine hooks take records (`_sweep`, `_dispatch_one`, `_pre_dispatch_check`), so the
  `l2p` lists they took are gone.  Parameter shift replaces the stream of each circuit it
  rewrites and keeps every other record as compiled.
- **`Runnable.target` is typed `Target`**, not `Any`: the type is importable below the layer
  line now.
- **`plan_structure` reads an unbuilt `CompositeBlock` through `unbuilt_children()`**, the
  list `build()` will wire, which the block reports itself, so `qarp._program` imports no
  block class; it composes placements through nested frames with `Command.remap_qubits`.  A
  block whose children are not readable before build (a composite subclass populating itself
  in `build_vanilla`, for which `unbuilt_children()` is None) is a plain part, built on its
  own.  A composite is descended only when its class builds as `CompositeBlock` does, and an unbuilt
  declaration is read only from a class that also defines `build_vanilla`: a subclass that
  builds differently under an inherited `structure()` is a plain part too.  A built block,
  declaring or not, is planned from its tree with its declaration checked by digest.
- **No lint gate on comparing a `Target` by name.**  The test row asked for one; the engines
  now compare by identity, and a standing "must not appear" test is not kept after a
  one-time change.  The identity of the two `Target` imports is still tested, and Phase 1's
  first box names that test where it named the gate.
- **`MirrorTest` accepts an input narrower than the working register**, placed on its low
  qubits; the built composition placed every input on the full register and refused one.
- **`batch_run` steps seeds as `run()` does**: an `EXACT` readout draws none, so a
  finite-shot primitive after it samples the same tape in both calls.
- **`Engine.kernels(primitive)` is public**, beside `Block.kernels()`: the kernels `run()`
  and `batch_run` execute for each circuit of a primitive compiled under this engine's
  settings, `["gates"]` where the gate stream runs, `RuntimeError` for a primitive compiled
  under other settings, and an empty list for one with no circuits, built or not (the engine
  cannot tell a constant operator from a primitive it never built, and neither has a kernel).
  Whether a block handed over unbuilt was lowered could
  otherwise be read only from private state; the tests read the accessor, and §14, the
  configuration page and the notebook name it.
- **The built-block hook is `_plan_built`**, not `_plan_one`, paired with `_plan_unbuilt`.
- **The unbuilt path declines a composition whose cbits would move.**  An unbuilt part's
  commands carry its own cbits, while the built stream relabels them by a node's
  `target_cbits` and stacks each part's cbits after the earlier parts'.  `plan_structure`
  therefore returns None when any node of the composition sets `target_cbits`, or when a part
  taking sequential cbits (`n_cbits > 0`, no `target_cbits`) is followed by another; only a
  cbit owner ending the composition is planned.  The circuit is then built and planned from
  its gates.
- **Planning sits in `Engine.build`, not in `QarpEngine._compile_one`.**  The base build asks
  the `_plan_unbuilt` hook before building a circuit and `_plan_built` after compiling it;
  `_compile_one` only transpiles and routes, as before.  A future planning engine overrides
  the two hooks.
- **`accepts_initial_state` and `requires_noiseless` stay off `Runnable`.**  The engines read
  them with `getattr` and a default, so a runnable that does not define them is refused
  `initial_state` and runs on a noisy engine, as before.
- **`PauliAveraging` over a QPE block skips the ladder only with `measure=False`.**  With the
  block's own readout, its measurements sit before the basis change, outside the last part,
  so planning declines and the block is built.
- **No QPE build-time bound.**  The row asked for 50 ms per case; a wall-clock bound fails
  on a loaded runner and never shows flatness.  The row's oracle is that the ladder stays
  unbuilt (`is_built`) and runs as one `controlled_powers` kernel at every ancilla count.
- **§14 gained more than the two edits named.**  Besides the `prepare_structured` paragraph
  and the `samples_block` wording: `batch_run` among the planned paths and its per-set
  seeds (scoped to `QarpEngine`), `Engine.kernels`, the rules for reading an unbuilt
  composition, and the `CompiledCircuit` record.  Each follows an entry above.
- **Files outside the Scope.**  `CHANGELOG.md`, `docs/contributions/to_do.md`,
  `docs/source/configuration.rst`, `errors.rst` and `tutorial.rst`,
  `examples/tutorial_03_engines.ipynb`, `qarp/algorithms/_composite/projected_vqe.py`, and the
  tests in `tests/test_blocks/test_unbuilt_planning.py` and `tests/test_engines/`
  (`test_compiled_circuits.py`, `test_initial_state_engine.py`,
  `test_seed_decorrelation.py`, `test_gradient_registry.py`, `test_cudaq_engine.py`) and
  `tests/conftest.py`: each documents, follows or tests an entry above.
