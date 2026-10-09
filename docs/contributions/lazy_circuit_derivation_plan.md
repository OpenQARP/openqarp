# Derive primitive circuits lazily and plan declared structure inside `Engine.build`

**Status:** Landed
**Author:** Stefano Scali
**Reviewer:** Stefano Scali
**Date:** 2026-10-07
**Tier:** Structural
**Branch:** improvement/lazy-circuit-derivation
**Green-lit:** 68831f4 (2026-10-09), plan blob aa0f219c5fb305e2fda24fa0259776cd048dfdc9
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
    # else block.build(), flatten, transpile, _plan_one as today

# removed: Engine.prepare_structured, StructuredRun, QPE._structured_run, QPE.block = None
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| No engine module compares a `Target` by name string; `qarp.algorithms.Target is qarp._types.Target` | lint gate, identity | `tests/test_engines/test_layering.py` |
| Every primitive's `circuits()` leaves its inputs unbuilt; once the engine builds them the estimates are unchanged | the primitives' existing analytic oracles (expectation values, overlaps, phases) | `tests/test_algorithms/test_primitives/` |
| `plan_structure` on an unbuilt composite with a declaring child lowers the child without building it | `unitary_matrix()` of a separately built copy | `tests/test_blocks/test_structure.py` |
| `engine.build([Sampler(ket=QPEBlock)])` never builds the ladder: `block.is_built` is False and build time is flat in the ancilla count (10, 12, 14 ancillas under 50 ms each) | analytic phase 3/8 from `P(2π·3/8)`; timing bound | `tests/test_algorithms/test_composite/test_qpe.py` |
| `engine.run()` and `batch_run` serve a planned QPE and DOSQPE | analytic phase; the Dicke sector DOS (1/4 and 1/2 at half weight) | `test_qpe.py`, `test_dos_qpe.py` |
| `PauliAveraging` over a declaring block is planned (`controlled_powers` in the program) and its expectation matches `StateVector` | `StateVector` (amplitude path, never planned) | `tests/test_engines/test_structured_execution.py` |
| A second `engine.build()` of the same circuit does not re-plan | planner call-count spy | `tests/test_engines/test_structured_execution.py` |
| `prepare_structured` and `StructuredRun` are gone from the public surface | `AttributeError` | `tests/test_engines/test_structured_execution.py` |

## Phases

### Phase 1 — `Target` below the layer line

- [x] `Target` in `qarp._types`, re-exported from `qarp.algorithms._primitives.target`; engines compare by identity; lint gate extended *(2026-10-09)*
- [x] `samples_block` retired for `target is Target.SAMPLING and consumes is Consumes.COUNTS`; §14 wording *(2026-10-09)*
- [x] `Runnable` docstring grouped by boundary *(2026-10-09)*

### Phase 2 — Lazy circuits

- [x] `PrimitiveAlgorithm.circuits()`; every primitive derives from unbuilt inputs; `build()` builds nothing *(2026-10-09)*
- [x] `plan_structure` descends an unbuilt composite's constructor list *(2026-10-09)*
- [x] `Engine.build()` plans before building; programs cached on the engine path *(2026-10-09)*
- [x] `prepare_structured`, `StructuredRun` and the QPE/DOSQPE special case removed; §14 rewritten *(2026-10-09)*
- [x] Tests per the table; the structured-execution notebook shows the pipeline without the hook *(2026-10-09)*

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
- **Programs of the built path are cached on the engine** by `(stream digest, width, fusion
  width, level)`, bounded to 64 entries; the unbuilt path is not cached, having no stream to
  digest and costing milliseconds.
- **`batch_run` with finite shots evaluates a lowered circuit per parameter set** through
  `program_run`, with one seed stream per circuit stepped per set, since the C++ sweep needs
  a gate stream; built circuits keep the C++ sweep.  `compiled_circuits` holds an empty list
  for a lowered circuit.
- **`Runnable.target` is typed `Target`**, not `Any`: the type is importable below the layer
  line now.
- **`plan_structure` reads an unbuilt `CompositeBlock` through its `blocks` list** (the
  constructor list it wires at build) and composes placements through nested frames with
  `Command.remap_qubits`; a block whose children are not readable before build (a composite
  subclass populating itself in `build_vanilla`) is a plain part, built on its own.
