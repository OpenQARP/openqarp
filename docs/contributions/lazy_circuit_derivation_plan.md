# Derive primitive circuits lazily and plan declared structure inside `Engine.build`

**Status:** Draft
**Author:** Stefano Scali
**Reviewer:** to be named at the sit-down
**Date:** 2026-10-07
**Tier:** Structural
**Branch:** improvement/lazy-circuit-derivation
**Green-lit:** —
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

Open decisions for the sit-down, not guessed here:

- Whether `circuits()` is public (`PrimitiveAlgorithm.circuits`) or `_circuits`.
- Whether `returns_probability` becomes a fifth `Target` member (`OVERLAP_PROBABILITY`) in
  Phase 1 or stays a flag.
- Whether `Sampler(ket=...)` must still accept a built ket whose structure cannot be planned
  (yes by default: a built block is planned from its tree as today).
- Whether `prepare_structured` gets a one-release deprecation shim or the hard break.

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

- [ ] `Target` in `qarp._types`, re-exported from `qarp.algorithms._primitives.target`; engines compare by identity; lint gate extended
- [ ] `samples_block` retired for `target is Target.SAMPLING and consumes is Consumes.COUNTS`; §14 wording
- [ ] `Runnable` docstring grouped by boundary

### Phase 2 — Lazy circuits

- [ ] `PrimitiveAlgorithm.circuits()`; every primitive derives from unbuilt inputs; `build()` builds nothing
- [ ] `plan_structure` descends an unbuilt composite's constructor list
- [ ] `Engine.build()` plans before building; programs cached on the engine path
- [ ] `prepare_structured`, `StructuredRun` and the QPE/DOSQPE special case removed; §14 rewritten
- [ ] Tests per the table; the structured-execution notebook shows the pipeline without the hook

## Deviations log

- (empty)
