# Add reusable grouped transition measurement layouts

**Status:** Planned
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-11-03
**Tier:** Standard
**Branch:** `improvement/qmc-measurement-layouts`
**Green-lit:**
**Scope:** This is the separate contribution identified by P4 of the GPU design
review. Only this planning document and its index entry are prepared with the
GPU plan; implementation requires its own branch, Draft PR and Gate 1. The GPU
implementation merges first. Rebase this contribution on that result; shared RST
pages and `qarp/operators/__init__.py` receive only the layout additions. Its
plan and index entry already land with the GPU plan; the index row is not edited
again. The proposed implementation scope is:

- `docs/contributions/qmc_measurement_layouts_plan.md`
- `docs/source/algorithms.rst`
- `docs/source/blocks.rst`
- `docs/source/operators.rst`
- `examples/algorithms/primitives/mwe_grouped_transition_layouts.ipynb`
- `qarp/algorithms/_primitives/grouped_transition_hadamard_test.py`
- `qarp/blocks/_primitives/interferometric_state_block.py`
- `qarp/operators/__init__.py`
- `qarp/operators/_grouping.py`
- `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py`
- `tests/test_algorithms/test_primitives/test_lazy_circuits.py`
- `tests/test_operators/test_grouping.py`

---

## Why

`GroupedTransitionHadamardTest` currently supports QWC measurement groups. Add
general commuting measurement bases, an optional DSATUR grouping strategy and
reusable layout descriptions for transition measurements. These features are
independent of resident GPU walker evolution. The GPU contribution uses the
existing QWC `circuits()` surface for optional accounting and does not depend on
this plan. A concrete consumer is `GroupedTransitionHadamardTest` for
real/imaginary transitions of the two-qubit Heisenberg Hamiltonian H=XX+YY+ZZ:
QWC requires three groups (six circuits for both quadratures), while
`FullyCommuting` needs one group (two circuits). General commuting bases allow
that existing strategy to serve transitions.

## Design

Follow conventions §§13–15, 17–19 without a convention change. Extend the
existing grouping strategy and block/primitive classes; introduce no engine or
block hierarchy. Keep QWC as the default. Add `FullyCommutingDSATUR`, a
deterministic saturation-degree graph-coloring strategy, as an optional
alternative to the existing deterministic greedy `FullyCommuting` strategy. Both
operate on the anticommutation graph; they differ in vertex selection. DSATUR
chooses greatest saturation, then greatest graph degree, then existing canonical
term order. Assign the lowest available color. Return groups in color order,
with members in canonical term order, mapping indices back to the input as
required by `GroupingStrategy`. Extend `test_grouping_is_order_insensitive` with
the new strategy.

Use existing `diagonalise_group` for the Clifford commands, Z parity masks and
signs; add no second diagonalizer. Preserve exact phase when constructing the
interferometric measurement basis from those commands. Derive circuits lazily
through `circuits()`; `build()` must not eagerly fill `sub_blocks`. Use the
public base lifecycle for target inference and derivation. Build no caller input
merely to count circuits. Cache only immutable plain layout descriptions
globally, never native blocks or mutable bound parameter values. Cache keys
include operator terms/coefficients, register width and grouping strategy;
layouts are reused across bra/ket pairs without caching their state preparations
globally.

Within one derivation, reuse the interferometric state-preparation prefix by
copying it before placement. Preserve pending substitutions/daggers, exact
global phase, target/cbit mapping, live bra/ket defaults, explicit rebinding and
`SamplingDistribution` results. Read Python symbol/parameter members through
`QarpBlock` and use `parameter_map`; use `AnyBlock` only where native members
suffice. Retain existing regression cases.

This contribution adds no finite-shot Monte Carlo mode and does not change
`MonteCarlo` or `AggregatedMonteCarlo`. Any later integration beyond the
existing accounting interface needs an explicit scope decision. Tapering and
selected-pair work are outside this contribution.

## API sketch

```python
from qarp.operators import FullyCommutingDSATUR

# Existing grouping= argument additionally accepts FullyCommutingDSATUR().
# QubitWiseCommuting() remains the default.
```

Expose `GroupedTransitionHadamardTest.measurement_plan` after lazy circuit
derivation as a tuple of per-group tuples: `(term_indices, sorted_basis_pairs,
parity_masks, sign_flip_flags, commands)`. Every component contains only tuples
of plain ints/strings/bools/floats. Command descriptions are `(gate_name,
qubit_tuple, parameter_float_tuple)`. The zero operator yields `()`. This
property describes derived layouts; reading it must not execute or compile a
circuit.

Extend `InterferometricMeasurementBlock` with keyword-only `basis_change:
Optional[AnyBlock] = None` and `prepared_state:
Optional[InterferometricStateBlock] = None`. `basis_change` supplies a concrete
unitary on the data register and is mutually exclusive with a nonempty local
`basis`. `prepared_state` must match the required bra, ket and quadrature. Copy
before placement to preserve caller ownership and pending operations. Width,
basis/quadrature/preparation mismatches and free basis-change symbols raise
`ValueError`. Symbols remain those of bra/ket; no new parametric block class is
introduced.

`measurement_plan` raises `RuntimeError` before successful `circuits()`
derivation. It is a read-only snapshot of the current inputs, not an automatic
build. Fingerprint bra/ket identity plus their current parameter/placement/
pending-operation state, operator terms, quadratures, width and grouping
settings when deriving. The property compares those inputs without building; if
anything changed, raise `RuntimeError` requiring fresh `circuits()`. A later
derivation replaces the snapshot; previously returned immutable tuples remain
valid. Global caches retain only operator/grouping layout descriptions, so
changing bra/ket invalidates the per-primitive snapshot without poisoning the
shared layout cache. Test in-place block/operator mutations as well as attribute
replacement and grouping changes.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| DSATUR grouping | Hand-colored path and cycle anticommutation graphs with fixed canonical tie-breaking; literal Pauli commutators establish allowed groups; extend existing test_grouping_is_order_insensitive | `tests/test_operators/test_grouping.py` |
| General commuting bases | Analytic simultaneous eigenbases for XX/YY/ZZ, including YY parity sign; exact matrices including global phase and controlled form | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Layout descriptions | Hand-counted groups, masks, signs and commands; zero operator gives empty tuple; mutation cannot change cached descriptions; pre-derivation reads and stale reads raise RuntimeError, fresh derivation replaces snapshot | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Lazy lifecycle and binding | Unbuilt-input/compile/run sentinels; analytic transition matrices after rebinding, dagger and target/cbit remapping | `tests/test_algorithms/test_primitives/test_lazy_circuits.py` |
| Reused preparation and ownership | Analytic complex bra/ket transition amplitude; copied placements leave originals unchanged; width/quadrature/basis conflicts reject | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Existing QWC behavior | Existing independent analytic cases remain; default and explicit QWC give the same defined layouts and results | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Example | Small literal Pauli matrices and bra/ket vectors predict measured real/imaginary transitions using exact readout | `examples/algorithms/primitives/mwe_grouped_transition_layouts.ipynb` |

Run the grouping, primitive and lazy-lifecycle regressions, export checks, ruff,
mypy, unchanged coverage ratchets and the example in this contribution's own
checkout environment. Numerical tests use the independent matrices above; round
trips and repeated outputs are supplementary.

## Phases

### Phase 1 — resolve API and group deterministically

- [ ] Record this contribution's separate Gate 1 before implementation.
- [ ] Add the strategy and analytic grouping tests; preserve QWC defaults.

### Phase 2 — lazy layouts and phase-exact measurement

- [ ] Add general bases, immutable layout descriptions and copied preparation.
- [ ] Test exact phases, controlled form, ownership, binding and lazy lifecycle.

### Phase 3 — documentation and validation

- [ ] Document APIs and add the bounded example.
- [ ] Run regression, lint, type, coverage and example checks before Gate 2.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR and folded
  back here before merge)
