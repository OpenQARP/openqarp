# Return Sampler distributions as an array-backed `SamplingDistribution`

**Status:** Draft
**Author:** Stefano Scali (+ Claude Code)
**Reviewer:** <to be named>
**Date:** 2026-09-25
**Tier:** Structural
**Branch:** improvement/sampler-exact-speed
**Green-lit:**
**Scope:**
- `qarp/_sampling_distribution.py` (new) — `SamplingDistribution`, the shared bit-projection kernel, the result builder
- `qarp/_types.py` — `SamplingDictionary` becomes a read-only `Mapping` alias; `PrimitiveResult` alias
- `qarp/__init__.py` — export `SamplingDistribution`
- `qarp/algorithms/_primitives/sampler.py`, `qarp/algorithms/_primitives/primitive_algorithm.py`
- `qarp/engines/_engine.py` — `StructuredQPEPlan.sample`, `_reindex_exact`, result annotations
- `qarp/engines/_qarp_engine.py`, `qarp/engines/_cudaq_engine.py` — result annotations
- `qarp/_postselection.py`
- `qarp/algorithms/_composite/grover.py`, `amplitude_amplification.py`, `amplitude_estimation.py`, `pce.py`
- `tests/test_sampling_distribution.py` (new), `tests/test_postselection.py`, `tests/strategies.py`
- `tests/test_algorithms/test_primitives/test_sampler.py`
- `tests/test_pipeline/test_capability_contract.py`
- `tests/test_engines/test_capability_validation.py`, `tests/test_engines/test_cudaq_engine.py`
- `docs/contracts/qarp_conventions.md` — §14 result contract
- `docs/source/algorithms.rst`, `docs/source/configuration.rst`, `docs/source/tutorial.rst`, `docs/source/postselection.rst`, `docs/api/qarp.rst`
- `examples/tutorial_02_primitives.ipynb` — one cell on bulk access
- `docs/contributions/sampler_distribution_plan.md`, `docs/contributions/README.md` (index row, roadmap sentence)

---

## Why

`Sampler` returns `dict[tuple[int, ...], float]`: one LSB-first tuple per
outcome (§1).  Under `n_shots=qarp.EXACT` a dense state has an outcome per
basis state, so the result is millions of Python tuples.  A dense 22-qubit
state (2.7M outcomes) measured on every qubit:

| Stage | Time |
|---|---|
| `_exact_result` (statevector + \|ψ\|²) | 0.2–0.3 s |
| `Sampler.run` building the tuple-keyed dict | 3.6–9.3 s |
| qlbm's runner repacking tuples into register integers | 9.5 s |
| same circuit, 4096 shots, end to end | 0.31 s |

The reduction's numpy stages take about 0.3 s.  The rest is Python object
construction, and every consumer that wants integers or arrays (qlbm, any
histogram into a numpy array) then undoes it in a second Python loop.
Keeping the dict and speeding up its construction caps the gain at
1.2–1.4×.  An opt-in integer-keyed variant (a flag or a sibling class) forks
the output type, and a flag makes `Sampler.run`'s declared return a union
for every caller.

The data already exists as arrays: `ExactResult` carries sorted `keys` and
`probs`, and the Sampler's reduction produces sorted packed keys and their
probabilities.  The dict is built only to satisfy the return type.

## Design

**One result type.**  `Sampler.run` returns a `SamplingDistribution`: a read-only
`collections.abc.Mapping[tuple[int, ...], float]` backed by two aligned
arrays.  Reading it like the current dict keeps working: `d[bits]`, `.get`,
`in`, `len`, `.items()`, iteration, `==` against a dict literal or
`pytest.approx`.  Bulk consumers read the arrays and never build a tuple.

- `outcomes` — the §1 integers `Σ_i b_i · 2**i` of the keys, strictly
  ascending; `int64`, or `object` dtype of Python ints when `n_bits_measured > 63`.
- `probabilities` — `float64`, aligned with `outcomes`.
- `n_bits_measured` — bits per key, `len(measured_qubits)`.

Both arrays are read-only views.  Tuple keys are made on demand: iteration
streams them from two cached half-tables, and `d[bits]` packs the tuple and
binary-searches `outcomes`.  Nothing is materialised unless the caller asks
with `to_dict()`.  Iteration is always ascending in the packed integer, for
every register width.

**One projection kernel.**  `qarp/_sampling_distribution.py` owns
`pack_bits(outcomes, positions)`: bit `positions[i]` of each outcome moves to
bit `i`, one shift pass per position, identity when `positions` is
`range(n)`.  Three places use it: the Sampler's reduction, the structured
QPE sampler (`StructuredQPEPlan.sample`, which today builds tuples outcome by
outcome) and `_reindex_exact` (the same permutation, written separately).
`distribution_from_result(result, measured)` is the one builder: project,
`np.unique`, `np.bincount`, wrap.  The `> 63`-qubit loop moves inside it.

**Types.**  `SamplingDictionary` becomes `Mapping[tuple[int, ...], float]`.
Annotations that read results keep type-checking, dict literals still
satisfy it, and only writes through the alias stop checking.
`PrimitiveResult = Union[float, complex, SamplingDistribution]` replaces the
three-way union spelled out in `PrimitiveAlgorithm.run` and the engines.
`Sampler.run` returns exactly `SamplingDistribution`.

**Post-selection.**  `PostSelection.apply` accepts any `SamplingDictionary`.
On a `SamplingDistribution` it tests bits on `outcomes` in numpy, and fixed-bit specs
drop the selected positions with `pack_bits`.  On a plain dict it keeps its
current loop.  `PostSelected.distribution` becomes a `SamplingDistribution` either
way.

**In-tree consumers.**  An audit found no mutation, JSON or pickling of a
Sampler result in `qarp/`, `tests/`, `examples/` or `docs/`.  Two things
break and are fixed here:
- Grover, AmplitudeAmplification and AmplitudeEstimation check
  `isinstance(result, dict)`.  They check `Mapping` instead.
- PCE calls `np.array(engine.run(...))`.  numpy treats a non-dict `Mapping` as
  a sequence, so the call either builds an array of key bits or raises.  PCE
  builds its object array element by element instead.

**What breaks for users.**  `isinstance(result, dict)`, mutation, and
`np.array` over a list of results.  Mutation raises `TypeError` from the first
release: there is no deprecation shim.  `to_dict()` is the migration for all
three.  The repr changes to
`SamplingDistribution({...})`, truncated past 16 entries.

**Conventions.**  §1 is unchanged: keys stay LSB-first tuples, and `outcomes`
holds the §1 integer of each.  §14's "One result contract" gains a sentence:
`Sampler` returns a `qarp.SamplingDistribution`, a read-only mapping over LSB-first
bit tuples in ascending packed order, with `outcomes` / `probabilities`
arrays for bulk access.  That is the deliberate convention edit this plan
carries.

**Roadmap.**  This lands in the 0.2 cycle, which already breaks import
paths, so users migrate once.  The contributions README's 0.2 paragraph names
it, and its 1.0 paragraph no longer claims to be the only change that breaks
user code.

## API sketch

```python
# qarp/_sampling_distribution.py
class SamplingDistribution(Mapping[tuple[int, ...], float]):
    """Read-only sampling distribution over LSB-first bit tuples."""

    def __init__(self, outcomes: ArrayLike, probabilities: ArrayLike, n_bits_measured: int): ...
    # outcomes strictly ascending and < 2**n_bits_measured; same length as probabilities

    outcomes: np.ndarray        # read-only
    probabilities: np.ndarray   # read-only
    n_bits_measured: int

    def __getitem__(self, bits: tuple[int, ...]) -> float: ...   # KeyError if absent or wrong width
    def __iter__(self) -> Iterator[tuple[int, ...]]: ...           # ascending packed order
    def __len__(self) -> int: ...
    def __contains__(self, bits: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...   # array fast path for SamplingDistribution, Mapping otherwise
    def to_dict(self) -> dict[tuple[int, ...], float]: ...


def pack_bits(outcomes: np.ndarray, positions: Sequence[int]) -> np.ndarray: ...
def distribution_from_result(result, measured: Sequence[int]) -> SamplingDistribution: ...

# qarp/_types.py
SamplingDictionary = Mapping[tuple[int, ...], float]
PrimitiveResult = Union[float, complex, "SamplingDistribution"]

# qarp/algorithms/_primitives/sampler.py
class Sampler(PrimitiveAlgorithm):
    def run(self, results: list) -> SamplingDistribution: ...
```

## Test plan

The product state `⊗_q Ry(θ_q)|0⟩` gives the analytic distribution
`P(b) = Π_q (cos²(θ_q/2) if b_q = 0 else sin²(θ_q/2))`, and its marginals are
the same product over the measured qubits.

| Test | Oracle | Location |
|---|---|---|
| Lookup, `in`, `len`, `get`, iteration order on a hand-built `SamplingDistribution` | hand-written keys and values | `tests/test_sampling_distribution.py` |
| Absent key, wrong-width key and non-0/1 entries raise `KeyError` | behavioural | `tests/test_sampling_distribution.py` |
| Writes raise `TypeError`; arrays are not writeable | behavioural | `tests/test_sampling_distribution.py` |
| `==` against a dict literal, `pytest.approx(dict)` and another `SamplingDistribution` | hand-written dicts | `tests/test_sampling_distribution.py` |
| `deepcopy` and pickle keep equality | round-trip (additional, not the oracle) | `tests/test_sampling_distribution.py` |
| `pack_bits` on chosen outcomes and permuted positions | hand-computed integers | `tests/test_sampling_distribution.py` |
| EXACT Sampler, 6 qubits, measured all / subset / permuted subset | analytic product-state marginals, `1e-12` | `test_sampler.py` |
| `outcomes` / `probabilities` of the same runs | analytic marginals keyed by `Σ b_i 2**i` | `test_sampler.py` |
| EXACT GHZ, 2 of 4 qubits measured | analytic `{(0,0): ½, (1,1): ½}` | `test_sampler.py` |
| Sampled Bell state stays on its support | analytic support `{00, 11}`; values are `counts / n_shots` | `test_sampler.py` |
| `> 63`-qubit result: keys, values and ascending order | hand-computed Python ints | `test_sampler.py` |
| 31 / 32 / 33 / 40 measured bits across the half-table boundary | tuples listed from chosen set bits | `test_sampler.py` |
| `PostSelection` fixed-bit on a product-state `SamplingDistribution` | analytic conditional marginal; success `sin²(θ_0/2)` | `test_postselection.py` |
| `PostSelection.hamming_weight` on an EXACT GHZ `SamplingDistribution` | analytic sector weights | `test_postselection.py` |
| Structured QPE returns a `SamplingDistribution` peaked at the eigenphase | existing analytic-eigenphase QPE tests | existing `test_qpe.py` |
| Grover, amplitude amplification / estimation, PCE with a Sampler | their existing analytic targets | existing composite tests |

Speed is not asserted in tests.  The PR reports the 22-qubit table above
for develop and for this branch, plus the cost of a full `to_dict()`.

## Phases

### Phase 1 — `SamplingDistribution` and the kernel

- [ ] `qarp/_sampling_distribution.py`: `SamplingDistribution`, `pack_bits`, `distribution_from_result`
- [ ] `tests/test_sampling_distribution.py`

### Phase 2 — producers

- [ ] `Sampler.run` returns `SamplingDistribution`; `StructuredQPEPlan.sample` and `_reindex_exact` use the kernel
- [ ] `SamplingDictionary` and `PrimitiveResult` aliases; primitive and engine annotations
- [ ] Sampler tests against analytic marginals

### Phase 3 — consumers

- [ ] Grover, AmplitudeAmplification, AmplitudeEstimation check `Mapping`
- [ ] PCE builds its result array element by element
- [ ] `PostSelection.apply` on `SamplingDistribution`; `PostSelected.distribution` is a `SamplingDistribution`
- [ ] Test helpers that require `dict` accept `Mapping`

### Phase 4 — contract and docs

- [ ] §14 sentence in `qarp_conventions.md`; 0.2 and 1.0 roadmap paragraphs in the contributions README
- [ ] `algorithms.rst`, `configuration.rst`, `tutorial.rst`, `postselection.rst`, `docs/api/qarp.rst`
- [ ] Bulk-access cell in `tutorial_02_primitives.ipynb`
- [ ] 22-qubit timing table in the PR

## Decisions (sit-down, 2026-09-25)

- Release window: the 0.2 cycle.
- Mutation: hard break, no shim.
- Names: `SamplingDistribution`, with `outcomes`, `probabilities` and
  `n_bits_measured`.
- `PostSelected.distribution` is a `SamplingDistribution`, so the sampling
  pipeline returns one type end to end.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
