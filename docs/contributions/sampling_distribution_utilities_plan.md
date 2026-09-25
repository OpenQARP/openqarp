# Add analysis utilities and shot metadata to `SamplingDistribution`

**Status:** Draft
**Author:** Stefano Scali (+ Claude Code)
**Reviewer:** <to be named>
**Date:** 2026-09-25
**Tier:** Standard
**Branch:** improvement/sampler-exact-speed
**Green-lit:**
**Scope:**
- `qarp/_sampling_distribution.py` — the methods below, the `n_shots` attribute, public `from_dict`
- `qarp/_postselection.py` — `n_shots` of the kept distribution; `from_dict` in place of `_from_mapping`
- `tests/test_sampling_distribution.py`, `tests/test_postselection.py`
- `tests/test_algorithms/test_primitives/test_sampler.py` — `n_shots` from sampled and exact runs
- `docs/source/algorithms.rst` — a utilities subsection under Sampler; `docs/api/qarp.rst` — members
- `examples/tutorial_02_primitives.ipynb` — one cell using the utilities
- `docs/contributions/sampling_distribution_utilities_plan.md`, `docs/contributions/README.md` (index row)

Builds on [`sampler_distribution_plan.md`](sampler_distribution_plan.md) and
lands in the same PR, by the author's decision.  It is green-lit on its own.

---

## Why

`SamplingDistribution` gives array access to a sampling result.  Code in the
tree and in the examples still does common analyses by hand, one outcome at a
time:

| Analysis | Done by hand today |
|---|---|
| Most likely outcome | QPE `max(dist, key=dist.get)`; Grover filters for the maximum |
| ⟨Z…Z⟩ over chosen bits | PCE's ungrouping helper, with a bits × masks matrix product |
| Sub-register marginal | re-running with different `measured_qubits` |
| Dense probability vector | loops that scatter probabilities into `np.zeros(2**n)` |
| Distance between runs | `test_sampled_vs_exact_agreement`, device-validation notebooks |
| Building an expected distribution | hand-written dicts, compared key by key |

The distribution also cannot tell a sampled result from an exact one.  The
post-selection docs tell users to derive error bars from the shot count, which
the result does not carry.

## Design

Every method is a vectorised operation on `outcomes` / `probabilities`, and
every method that returns a distribution returns a `SamplingDistribution`.
**Positions** in this plan index the key tuple (`0 … n_bits_measured - 1`),
not physical qubits.  Bit `i` of a derived key comes from `positions[i]`,
matching `pack_bits` and `measured_qubits`.

**Shot metadata.**  `n_shots: Optional[int]` is the number of shots behind a
sampled distribution and `None` for an exact one.  The builder reads it from
the result (`None` when `result.is_exact`), `from_dict` and the constructor
take it as a keyword, `marginal` keeps it, `sample` sets it, and
`PostSelection.apply` sets the kept distribution's to the number of kept
shots, `round(n_shots · success_rate)`.  Equality, iteration and `repr` ignore
it: two distributions with the same keys and probabilities are equal as
mappings.

- `counts()` — `int64` counts aligned with `outcomes`, `rint(p · n_shots)`.
  `ValueError` on an exact distribution.
- `standard_errors()` — `sqrt(p(1 − p) / n_shots)` per outcome; zeros on an
  exact distribution, whose values carry no shot noise.

**Reductions.**
- `marginal(positions)` — the distribution over those positions:
  `pack_bits`, `np.unique`, `np.bincount`.  Positions must be distinct and in
  range.
- `to_dense(max_bits=28)` — the length-`2**n_bits_measured` probability vector,
  zeros where absent.  `ValueError` past `max_bits`, which bounds the default
  allocation at 2 GiB.
- `parity_expectation(positions)` — `Σ p · (−1)^popcount(outcome & mask)`,
  the expectation of `Z` on each chosen position.  An empty `positions`
  returns the total probability.
- `top(k)` — the `k` most probable `(bits, probability)` pairs, in
  descending probability, ties broken by ascending packed outcome.
  `most_likely()` is `top(1)[0]`; `ValueError` on an empty distribution.  The
  tie rule matches what `max(dist, key=dist.get)` returns today.

**Comparison and resampling.**  Both comparisons take any
`SamplingDictionary`, converted with `from_dict`, require equal
`n_bits_measured`, and use the probabilities as given, without
renormalising.
- `total_variation(other)` — `½ Σ |p − q|` over the union of outcomes.
- `hellinger_fidelity(other)` — `(Σ √(p q))²`, the definition qiskit uses.
- `sample(n_shots, seed=None)` — a multinomial draw of `n_shots` from the
  probabilities renormalised to sum to 1, since exact results prune below
  `1e-12`.  `seed` is an int, a `np.random.Generator` or `None`.  Outcomes
  drawn zero times are dropped, and the result's `n_shots` is set.

**Construction.**  `from_dict(mapping, *, n_bits_measured=None, n_shots=None)`
replaces the private `_from_mapping`.  The width comes from the keys, or from
`n_bits_measured`, which an empty mapping needs.  Keys must be 0/1 tuples of
one width.

No consumer is migrated in this plan: QPE, Grover and PCE keep their code,
and `QPE.distribution` still accepts a plain dict.  No convention edit: the
§14 result contract names the mapping and its arrays, and these are methods
on it.

## API sketch

```python
class SamplingDistribution(Mapping[tuple[int, ...], float]):
    def __init__(self, outcomes, probabilities, n_bits_measured: int, *, n_shots: Optional[int] = None): ...

    @classmethod
    def from_dict(cls, mapping: SamplingDictionary, *, n_bits_measured: Optional[int] = None,
                  n_shots: Optional[int] = None) -> "SamplingDistribution": ...

    n_shots: Optional[int]
    def counts(self) -> np.ndarray: ...
    def standard_errors(self) -> np.ndarray: ...

    def marginal(self, positions: Sequence[int]) -> "SamplingDistribution": ...
    def to_dense(self, max_bits: int = 28) -> np.ndarray: ...
    def parity_expectation(self, positions: Sequence[int]) -> float: ...
    def top(self, k: int) -> list[tuple[tuple[int, ...], float]]: ...
    def most_likely(self) -> tuple[tuple[int, ...], float]: ...

    def total_variation(self, other: SamplingDictionary) -> float: ...
    def hellinger_fidelity(self, other: SamplingDictionary) -> float: ...
    def sample(self, n_shots: int, seed: Union[int, np.random.Generator, None] = None) -> "SamplingDistribution": ...
```

## Test plan

The product state `⊗_q Ry(θ_q)|0⟩` has `P(b_q = 1) = sin²(θ_q/2)` independently
per qubit, so `⟨Z_q⟩ = cos θ_q`, and `⟨Z_a Z_b⟩ = cos θ_a cos θ_b`.  Its
marginals and dense vector are products of the per-qubit pairs.

| Test | Oracle | Location |
|---|---|---|
| `marginal` on all / subset / permuted positions of an EXACT product state | analytic product-state marginals | `test_sampling_distribution.py` |
| `to_dense` of an EXACT product state; `ValueError` past `max_bits` | Kronecker product of `[cos², sin²]` pairs, qubit 0 innermost | `test_sampling_distribution.py` |
| `parity_expectation` on single and paired positions of the product state | `cos θ_q`, `cos θ_a cos θ_b` | `test_sampling_distribution.py` |
| `parity_expectation` on GHZ | `⟨Z_0⟩ = 0`, `⟨Z_0 Z_1⟩ = 1` | `test_sampling_distribution.py` |
| `top` / `most_likely` with ties; most likely product-state outcome | hand-written distribution; per-qubit larger of `cos²`, `sin²` | `test_sampling_distribution.py` |
| `from_dict` packing, width inference, empty mapping, errors | hand-computed packed integers | `test_sampling_distribution.py` |
| `total_variation`, `hellinger_fidelity` on one-bit distributions `p`, `q` | `|p − q|` and `(√(pq) + √((1−p)(1−q)))²` | `test_sampling_distribution.py` |
| `sample` at 10⁶ shots from an EXACT product state | every frequency within 5σ of the analytic probability; counts sum to `n_shots` | `test_sampling_distribution.py` |
| `sample` with one seed twice | identical draws (reproducibility, additional) | `test_sampling_distribution.py` |
| `n_shots` of sampled and EXACT Sampler runs; `counts()` | the engine's shot count; `None` for EXACT; counts from the raw `qx.SamplingResult` of the same seed | `test_sampler.py` |
| `standard_errors` | `sqrt(p(1−p)/N)` on hand-written values; zeros when exact | `test_sampling_distribution.py` |
| Post-selected `n_shots` | kept shots summed by hand on a hand-built sampled distribution | `test_postselection.py` |

## Phases

### Phase 1 — shot metadata and construction

- [ ] `n_shots` on the class and the builder; `counts`, `standard_errors`
- [ ] Public `from_dict`; `PostSelection.apply` uses it and sets `n_shots`

### Phase 2 — reductions

- [ ] `marginal`, `to_dense`, `parity_expectation`, `top`, `most_likely`

### Phase 3 — comparison and resampling

- [ ] `total_variation`, `hellinger_fidelity`, `sample`

### Phase 4 — docs and example

- [ ] Utilities subsection in `algorithms.rst`; API members
- [ ] Tutorial cell

## Open items for the green-light

- `to_dense`'s default `max_bits = 28`.
- Equality ignoring `n_shots`.
- No consumer migration here: QPE, Grover and PCE could use `most_likely`
  and `parity_expectation` in a later change.
- Names: `marginal`, `to_dense`, `parity_expectation`, `top`,
  `most_likely`, `total_variation`, `hellinger_fidelity`, `sample`,
  `from_dict`, `counts`, `standard_errors`.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
