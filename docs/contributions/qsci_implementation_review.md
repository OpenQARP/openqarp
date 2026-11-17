# Review of #20 implementation — QSCI, Phase 1

Code review of `feature/qsci` at `067c595`: `qarp/algorithms/_composite/qsci.py`, its tests, the
`algorithms.rst` section and `mwe_qsci.ipynb`. Gates and plan conformance are out of scope. Every
snippet below was run against `067c595` with an editable install built from it (Python 3.12,
OpenMP on), and its output is pasted as printed.

Mutation runs apply one change at a time to a copy of `qsci.py` and run
`test_qsci.py` against it. A positive control (flipping two signs in `_Y_PHASE`) fails 2 tests, so
the harness tests the copy. The test sketches under T1 and T3 pass on `067c595` and fail on their
mutants. The fix sketches under B1 and B2 were applied to the copy: with them, the 46 existing tests
and the new ones all pass (60 of 60).

## Summary

The numbers are right. All 46 tests pass, ruff, `ruff format` and mypy are clean on the touched
files, and the notebook passes `pytest --nbmake` (6 s). Branch coverage of `qsci.py` is 97.7%. No
probe found a wrong energy, a wrong phase or a wrong eigenvector (see *What checks out*).

What's left:

**Bugs**

- **B1.** Ties at the `max_subspace_dim` boundary depend on block order, not on the label.
- **B2.** A `run()` that raises after sampling throws the samples away.

**Design**

- **A1.** There is no per-pass method for Phase 3's outer loop. Only `_solve` exists as a seam.
- **A2.** The selection pipeline makes one Python object per sampled outcome.

**Tests**

- **T1.** Eleven mutants pass all 46 tests.
- **T2.** A test comment gives the wrong reason for the dense fallback.
- **T3.** No regression test pins B1 or B2.

Minor items are at the end.

## Bugs

### B1. Ties at the cap depend on block order

The tie-break rule is ascending `bits_to_label`, so the selected subspace never depends on input
order. `_pool` (`qsci.py:392`) sums each key's weights as floats in block order. Two configurations
that tie exactly can then differ in the last bit, and the cap picks by rounding instead of by label.
The weights below are what three blocks give at 2000 shots, with counts 200/400/1400 spread across
two configurations:

```python
from qarp.algorithms._composite.qsci import _pool, _select

lo, hi = (1, 0, 0, 0), (0, 0, 1, 0)  # bits_to_label 1 and 4
blocks = [{hi: 0.1, lo: 0.2}, {hi: 0.2, lo: 0.7}, {hi: 0.7, lo: 0.1}]
for order in ([0, 1, 2], [1, 2, 0]):
    pooled = _pool([blocks[i] for i in order])
    kept = _select(pooled, 4, None, 1, None).configurations
    print(order, pooled[lo], pooled[hi], "kept", kept)
```

```
[0, 1, 2] 0.49999999999999994 0.5 kept ((0, 0, 1, 0),)
[1, 2, 0] 0.5 0.49999999999999994 kept ((1, 0, 0, 0),)
```

Both totals are exactly 1.0, so the rule says to keep label 1 in either order. In the first order
the cap keeps label 4.

`math.fsum` is exactly rounded, so it gives the same total in any order. The division by the
common mass that follows keeps equal weights equal:

```python
def _pool(samples: Sequence[SamplingDictionary]) -> SamplingDictionary:
    parts: dict[tuple[int, ...], list[float]] = {}
    for sample in samples:
        for key, weight in sample.items():
            parts.setdefault(key, []).append(float(weight))
    # fsum is exactly rounded, so equal pooled weights stay equal in any block order.
    totals = {key: math.fsum(weights) for key, weights in parts.items()}
    mass = math.fsum(totals.values())
    return {key: weight / mass for key, weight in totals.items()}
```

Pooling integer counts would also work, but the published pool is a probability dict, so `fsum` is
the smaller change. The inclusive `prob_threshold` has the same exposure at non-dyadic values
(`weight / in_mass >= prob_threshold`). That one is inherent and only needs a line in the docstring.

### B2. A failed `run()` throws the samples away

`run()` assigns `distribution`, `in_sector_mass` and `rejected` only after the solve succeeds
(`qsci.py:267-271`). The empty-subspace and `n_roots` checks sit between sampling and that
assignment, so a run that fails them loses the pool. The pool is what makes a re-run cheap: you
pass `qsci.distribution` back with smaller `n_roots` or looser cuts.

```python
qsci = QSCI(
    FermionOperator("0^ 0"),
    ket=ComputationalBasisStateBlock([1, 0]),
    primitive=Sampler(n_shots=100),
    n_electrons=(1, 0),
    n_roots=2,  # the sector holds one configuration
).build()
try:
    qsci.run()
except ValueError as error:
    print("ValueError:", error)
print("distribution:", qsci.distribution, "| in_sector_mass:", qsci.in_sector_mass)
```

```
ValueError: n_roots=2 exceeds the selected subspace dimension 1
distribution: None | in_sector_mass: None
```

Second run in block mode: if it fails, the attributes keep the *previous* run's outputs and give
no sign that they are stale. Publishing the selection before the checks fixes both, as long as the
solve outputs are cleared first:

```python
        self.energies = self.coefficients = self.configurations = None
        selection = _select(
            pooled, self.n_qubits, self.n_electrons, self.max_subspace_dim, self.prob_threshold
        )
        self.distribution = pooled
        self.in_sector_mass = selection.in_sector_mass
        self.rejected = selection.rejected
```

The three later assignments in `run()` then go. Regression: the snippet above with
`pytest.raises(ValueError, match="exceeds")`, then `qsci.distribution == {(1, 0): 1.0}` and
`qsci.energies is None`.

## Design

### A1. There is no per-pass method for Phase 3

Phase 3 runs selection and solve repeatedly on one pool, without re-sampling. `run()` does
sampling, selection, solve and publishing inline (`qsci.py:226-277`). The only reusable piece is
`_solve(configurations)`. A split such as `_sample() -> pooled` and `_pass(pooled) -> _Selection,
energies, coefficients` gives Phase 3 its loop body. It also leaves `run()` as sample, then pass,
then publish, which is where B2's fix lands. Better done now, while `run()` has one caller.

### A2. One Python object per sampled outcome

`Sampler` returns an array-backed `SamplingDistribution`. QSCI turns it back into per-outcome
Python objects at four points:

- `_validate_distribution` checks each bit of every key, even on qarp's own Sampler output, which
  is well-formed by construction;
- `_pool` builds a tuple-keyed dict;
- `_select` sends the in-sector set through `PostSelection` (dict, then `SamplingDistribution`,
  then a keys view) and builds a tuple and a label per candidate;
- the published `distribution` is a plain dict, so it drops `n_shots`.

Timing at 60 qubits, sector (15, 15), cap 50:

```
200000 outcomes: validate 1.16 s, pool 0.26 s, select 1.51 s
1000000 outcomes: validate 5.67 s, pool 1.70 s, select 8.04 s
```

At 200k outcomes the run peaks at 152 MiB under `tracemalloc`. The same outcomes as a
`SamplingDistribution` take 3.1 MiB.

That's fine for Phase 1's sizes. Phase 3 draws `n_batches` subsets per iteration from the
in-sector set, and SQD's sample counts are 10⁵–10⁷, so it is worth settling before then. Three
changes would get there:

- keep the pool as a `SamplingDistribution`, which carries `n_shots` and so an error bar on
  `in_sector_mass`;
- filter on the packed integers;
- check the structure of external input only.

## Tests

### T1. Eleven mutants pass all 46 tests

| Mutant | What it shows | A test that fails on it |
|---|---|---|
| `if weight > 0` → `>= 0` (`:430`) | Nothing asserts zero-weight exclusion: in `test_selection_pipeline` the zero entry already falls under `prob_threshold` | `{(1, 0): 1.0, (0, 0): 0.0}`, no cuts: `configurations == ((1, 0),)` |
| drop `or n_roots >= dim - 1` (`:583`) | The test's real matrix at `n_roots = dim − 1` is accepted by ARPACK (T2) | cutoff 1: complex `Y0 + 0.5 Z1` at `n_roots=3`, and real `X0 + 0.5 Z1` at `n_roots=4`, against `±1 ± 0.5` |
| drop the `n_roots > dim` guard (`:260`) | `test_input_validation` has no `match=`: `eigh` raises its own `ValueError` | `match="exceeds the selected subspace dimension"` |
| drop the empty-subspace guard (`:255`) | Same: something downstream raises a `ValueError` | `match="selected an empty subspace"` |
| drop the duplicate check in `_solve` (`:290`) | `run()` can't reach it; it guards the Phase 2/3 seam | `_solve([c, c])` raises `match="duplicates"` |
| `coefficients[order] = vectors` → `[:]` (`:295`) | `run()` always passes sorted configurations | `_solve` on a shuffled sector: `H_block @ c == c * e`, with `H_block` from `sparse_matrix()` in the shuffled order |
| drop the full-register check on sampler output (`:237`) | `build()` forces `measured_qubits`, so no public path reaches it | monkeypatch `engine.run` to return `[{(1,): 1.0}]`: `match="full register"` |
| wide build: an image outside the subspace lands on row 0 (`:510`) | Every wide-register test uses a closed subspace; `:511` is the one uncovered branch | 64 qubits, `X63 + Z0` on `{0…0: 1.0}`: energy `+1` (the mutant gives 2) |
| the qubit path ignores the opt-in `n_electrons` | `n_electrons` on a `QubitOperator` appears only in a validation case | `Z1` on `{(1,0): .5, (0,1): .25, (1,1): .25}`, sector (1, 0): `configurations == ((1, 0),)`, `in_sector_mass == 0.5` |
| the leakage warning always fires (`:247`) | No test asserts silence on a clean sector | `warnings.simplefilter("error")` around a clean `(1, 0)` run |
| eigsh eigenvectors replaced by ones (`:586`) | Above the cutoff only energies are compared; Phase 3 consumes the vectors | cutoff 1, 4-site Hubbard (2, 2), 3 roots: residual `H_block @ c − c e` < 1e-8 and `c†c = I` |

The full module is in the appendix.

### T2. A test comment gives the wrong reason for the dense fallback

`test_solver_dispatch` says of `n_roots ≥ dim − 1`: "ARPACK would refuse it". For its own real 4×4
input, ARPACK doesn't:

```
real k=n-1 accepted
real k=n TypeError Cannot use scipy.linalg.eigh for sparse A with k >= N. Use scipy.linal
complex k=n-1 TypeError Cannot use scipy.linalg.eig for sparse A with k >= N - 1. Use scipy.li
```

The clause is needed for real input at `n_roots = dim` and for complex input at `dim − 1`. The code
docstring says so correctly. The T1 row above tests both.

### T3. No regression test pins B1 or B2

B1: pool the three blocks above in each of the three rotations; each time, `max_subspace_dim=1`
keeps label 1. On `067c595` the first order fails. B2: the regression under B2.

## Minor

- `QSCI` accepts only a `Sampler`, checked with `isinstance` (`qsci.py:162`). That's the right
  rule: `HadamardTest` also carries `Target.SAMPLING` and returns no distribution. The error
  message cites `Target.SAMPLING`, though, which suggests a target check.
- Hermiticity is checked on the projected matrix only. An anti-Hermitian part of `H` that
  projects to zero passes. Worth a docstring line.
- `build()` deep-copies the primitive once per block, `initial_state` included: K copies of 2ⁿ
  amplitudes.
- The notebook's "Samples gathered elsewhere" cell compares two routes to the same pool. That's a
  consistency check, which is fine for an example. The CASCI limits cell is the oracle.

## What checks out

- **Oracles.** Every energy test has an independent reference: closed forms (Hubbard, TFIM),
  pyscf (`mf.e_tot`, FCI with `singlet=False`), and interlacing against `sparse_matrix()`, which
  shares no code with the Pauli walk. The reversed Hartree-Fock string pins LSB packing.
- **Selection.** `test_selection_pipeline` tells the product of the two filter rates (0.5) from the
  last filter's rate alone (2/3), and the LSB tie-break from MSB. Pooling before filtering is
  tested at 1/3 vs 2/3.
- **Pauli walk.** The `i**n_Y` phase with Y in both masks, the int64 parity fold and the
  `searchsorted` image lookup all match `sparse_matrix()`. The over-63-bit path matches the
  closed form when forced, and the sector filter is right on a 66-qubit register.
- **Solver.** On the eigsh path, residuals and orthonormality hold to about 1e-13 for a complex
  Hermitian 12-qubit operator (scipy routes it to `eigs`) and for a degenerate real Hubbard 3×2
  spectrum (dimension 400, 6 roots, equal to dense to 1e-14). Constant and degenerate-diagonal
  matrices at dimension 2048 and 4096 solve without an ARPACK failure. The zero-matrix shortcut
  covers stored zeros.
- **Inputs.** A `SamplingDistribution` passed as `distribution=` is accepted and copied. Its
  zero-weight keys survive `PostSelection`, so they are not misfiled into `rejected`. The
  exact-sampling pool replays through `distribution=` with the same configurations and energy.
- **Conventions.** No SDK imports, no `zip`, `AnyBlock` typing, LSB throughout, no qarpx
  objects built at module scope in the tests.

## Appendix: tests for T1 and T3

They pass on `067c595`, except the B1 test, which fails in its first block order until the
`fsum` fix lands. Each T1 mutant fails at least one of them.

```python
import itertools
import warnings

import numpy as np
import pytest
import scipy.linalg

from qarp.algorithms import QSCI
from qarp.algorithms._composite import qsci as qsci_module
from qarp.algorithms._composite.qsci import _pool, _select
from qarp.blocks import ComputationalBasisStateBlock
from qarp.endianness import bits_to_label
from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.models import fermi_hubbard


def _sector(n_qubits, n_alpha, n_beta):
    return [
        bits
        for bits in itertools.product((0, 1), repeat=n_qubits)
        if sum(bits[0::2]) == n_alpha and sum(bits[1::2]) == n_beta
    ]


def test_zero_weight_never_selected_without_cuts():
    algorithm = QSCI(QubitOperator("Z0"), distribution={(1, 0): 1.0, (0, 0): 0.0}).build()
    algorithm.run()
    assert algorithm.configurations == ((1, 0),)


def test_dense_fallback_at_arpack_limits(monkeypatch):
    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    both = {(0, 0): 0.25, (1, 0): 0.25, (0, 1): 0.25, (1, 1): 0.25}
    # Complex Hermitian, n_roots = dim - 1: ARPACK refuses it.
    complex_h = QubitOperator("Y0") + QubitOperator("Z1", 0.5)
    energies, _ = QSCI(complex_h, distribution=both, n_roots=3).build().run()
    np.testing.assert_allclose(energies, [-1.5, -0.5, 0.5], atol=1e-12)
    # Real, n_roots = dim: ARPACK refuses that too.
    real_h = QubitOperator("X0") + QubitOperator("Z1", 0.5)
    energies, _ = QSCI(real_h, distribution=both, n_roots=4).build().run()
    np.testing.assert_allclose(energies, [-1.5, -0.5, 0.5, 1.5], atol=1e-12)


@pytest.mark.parametrize(
    "case, match",
    [
        ("n_roots", "exceeds the selected subspace dimension"),
        ("empty", "selected an empty subspace"),
    ],
)
def test_guard_messages(case, match):
    if case == "n_roots":
        algorithm = QSCI(QubitOperator("Z0"), distribution={(1, 0): 1.0}, n_roots=2)
    else:
        algorithm = QSCI(FermionOperator("0^ 0"), distribution={(0, 0): 1.0}, n_electrons=(1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match=match):
            algorithm.build().run()


def test_solve_follows_input_order_and_rejects_duplicates():
    t, u = 1.0, 4.0
    sector = _sector(4, 1, 1)
    algorithm = QSCI(
        fermi_hubbard([2], t, u), distribution={b: 0.25 for b in sector}, n_electrons=(1, 1)
    ).build()
    algorithm.run()
    shuffled = [sector[i] for i in (2, 0, 3, 1)]
    energies, coefficients = algorithm._solve(shuffled)
    dense = fermi_hubbard([2], t, u).sparse_matrix(4).toarray()
    index = [bits_to_label(b) for b in shuffled]
    block = dense[np.ix_(index, index)]
    np.testing.assert_allclose(block @ coefficients, coefficients * energies, atol=1e-10)
    with pytest.raises(ValueError, match="duplicates"):
        algorithm._solve([sector[0], sector[0]])


def test_narrow_sampler_output_rejected(monkeypatch):
    algorithm = QSCI(QubitOperator("Z0"), ket=ComputationalBasisStateBlock([1, 0])).build()
    monkeypatch.setattr(algorithm.engine, "run", lambda *a, **k: [{(1,): 1.0}])
    with pytest.raises(ValueError, match="full register"):
        algorithm.run()


def test_wide_build_drops_images_outside_the_subspace():
    empty = (0,) * 64
    # X63 maps |0...0> outside the one-configuration subspace; <Z0> = +1.
    hamiltonian = QubitOperator("X63") + QubitOperator("Z0")
    energies, _ = QSCI(hamiltonian, distribution={empty: 1.0}).build().run()
    assert energies[0] == pytest.approx(1.0, abs=1e-12)


def test_qubit_path_sector_filter_is_applied():
    distribution = {(1, 0): 0.5, (0, 1): 0.25, (1, 1): 0.25}
    algorithm = QSCI(QubitOperator("Z1"), distribution=distribution, n_electrons=(1, 0)).build()
    with pytest.warns(UserWarning):
        algorithm.run()
    assert algorithm.configurations == ((1, 0),)
    assert algorithm.in_sector_mass == 0.5


def test_clean_sector_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QSCI(FermionOperator("0^ 0"), distribution={(1, 0): 1.0}, n_electrons=(1, 0)).build().run()


def test_sparse_path_eigenvectors(monkeypatch):
    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    hamiltonian = fermi_hubbard([4], 1.0, 3.0)
    sector = _sector(8, 2, 2)
    algorithm = QSCI(
        hamiltonian,
        distribution={b: 1 / len(sector) for b in sector},
        n_electrons=(2, 2),
        n_roots=3,
    )
    energies, coefficients = algorithm.build().run()
    dense = hamiltonian.sparse_matrix(8).toarray()
    index = [bits_to_label(b) for b in algorithm.configurations]
    block = dense[np.ix_(index, index)]
    np.testing.assert_allclose(energies, scipy.linalg.eigvalsh(block)[:3], atol=1e-10)
    np.testing.assert_allclose(block @ coefficients, coefficients * energies, atol=1e-8)
    np.testing.assert_allclose(coefficients.conj().T @ coefficients, np.eye(3), atol=1e-10)


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_tie_break_independent_of_block_order(order):
    lo, hi = (1, 0, 0, 0), (0, 0, 1, 0)  # labels 1 and 4, tied at 1/2
    blocks = [{hi: 0.1, lo: 0.2}, {hi: 0.2, lo: 0.7}, {hi: 0.7, lo: 0.1}]
    pooled = _pool([blocks[i] for i in order])
    assert _select(pooled, 4, None, 1, None).configurations == (lo,)
```
