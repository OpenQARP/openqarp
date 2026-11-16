# Respect PySCF integral settings and extract active spaces directly

**Status:** Planned
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-11-16
**Tier:** Standard
**Branch:** `fix/molecular_integrals`
**Green-lit:**
**Scope:**

- `docs/contributions/molecular_integrals_plan.md` (this plan)
- `docs/contributions/README.md` (index row)
- `src/qarp/operators/pyscf.py` (backend selection and direct active-space extraction)
- `src/qarp/operators/integrals.py` (reduction docstrings only)
- `tests/test_operators/test_pyscf_helpers.py` (adapter regressions and extraction guards)
- `tests/test_operators/test_active_space.py` (DF frozen-core reduction oracles)
- `docs/source/operators.rst` (integral settings, scaling and reference extraction)
- `examples/operators/mwe_density_fitted_active_space.ipynb` (new executable example)
- `CHANGELOG.md` (Unreleased fix entry)

---

## Why

Comparing QARP Hamiltonians with density-fitted CASCI revealed that
`integrals_from_mf` recomputes conventional two-electron integrals from
`mf.mol`, even when the supplied HF object uses density fitting. The unrestricted
adapter does the same. This can produce a different Hamiltonian from the CASCI
reference and also bypasses user-supplied AO integrals in `mf._eri`.

The investigation on PySCF 2.14.0 reproduced a difference of
`-8.379006e-6` hartree for LiH at 1.6 angstrom, cc-pVDZ, Weigend auxiliary basis,
CAS(2 electrons, 3 orbitals). Building the QARP operator from that DF-CASCI
object's effective integrals reproduced its energy to numerical precision.
The conventional calculation agreed within `1e-14` hartree. These are incident
measurements, not portable hard-coded energy expectations.

`active_space_from_mf` currently calls `integrals_from_mf` and then
`active_space_integrals`, materializing the full `N_MO**4` tensor before reducing
it. The generic reducer itself only consumes supplied arrays and folds frozen
cores into the constant and one-electron term; it does not compute integrals.

## Design

Keep chemistry extraction in `qarp.operators.pyscf` and the driver-independent
array reductions in `qarp.operators.integrals`. Follow conventions §§1, 15 and
18: abab spin orbitals and LSB operators, lazy optional imports, independent
numerical oracles, and an executed example. No convention changes are needed;
blocks, engines and symbol binding (§§13, 14, 17) are unaffected. The landed
repository-layout plan supplies the current `src/` paths. The planned QSCI and
measurement-layout contributions need no implementation dependency; reconcile
their shared narrative documentation when rebasing.

**Full-space extraction.** Share a private AO-to-MO dispatch within `pyscf.py`:

1. When `mf.with_df` is enabled, use that exact object's `ao2mo` method, retaining
   its auxiliary basis and configured factors/files. Do not create a replacement
   fitting object or fall back to conventional integrals after a DF failure.
2. Otherwise transform `mf._eri` when it is available, respecting supplied or
   cached AO integrals.
3. Otherwise transform from `mf.mol`, preserving the conventional fallback.

Request unpacked output (`compact=False`) and return dense chemists'-notation
tensors with the existing shapes. Apply the dispatch to restricted integrals
and all three UHF blocks, including `(alpha, alpha, beta, beta)`. Preserve the
existing one-electron and nuclear terms. `fermion_operator_from_mf` inherits
the fix. Full-space output still necessarily requires quartic MO storage.

**Active-space extraction.** In `active_space_from_mf`, validate the contiguous
frozen-core window before expensive work: nonnegative counts, active electrons
within the total, even frozen-electron count, sufficient active orbitals, and
`ncore + active_orbitals <= nmo`. Preserve existing valid inputs and ONV layout.
Construct `pyscf.mcscf.CASCI(mf, active_orbitals, active_electrons)` using the
public factory so its conventional/DF dispatch follows PySCF. Reuse the supplied
DF object explicitly if required by a supported PySCF version. Extract
`get_h1eff(mf.mo_coeff)` and `get_h2eff(mf.mo_coeff)` without calling SCF, CASCI
`kernel`, or orbital optimization. Restore the two-electron result with
`ao2mo.restore(1, ..., active_orbitals)`.

The returned constant already includes nuclear repulsion and the frozen-core
energy. Both it and the effective one-electron matrix must use the same
conventional/DF Hamiltonian as the active two-electron tensor. Only active MO
columns reach the two-electron transformation. PySCF computes the core potential
in the AO basis; this avoids a full MO tensor but does not eliminate AO/DF work
or the returned `N_active**4` tensor. Keep the existing ONV construction.

The extraction mechanism follows PySCF's [CASCI getters](
https://pyscf.org/_modules/pyscf/mcscf/casci.html) and [DF-CASCI implementation](
https://pyscf.org/_modules/pyscf/mcscf/df.html). Test the supported PySCF range
(`>=2.10`), including its lower bound, rather than relying solely on 2.14.0.

**Documentation and boundaries.** Explain the behavior change for DF inputs,
the difference between extraction and array reduction, and the remaining dense
storage cost. Label the inline `ao2mo.full(mf.mol, ...)` recipe as conventional
integrals rather than describing it as equivalent to every adapter input.
Document extracting all three terms from the actual reference CASCI/CASSCF
object when it uses different orbitals or fitting settings from `mf`.
Include the core constant exactly once and compare in the same spin sector.

This contribution covers real molecular restricted HF/ROHF extraction, restricted
active spaces, and full-space UHF extraction. It adds no general promise to
reproduce every SCF setting: J-only fitting (`only_dfj=True`) mixes fitted J and
exact K, so its HF energy need not equal a determinant expectation of the fully
DF Hamiltonian. State that distinction explicitly. Periodic, complex/spinor,
DFT functional energies, embedding corrections and a new direct UHF active-space
API are outside this contribution. Generic reducers continue to accept already
prepared conventional or DF tensors without knowing their provenance.

## API sketch

Existing public signatures and return structures remain:

```python
def integrals_from_mf(mf: Any) -> tuple[float, NDArray, NDArray]: ...

def unrestricted_integrals_from_mf(
    mf: Any,
) -> tuple[float, tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]: ...

def active_space_from_mf(
    mf: Any, active_electrons: int, active_orbitals: int
) -> tuple[tuple[float, NDArray, NDArray], Onv]: ...
```

No new public helper or integral-mode flag. The reference-object recipe uses
existing APIs:

```python
from pyscf import ao2mo
from qarp.operators.integrals import restricted_integrals_to_fermion_operator

# mc is the reference restricted CASCI/CASSCF object, with its selected orbitals.
mo = mc.mo_coeff
h1, constant = mc.get_h1eff(mo)
h2 = ao2mo.restore(1, mc.get_h2eff(mo), mc.ncas)
hamiltonian = restricted_integrals_to_fermion_operator(float(constant), h1, h2)
```

## Test plan

Use converged small molecules, `rtol=0`, integral tolerances of `1e-10` and
energy tolerances of `1e-9` hartree, tightening solver convergence accordingly.
Keep the LiH/cc-pVDZ/Weigend CAS(2,3) incident as a named regression alongside
parameterized conventional/DF cases. Hold orbitals fixed for tensor comparisons;
disable CASCI canonicalization or retain the input coefficient matrix. Build
qarpx objects inside tests. Guard numerical tests when PySCF is absent.

| Test | Oracle | Location |
|---|---|---|
| Restricted DF integrals use the configured auxiliary basis/factors, including when `_eri` also exists; conventional fallback remains correct | Independently transform AO DF factors and contract `sum_L B[L,p,q] B[L,r,s]`; conventional AO tensor contracted with NumPy | `tests/test_operators/test_pyscf_helpers.py` |
| UHF DF blocks include the correct cross-spin block | Separate alpha/beta transformations of AO DF factors and their NumPy contractions; UHF determinant expectation agrees with fully DF PySCF HF | `tests/test_operators/test_pyscf_helpers.py` |
| Custom `_eri` takes precedence over molecular recomputation with DF disabled (`with_df=None` included) | Deliberately modified symmetric AO tensor with direct NumPy MO contraction; CASCI effective integrals for the same custom Hamiltonian | `tests/test_operators/test_pyscf_helpers.py` |
| Conventional RHF/ROHF/UHF behavior and reference ONVs remain correct | Existing independent HF/FCI oracles and analytic occupation patterns, retained | `tests/test_operators/test_pyscf_helpers.py` |
| Direct active extraction agrees term by term for conventional/DF inputs, frozen cores and no core | PySCF CASCI effective terms plus QARP operator eigenenergy versus independently run CASCI in fixed `(n_alpha, n_beta)` sector | `tests/test_operators/test_pyscf_helpers.py` |
| Named LiH incident: fully DF Hamiltonian matches DF-CASCI; changing the auxiliary basis is respected | Independently run DF-CASCI total energies; conventional recomputation on the same DF orbitals is a negative control exceeding the energy tolerance | `tests/test_operators/test_pyscf_helpers.py` |
| Generic restricted/unrestricted frozen-core reductions also accept DF tensors | PySCF effective core/one-electron terms for supported reference cases; direct factor contractions for active two-electron blocks | `tests/test_operators/test_active_space.py` |
| Active extraction never builds full-MO ERIs or runs an electronic solver | Instrument conventional and DF transformations to require `ncas` MO columns; make `integrals_from_mf` and solver kernels fail if called (structural guard, additional to energy oracles) | `tests/test_operators/test_pyscf_helpers.py` |
| Invalid active-space counts/windows fail before integral transformation; optional dependency remains lazy | Explicit count/window rules and an import-blocked missing-PySCF test | `tests/test_operators/test_pyscf_helpers.py` |
| Notebook demonstrates DF extraction and extraction from the reference object | QARP sector eigenenergy versus PySCF DF-CASCI total energy | `examples/operators/mwe_density_fitted_active_space.ipynb` |

## Phases

### Phase 1 — Honor the full-space integral source

- [ ] Add the restricted/UHF DF and custom-AO regressions with independent oracles.
- [ ] Implement private dispatch and update adapter docstrings; retain conventional tests.

### Phase 2 — Extract active-space integrals directly

- [ ] Replace full-space extraction/reduction with validated CASCI effective getters.
- [ ] Add the named incident, frozen/no-core and ROHF cases, fixed-sector energy
  comparisons, and guards against full-MO transformations and solver calls.
- [ ] Check generic DF tensor reductions without changing their numerical algorithms.

### Phase 3 — Document and validate the behavior

- [ ] Update narrative docs and reducer docstrings; add the executable notebook and
  changelog entry, including the DF behavior change and J-only fitting limitation.
- [ ] Run the two affected operator test modules and existing integral tests in the
  checkout's `.venv`; run the new notebook with `pytest --nbmake` and
  `SKBUILD_EDITABLE_VERBOSE=0`.
- [ ] Run Ruff, mypy and documentation code checks appropriate to the edited files;
  verify >=80% changed-line coverage and retain the coverage floor. Check numerical
  regressions on PySCF 2.10 and the development environment's version.
- [ ] Record validation and any deviations in the same PR; finish the plan before
  requesting Gate 2 review.

## Deviations log

- (empty — declare deviations from the green-lit plan in the PR and fold them
  back here before merge)
