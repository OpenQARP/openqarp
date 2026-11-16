# Add `LUCJBlock` — the local unitary cluster Jastrow ansatz, ffsim-exact

**Status:** Planned
**Author:** Jonathan Fetherolf (+ Claude Code)
**Date:** 2026-11-16
**Tier:** Structural
**Branch:** feature/lucj-block
**Green-lit:**
**Scope:**
- `src/qarp/blocks/_primitives/orbital_rotation_block.py` (complex unitaries, parametric mode, brickwork decomposition)
- `src/qarp/blocks/_primitives/lucj_block.py` (new: `LUCJBlock`, `lucj_interaction_pairs`)
- `src/qarp/blocks/_primitives/__init__.py`, `src/qarp/blocks/__init__.py` (exports, `__all__`)
- `src/qarp/algorithms/_composite/vqa.py` (`parameter_transform` hook)
- `src/qarp/algorithms/_composite/projected_vqe.py` (reject a transform)
- `tests/test_blocks/test_primitives/test_orbital_rotation_block.py`
- `tests/test_blocks/test_primitives/test_lucj_block.py` (new)
- `tests/test_algorithms/test_composite/test_vqa.py`
- `tests/strategies.py` (`FACTORIES` entries)
- `tests/assets/lucj/generate.py`, `tests/assets/lucj/*.npz` (new: ffsim golden fixtures)
- `docs/source/blocks.rst`, `docs/source/algorithms.rst`
- `examples/blocks/mwe_lucj.ipynb` (new)
- `CHANGELOG.md`, `docs/contributions/to_do.md`
- `docs/contributions/lucj_block_plan.md`, `docs/contributions/README.md` (this plan + index row)

---

## Why

qarp has QSCI/SQD (`qsci_plan.md`) but not the ansatz SQD was demonstrated with:
the local unitary cluster Jastrow (LUCJ) of Motta et al., [Chem. Sci. 14, 11213
(2023)](https://doi.org/10.1039/d3sc02516k), used by Robledo-Moreno et al.,
[Sci. Adv. 11, eadu9991 (2025)](https://doi.org/10.1126/sciadv.adu9991) for
N2, [2Fe-2S] and [4Fe-4S]. It is also a compact VQE ansatz that reaches FCI
where qUCCSD does not (Motta Figs. 6–9). Both papers' circuits come from ffsim,
so ffsim is the reference: same parameters must give the same state.

Pre-plan measurements (scratch prototypes, not committed):
- The planned circuit matches ffsim's LUCJ state to 1e-15 (H2O from t2/t1;
  random complex operators; square, heavy-hex and all-to-all pairs; L = 1, 2;
  open shell), up to the reference-sign convention below. Bound symbolic
  circuit equals numeric. Full unitary equals the first-principles oracle.
- Optimising gate angles from the CCSD start stalled 8–22 mEh above K-matrix
  coordinates in 4 of 6 runs (H2O, N2 at 1.1 and 2.0 Å; L-BFGS-B). Hence the
  K-coordinate hook below is necessary.

## Design

**Register.** `LUCJBlock` is a pure unitary on the abab Jordan-Wigner register
(qubit 2p = α_p, 2p+1 = β_p), so it composes with `MappedONVStateBlock`, qarp
Hamiltonians and `QSCI` unchanged. The reference determinant is composed in
front by the user, as with `UCCBlock`. Inside, an odd-even fermionic-swap
network (FSWAP = SWAP·CZ, depth N−1) moves to the paper's chain layout
(α_p → p, β_p → N+p), the paper's circuit runs, and the inverse network moves
back. Child blocks therefore act in chain layout; the composite is the abab
unitary. aabb never reaches the public surface, so no conventions edit. In-circuit transformation
from abab to aabb and back is the only way to avoid touching QARP conventions.

**Circuit.** Spin-balanced UCJ with an optional final rotation F:
`F·Π_k U_k e^{iJ_k} U_k†` (ffsim `UCJOpSpinBalanced` order). Adjacent rotations
are merged: `R_0 = U_1†`, `R_k = U_{k+1}†U_k`, `R_L = F·U_L` (F = I if absent),
giving `R_L J_L … J_1 R_0` with L+1 rotations. Each `R_k` is a parametric
`OrbitalRotationBlock` on the α chain and a second one, same symbols, on the
β chain. `J_k` follows ffsim's `exp(i Σ J^{στ}_ij n_iσ n_jτ / 2)` with
`J^{αα} = J^{ββ}`: `CP(J_ij)` per same-spin pair i<j and per opposite-spin
pair (both orientations off the diagonal), `P(J_ii/2)` per same-spin diagonal.
`CP`/`P` are exactly `e^{iφ n n}`/`e^{iφ n}`, so the block is phase-exact
(§13) and controllable.

**`OrbitalRotationBlock` extensions.** Accepts any unitary `u`; a real `u`
emits today's circuit unchanged. Complex `u` and the new parametric mode use
ffsim's brickwork Givens layout (`GivensAnsatzOp`: N layers of nearest-neighbour
Givens, each `G(θ)` followed by `P(φ)`, then a phase layer; N² parameters, depth
N). The parametric mode has fixed gate positions (no skipped rotations), so a
numeric `u` decomposes into values for its symbols.

**Parameters.** Circuit symbols are gate angles (§10, §13: one linear symbol
per gate). `LUCJBlock.parameter_map_from_ucj(params)` maps a K-coordinate vector
in ffsim's `UCJOpSpinBalanced.to_parameters` layout — orbital-rotation
generators and J entries — to `{symbol: angle}` via `U = expm(K)`, the merge
above and the brickwork decomposition. Deterministic; §17-safe (by name).
`symbol_postfix` disambiguates multiple blocks, as in `UCCBlock`.

**CCSD start.** `LUCJBlock.from_t_amplitudes(t2, t1=None, n_reps=None,
interaction_pairs=None)` reimplements ffsim's
`UCJOpSpinBalanced.from_t_amplitudes` (double factorization of t2, ffsim's
ordering and truncation, t1 → final rotation), not vendored; credited in the
docstring. It returns the block with `initial_ucj_parameters` set (the K vector).
Restricted closed-shell t2; norb and active space must match t2's.

**Interaction pairs.** Explicit `(pairs_aa, pairs_ab)`, upper-triangular, with
ffsim's semantics and validation (`None` = all pairs). `lucj_interaction_pairs(
topology, norb, *, max_ab_orbital=None)` returns the papers' pairs: same-spin
(p, p+1) plus opposite-spin diagonals p — `"square"` all p, `"hex"` even p,
`"heavy-hex"` p%4==0 (Motta eq. 14, Robledo-Moreno eq. 26; Motta's N=6 loop
exception {0, 5} noted, not used), `"linear"` {0}; `"all-to-all"` returns
`(None, None)`. `max_ab_orbital=16` gives Robledo-Moreno's cut.

**`VQA` hook.** `parameter_transform: Callable[[NDArray], Mapping[Symbol,
float]]`. The optimizer works on y; each evaluation binds `transform(y)`.
With a transform, `initial_parameters` is required and lives in y-space;
`optimal_parameters` stays the circuit-symbol map; `optimal_transform_parameters`
holds y*. Gradients are finite differences by the optimizer
(`gradient=True` with a transform raises — an exact K-gradient is an open
item). Applies to `VQE`/`QAOA` (inherited `run`); `ProjectedVQE` raises.
`LUCJBlock.ucj_transform` is `parameter_map_from_ucj`.

**Reference sign.** ffsim's Hartree-Fock state is +1 in α-then-β ordering;
qarp's X-gate reference is +1 in abab. The same determinant differs by
`(−1)^{#(p<q): β_p, α_q occupied}`, a global sign; parity tests apply it.

**Limits.** Spin-balanced only, Jordan-Wigner only, numeric initialiser from
restricted t2 only.

## API sketch

```python
def lucj_interaction_pairs(
    topology: Literal["square", "hex", "heavy-hex", "linear", "all-to-all"],
    norb: int,
    *,
    max_ab_orbital: int | None = None,
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]: ...

class LUCJBlock(CompositeBlockBase):
    def __init__(
        self,
        norb: int,
        n_reps: int,
        *,
        interaction_pairs: tuple[list[tuple[int, int]] | None,
                                 list[tuple[int, int]] | None] | None = None,
        with_final_orbital_rotation: bool = False,
        symbol_postfix: str = "",
        target_qubits: list[int] | None = None,
        name: str = "LUCJ",
    ): ...                                    # n_qubits = 2 * norb
    initial_ucj_parameters: NDArray | None    # set by from_t_amplitudes
    @classmethod
    def from_t_amplitudes(cls, t2, *, t1=None, n_reps=None, interaction_pairs=None,
                          symbol_postfix="") -> "LUCJBlock": ...
    def n_ucj_parameters(self) -> int: ...    # == ffsim UCJOpSpinBalanced.n_params
    def parameter_map_from_ucj(self, params: NDArray) -> dict[Symbol, float]: ...
    ucj_transform = parameter_map_from_ucj

class OrbitalRotationBlock(CompositeBlockBase):
    def __init__(self, u, dagger=False, target_qubits=None, name="OrbitalRotation"): ...  # any unitary
    @classmethod
    def parametric(cls, n_modes: int, *, prefix: str = "u", real: bool = False,
                   target_qubits=None, name="OrbitalRotation") -> "OrbitalRotationBlock": ...
    def parameter_map_from_matrix(self, u) -> dict[Symbol, float]: ...  # parametric mode

class VQA(CompositeAlgorithm):
    def __init__(..., parameter_transform: Callable[[NDArray], Mapping[Symbol, float]] | None = None): ...
    optimal_transform_parameters: NDArray     # set by run() when a transform is given
```

Typical use:

```python
ansatz = LUCJBlock.from_t_amplitudes(ccsd.t2, t1=ccsd.t1, n_reps=2,
                                     interaction_pairs=lucj_interaction_pairs("square", norb))
ket = CompositeBlock([MappedONVStateBlock(onv_from_mf(mf), JordanWigner()), ansatz])
vqe = VQE(hamiltonian, ket, initial_parameters=ansatz.initial_ucj_parameters,
          parameter_transform=ansatz.ucj_transform, optimizer=ScipyOptimizer("L-BFGS-B"))
```

## Test plan

pyscf rows use `tests/pyscf_recipes.py`. Parametrise over plain specs; build
qarpx objects in the test body. ffsim fixtures are generated by
`tests/assets/lucj/generate.py` (needs ffsim; version stamped in each file) and
stored in the fixed-particle basis, so they run in every PR without ffsim.

| Test | Oracle | Location |
|---|---|---|
| Complex `u` (det ≠ 1 included): full unitary, exact equality | `thouless_unitary` (ladder-matrix `expm(logm u)`) | `test_orbital_rotation_block.py` |
| Complex `u` under `ControlledBlock`, both polarities | §6.1 analytic matrix | `test_orbital_rotation_block.py` |
| Parametric mode with `parameter_map_from_matrix(u)` bound | `thouless_unitary(u)`, exact | `test_orbital_rotation_block.py` |
| Real `u` emits the pre-change gate stream | regression (additional) | `test_orbital_rotation_block.py` |
| LUCJ full unitary, numeric parameters: square, heavy-hex, all-to-all; L = 1, 2; with and without F | first-principles product of Thouless rotations and the diagonal Jastrow, exact | `test_lucj_block.py` |
| LUCJ under `ControlledBlock` | §6.1 analytic matrix | `test_lucj_block.py` |
| Jastrow `CP` counts per topology | Motta Table 1 | `test_lucj_block.py` |
| Zero parameters give the identity; N and Sz conserved | analytic | `test_lucj_block.py` |
| Same K vector gives ffsim's state (with the reference sign) and energy | golden ffsim fixtures | `test_lucj_block.py` |
| Same, live, when ffsim is installed | ffsim via `importorskip` | `test_lucj_block.py` |
| `from_t_amplitudes` K vector equals ffsim's for H2O and N2 | golden ffsim fixtures | `test_lucj_block.py` |
| Untruncated factors reconstruct t2 | double-factorization identity (analytic) | `test_lucj_block.py` |
| Scaled-down t2, all-to-all, full rank: doubles amplitudes equal t2 to first order | first-order UCCD (analytic) | `test_lucj_block.py` |
| H2 (2e,2o), L = 1, all-to-all, with F: VQE reaches FCI to 1e-8 | pyscf FCI (Motta Fig. 9) | `test_lucj_block.py` |
| Engine gradient w.r.t. shared angle symbols | central finite difference of the energy | `test_lucj_block.py` |
| `lucj_interaction_pairs` lists | Motta eq. 14, Robledo-Moreno eq. 26 | `test_lucj_block.py` |
| Invalid pairs, `n_reps`, t2 shape raise | contract (ffsim's rules) | `test_lucj_block.py` |
| Symbol contract (sorted tuple, postfix, rename) | §17 registry | `tests/strategies.py` `FACTORIES` |
| Hook with x = 2y on a one-parameter ansatz | analytic optimum | `test_vqa.py` |
| Hook + LUCJ on H2O from the CCSD start: converged energy | ffsim K-coordinate L-BFGS-B optimum (fixture), within 1e-6 Eh | `test_vqa.py` |
| `gradient=True` with a transform raises; `ProjectedVQE` with a transform raises | contract | `test_vqa.py` |

## Phases

One PR, one phase.

### Implementation

- [ ] `OrbitalRotationBlock`: complex unitaries, brickwork decomposition, parametric mode
- [ ] `LUCJBlock`: swap networks, merged rotations, Jastrow, symbols, `parameter_map_from_ucj`
- [ ] `LUCJBlock.from_t_amplitudes`
- [ ] `lucj_interaction_pairs`
- [ ] `VQA.parameter_transform`; `ProjectedVQE` rejection
- [ ] Exports and `__all__`; `FACTORIES` entries
- [ ] ffsim fixture generator and fixtures
- [ ] Test rows above
- [ ] `docs/source/blocks.rst` (LUCJ section, `OrbitalRotationBlock` update), `docs/source/algorithms.rst` (hook)
- [ ] `examples/blocks/mwe_lucj.ipynb`: VQE from the CCSD start in K coordinates; SQD through `QSCI`
- [ ] `CHANGELOG.md`; `to_do.md` entry for 2-CX Givens/FSWAP decompositions

## Open items

- Exact K-coordinate gradient: forward/back-propagated states at each rotation
  via `Block.statevector` (statevector engines only), or an engine adjoint
  extension.
- Opt-in chain-layout register (drop the swap networks) for hardware SQD; needs
  a conventions edit.
- Robledo-Moreno's orbital permutation to place the largest `J^{αβ}` on kept
  pairs (eqs. 27–31).
- Motta's bootstrapping (all-to-all → square → hex → heavy-hex): a helper that
  remaps a converged K vector onto another interaction-pair set.
- Spin-unbalanced UCJ (ffsim `UCJOpSpinUnbalanced`); complex `SlaterDeterminantBlock`.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
