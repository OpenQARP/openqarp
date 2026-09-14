<!-- WORKED EXAMPLE.  This plan exists to show the format: a filled-in
     _template.md for a small standard-tier feature.  Nothing in it is
     implemented or scheduled.  Delete it, and its index row, once the first
     real plan lands. -->

# Add `WStateBlock` — linear-depth W-state preparation

**Status:** 🗄️ Example (not for implementation)
**Author:** OpenQARP maintainers (+ Claude)
**Reviewer:** <a named maintainer — their green-light on this plan in the Draft PR is the design approval>
**Date:** 2026-09-15
**Tier:** Standard
**Branch:** `feature/w-state-block`
**Scope:**
- `qarp/blocks/_state_preparation/w_state_block.py` (new)
- `qarp/blocks/_state_preparation/__init__.py`, `qarp/blocks/__init__.py` (export)
- `docs/api/blocks.rst` (one entry)
- `tests/test_blocks/test_state_preparation/test_w_state_block.py` (new)
- `examples/blocks/mwe_blocks.ipynb` (one cell)

---

## Why

The W state `(|10…0⟩ + |01…0⟩ + … + |00…1⟩)/√n` is the Hamming-weight-1 Dicke
state, so `DickeStateBlock(n, 1)` already prepares it — through the general
Bärtschi–Eidenbenz network, which costs O(n) CNOT layers with a large constant.
Users preparing W states as inputs to entanglement-witness and swap-test
examples asked for a block that is cheap, reads as "W" in the circuit plot,
and needs no Hamming-weight argument.  This adds one; it does not touch
`DickeStateBlock`.

## Design

`WStateBlock(SimpleBlock)` in `qarp/blocks/_state_preparation/`, alongside
`GHZLikeStateBlock` and `DickeStateBlock` (§13: state-preparation blocks are
leaf `SimpleBlock`s with a `build_vanilla()` that emits gates).  Circuit: the
cascade of Cruz et al. (arXiv:1807.05572, Fig. 2) — an `X` on qubit 0, then
for `k = 0 … n−2` a controlled-`Ry(2·arccos(√(1/(n−k))))` from qubit `k` to
`k+1` followed by `CX(k+1, k)`.  Depth O(n), `n−1` controlled rotations, no
ancilla.  Angles are radians in the `exp(−iθP/2)` convention (§2), so the
`Ry` argument is exactly `2·arccos(…)` — the half-turn form of the paper is
NOT copied.

The block is phase-exact: every gate is a real rotation or a Pauli, so the
prepared state has no global phase and the block is controllable under
`ControlledBlock` (§13 modulo-global-phase rule).  No convention changes.

## API sketch

```python
class WStateBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        n_controls: Optional[int] = None,
        control_state: Optional[List[bool]] = None,
        name: Optional[str] = None,          # default f"W{n_qubits}"
    ): ...
    def build_vanilla(self) -> None: ...
```

Exported flat as `qarp.blocks.WStateBlock` (§15 R1: `__all__` is the surface).

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Statevector equals `(1/√n) Σ_k |e_k⟩` for n = 2 … 6, LSB layout | analytic amplitudes | `test_w_state_block.py::test_amplitudes` |
| Equals `DickeStateBlock(n, 1)` statevector up to 1e-12 | independent implementation of the same state | `…::test_matches_dicke` |
| Controlled version: `ControlledBlock` unitary on `|1⟩⊗|0…0⟩` equals W, on `|0⟩⊗|0…0⟩` is identity, exact equality (§18) | analytic | `…::test_controlled_exact` |
| Gate count is `1 + 2(n−1)` and no ancilla qubit | structural, from the construction | `…::test_gate_count` |
| `n_qubits < 2` raises `ValueError` at construction | contract | `…::test_rejects_small_n` |

## Phases

### Phase 1 — block + tests

- [ ] `WStateBlock` with `build_vanilla()` per Design
- [ ] Export + API page entry
- [ ] The five tests above

### Phase 2 — example

- [ ] One cell in `examples/blocks/mwe_blocks.ipynb` preparing
      `W4`, sampling it, and asserting four equiprobable one-hot outcomes

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
