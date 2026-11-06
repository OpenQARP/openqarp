# Make `.symbols` a tuple on every block that has a registry, and an error before

**Status:** Landed
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-23
**Tier:** Structural
**Branch:** improvement/symbols-always-tuple
**Green-lit:** 2026-10-28, plan blob 7594a513dd769952af9cfc8396accdbb5aa6a4d0
**Scope:**
- `qarp/blocks/_block.py`: the `symbols` property and setter, the reset in
  `_init_python_state`, and the probes in `parameter_map`, `_publish_symbols`,
  `set_symbols`, `replace_symbols`, `refresh_symbols`
- `qarp/blocks/_composite_block.py` (the reset at construction),
  `qarp/blocks/_primitives/ucc_block.py` (one probe), `trotter_block.py` (two)
- `qarp/algorithms/_composite/vqa.py`, `vff.py`, `ssvqe.py`, `pce.py`, `vqd.py`
  (error text and the `assert symbols is not None` narrowings), `adapt_vqe.py`,
  `projected_vqe.py` (one `assert` each)
- `tests/test_blocks/test_block.py`, `test_block_refactor.py`,
  `test_symbols_contract.py`, `tests/test_blocks/test_primitives/test_a_gate_block.py`,
  `tests/test_engines/test_threading.py`, `tests/test_stubs.py` (one snippet)
- `docs/source/qarp_philosophy.rst`: one snippet drops its `or ()`
- `docs/contracts/qarp_conventions.md`: one sentence in §17
- `CHANGELOG.md`, `docs/contributions/to_do.md`

---

## Why

`block.symbols` is typed `Optional[Tuple[Symbol, ...]]`: `None` until the block
publishes its registry, at `build()` for most blocks and at construction for
the ansatz blocks that declare their symbols (`HEABlock`, `UCCBlock`,
`TrotterAnsatzBlock`, `UPCCDBlock`, `BrickworkPCEBlock`).  Every read must
narrow first.  `len(block.symbols)` fails mypy in 22 notebook cells and 3 docs
snippets (the count in `to_do.md`'s *`qarp` typed for its users* entry), and a
`py.typed` qarp would push that narrowing onto every user.

The `None` carries one rule worth keeping: a block that failed to build has no
registry, and must not look like a parameter-free block.  `QAOABlock` sized
below its labels once published `()` after a remap error, and VQE reported "ket
has no parameters to optimize"; `tests/test_blocks/test_block.py` pins that a
failed composite build leaves `symbols` unpublished, "never published as ()".

The silent side of `None` is a hazard.  A truthiness probe on an unpublished
registry is False, so a composite that forgot to build its ket would accept an
unbuilt parametric block as concrete.  Every in-tree probe builds first
(`VQA.__init__`, `QITE`, `PauliShadow`: "`.symbols` exists on a built block
only, so the ket is built here"); the type does not say so, and
`parameter_map` and `refresh_symbols` already raise `RuntimeError` on the
unpublished case instead of returning something.

## Design

**The registry is a tuple or it is an error.**  `symbols` returns
`Tuple[Symbol, ...]`.  Reading it on a block that has not published a registry
raises `RuntimeError`, the condition `parameter_map` and `refresh_symbols`
already raise on, moved up to the read.  Nothing else about the registry
changes: canonical order, the setter as the sole write path, `()` for a built
block with no free parameter (§17).

- **Private probe.**  `_symbols is None` is the "unpublished" test inside the
  block layer: `_publish_symbols`, `set_symbols` and `replace_symbols` (both
  run on unbuilt blocks and skip a `None` registry), `refresh_symbols`,
  `UCCBlock.build_vanilla`, `TrotterAnsatzBlock.build_vanilla` and
  `TrotterAnsatzBlock.replace_symbols`.  `parameter_map` keeps no probe of its
  own: the property raises on the same condition.  The setter keeps accepting
  `None`: the reset in `_init_python_state` and `CompositeBlock.__init__`
  stays.
- **A failed build publishes nothing**, so reading `symbols` afterwards raises
  rather than returning `None`; the incident test asserts the raise.
- **Composites.**  The five "no parameters to optimize" probes read a ket they
  have just built, so `if not ket.symbols:` stays; the message drops "(symbols
  is None or empty)".  The eight `is not None` narrowings in
  the composites go: mypy no longer needs them.
- **Alternatives, rejected.**  `()` before build collapses "unbuilt or failed"
  into "parameter-free", the incident.  Keeping `Optional` is the gap this plan
  closes.  Raising `AttributeError` would let `getattr(block, "symbols", None)`
  probe softly, but the attribute exists and the existing paths raise
  `RuntimeError`.

**§17 edit, deliberate.**  The invariant paragraph gains one sentence:
"Reading `symbols` on a block that has not published a registry raises
`RuntimeError`; it is never `None`, and never `()` for a block that failed to
build."

**Who sees a change.**  `block.symbols is None` as a built-ness probe raises;
use `block.is_built`, or catch `RuntimeError` for a block that may declare its
symbols at construction.  User code that narrowed `Optional` can stop.  A read
on an unbuilt block that declared no symbols raises where it returned `None`.
This closes one of the five gaps of the *`qarp` typed for its users* entry;
the other four stay there.

## API sketch

```python
@property
def symbols(self) -> Tuple[Symbol, ...]:
    """Canonically ordered free-parameter registry (§17).

    Raises ``RuntimeError`` before the block has one: build it, or declare
    the symbols in the constructor.
    """
    symbols = getattr(self, "_symbols", None)
    if symbols is None:
        raise RuntimeError(
            f"'{self.name}' has no symbols yet: build it first, or define "
            "symbols in the constructor."
        )
    return symbols

@symbols.setter
def symbols(self, value: Optional[Iterable[Symbol]]) -> None:
    # None resets the registry (a composite before build); any sequence is
    # canonicalized, so an unsorted published order is unreachable.
    self._symbols = None if value is None else _sorted_symbols(value)
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| `SimpleBlock(2).symbols` before `build()` raises `RuntimeError` naming build; after `build()` it is `()` | the contract; `()` is the registry of a stream with no `Param` | `tests/test_blocks/test_symbols_contract.py` |
| A composite whose child sits on an out-of-range wire raises `IndexError` at `build()`, and reading `symbols` afterwards raises `RuntimeError`, never `()` | the incident | `tests/test_blocks/test_block.py` (existing test, assertion adapted) |
| Constructor-declared symbols are readable before `build()`, in canonical order | `UCCBlock`: the constructor's `symbol_qop_pairs` names sorted by `str` (§17); `HEABlock`: the pre-build tuple against the built stream's free symbols, a cross-check of two paths | `test_symbols_contract.py`, the `UCCBlock` and `HEABlock` factories |
| `set_symbols` and `replace_symbols` on an unbuilt block still queue and build to the right unitary | the analytic unitary of `Rz(θ)` at the bound value | `test_symbols_contract.py` (existing partial-bind tests) |
| `parameter_map` and `refresh_symbols` on an unbuilt block raise the same `RuntimeError` | the contract | `test_symbols_contract.py`, `test_symbols_before_build_raises_and_is_empty_after` |
| Every `FACTORIES` block: `len(block.symbols)` after `build()`, with no narrowing, type-checks and runs | mypy on `qarp/` and `tests/`; the suite | existing suite |
| The five composites still reject a built parameter-free ket with "no parameters to optimize" | existing tests | `tests/test_algorithms/test_composite/test_pce.py`, `test_vqd.py` |

## Phases

### Phase 1 — Code and tests

- [x] `_block.py`: property, setter, the private probes *(2026-10-29)*
- [x] `ucc_block.py`, `trotter_block.py` probes; composite messages; the
      `assert`s *(2026-10-29)*
- [x] The three new tests; the five existing `is None` / `is not None`
      assertions in `tests/` adapted *(2026-10-29)*
- [x] `ruff`, `mypy`, `pytest` green *(2026-10-29)*

### Phase 2 — Docs and records

- [x] §17 sentence *(2026-10-29)*
- [x] `CHANGELOG.md`: Changed, with the `is_built` migration *(2026-10-29)*
- [x] `to_do.md`: the `symbols` clause leaves the *typed for its users* entry *(2026-10-29)*

## Deviations log

- **Eight dead narrowings, not two.**  `vqa.py`, `ssvqe.py`, `pce.py` (twice)
  and `vqd.py` carry the same `assert symbols is not None` as `adapt_vqe.py`
  and `projected_vqe.py`, and `VFF._all_symbols` asserts both its registries
  are not `None`; all eight go.
- **A second probe in `trotter_block.py`.**  `TrotterAnsatzBlock.replace_symbols`
  truth-tests the copy's registry before renaming its pairs; it reads
  `_symbols` with the other private probes.
- **`parameter_map` reads the property.**  Its own `None` check raised on the
  condition the property now raises on, so the duplicate went.
- **One stub snippet builds first.**  `tests/test_stubs.py` ran
  `SimpleBlock(1).h(0).symbols` to check that a builder returns its block;
  the chain is `SimpleBlock(1).h(0).build().symbols` now, which checks the
  same thing on a block that has a registry.
- **The test table states what the tests check.**  The `HEABlock` half of the
  constructor-declared row is a cross-check of the pre-build tuple against the
  built stream's free symbols, not an independent oracle; the `UCCBlock` half
  checks the constructor's `symbol_qop_pairs` names sorted by `str`.
- **One docs snippet, out of the green-lit Scope.**  The `n_parameters`
  example in `docs/source/qarp_philosophy.rst` read `len(block.symbols or ())`;
  it reads `len(block.symbols)` now, and the file joins Scope.
