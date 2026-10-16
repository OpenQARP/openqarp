# Ship type stubs for the compiled backend and type-check `qarp` against them

**Status:** Planned
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-15
**Tier:** Structural
**Branch:** improvement/compiled-backend-stubs
**Green-lit:**
**Scope:** measured at `develop` `4093ad5` with the stub as designed (`-P`); anything the
re-measure after #13 adds is declared as a deviation.
- New: `cpp/libqarpx/python/qarpx.pyi` (generated, committed),
  `cpp/libqarpx/python/stubgen_patterns.txt`, `scripts/generate_stubs.py`,
  `tests/test_stubs.py`.
- C++: `cpp/libqarpx/python/bindings.cpp`, `bindings_operators.cpp`,
  `cpp/libqarpx/python/CMakeLists.txt`; the binding code in
  `cpp/libqarpx/src/emit/{qiskit,pytket,pennylane,qulacs}_emitter.cpp` and
  `cpp/libqarpx/src/absorb/{qiskit,pytket,pennylane,qulacs}_absorber.cpp`.
- Python: `qarp/_abi.py`; `qarp/blocks/__init__.py` (exports `QarpBlock`);
  `qarp/engines/_engine.py` (`Engine.build` takes a `Sequence`);
  `qarp/blocks/_primitives/upccd_block.py`; the files listed under *Phase 2*.
- Tests: `tests/test_blocks/test_block.py`, `test_bridge_cleanup.py`,
  `test_pauli_exp_builder.py`, `test_structure.py`;
  `tests/test_blocks/test_primitives/test_swap_test_block.py`, `test_hadamard_test_block.py`,
  `test_qpe_block.py`, `test_dos_qpe_block.py`, `test_amplitude_amplification_block.py`,
  `test_amplitude_estimation_block.py`, `test_orbital_rotation_block.py`;
  `tests/test_algorithms/test_composite/test_vff.py`;
  `tests/test_algorithms/test_utils.py` (new, if no file holds `_utils` tests);
  `tests/test_resources/test_estimator.py`; `tests/test_operators/test_qubit_operator.py`,
  `test_fermion_operator.py`; and the test files listed under *Phase 2*.
- Config and CI: `mypy.ini`, `pyproject.toml` (`nanobind` in `[dev]` and `[full-dev]`, ruff
  `extend-exclude` and `force-exclude`), `.gitattributes`, `.github/workflows/ci.yml`.
- Docs: `docs/contracts/qarp_conventions.md` (§13, §15), `AGENTS.md` (`CLAUDE.md` is a
  symlink to it), `docs/source/qarp_philosophy.rst` (`QarpBlock` beside `AnyBlock`),
  `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/contributions/README.md` (index
  row, roadmap), `docs/contributions/to_do.md`.

---

## Why

`qarpx` ships no type information, so every `qx.` expression in `qarp` is `Any`.  `mypy.ini`
silences the missing module (`ignore_missing_imports` for `qarpx`), and the CI mypy job passes
while checking nothing that crosses the bindings: 558 lines with a `qx.` reference in 63
`qarp` modules, and 146 test modules importing `qarpx`.

With a stub generated as designed (nanobind 3.0.1's `stubgen`, private members included),
mypy reports 212 errors in 70 files at `4093ad5`, against none today: 145 in 54 `qarp`
files, 47 in 15 test files and 20 in the stub itself.  Behind them, and behind what the review
of this plan found, are bugs that run today, each reproduced:

- Planning a raw `qx.ControlledBlock` reads `n_controls`, which only the Python wrapper has
  (`qarp/_program.py`).  Once the planner reaches it, at 3 or more gates in the span, it
  raises `AttributeError`.
- `ResourceEstimator.estimate` reads `is_built` without calling it, so an unbuilt raw block
  counts as built and `flatten()` raises.  On a raw block `is_built` is a method; on the
  Python blocks it is a property.
- C++ `build()` returns `None` and the Python blocks return `self`.  `SWAPTestBlock`,
  `HadamardTestBlock`, `QPEBlock`, `DOSQPEBlock`, `generate_states_new_basis` and `VFF` use
  the return value, so each crashes on a raw block.
- Variational algorithms read `.symbols` from a parameter typed `AnyBlock`, which admits a raw
  block, and `VQE(op, raw_block)` raises `AttributeError`.
- A `ConditionalBlock` with only an `else` branch cannot be deep-copied: the constructor
  requires `then_body`.
- `AmplitudeAmplificationBlock` and `AmplitudeEstimationBlock` store their input as
  `self.state_preparation`, and `OrbitalRotationBlock` stores `self.u`.  Each hides the gate
  method of that name, so calling the gate raises `TypeError`.
- `qx.Command(GateType.Rz, 0, 1)` reads the `1` as a second qubit and drops the angle; the
  error surfaces only when the command runs.

The other four `truthy-function` sites (`grover.py`, `amplitude_estimation.py`,
`amplitude_amplification.py`, `haar_random_block.py`) only ever hold Python blocks, and the
`adapt_vqd.py` and `montecarlo.py` errors are inference artefacts.  They are fixed as typing,
without regression tests.

The stubs also cover what a user sees: `import qarpx as qx` in an editor shows no signatures
today.

`mypy.ini` names this work as the precondition for its empty `[mypy]` section; its figure,
156 errors in 49 files, is stale.  Enabling `check_untyped_defs` with the stub in place
reports 2162 errors in 273 files: 329 in 70 `qarp/` files and 1813 in 202 test files.  Test
bodies run on every CI pass, so a wrong call there fails anyway; library branches the tests
miss are checked by nothing.  This plan turns the flag on for `qarp/`; `tests/` follows in its
own plan.

## Design

**Generation.**  `nanobind.stubgen` writes the stub from the built module, with private
members (`-P`).  `scripts/generate_stubs.py` checks the ABI and source-dir stamps
(`qarp._abi`) first, so a shared venv cannot write another checkout's surface, then writes
`cpp/libqarpx/python/qarpx.pyi`.  The generator is the `nanobind` Python package, pinned in
both `[dev]` and `[full-dev]` (CI and `AGENTS.md`'s install use `[full-dev]`) to the version
CMake fetches.  A test checks that every extra listing it pins the CMake `GIT_TAG`, since
stubgen's output changes between releases.

**The stub is committed, and guarded.**  A binding change then shows its type change in the
same diff.  A pytest test regenerates the stub and compares it with the committed file:

- It runs under Python 3.12, the CI version the stub is generated with, and skips elsewhere
  with that reason, since stubgen branches on the Python version.  Where `CI` is set, a
  missing generator fails the test instead of skipping it.
- The stub must not depend on the build.  `__source_dir__` (the configuring checkout's path)
  is replaced by a pattern entry, `__source_dir__: str`.  `substitute` and `free_symbols`,
  compiled only with SymEngine, are bound unconditionally and raise without it, the pattern
  `bindings_operators.cpp`'s header already describes.
- `.gitattributes` pins `qarpx.pyi` to `eol=lf`.  Ruff excludes the stub with
  `force-exclude = true`, so a direct `ruff check` on the file skips it too.

The pattern file holds what nanobind cannot express from C++ and values that vary by build,
and each pattern says why.

**Precision lives in the bindings.**  Where the stub is wrong or vague, the binding changes so
that stubgen emits the right type:

- Argument names everywhere stubgen emits `arg0`, using the Python overrides' names (`q`,
  `angle`, …), so raw and Python blocks take the same keywords.
- `Param | float` wherever `nb::implicitly_convertible` accepts a number.  stubgen ignores
  implicit conversions, so each gate overload gets an `nb::sig`, generated by one helper per
  gate shape (one qubit and an angle, two qubits and an angle, the bulk list form) so each
  signature text is written once.  In `Command.__init__` the integer overloads come first.
- Return types: the 13 bindings that return `object` (`QubitOperator.terms`, the absorbers'
  `absorb()`, …) get their real type, and nullable returns say `| None`
  (`ConditionalBlock.then_body`, `else_body`).
- `build()` returns the block, as the Python blocks do, and its signature says
  `typing.Self`.  This fixes the six crashes above and the mixin's `build` conflict.
- `is_built` becomes a read-only property, as on the Python blocks.  The same commit changes
  `_block.py`'s read of the C++ flag to `self._cpp_base.is_built.fget(self)`, and the five
  `.is_built()` calls in tests (`test_block.py` ×3, `test_pauli_exp_builder.py`, and a
  subprocess source string in `test_bridge_cleanup.py`).  The property canary in
  `test_block.py` gains `is_built`.
- `sparse_matrix` is bound in C++ on `QubitOperator` and `FermionOperator`: it calls the
  existing `qubit_operator_coo` / `fermion_operator_coo` and builds the `scipy.sparse` matrix
  through a call-time import (scipy is a core dependency).  The assignments in
  `_qubit_operator.py` and `_fermion_operator.py` go, and the method exists on a bare
  `import qarpx`.
- The `ConditionalBlock` constructor accepts `then_body=None`, so an else-only conditional can
  be copied.
- `Command` rejects a parametric gate constructed without a parameter.
- The stub's own errors are fixed without changing behaviour.  `+=` with a symbolic operand
  falls back to `__add__` and rebinds, and operators stay unhashable; both are declared as
  they are, with `nb::sig` or a pattern entry, and a test pins both.

These are binding changes, so `QARPX_ABI_VERSION` and `EXPECTED_QARPX_ABI` move from 14 to 15
together (§15).

**Distribution.**  PEP 561 does not cover a stub beside a single-module extension: a
`qarpx.pyi` next to `qarpx.so` in site-packages is not read by mypy, and a `qarpx-stubs`
package is (both checked).  `cpp/libqarpx/python/CMakeLists.txt` installs the committed stub as
`qarpx-stubs/__init__.pyi` in the `python` component, so wheels carry it.

The repository's own mypy does not rely on that install.  A `-C editable.rebuild=true`
install lands it off mypy's path, the bootstrap scripts run no CMake install, and a plain
editable install holds the stub as it was at install time.  `mypy.ini` therefore sets
`mypy_path = cpp/libqarpx/python` and drops its `qarpx` sections.  With the stub on the path,
mypy also reports errors inside it.  The `ci.yml` mypy job, which installs the wheel, also runs
mypy on a probe file from a temporary directory, so the wheel's own stub is resolved.
`mypy.stubtest` is not used: on nanobind classes it reports 175 differences that are
artefacts (metaclass, disjoint bases, `nb_func` not being a function).

**The Python blocks** (§13).  `_attach_mixin` copies `_BlockMixin` onto each concrete class at
runtime, and nanobind refuses multiple inheritance, so the mixin is shown to mypy through a
base alias that exists only under `TYPE_CHECKING`; the runtime classes are unchanged:

```python
if TYPE_CHECKING:
    class _SimpleBlockBase(_BlockMixin, qx.SimpleBlock): ...
else:
    _SimpleBlockBase = qx.SimpleBlock


@_attach_mixin
class SimpleBlock(_SimpleBlockBase): ...
```

The split surfaces conflicts between the mixin and its C++ base in all six classes.  `build`
and `is_built` agree once the bindings change.  `set_symbols` and `replace_symbols` take `Dict`
in the mixin and `Mapping` in C++; the mixin accepts both key types at runtime, so it widens to
`Mapping`.  `target_qubits` is declared in the mixin as the binding's property type.

**Which blocks a parameter accepts** (§13).  `AnyBlock` (= `qx.Block`) keeps meaning any
block, raw `qarpx` blocks included, and annotates code that reads only C++ members: engines,
the planner, the estimator.  A new public type, `QarpBlock`, annotates parameters that read
members only the Python blocks have; about 45 sites, 27 of them reading `.symbols` in the
variational algorithms.  It is the union of the six Python block classes, so it is a real
object at runtime and works with `isinstance`:

```python
QarpBlock: TypeAlias = Union[
    SimpleBlock, CompositeBlockBase, ControlledBlock, MeasureBlock, ResetBlock, ConditionalBlock
]
```

mypy then rejects a raw block passed where Python members are read, such as
`VQE(op, raw_block)`.  At runtime such a call fails as it does today.  A raw block reaches
those entry points by wrapping, `CompositeBlock([raw_block])`, which carries its symbols
through; the absorbers already return Python blocks.

**Renamed attributes.**  The three attributes that hide gate methods get public names that do
not collide: `state_preparation` becomes `state_preparation_block` on both amplitude blocks,
and `OrbitalRotationBlock.u` becomes `rotation_matrix`.  Nothing in the repository reads them
from outside.  Names to confirm: see *Open decisions*.

**Library strictness.**  `mypy.ini` gains `[mypy-qarp.*]` with `check_untyped_defs = True`;
`[mypy]` stays empty, and `tests/` keeps today's setting.  The flag adds 184 errors in
`qarp/`.  Two patterns make up about 100 of the errors the binding and mixin fixes do not
clear: callbacks that carry iteration state as function attributes
(`callback.previous_energy`, `.iter`), and `Optional` values used without a `None` check.
Each pattern is fixed one way across its files.  `Engine.build` takes a `Sequence`, since a
`list` parameter is invariant.  The `# type: ignore` comments the stub makes unused, 53 of
the 77 in `qarp/`, are removed.

`mypy.ini`'s note says not to add strictness flags piecemeal.  It is replaced by the rule
this plan follows: a strictness flag is switched on for a whole tree at once, and the note
records the count measured when it was.

**Who sees a change.**  `qarp` has no `py.typed`, so code that only imports `qarp` sees no
typing change.  Code that imports `qarpx` and runs mypy does: a user's
`ignore_missing_imports` for `qarpx` no longer applies once the stub is found, and calls that
run can be rejected (an `np.int64` qubit index, attributes set through `nb::dynamic_attr`).
The opt-out that works is `follow_imports = skip` with `follow_imports_for_stubs = True`.
Runtime changes: `is_built` is a property, `build()` returns the block, three block
attributes are renamed, and the ABI is 15.  The CHANGELOG lists each under Changed with a
one-line migration.  The README roadmap moves this work from the window with no public surface
change into the 0.2 cycle.

**Conventions.**  §15's ABI bullet gains one rule: a commit that changes a binding also
regenerates `qarpx.pyi`.  §13 records `is_built` as a property on every block and defines
`QarpBlock` beside `AnyBlock`: which of the two a parameter takes depends on whether it reads
members only the Python blocks have.  Both edits come first, before the code they govern.

**No example.**  The user-visible result is what an editor and a type checker show, which a
notebook cannot exercise.  The runtime changes are fixes, each with a regression test.

### Decisions

- **The `is_built` entry from `to_do.md` is folded in.**  The stubs surface it as
  `truthy-function` errors, it needs the same ABI bump, and leaving it would type a method
  that every Python block overrides with a property.  *(2026-10-15)*
- **The stub ships in the wheel.**  Users get signatures in their editors; the price is a
  public typing surface for `qarpx` that changes with each ABI bump.  *(2026-10-15)*
- **Based on `develop`.**  #13 changes no C++, PRs land squashed, and a branch based on #13
  would carry its diff in this PR until #13 lands.  *(2026-10-15)*
- **Private members are in the stub** (`-P`).  `qarp` calls ten underscored bindings (the
  command digests, `_permutation_table`, `_local_unitary`, the thread-budget functions,
  `Block._copy_base_state_from`), and those are the calls most worth checking.  The
  underscore keeps them private; `dir(qx)` already lists them.  *(2026-10-15)*
- **`check_untyped_defs` is on for `qarp/` in this plan; `tests/` gets its own.**  The
  library is where unchecked bodies hide bugs; the tests' 1813 errors would touch nearly
  every test file and are parked in `to_do.md`, for after the layout move.  *(2026-10-15)*
- **This plan lands after #13.**  The `is_built` entry it removes from `to_do.md` arrives
  with #13, and #13 edits 18 files in this Scope, among them `_program.py`, `_qarp_engine.py`,
  `cutting_primitive.py`, `test_structure.py`, `CHANGELOG.md` and `to_do.md`.  Once #13 lands,
  `develop` is merged in, the Phase 2 lists are measured again with the difference declared,
  and this header follows #13's plan template.  *(2026-10-16)*
- **Sit-down held.**  *(2026-10-16)*
- **`sparse_matrix` is bound in C++** rather than declared in the stub, so the stub never
  advertises a method a bare `import qarpx` lacks.  *(2026-10-16)*
- **The shadowing attributes are renamed to public names**, not made private.
  *(2026-10-16)*
- **`AnyBlock` keeps meaning any block; a new type covers the Python blocks.**  Making every
  site work on raw blocks would be a feature, with about 45 regression tests, and sites such
  as `mark_built` and `_publish_symbols` only make sense on Python blocks anyway.
  *(2026-10-16)*

### Open decisions

To settle before the green-light; each moves to *Decisions* when it is.

1. **Names**: `QarpBlock` for the new type; `state_preparation_block` and `rotation_matrix`
   for the renamed attributes.

## API sketch

```text
# Wheel
site-packages/qarpx.cpython-*.so
site-packages/qarpx-stubs/__init__.pyi      # from cpp/libqarpx/python/qarpx.pyi

# Regenerate after a binding change (Python 3.12)
python scripts/generate_stubs.py            # writes cpp/libqarpx/python/qarpx.pyi
```

```python
# qarpx.pyi (excerpts, as the changed bindings make stubgen emit them)
__abi_version__: int = 15
__source_dir__: str

class Block:
    @property
    def is_built(self) -> bool: ...
    def build(self) -> Self: ...
    def rz(self, q: int, angle: Param | float) -> Block: ...

class ConditionalBlock(Block):
    def __init__(self, cbits: Sequence[int], values: Sequence[bool], then_body: Block | None,
                 else_body: Block | None = None, name: str = "Conditional") -> None: ...
    @property
    def then_body(self) -> Block | None: ...

class QubitOperator:
    def sparse_matrix(self, n_qubits: int | None = None) -> scipy.sparse.csc_matrix: ...
    __hash__: None
```

```ini
# mypy.ini
[mypy]
mypy_path = cpp/libqarpx/python

[mypy-qarp.*]
check_untyped_defs = True
```

Argument names and order in the sketch follow the current bindings and Python overrides;
the stub records whatever the bindings end up saying, and the drift test keeps the two
together.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| The committed stub equals a fresh stubgen run, under Python 3.12 | the built module, read by the pinned generator | `tests/test_stubs.py` |
| Every extra that lists `nanobind` pins the CMake `GIT_TAG` | `cpp/libqarpx/CMakeLists.txt` | `tests/test_stubs.py` |
| Every `qx` call mypy accepts in a table also runs (`rz(0, 0.5)`, `rz([(0, 0.5)])`, keyword arguments, `ConditionalBlock` with no `then_body`, …); the known false rejections (an `np.int64` qubit, `nb::dynamic_attr` attributes) are listed in the test | the runtime: each accepted call is run | `tests/test_stubs.py` |
| `qarp/` type-checks with the stub and `check_untyped_defs`, `tests/` with the stub, and the stub itself has no errors | mypy over the code, with the stub on `mypy_path` | CI `mypy` job |
| mypy run from a temporary directory against the installed wheel reveals `qx.SimpleBlock(1).n_qubits` as `int` | the binding's C++ type | `ci.yml` `mypy` job |
| `is_built` is `False` on a fresh raw block and `True` after `build()`; `build()` returns the same object | the build lifecycle | `tests/test_blocks/test_block.py` |
| Each of the six Python block classes keeps its runtime MRO: `_BlockMixin` is not in it, and its second entry is the `qx` class | the runtime class | `tests/test_blocks/test_block.py` |
| Planning a raw `qx.ControlledBlock` over three X gates (`_program.plan(block, 12)`) gives CX | the analytic CX matrix | `tests/test_blocks/test_structure.py` |
| `ResourceEstimator.estimate` on an unbuilt raw composite of H then CX counts one H and one CX | counted by hand | `tests/test_resources/test_estimator.py` |
| `SWAPTestBlock` on two raw blocks preparing \|0⟩ measures ancilla 0 with probability 1 | `(1 + \|⟨a\|b⟩\|²)/2` | `test_swap_test_block.py` |
| `HadamardTestBlock` with a raw Z on \|0⟩ gives ⟨Z⟩ = 1 | analytic | `test_hadamard_test_block.py` |
| `QPEBlock` with a raw X as eigenstate and a raw P(π/2) as unitary, 2 ancillas, reads phase 1/4 | analytic eigenphase | `test_qpe_block.py` |
| `DOSQPEBlock` over a raw Z on one qubit gives eigenphases 0 and 1/2 with weight 1/2 each | analytic spectrum of Z | `test_dos_qpe_block.py` |
| `generate_states_new_basis` with a raw X on one qubit gives \|1⟩ and \|0⟩ | analytic | `tests/test_algorithms/test_utils.py` |
| `VFF` built over a raw H holds, as the unitary it diagonalises, the Hadamard matrix | analytic H | `test_vff.py` |
| A deep copy of an else-only conditional flattens to a conditional X | the hand-written command | `tests/test_blocks/test_block.py` |
| `state_preparation(...)` and `u(...)` on the three renamed blocks add the gate's command | the gate's command, written by hand | each block's test file |
| `Command(GateType.Rz, 0, 1)` raises at construction; `Command(GateType.Rz, 0, Param(1.0))` builds | Rz takes one parameter | `tests/test_blocks/test_block.py` |
| `sparse_matrix` on `QubitOperator("Z0")` in a subprocess that imports only `qarpx` equals diag(1, −1) | the Pauli Z matrix | `test_qubit_operator.py`, `test_fermion_operator.py` (n̂ = diag(0, 1)) |
| `op += Symbol("a")` rebinds `op`, and `hash(op)` raises `TypeError` | today's runtime behaviour | `test_qubit_operator.py` |
| mypy rejects `VQE(op, raw_block)` and accepts `VQE(op, CompositeBlock([raw_block]))`; the first raises at runtime and the second runs | the runtime: both calls are run | `tests/test_stubs.py` |

## Phases

The phases are review stages inside one PR.  CI stays green at each: the stub becomes visible
to mypy, through `mypy_path` and the CMake install, only at the end of Phase 2.

### Phase 1 — Conventions, bindings and the stub

- [ ] §13 and §15 edits
- [ ] `nanobind` pinned in `[dev]` and `[full-dev]`; `scripts/generate_stubs.py`; pattern file;
      drift and pin tests; `.gitattributes`; ruff exclusion
- [ ] Bindings: argument names, `Param | float`, return types, `build()` returning the block,
      `is_built` as a property, `sparse_matrix`, nullable `then_body`, the `Command` check,
      SymEngine methods bound unconditionally; the stub's own errors gone
- [ ] Same commit as `is_built`: `_block.py`'s `fget` read, the five test call sites, the canary
- [ ] ABI 14 → 15 in `bindings.cpp` and `qarp/_abi.py`
- [ ] Runtime-agreement table and the binding regression tests from the Test plan

### Phase 2 — Type-check `qarp` against the stub

- [ ] The `TYPE_CHECKING` base aliases for the six classes; the mixin conflicts resolved
- [ ] `QarpBlock` defined and exported; the parameters that read Python-only members take it
- [ ] Each flagged file fixed; the regression tests for the Python-side bugs.  Flagged at
      `4093ad5`: `qarp/_program.py`, `qarp/absorb/_reconstruct.py`;
      `qarp/algorithms/_composite/`: `adapt_vqe.py`, `amplitude_amplification.py`,
      `amplitude_estimation.py`, `grover.py`, `mmqcels.py`, `montecarlo.py`, `pce.py`,
      `qite.py`, `qmegs.py`, `ssvqe.py`, `vff.py`, `vqa.py`, `vqd.py`;
      `qarp/algorithms/_primitives/`: `pauli_averaging.py`, `shadows/estimator.py`,
      `shadows/pauli.py`, `state_vector.py`, `termwise_hadamard_test.py`;
      `qarp/algorithms/_utils.py`; `qarp/blocks/_block.py`; `qarp/blocks/_primitives/`:
      `amplitude_amplification_block.py`, `amplitude_estimation_block.py`,
      `block_encoding_block.py`, `dos_qpe_block.py`, `haar_random_block.py`,
      `hadamard_test_block.py`, `orbital_rotation_block.py`, `order_finding_block.py`,
      `pauli_block.py`, `phase_shift_block.py`, `projector_blocks.py`, `qpe_block.py`,
      `qrom_block.py`, `qsvt_block.py`, `reflection_block.py`, `select_block.py`,
      `swap_test_block.py`, `synthesized_time_evolution_block.py`, `trotter_block.py`,
      `ucc_block.py`; `qarp/blocks/_state_preparation/slater_determinant_block.py`;
      `qarp/cutting/_qpd_decomposition.py`; `qarp/engines/_qarp_engine.py`;
      `qarp/factories/_pauli_block_factory.py`; `qarp/operators/`:
      `_basis_rotation_grouping.py`, `_fermion_operator.py`, `_mappings.py`, `_qdrift.py`,
      `_qubit_operator.py`; `qarp/resources/_estimator.py`, `qarp/resources/_synthesis.py`;
      `qarp/utils/_random_instances.py`; `qarp/plotting/_circuit_adapter.py` and
      `qarp/algorithms/_primitives/cutting_primitive.py` (`is_built` reads mypy cannot see);
      `tests/test_absorb/conftest.py`,
      `tests/test_algorithms/test_primitives/test_termwise_hadamard_test.py`,
      `tests/test_blocks/test_interop_methods.py`,
      `tests/test_blocks/test_primitives/test_block_encoding_block.py`,
      `tests/test_blocks/test_primitives/test_trotter_block.py`,
      `tests/test_blocks/test_primitives/test_upccd_block.py`,
      `tests/test_blocks/test_structure.py`, `tests/test_cutting/test_qpd_decomposition.py`,
      `tests/test_emit/conftest.py`, `tests/test_emit/test_capability_contract.py`,
      `tests/test_engines/test_gradient_shift_rules.py`,
      `tests/test_engines/test_structured_execution.py`, `tests/test_operators/test_qdrift.py`,
      `tests/test_resources/test_modelers.py`, `tests/test_resources/test_synthesis.py`
- [ ] `check_untyped_defs` on for `qarp/`; the files above and, flagged only with it at
      `4093ad5`: `qarp/absorb/__init__.py`; `qarp/algorithms/_composite/`: `adapt_vqd.py`,
      `dos_qpe.py`, `projected_vqe.py`, `qpe.py`, `qse.py`;
      `qarp/blocks/_primitives/spa_block.py`,
      `qarp/blocks/_state_preparation/cvo_qram_state_block.py`,
      `qarp/cutting/_post_processing.py`, `qarp/devices/__init__.py`,
      `qarp/graphs/_graph.py`, `qarp/graphs/_simplicial_complex.py`,
      `qarp/operators/compat.py`, `qarp/operators/functions.py`,
      `qarp/optimizers/_gradient_descent_optimizer.py`, `qarp/plotting/_circuit_plotter.py`
- [ ] Unused `# type: ignore` comments removed
- [ ] Last: `mypy.ini` gets `mypy_path` and drops its `qarpx` sections, its note carries the
      new rule and numbers; CMake installs `qarpx-stubs/__init__.pyi`; the `ci.yml` wheel probe
- [ ] After #13 lands: `develop` merged in, the lists measured again, the difference declared

### Phase 3 — Docs

- [ ] `AGENTS.md` and `CONTRIBUTING.md`: the regenerate command beside the mypy command,
      `[dev]`'s description, and `QarpBlock` beside `AnyBlock` in `AGENTS.md`'s landmines
- [ ] `qarp_philosophy.rst`: `QarpBlock` beside `AnyBlock`
- [ ] `CHANGELOG.md`: Added (typed `qarpx`), Changed (each runtime change with its migration),
      Fixed (each bug in *Why*)
- [ ] `docs/contributions/README.md`: the roadmap moves this work into the 0.2 cycle
- [ ] `to_do.md`: the `is_built` entry leaves; the `tests/` strictness entry stays

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
