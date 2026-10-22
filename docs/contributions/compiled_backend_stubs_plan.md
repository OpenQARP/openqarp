# Ship type stubs for the compiled backend and type-check `qarp` against them

**Status:** Landed
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-15
**Tier:** Structural
**Branch:** improvement/compiled-backend-stubs
**Green-lit:** 2026-10-16, plan blob d01807223f141e4e496a998cc803fe37129574b2
**Scope:** measured at `develop` `4093ad5` with the stub as designed (`-P`), plus the three
test files #13 adds; anything the re-measure after #13 adds is declared as a deviation.
- New: `cpp/libqarpx/python/qarpx.pyi` (generated, committed),
  `cpp/libqarpx/python/stubgen_patterns.txt`, `scripts/generate_stubs.py`,
  `tests/test_stubs.py`.
- C++: `cpp/libqarpx/python/bindings.cpp`, `bindings_operators.cpp`,
  `cpp/libqarpx/python/CMakeLists.txt`; the binding code in
  `cpp/libqarpx/src/emit/{qiskit,pytket,pennylane,qulacs}_emitter.cpp` and
  `cpp/libqarpx/src/absorb/{qiskit,pytket,pennylane,qulacs}_absorber.cpp`;
  `cpp/libqarpx/include/qarpx/core/errors.h`, `cpp/libqarpx/include/qarpx/block/block.h`,
  `composite_block.h`, `cpp/libqarpx/src/block/block.cpp`, `composite_block.cpp`
  (composites take no gates); `cpp/libqarpx/tests/cpp/test_core.cpp`.
- Python: `qarp/_abi.py`; `qarp/blocks/__init__.py` (imports `QarpBlock`);
  `qarp/engines/_engine.py` (`Engine.build` takes a `Sequence`);
  `qarp/algorithms/_composite/vqe.py`; `qarp/blocks/_primitives/upccd_block.py`; the files
  listed under *Phase 2*.
- Tests: `tests/test_blocks/test_block.py`, `test_block_refactor.py`, `test_bridge_cleanup.py`,
  `test_pauli_exp_builder.py`, `test_structure.py`;
  `tests/test_blocks/test_primitives/test_swap_test_block.py`, `test_hadamard_test_block.py`,
  `test_qpe_block.py`, `test_dos_qpe_block.py`, `test_amplitude_amplification_block.py`,
  `test_amplitude_estimation_block.py`, `test_orbital_rotation_block.py`;
  `tests/test_algorithms/test_composite/test_vff.py`; `tests/test_algorithms/test_utils.py`;
  `tests/test_algorithms/test_primitives/test_cutting_primitive.py`;
  `tests/test_plotting/test_circuit_adapter_residuals.py`;
  `tests/test_resources/test_estimator.py`; `tests/test_operators/test_qubit_operator.py`,
  `test_fermion_operator.py`, `test_tensor_operators.py`, `test_symbolic_coefficients.py`;
  #13's `test_lazy_circuits.py`, `test_unbuilt_planning.py`, `test_compiled_circuits.py`;
  and the test files listed under *Phase 2*.
- Config and CI: `mypy.ini`, `pyproject.toml` (`nanobind` in `[dev]` and `[full-dev]`, ruff
  `extend-exclude` and `force-exclude`, `[tool.pyright] stubPath`), `.gitattributes`,
  `.github/workflows/ci.yml`.
- Docs: `docs/contracts/qarp_conventions.md` (§13, §15), `AGENTS.md` (`CLAUDE.md` is a
  symlink to it), `docs/source/qarp_philosophy.rst` (`QarpBlock` beside `AnyBlock`),
  `docs/source/installation.rst` (`[dev]`'s description), `docs/api/qarp.blocks.rst`,
  `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/contributions/README.md` (index row, roadmap),
  `docs/contributions/to_do.md`.

---

## Why

`qarpx` ships no type information, so every `qx.` expression in `qarp` is `Any`.  `mypy.ini`
silences the missing module (`ignore_missing_imports` for `qarpx`), and the CI mypy job passes
while checking nothing that crosses the bindings: 558 lines with a `qx.` reference in 72
`qarp` modules (63 of which import `qarpx`), and 146 test modules importing `qarpx`.

With a stub generated as designed (nanobind 3.0.1's `stubgen`, private members included),
mypy reports 212 errors in 70 files at `4093ad5`, against none today: 145 in 54 `qarp`
files, 47 in 15 test files and 20 in the stub itself.  Behind them, and behind what the
reviews of this plan found, are bugs that run today, each reproduced:

- Planning a raw `qx.ControlledBlock` reads `n_controls`, which only the Python wrapper has
  (`qarp/_program.py`).  Once the planner reaches it, at 3 or more gates in the span, it
  raises `AttributeError`.
- `ResourceEstimator.estimate`, `CircuitAdapter` and `CuttingPrimitive.build` read
  `is_built` without calling it, so an unbuilt raw block counts as built and `flatten()`
  raises.  On a raw block `is_built` is a method; on the Python blocks it is a property.
- C++ `build()` returns `None` and the Python blocks return `self`.  `SWAPTestBlock`,
  `HadamardTestBlock`, `QPEBlock`, `DOSQPEBlock`, `generate_states_new_basis`, `VFF` and
  `CuttingPrimitive` use the return value, so each crashes on a raw block.
- Variational algorithms read `.symbols` from a parameter typed `AnyBlock`, which admits a raw
  block, and `VQE(op, raw_block)` raises `AttributeError`.
- A `ConditionalBlock` with only an `else` branch cannot be deep-copied: the constructor
  requires `then_body`.
- `AmplitudeAmplificationBlock` and `AmplitudeEstimationBlock` store their input as
  `self.state_preparation`, and `OrbitalRotationBlock` stores `self.u`.  Each hides the gate
  method of that name, so calling the gate raises `TypeError`.
- A gate method called on a composite block records the gate in a buffer that nothing
  executes: `comp.z(0)` is listed by `commands()` and absent from `flatten()`, silently.
- `qx.Command` picks its overload by argument type, not by the gate's arity.
  `Command(Rz, 0, 1)` reads the `1` as a second qubit and drops the angle, and
  `Command(CX, 0, np.int64(1))` builds a one-qubit CX carrying angle 1, which runs and gives
  the wrong state.  `Command(H, 0, 0.5)` is accepted too.
- `set_symbols` and `replace_symbols` with string keys substitute in the gate stream but
  leave the canonical `.symbols` tuple (§17) unchanged, so `parameter_map` counts a bound
  symbol as free.

The other four `truthy-function` sites (`grover.py`, `amplitude_estimation.py`,
`amplitude_amplification.py`, `haar_random_block.py`) only ever hold Python blocks, and the
`adapt_vqd.py` and `montecarlo.py` errors are inference artefacts.  They are fixed as typing,
without regression tests.

The stubs also cover what a user sees: `import qarpx as qx` in an editor shows no signatures
today.

`mypy.ini` names this work as the precondition for its empty `[mypy]` section; its figure,
156 errors in 49 files, is stale.  Enabling `check_untyped_defs` with the stub in place
reports 2161 errors in 272 files on Linux: 329 in 70 `qarp/` files, 1812 in 201 test files
and the stub's 20.  macOS adds one (`os.sched_getaffinity` in `test_blas_threads.py`).  Test
bodies run on every CI pass, so a wrong call there fails anyway; library branches the tests
miss are checked by nothing.  This plan turns the flag on for `qarp/`; `tests/` follows in its
own plan.

## Design

**Generation.**  `nanobind.stubgen` writes the stub from the built module, with private
members (`-P`).  The generator is the `nanobind` Python package, pinned in both `[dev]` and
`[full-dev]` (CI and `AGENTS.md`'s install use `[full-dev]`) to the version CMake fetches.  A
test checks that every extra listing it pins the CMake `GIT_TAG`, since stubgen's output
changes between releases.

`scripts/generate_stubs.py` refuses to write unless:

- the imported `qarpx` carries this checkout's ABI and source-dir stamps, checked with the
  source tree passed explicitly and `QARP_SKIP_ABI_CHECK` ignored, so a shared venv or a
  site-packages `qarp` cannot write another checkout's surface;
- it runs under Python 3.12, CI's `PYTHON_VERSION`;
- the installed `nanobind` equals the CMake `GIT_TAG`;
- every pattern in the pattern file matches something.

It writes `cpp/libqarpx/python/qarpx.pyi` with `\n` line endings.  `--check` compares
instead of writing and exits non-zero on a difference.

**The stub is committed, and guarded.**  A binding change then shows its type change in the
same diff.  A pytest test runs `generate_stubs.py --check` in a subprocess:

- Outside Python 3.12 it skips with that reason, since stubgen branches on the Python
  version.  Where `CI` is set, a Python mismatch or a missing generator fails the test
  instead of skipping it, so moving CI's Python cannot silently turn the guard off.
- The stub must not depend on the build.  `__source_dir__` (the configuring checkout's path)
  is replaced by a pattern entry, `__source_dir__: str`.  `substitute` and `free_symbols`,
  compiled only with SymEngine, are bound unconditionally and raise without it, the pattern
  `bindings_operators.cpp`'s header already describes.  Code that detected SymEngine by the
  methods' presence reads a new flag instead, `qx.__with_symengine__`, typed `bool` through
  a pattern entry since its value varies by build; `test_symbolic_coefficients.py` switches
  to it in the same commit.
- `.gitattributes` pins `qarpx.pyi` to `eol=lf`.  Ruff excludes the stub with
  `force-exclude = true`, so a direct `ruff check` on the file skips it too.

The pattern file holds what nanobind cannot express from C++ and values that vary by build.
Each pattern is anchored (`^…$`) and says why.

**Precision lives in the bindings.**  Where the stub is wrong or vague, the binding changes so
that stubgen emits the right type:

- Argument names everywhere stubgen emits `arg0`, using the Python overrides' names (`q`,
  `angle`, …), so raw and Python blocks take the same keywords.  One helper emits both the
  `nb::arg` list and the `nb::sig` text from a single list of names, so the displayed
  signature and the runtime keywords cannot disagree.
- `Param | float` wherever `nb::implicitly_convertible` accepts a number: the gate methods,
  `u`, `cu`, `gphase`, `pauli_exp`, `commuting_pauli_set_exp`, the `Command` constructor
  forms, `Param` arithmetic, `approx_equal` and `affine_coefficients`.  stubgen ignores
  implicit conversions, so each overload gets an `nb::sig` from the helper above, one per
  shape, so each signature text is written once.
- Return types: of the 13 bindings that return `object`, `QubitOperator.terms`,
  `.coefficient` (both operator classes) and `Block.__deepcopy__` get their real types.  The
  raw absorbers' `absorb()` returns `tuple[list[Command], int, int]`, the commands and
  register sizes the Python absorbers build the block from.  The emitters' `emit()` return
  and the absorbers' framework argument are typed `Any`, so the stub imports no optional
  framework.  Nullable returns say `| None` (`ConditionalBlock.then_body`, `else_body`).
- `build()` returns the block, as the Python blocks do, and its signature says
  `typing.Self`.  This fixes the seven crashes above and the mixin's `build` conflict.
- `is_built` becomes a read-only property, as on the Python blocks.  The same commit changes
  `_block.py`'s read of the C++ flag to `self._cpp_base.is_built.fget(self)`, and the five
  `.is_built()` calls in tests (`test_block.py` ×3, `test_pauli_exp_builder.py`, and a
  subprocess source string in `test_bridge_cleanup.py`).  The property canary in
  `test_block.py` gains `is_built`.
- `sparse_matrix` is bound in C++ on `QubitOperator` and `FermionOperator`: it calls the
  existing `qubit_operator_coo` / `fermion_operator_coo` and builds the same
  `scipy.sparse.csc_matrix` (complex128, LSB) through a call-time import (scipy is a core
  dependency).  `n_qubits` keeps today's meaning: `None` or a negative value infers it, and
  too small a value raises `ValueError`.  A non-integer now raises `TypeError` instead of
  `RuntimeError: std::bad_cast`.  The docstring is ported.  The assignments in
  `_qubit_operator.py` and `_fermion_operator.py` go, and the method exists on a bare
  `import qarpx`.  As a nanobind method it can no longer be pickled or read by
  `inspect.signature`, like every other `qarpx` method.
- `ConditionalBlock` takes two constructor overloads, in the binding and in the Python
  class: `then_body` with an optional `else_body`, or `then_body=None` with `else_body`
  required.  An else-only conditional can then be copied, and a call with neither body is
  rejected by mypy as it is by the constructor.
- `Command`'s Python constructor forms check, for every gate with a fixed qubit count, the
  qubit count against `gate_num_qubits` and the parameter count against `gate_num_params`
  (`core/gates.h`), and raise `ValueError`.  The integer overloads are registered first.  That
  changes runtime dispatch: a numpy integer or a `bool` in the second position is now read as
  a qubit, so `Command(CX, 0, np.int64(1))` builds the CX, and `Command(Rz, 0, np.int64(1))`
  and `Command(Rz, 0, True)`, which built an Rz with angle 1, now raise.
- `Command`'s duplicate `__eq__`/`__ne__` lambdas go; the `nb::self` operators remain, so
  `Command() == 5` returns `False`, as on every other binding, instead of raising.
- The stub's own 20 errors, each fixed:

  | Errors | Where | Fix |
  |---|---|---|
  | 10 `[override]` | `__eq__`/`__ne__` on `QubitOperator`, `FermionOperator`, `Param`, `Block`, `Command` | `nb::sig` declares `other: object`; each returns `False` for a foreign type at runtime (`Command` after the change above) |
  | 2 `[overload-cannot-match]` | `Command.__eq__`/`__ne__` bound twice | the duplicate lambdas removed |
  | 6 `[misc]` | `__iadd__`/`__isub__`/`__imul__` on both operator classes | a pattern entry adds the overload `__add__` (resp. `__sub__`, `__mul__`) has and the in-place form lacks, since a symbolic operand falls back to `__add__` and rebinds; a test pins the rebind |
  | 2 `[assignment]` | `__hash__` on both operator classes | a pattern entry, `__hash__: ClassVar[None]  # type: ignore[assignment]`, with the `ClassVar` import; `__hash__` is set as an attribute, so `nb::sig` cannot reach it |

These are binding changes, so `QARPX_ABI_VERSION` and `EXPECTED_QARPX_ABI` move from 14 to 15
together (§15).

**Blocks that hold blocks take no gates of their own.**  A gate method, or `set_commands`
with any command, called on a `CompositeBlock`, `ControlledBlock` or `ConditionalBlock` raises
`TypeError`, in C++ so raw blocks raise too: `Block`'s append path and a non-empty
`set_commands` throw a new error from `core/errors.h` when the block's own buffer never
executes (`accepts_gates()`), and the binding translates it.  Any library site that adds a
gate to such a block is a bug of the same kind and is fixed in the same commit.

**Renamed attributes.**  The three attributes that hide gate methods get public names that do
not collide: `state_preparation` becomes `state_preparation_block` on both amplitude blocks,
and `OrbitalRotationBlock.u` becomes `rotation_matrix`.  All three blocks are composites, so
after the rename the gate methods raise as above; the rename is for typing (the stub types
those names as methods).  `tests/test_operators/test_tensor_operators.py` reads `block.u` and
migrates.  There is no compatibility shim.

**Distribution.**  PEP 561 does not cover a stub beside a single-module extension: a
`qarpx.pyi` next to `qarpx.so` in site-packages is not read by mypy, and a `qarpx-stubs`
package is (both checked).  `cpp/libqarpx/python/CMakeLists.txt` installs the committed stub as
`qarpx-stubs/__init__.pyi` in the `python` component, so wheels carry it.

The repository's own mypy does not rely on that install.  A `-C editable.rebuild=true`
install lands it off mypy's path, the bootstrap scripts run no CMake install, and a plain
editable install holds the stub as it was at install time.  `mypy.ini` therefore sets
`mypy_path = $MYPY_CONFIG_FILE_DIR/cpp/libqarpx/python`, which resolves from any working
directory, and drops its `qarpx` sections.  pyright and Pylance do not read `mypy.ini`, so
`pyproject.toml` sets `[tool.pyright] stubPath = "cpp/libqarpx/python"`.  With the stub on the
path, mypy also reports errors inside it.  The `ci.yml` mypy job, which installs the wheel,
also runs mypy on a probe file from a temporary directory, so the wheel's own stub is
resolved.  `mypy.stubtest` is not used: on nanobind classes it reports about 190 differences,
all artefacts (metaclass, disjoint bases, `nb_func` not being a function).

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

The split surfaces conflicts between the mixin and its C++ base in all six classes:

- `build` and `is_built` agree once the bindings change.
- `set_symbols` and `replace_symbols` take `Dict` in the mixin and `Mapping` in C++.  The
  mixin converts string keys to `Symbol` on entry, so `.symbols` and `parameter_map` stay
  canonical, and then widens to `Mapping[Symbol | str, …]`.
- `target_qubits` is declared in the mixin as the binding's property type,
  `Optional[List[int]]`, with `None` checks at its two reads in `_block.py`.
- Four sites pass the mixin's `self` as a block (`_flatten_digest`, and `[self, other]` in
  `__mul__`, `__or__` and `_repeat`); each gets a `cast` to `QarpBlock`.

With the mixin first in the method order, its untyped gate overrides would hide the
bindings' types from every Python block.  The 15 overrides (`rz`, `ry`, …), `mcx` and `ccz` are
annotated to match the stub.  Their angles are `qx.Param | float | Expr`, and sympy ships no
types, so mypy accepts any angle on a Python block; the runtime check stays the only one there.

**Which blocks a parameter accepts** (§13).  `AnyBlock` (= `qx.Block`) keeps meaning any
block, raw `qarpx` blocks included, and annotates code that reads only C++ members: engines,
the planner, the estimator.  A new public type, `QarpBlock`, annotates parameters that read
members only the Python blocks have; about 45 sites, 27 of them reading `.symbols` in the
variational algorithms.  It is the union of the six Python block classes, written with `|` so
mypy narrows on it, and it works with `isinstance` at runtime, rejecting a raw block:

```python
QarpBlock: TypeAlias = (
    SimpleBlock | CompositeBlockBase | ControlledBlock | MeasureBlock | ResetBlock | ConditionalBlock
)
```

It is defined in `qarp/blocks/_block.py` and imported into `qarp/blocks/__init__.py`, so
the package namespace gains no `typing` names.  A `#:` doc comment, and an entry in
`docs/api/qarp.blocks.rst` if autodoc skips the alias, keeps the docs job's API inventory
check passing.

mypy then rejects a raw block passed where Python members are read, such as
`VQE(op, raw_block)`.  At runtime such a call fails as it does today.  A raw block reaches
those entry points by wrapping, `CompositeBlock([raw_block])`, which carries its symbols
through; the absorbers already return Python blocks.  So that valid calls stay accepted:

- `_BlockMixin.__mul__` and `__or__` return the Python `CompositeBlock`, and `_repeat`
  returns `QarpBlock` (a count of one returns the block itself);
  `get_final_state_block()` and ADAPT-VQE's `_ansatz_wfn()` return `QarpBlock`.
- Parameters that take several blocks take `Sequence[QarpBlock]`, since `List` is invariant
  (`VQD`'s `kets`).
- `children()` and `inner()` keep returning `Block`: a Python composite can hold raw blocks.
  Callers narrow with `isinstance(child, QarpBlock)`.
- Two sites read members of one block class, not of every Python block, and are typed with
  that class: `termwise_hadamard_test.py`'s `PauliBlock` list (`.coefficient`) and
  `qmegs.py`'s time-evolution unitary (`.operator`).

**Library strictness.**  `mypy.ini` gains `[mypy-qarp,qarp.*]` with `check_untyped_defs = True` and
`warn_unused_ignores = True`; `[mypy]` carries no strictness flags, and `tests/` keeps today's
setting.  The flag adds 184 errors in `qarp/`.  Two patterns make up about 100 of the errors
the binding and mixin fixes do not clear, and each is fixed one way across its files:

- Callbacks that carry iteration state as function attributes (`callback.previous_energy`,
  `.iter`) keep it in closure locals with `nonlocal`.
- An `Optional` value used without a `None` check gets `assert x is not None` where `None`
  cannot reach that point (the existing house form, and no new branch against the 80% patch
  gate), and a `ValueError` with a test where a caller can pass `None`.

`Engine.build` takes a `Sequence`, since a `list` parameter is invariant.  Of the 77
`# type: ignore` comments in `qarp/`, 64 are unused today and the stub makes about 11
needed; the unused ones, measured again after the fixes, are removed.

`mypy.ini`'s note says not to add strictness flags piecemeal.  It is replaced by the rule
this plan follows: a strictness flag is switched on for a whole tree at once, and the note
records the count measured when it was.

CI, `AGENTS.md` and `CONTRIBUTING.md` drop `--disable-error-code=method-assign` from the mypy
command.  With the stub, its only errors are the three attributes this plan renames, so the
flag would hide the next one.

**Who sees a change.**  `qarp` has no `py.typed`, so code that only imports `qarp` sees no
typing change, and `QarpBlock` rejecting `VQE(op, raw_block)` happens only inside this
repository.  Code that imports `qarpx` and runs mypy does see a change: a user's
`ignore_missing_imports` for `qarpx` no longer applies once the stub is found, and calls that
run can be rejected (an `np.int64` qubit index, attributes set through `nb::dynamic_attr`).
The opt-out that works is `follow_imports = skip` with `follow_imports_for_stubs = True`.
The runtime changes, each listed in the CHANGELOG with a one-line migration:

- `is_built` is a property, and the unbound `qx.Block.is_built(b)` form breaks.
- `build()` returns the block.
- Three block attributes are renamed, without a shim.
- Gate methods on a composite raise.
- `Command` dispatches a numpy integer or `bool` second argument as a qubit, and rejects
  wrong qubit or parameter counts.
- `Command == other` returns `False` for a foreign type.
- `substitute` and `free_symbols` exist on every build.
- `sparse_matrix` is a nanobind method: not picklable, and a non-integer `n_qubits` raises
  `TypeError`.
- The ABI is 15.

The README roadmap moves this work from the window with no public surface change into the
0.2 cycle.

**Conventions.**  §15 gets a bullet of its own: a commit that changes a binding also
regenerates `qarpx.pyi`, a wrong or vague type is fixed in the binding, and the pattern file
holds only what nanobind cannot express and values that vary by build.  §13 records that every block has `is_built` as a property, that
`build()` returns the block, and that a composite takes no gates of its own.  It also defines
`QarpBlock` beside `AnyBlock`: which of the two a parameter takes depends on whether it reads
members only the Python blocks have.  Both edits come first, before the code they govern.

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
  library is where unchecked bodies hide bugs; the tests' 1812 errors would touch nearly
  every test file and are parked in `to_do.md`, for after the layout move.  *(2026-10-15)*
- **This plan lands after #13, and Phase 2 starts after it.**  #13 edits 19 files in this
  Scope, among them `_engine.py`, `_qarp_engine.py`, `_program.py`, `cutting_primitive.py`,
  `test_structure.py`, `CHANGELOG.md` and `to_do.md`, and the `is_built` entry this plan
  removes from `to_do.md` arrives with it.  #13 changes no C++, so Phase 1 proceeds now.  Phase
  2 opens by merging `develop` once #13 lands; the Phase 2 lists are then measured again and
  the difference declared.  Until that merge, this header and the index row follow
  `develop`'s format (Green-lit with the sha; 🚧 in the index); the merge moves both to
  #13's (Green-lit with the date and blob; the state in the Status header only).
  *(2026-10-16)*
- **Sit-down held.**  *(2026-10-16)*
- **`sparse_matrix` is bound in C++** rather than declared in the stub, so the stub never
  advertises a method a bare `import qarpx` lacks and holds no hand-written method.  It
  loses pickling and `inspect.signature`, as every other `qarpx` method has; the change is
  declared.  *(2026-10-16)*
- **The shadowing attributes are renamed to public names**, not made private.
  *(2026-10-16)*
- **`AnyBlock` keeps meaning any block; a new type covers the Python blocks.**  Making every
  site work on raw blocks would be a feature, with about 45 regression tests, and sites such
  as `mark_built` and `_publish_symbols` only make sense on Python blocks anyway.
  *(2026-10-16)*
- **Names:** `QarpBlock` for the new type; `state_preparation_block` and `rotation_matrix`
  for the renamed attributes.  *(2026-10-16)*
- **One PR.**  The second review's four-way split (tooling, bindings, library sweep,
  strictness) was weighed; the work stays one PR, reviewed phase by phase.  *(2026-10-16)*
- **Gate methods on a composite raise**, rather than the rename only dropping its claim to
  free them.  *(2026-10-16)*
- **`Command == other` returns `False`** for a foreign type, as every other binding does,
  rather than keeping today's `TypeError`.  *(2026-10-16)*
- **SymEngine is detected by `qx.__with_symengine__`**, so the operator methods can be bound
  on every build.  *(2026-10-16)*
- **No example.**  §15 asks structural work for one; the user-visible result is what an
  editor and a type checker show, which a notebook cannot exercise, and each runtime change
  has a regression test.  *(2026-10-16)*

## API sketch

```text
# Wheel
site-packages/qarpx.cpython-*.so
site-packages/qarpx-stubs/__init__.pyi      # from cpp/libqarpx/python/qarpx.pyi

# Regenerate after a binding change (Python 3.12)
python scripts/generate_stubs.py            # writes cpp/libqarpx/python/qarpx.pyi
python scripts/generate_stubs.py --check    # what the drift test runs
```

```python
# qarpx.pyi (excerpts, as the changed bindings make stubgen emit them)
__abi_version__: int = 15
__source_dir__: str
__with_symengine__: bool

class Block:
    @property
    def is_built(self) -> bool: ...
    def build(self) -> Self: ...
    def rz(self, q: int, angle: Param | float) -> Block: ...
    def __eq__(self, other: object, /) -> bool: ...

class ConditionalBlock(Block):
    @overload
    def __init__(self, cbits: Sequence[int], values: Sequence[bool], then_body: Block,
                 else_body: Block | None = None, name: str = "Conditional") -> None: ...
    @overload
    def __init__(self, cbits: Sequence[int], values: Sequence[bool], then_body: None, *,
                 else_body: Block, name: str = "Conditional") -> None: ...
    @property
    def then_body(self) -> Block | None: ...

class QubitOperator:
    def sparse_matrix(self, n_qubits: int | None = None) -> scipy.sparse.csc_matrix: ...
    __hash__: ClassVar[None]  # type: ignore[assignment]
```

```ini
# mypy.ini
[mypy]
mypy_path = $MYPY_CONFIG_FILE_DIR/cpp/libqarpx/python

[mypy-qarp,qarp.*]
check_untyped_defs = True
warn_unused_ignores = True
```

The sketch is binding: argument names and order follow the current bindings and Python
overrides, and an implementation that makes stubgen emit something else declares it as a
deviation.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| `generate_stubs.py --check` passes in a subprocess under Python 3.12; under `CI`, a Python mismatch or missing generator fails | the built module, read by the pinned generator | `tests/test_stubs.py` |
| Every extra that lists `nanobind` pins the CMake `GIT_TAG` | `cpp/libqarpx/CMakeLists.txt` | `tests/test_stubs.py` |
| Generated from the stub: every `Block` method is called on a raw and on a Python block by keyword, with a float and a `Param` for each `Param \| float` argument, and runs; each of the 40 single-gate builders appends one command with the stub's gate, qubits in argument order and parameter count | the runtime, and the stub's argument order | `tests/test_stubs.py` |
| The generator refuses another Python, nanobind version, ABI or checkout, and a pattern that matches nothing; `--check` fails on a stale stub and leaves it | each guard's message | `tests/test_stubs.py` |
| mypy snippets through `mypy.api.run` give the expected error codes: a wrong argument type rejected, `isinstance(x, QarpBlock)` narrowing, `ConditionalBlock` with neither body rejected; the known false rejections (an `np.int64` qubit, `nb::dynamic_attr` attributes) are listed | each snippet is also run, and mypy's verdict matches the runtime's | `tests/test_stubs.py` |
| `qarp/` type-checks with the stub, `check_untyped_defs` and `warn_unused_ignores`, `tests/` with the stub, and the stub itself has no errors | mypy over the code, with the stub on `mypy_path` | CI `mypy` job |
| mypy run from a temporary directory against the installed wheel reveals `qx.SimpleBlock(1).n_qubits` as `int` | the binding's C++ type | `ci.yml` `mypy` job |
| `is_built` is `False` on a fresh raw block and `True` after `build()`; `build()` returns the same object | the build lifecycle | `tests/test_blocks/test_block.py` |
| Each of the six Python block classes keeps its runtime MRO: `_BlockMixin` is not in it, and its second entry is the `qx` class | the runtime class | `tests/test_blocks/test_block.py` |
| `isinstance(b, QarpBlock)` is true for each Python block class and false for a raw `qx.SimpleBlock` | the runtime classes | `tests/test_blocks/test_block.py` |
| Planning a raw `qx.ControlledBlock` over three X gates (`_program.plan(block, 12)`) gives CX | the analytic CX matrix | `tests/test_blocks/test_structure.py` |
| A ladder of nine raw C-P(θ) blocks onto qubit 11 plans as one controlled-powers kernel | \|x⟩ gains e^{iθ(x0+…+x8)} when x11 = 1 | `tests/test_blocks/test_structure.py` |
| A raw `ControlledBlock` with controls on \|1⟩ and \|0⟩ plots as a controlled box with two controls and state 0b01 | §6 (LSB) | `tests/test_plotting/test_circuit_adapter_residuals.py` |
| `MirrorTest` with an unbuilt raw H as bra and a raw H as ket gives 1 | \|⟨+\|+⟩\|² = 1; 1/2 without the bra† | `test_mirror_test.py` |
| `PauliShadow` rejects a raw Bell ket with `TypeError`; wrapped, ⟨Z0 Z1⟩ is within 0.2 of 1 | ⟨Φ+\|Z0 Z1\|Φ+⟩ = 1 | `test_shadows/test_base.py` |
| Filtering a four-qubit X…X · CCZ(1, 2, 3) to qubits 1–3 keeps the CCZ; a CCZ across the cut raises | CCZ flips \|111⟩ only | `tests/test_cutting/test_reconstructer.py` |
| A string-key rename reaches `TrotterAnsatzBlock`'s and `UCCBlock`'s pairing | the pairing written by hand | `test_trotter_block.py`, `test_ucc_block.py` |
| `ResourceEstimator.estimate` on an unbuilt raw composite of H then CX counts one H and one CX | counted by hand | `tests/test_resources/test_estimator.py` |
| `CircuitAdapter` over an unbuilt raw block holding one H lists one H gate | the hand-written circuit | `tests/test_plotting/test_circuit_adapter_residuals.py` |
| `CuttingPrimitive` with an unbuilt raw ket of two Bell pairs and `X0 X1` as operator gives 1 | ⟨Φ+\|XX\|Φ+⟩ = 1 | `test_cutting_primitive.py` |
| `SWAPTestBlock` on raw blocks preparing \|+⟩ and \|−⟩ measures ancilla 0 with probability 1/2; losing either input's gates gives 3/4 | `(1 + \|⟨a\|b⟩\|²)/2` | `test_swap_test_block.py` |
| `HadamardTestBlock` with a raw Z on \|1⟩ gives −1 | ⟨1\|Z\|1⟩ = −1 | `test_hadamard_test_block.py` |
| `QPEBlock` with a raw X as eigenstate and a raw P(π/2) as unitary, 2 ancillas, reads phase 1/4 | analytic eigenphase | `test_qpe_block.py` |
| `DOSQPEBlock` with a raw H as the state block (\|+⟩), a raw Z as the unitary and one ancilla gives eigenphases 0 and 1/2 with weight 1/2 each | analytic spectrum of Z, \|⟨±\|+⟩\|² weights | `test_dos_qpe_block.py` |
| `generate_states_new_basis` with a raw X on one qubit gives \|1⟩ and \|0⟩ | analytic | `tests/test_algorithms/test_utils.py` |
| `VFF` over a raw H, at a = π/4, θ = π, reads \|00⟩ with probability 1 | \|Tr(V†U)\|²/4 with V = −iH; 0 without U | `test_vff.py` |
| A deep copy of an else-only conditional flattens to a conditional X | the hand-written command | `tests/test_blocks/test_block.py` |
| A gate method and `set_commands` on a Python and on a raw composite, controlled and conditional block raise `TypeError`; on each renamed block, `state_preparation(...)` and `u(...)` raise `TypeError` and the renamed attribute holds the input | blocks that hold blocks take no gates (§13); the constructor's input | `test_block.py`, each block's test file, `test_tensor_operators.py` |
| Composite rejection in C++ | as above | `cpp/libqarpx/tests/cpp/test_core.cpp` |
| `Command(CX, 0, np.int64(1))` after X on q0 gives counts {3}; `Command(Rz, 0, 1)`, `Command(Rz, 0, np.int64(1))`, `Command(Rz, 0, True)`, `Command(H, 0, 0.5)`, `Command(CX, 0, Param(1.0))`, `Command(MCZ, 0, 1, 0.5)`, `Command(Barrier, 0, 0.5)` and `Command(GPhase, 3)` raise `ValueError`; `Command(Rz, 0, Param(1.0))` builds | CX\|10⟩ = \|11⟩; each gate's qubit and parameter count | `tests/test_blocks/test_block.py` |
| `Command() == 5` is `False` | Python's equality protocol | `tests/test_blocks/test_block.py` |
| `set_symbols({"a": 0.5})` and `replace_symbols` with a string key leave `.symbols` = `(b,)` and `parameter_map` over `b` only, on a block with `rz(a)`, `ry(b)` | counted by hand | `tests/test_blocks/test_block.py` |
| In a subprocess that imports only `qarpx`, two-qubit `Z0.sparse_matrix(2)` = diag(1, −1, 1, −1); `2·I` gives 2·𝟙 (1×1 inferred, 4×4 at `n_qubits=2`); the empty operator gives zeros; `Z1` at `n_qubits=1` and `Z0` at `n_qubits=0` raise `ValueError`; `n_qubits=2.0` raises `TypeError` | LSB Kronecker products, written by hand | `test_qubit_operator.py`; `test_fermion_operator.py` (two-mode n̂₀ = diag(0, 1, 0, 1)) |
| `qx.__with_symengine__` equals whether `QubitOperator("X0", Symbol("a"))` constructs | the build's actual behaviour | `test_symbolic_coefficients.py` |
| `op += Symbol("a")` rebinds `op`, and `hash(op)` raises `TypeError` | today's runtime behaviour | `test_qubit_operator.py` |
| mypy rejects `VQE(op, raw_block)` and accepts `VQE(op, CompositeBlock([raw_block]))`, `VQE(op, hb \| hb)` and `VQD(op, kets, [])` for a built Python block `hb` and `kets = [hb]`; each call is run, and only the first fails | the runtime | `tests/test_stubs.py` |

## Phases

The phases are review stages inside one PR.  CI stays green at each: the stub becomes visible
to mypy, through `mypy_path` and the CMake install, only at the end of Phase 2.

### Phase 1 — Conventions, bindings and the stub

- [x] §13 and §15 edits *(2026-10-16)*
- [x] `nanobind` pinned in `[dev]` and `[full-dev]`; `scripts/generate_stubs.py` with its
      guards and `--check`; pattern file; drift and pin tests; `.gitattributes`; ruff
      exclusion *(2026-10-16)*
- [x] Bindings: the argument helper, `Param | float`, return types, `build()` returning the
      block, `is_built` as a property, `sparse_matrix` (the Python assignments removed),
      `ConditionalBlock` overloads, the `Command` checks and dispatch, `Command`'s duplicate
      equality removed; the stub's own 20 errors gone *(2026-10-16)*
- [x] Same commit as `is_built`: `_block.py`'s `fget` read, the five test call sites, the
      canary *(2026-10-16)*
- [x] SymEngine methods bound unconditionally and `__with_symengine__`, with
      `test_symbolic_coefficients.py` switched in the same commit *(2026-10-16)*
- [x] Composites take no gates (C++ and the error translation), with the C++ test
      *(2026-10-16)*
- [x] The three renames, with `test_tensor_operators.py` *(2026-10-16)*
- [x] ABI 14 → 15 in `bindings.cpp` and `qarp/_abi.py` *(2026-10-16)*
- [x] The generated keyword table and the binding regression tests from the Test plan
      *(2026-10-16)*

### Phase 2 — Type-check `qarp` against the stub

- [x] After #13 lands: `develop` merged in; header and index row moved to #13's format; the
      lists below measured again, the difference declared; #13's `_is_built` helpers in
      `_program.py` and `_circuit_adapter.py` and its `callable(...)` shims for `n_controls`
      and `ctrl_state` removed *(2026-10-20)*
- [x] The `TYPE_CHECKING` base aliases for the six classes; the mixin conflicts resolved;
      string keys converted in `set_symbols`/`replace_symbols`; the mixin's gate overrides
      annotated; `upccd_block.py`'s `symbols` attribute *(2026-10-20)*
- [x] `QarpBlock` defined and exported; the parameters that read Python-only members take
      it; the Python-block returns and `Sequence` parameters typed; `vqe.py` *(2026-10-20)*
- [x] The `is_built` regression tests (estimator, plotter, cutting primitive) *(2026-10-20)*
- [x] Each flagged file fixed; the regression tests for the Python-side bugs.  Flagged at
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
      `tests/test_engines/test_structured_execution.py`,
      `tests/test_operators/test_qdrift.py`, `tests/test_resources/test_modelers.py`,
      `tests/test_resources/test_synthesis.py`; flagged on #13's tip:
      `test_lazy_circuits.py`, `test_unbuilt_planning.py`, `test_compiled_circuits.py`
      *(2026-10-20)*
- [x] `check_untyped_defs` on for `qarp/`; the files above and, flagged only with it at
      `4093ad5`: `qarp/absorb/__init__.py`; `qarp/algorithms/_composite/`: `adapt_vqd.py`,
      `dos_qpe.py`, `projected_vqe.py`, `qpe.py`, `qse.py`;
      `qarp/blocks/_primitives/spa_block.py`,
      `qarp/blocks/_state_preparation/cvo_qram_state_block.py`,
      `qarp/cutting/_post_processing.py`, `qarp/devices/__init__.py`,
      `qarp/graphs/_graph.py`, `qarp/graphs/_simplicial_complex.py`,
      `qarp/operators/compat.py`, `qarp/operators/functions.py`,
      `qarp/optimizers/_gradient_descent_optimizer.py`, `qarp/plotting/_circuit_plotter.py`
      *(2026-10-20)*
- [x] Unused `# type: ignore` comments removed, measured after the fixes *(2026-10-20)*
- [x] `test_block_refactor.py`'s note that mypy sees `AnyBlock` as `Any` corrected
      *(2026-10-20)*
- [x] Last: `mypy.ini` gets `mypy_path`, `warn_unused_ignores` and drops its `qarpx`
      sections, its note carries the new rule and numbers; `[tool.pyright] stubPath`; CMake
      installs `qarpx-stubs/__init__.pyi`; `method-assign` dropped from the CI mypy command;
      the `ci.yml` wheel probe *(2026-10-20)*

### Phase 3 — Docs

- [x] `AGENTS.md` and `CONTRIBUTING.md`: the regenerate command beside the mypy command, the
      mypy command without `method-assign`, `[dev]`'s description, and `QarpBlock` beside
      `AnyBlock` in `AGENTS.md`'s landmines *(2026-10-20)*
- [x] `installation.rst`: `[dev]`'s description *(2026-10-20)*
- [x] `qarp_philosophy.rst`: `QarpBlock` beside `AnyBlock` *(2026-10-20)*
- [x] `CHANGELOG.md`: Added (typed `qarpx`, keyword arguments on the raw methods, `QarpBlock`,
      `qx.__with_symengine__`, `ConditionalBlock` without `then_body`); Changed (each runtime
      change under *Who sees a change* with its migration, `Engine.build` taking a
      `Sequence`, `VQE(op, raw_block)` rejected by mypy); Fixed (each bug in *Why*)
      *(2026-10-20)*
- [x] `docs/contributions/README.md`: the roadmap moves this work into the 0.2 cycle
      *(2026-10-20)*
- [x] `to_do.md`: the `is_built` entry leaves; the `tests/` strictness entry stays
      *(2026-10-20)*

## Deviations log

Deviations from the green-lit plan are declared in the PR's "Deviations from plan"
section and folded back here before merge; silent drift is the violation.

- **Phase 1, absorber return type.**  The plan typed the raw absorbers' `absorb()` as
  returning `Block`; it returns `(commands, n_qubits, n_cbits)`, and the Python absorbers
  build the block.  Typed `tuple[list[Command], int, int]`; Design updated.

- **Phase 2, re-measure after #13.**  With Phase 1's bindings, mypy reported 77 errors in 29
  files on the merged tree (65 in 23 `qarp` files, 12 in 6 test files), and with
  `check_untyped_defs` 221 in 40 `qarp` files; no file outside the lists was flagged.  The
  mixin aliases then flagged `tests/test_engines/test_threading.py` (an unchecked
  `Optional`), fixed with an `assert`.
- **Phase 2, files outside the Scope.**  `composite_algorithm.py` gains `_real_part`, the one
  narrowing of a scalar primitive result the variational algorithms share;
  `layer_block.py`'s `parameters` takes `Sequence[Angle]`, since `VFF` passes `qx.Param`s;
  unused `# type: ignore` comments go from `grouped_transition_hadamard_test.py`,
  `_spectral_estimation.py`, `_vumpo.py` and the `boxed` and `controlled_block` renderers.
- **Phase 2, two runtime changes.**  `AdaptVQD.run()` returns the energy as a float, as
  documented and as `AdaptVQE` does, instead of `complex(E+0j)`; pinned in
  `test_adapt_vqd.py`.  `CircuitAdapter` reads `num_controls` and `ctrl_state`, which both
  block kinds carry, so a raw `ControlledBlock` plots as a controlled box instead of a plain
  one.
- **Phase 2, `_repeat`.**  Returns `QarpBlock`, not `CompositeBlock`: a count of one returns
  the block itself.  Design updated.
- **Phase 2, `mypy.ini` section.**  `[mypy-qarp,qarp.*]`: `qarp.*` alone does not match the
  top-level package.  Design updated.
- **Phase 2, two test rows.**  The cutting-primitive test uses two raw Bell pairs and
  `X0 X1`: cutting needs two subcircuits and an entangling gate.  The `VQD` snippet passes a
  built parameterized block with `weights=[]`: a `ComputationalBasisStateBlock` has no
  symbols, so `VQD` rejects it at runtime.  Test plan updated.
- **Phase 2, a `qarpx` object built at import in a test.**  A `qx.Param` held by a
  `parametrize` list outlives nanobind's shutdown and is reported as a leak; Phase 1's two
  such lists (`test_block.py`, `test_stubs.py`) build their values inside the test.
- **Phase 3, `to_do.md`.**  The `tests/` strictness entry carries the count measured on the
  landed tree, 762 errors in 134 test files on macOS, in place of the count at `4093ad5`.
  The plan-only branches entry names ABI 15.
- **Phase 3, CHANGELOG.**  Bugs that are also runtime changes (gate methods on a composite,
  `Command`'s dispatch, the renamed attributes) are listed once, under Changed with their
  migration.  The two Phase 2 runtime changes are under Fixed.
- **Gate 2 review, notebooks.**  Three nightly notebooks
  (`mwe_qaoa_max_cut.ipynb`, `finance_portfolio_diversification.ipynb`,
  `mwe_grouping_strategy.ipynb`) added a measurement to a composite, which now raises; the
  measurement never ran, and the `Sampler` that follows measures every qubit, so the four
  lines go and the outputs are unchanged.
- **Gate 2 review, `QarpBlock` sites.**  About 21 parameters take `QarpBlock`, not about 45:
  mypy found the sites, and a grep for `.symbols`, `parameter_map`, `set_symbols` and
  `replace_symbols` read from an input finds one more, `PauliShadow`'s ket
  (`shadows/base.py`), behind `PrimitiveAlgorithm.ket: Any`.  It takes `QarpBlock` and raises
  `TypeError` on a raw ket; the stale comment on `PrimitiveAlgorithm.ket` goes.
- **Gate 2 review, blocks that hold blocks.**  `ControlledBlock` and `ConditionalBlock` take no
  gates either, and `set_commands` is guarded like the builders, so the composite rule covers
  every block whose own buffer never runs.  The error message no longer says "composite".
  §13 and its summary row are edited to match; Design updated.
- **Gate 2 review, files outside the Scope.**  `mirror_test.py` builds a raw bra that isn't
  built before daggering it, which #13's lazy rewrite had dropped.  `cutting/_reconstructer.py`
  remaps each kept command instead of rebuilding it from at most two qubits, which turned a
  CCZ into a CZ, and checks gates of any width for straddling the cut.
  `controlled_block.h` and `conditional_block.h` override `accepts_gates()`.  Tests in
  `test_mirror_test.py`, `test_shadows/test_base.py`, `test_reconstructer.py`,
  `test_ucc_block.py` and `test_device.py`.
- **Gate 2 review, `Command`'s shape check.**  MCZ and Barrier are checked for taking no
  parameter, and GPhase, of fixed arity, is checked in full, so no constructor form builds it.
- **Gate 2 review, `mcz`.**  The override takes `qubits` first, as the stub does, so
  `mcz(qubits=…)` works on a Python block.  The keyword table also runs on Python blocks, and
  checks each single-gate builder's command against the stub's argument order.
- **Gate 2 review, tests that could not fail.**  The raw SWAP test compares |+⟩ with |−⟩
  and the raw VFF test reads the Hilbert–Schmidt circuit at the analytic point.  Tests are
  added for the ladder planner site, the Trotter and UCC renames and the raw controlled box,
  whose fixes had none, and for the generator's refusals.  Test plan updated.
- **Gate 2 review, smaller items.**  `cu`'s `gamma` stays required on a raw block, since the
  argument helper has no defaults; the CHANGELOG says so.  `compile_for_device` raises
  `TypeError` on a wrong argument count again.  The absorber docstrings name the tuple they
  return.  The Python gate methods' angles are effectively `Any` to mypy; Design says so.  The
  `mypy.ini` note gives its baseline: at `ca09f06` on macOS, each flag on its own, 65 errors,
  221 with `check_untyped_defs`, and 55 unused ignores.
- **Gate 2 follow-up, `_real_part`.**  A `Sampler` passed as a variational algorithm's
  primitive reaches `_real_part`, so its `SamplingDistribution` check raises `TypeError`, as
  `float()` did, with a message naming the cause, in place of an `assert`.  Test in
  `test_vqa.py`.
- **Gate 2 follow-up, `Param.approx_equal`.**  Its `nb::sig` is written by hand beside its
  `nb::arg` list, since the argument helper renders no defaults: the one `Param | float` site
  outside the helper, with `cu`'s `gamma` the other place that limit shows.
- **Gate 2 follow-up, §15.**  The stub rule is a bullet of its own, and it also carries two
  rules from Design: a wrong or vague type is fixed in the binding, not the stub, and the
  pattern file holds only what nanobind cannot express and values that vary by build.
  Design updated.
- **Gate 2 follow-up, bulk builders.**  Each bulk overload in `block.cpp` fetches
  `gate_buffer()` once instead of per command, and the section comment says so.
