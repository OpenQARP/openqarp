# Review of #15 plan — Ship type stubs for the compiled backend

Gate 1 review of `compiled_backend_stubs_plan.md` at `18c5034` (based on `develop` at `2167d69`).
No implementation exists yet, so this checks the plan's claims against the code and its design
against what it will meet.

## Summary

The direction is sound. The distribution design works (checked below), stub generation is
deterministic, and the tooling exists. But implemented as written, the plan breaks every Python
block, ships a drift test that can only pass on the machine that generated the stub, and leaves
its central typing decision open. Its error counts and some of its bug claims need redoing.

Blockers:

- **B1.** Making `is_built` a property breaks every Python block, and five test call sites.
- **B2.** The drift test can't pass outside the machine that generated the stub: the stub records
  that checkout's path, CI doesn't install the generator, and the stub varies with the build.
- **B3.** The `TYPE_CHECKING` mixin split doesn't clear the errors the plan expects, because
  `AnyBlock` parameters accept raw blocks, which really lack the mixin. The plan must decide
  whether they do.
- **B4.** Process: structural tier with no sit-down, a Scope defined by mypy's output, and an
  undeclared dependency on #13.
- **B5.** Phase 1 can't land on its own: CI's mypy job is red from the moment the stub is visible
  until Phase 2 ends.

Should fix: S1–S12, below. Bugs the plan misses: N1–N4, below.

## Blockers

### B1. Making `is_built` a property breaks every Python block

The plan turns `qx.Block.is_built` from a method into a read-only property (`def_prop_ro`), so it
matches the Python blocks. But the Python blocks read the C++ value by calling the method through
the class (`qarp/blocks/_block.py:508`):

```python
return self._cpp_base.is_built(self)
```

Read through the class, a nanobind property is a `property` object, and calling it raises:

```python
import qarpx as qx

b = qx.SimpleBlock(2, "b")
print(qx.Block.n_qubits.fget(b))     # 2: a property, read through the class
qx.Block.n_qubits(b)                 # what _block.py:508 would do to is_built
# TypeError: 'property' object is not callable
```

With `is_built` simulated as a property, `SimpleBlock(1).is_built` and `.build()` both raise,
and 56 of 87 tests fail in the three affected test files. Also broken:

- `.is_built()` called with parentheses at `tests/test_blocks/test_block.py:280`, `:302` and `:414`,
  `tests/test_blocks/test_pauli_exp_builder.py:52`, and `tests/test_blocks/test_bridge_cleanup.py:46`.
  The last one is inside a subprocess source string, so no type checker sees it.
- `tests/test_blocks/test_block.py:393`, which asserts the exact set of properties on `qx.Block`.
- Raw `qarpx` users' `b.is_built()`. Phase 3's CHANGELOG plan lists only Added and Fixed (S9).

mypy can't catch `_block.py:508`: `_cpp_base` is typed `Optional[Any]`.

**Fix.** Read the property's getter, and land it in the same commit as the binding change, in
Phase 1:

```python
return self._cpp_base.is_built.fget(self)
```

List the five test sites and the canary in Phase 1 and in Scope.

### B2. The drift test can't pass outside the machine that generated the stub

The plan commits the stub and adds a test that regenerates it and compares, so that "a binding
edit without a regenerated stub fails in every environment that runs the suite, CI included".
Three things stop that.

**(a) The stub records the checkout's path.** The module exports `__source_dir__`
(`bindings.cpp:230`), and stubgen writes its value:

```bash
python -m nanobind.stubgen -m qarpx -P -o /tmp/qarpx.pyi
grep __source_dir__ /tmp/qarpx.pyi
# __source_dir__: str = '/path/to/your/checkout/cpp/libqarpx'
```

Every other checkout, worktree and CI runner generates a different line. On CI's longer path the
value passes stubgen's 50-character limit and is written as `= ...`, so it differs either way.
The wheel would also ship the committer's path. The plan allows a pattern file only "for types the
bindings cannot express", so it has no sanctioned fix. Widen that rule to build-dependent values,
and add:

```text
qarpx.__source_dir__:
    __source_dir__: str
```

**(b) CI doesn't install the generator.** The plan pins `nanobind` in `[dev]`. Every CI job that
runs pytest or mypy installs `[full-dev]` (`ci.yml:127`, `:145`), which repeats the ruff, mypy and
pre-commit pins but doesn't include `[dev]`:

```toml
dev = ["ruff==0.15.22", "mypy==1.19.1", "pre-commit>=3.5"]     # pyproject.toml:110; nanobind goes here
full-dev = [ ..., "pytest", ..., "ruff==0.15.22", "mypy==1.19.1", "pre-commit>=3.5"]   # :164, what CI installs
```

So in CI the drift test either errors or, behind `importorskip`, never runs. A developer who
follows `CLAUDE.md`'s `[full-dev]` install can't run `scripts/generate_stubs.py`, and so can't
follow the new §15 rule. Pin `nanobind` in `[full-dev]` too, have the pin test check every extra
that lists it, and make the drift test fail rather than skip when `CI` is set.

**(c) The stub varies with the build.** `substitute` and `free_symbols` on the operator classes
exist only when SymEngine is compiled in (`bindings_operators.cpp:519`, `:536`). A pip build turns
it on (`pyproject.toml:311`), but `bootstrap_qarpx.py --no-symengine` and the CUDA-Q bootstrap
leave it at the CMake default, off:

```python
import qarpx as qx

print(hasattr(qx.QubitOperator, "substitute"))   # True from a pip build, False from those
```

The drift test fails in those builds, and the stub advertises methods they lack. Either bind both
methods unconditionally and raise when SymEngine is absent (the pattern the file's header already
describes, `bindings_operators.cpp:16`), or skip the drift test there with a reason.

**(d) Python version and line endings (suspected).** stubgen branches on the Python version, and
only 3.12 was tested here; the package supports 3.11 and up, and the wheels cover 3.11–3.14.
stubgen also writes in text mode, so Windows gets CRLF. Either generate under one pinned Python and
skip elsewhere, or show the output is identical. Add `cpp/libqarpx/python/qarpx.pyi text eol=lf`
to `.gitattributes`.

### B3. The mixin split doesn't do what the plan says, and the raw-block contract is undecided

The plan expects the `TYPE_CHECKING` split to clear the 17 `"Block" has no attribute "symbols"`
errors and most `attr-defined` errors. Prototyped, 16 of the 17 remain, and `attr-defined` drops
only from 76 to 54. The receivers are parameters typed `AnyBlock`, which is the raw `qx.Block`,
and a raw block really has no `.symbols`:

```python
import qarpx as qx
from qarp.algorithms import VQE
from qarp.operators import QubitOperator

raw = qx.SimpleBlock(1, "raw")
raw.x(0)
raw.build()
VQE(QubitOperator("Z0"), raw)
# AttributeError: 'qarpx.SimpleBlock' object has no attribute 'symbols'
```

With `check_untyped_defs`, 31 more errors in `qarp/` are mixin members read through `AnyBlock`.
**The plan has to decide whether `AnyBlock` accepts raw blocks.** If it does, these ~45 sites are
bugs needing regression tests. If it doesn't, the annotations narrow, to a Protocol or a union of
the six Python classes. That may be a §13 change, which conflicts with "No other convention
changes".

Two more problems with the split as sketched:

- **Read literally, the API sketch makes things worse.** `if TYPE_CHECKING: class
  SimpleBlock(_BlockMixin, qx.SimpleBlock): ...` hides the real class bodies from mypy: 293 errors
  in 104 files, against 212 without it. A base alias works, and leaves the runtime MRO unchanged
  (prototyped: identical `__mro__`, block tests pass):

  ```python
  if TYPE_CHECKING:
      class _SimpleBlockBase(_BlockMixin, qx.SimpleBlock): ...
  else:
      _SimpleBlockBase = qx.SimpleBlock


  @_attach_mixin
  class SimpleBlock(_SimpleBlockBase): ...
  ```

- **It adds about 30 conflicts.** Each of the six classes gets `Definition of "X" in base class
  "_BlockMixin" is incompatible with definition in base class "Block"` for `build`, `is_built`,
  `set_symbols`, `replace_symbols` and `target_qubits`. The plan names only `build` and
  `is_built`. `set_symbols` and `replace_symbols` take `Dict` in the mixin and `Mapping` in C++;
  the mixin accepts both key types at runtime, so widening it to `Mapping` is safe. `build` is S4.

### B4. Process: no sit-down, an unbounded Scope, and an undeclared dependency on #13

- **No sit-down.** #15 says "Sit-down held: no". Structural tier requires one before green-light.
  The five Decisions are dated and read as settled, with no open items.
- **Scope isn't a bounded list.** It's "the files mypy flags once the stubs and library strictness
  are in place", plus unnamed regression tests, re-measured after #13. Whatever mypy flags at the
  end is then in scope by definition, so the Gate 2 check that the PR touches nothing outside
  Scope proves nothing. Freeze the measured list in Scope (after S3's re-measure), and declare
  anything the post-#13 re-measure adds as a deviation.
- **#13.** The `to_do.md` entry the plan folds in (Decision 1) and later removes (Phase 3) exists
  only on #13's branch (`cfa2e0c`), not on `develop`. 18 in-scope files are also edited by #13,
  including `_program.py` (both `n_controls` sites), `_qarp_engine.py`, `cutting_primitive.py`,
  `test_structure.py` (where the planned `n_controls` test goes), `test_structured_execution.py`,
  `CHANGELOG.md` and `to_do.md`. #13 also changes the plan-file format: Status `Planned`, a
  sha-less `Green-lit:` header and a new index layout. State the landing order as a Decision,
  "lands after #13", or stop relying on #13's content.

### B5. Phase 1 can't land on its own

`ignore_missing_imports` doesn't hide a stub mypy can find, so CI's mypy job goes red the moment
the stub is visible, and stays red until Phase 2 is done:

```bash
mypy qarp/ tests/ --disable-error-code=import-untyped --disable-error-code=method-assign
# Success: no issues found in 514 source files                (today)
MYPYPATH=<stub dir> mypy qarp/ tests/ --disable-error-code=import-untyped --disable-error-code=method-assign
# Found 212 errors in 70 files (checked 514 source files)     (stub visible, before Phase 2)
```

Phase 1's binding change also needs the `_block.py:508` fix (B1), which is listed under Phase 2.
Move the `_block.py` fix and the test call sites into Phase 1, and make the stub visible to mypy
only at the end of Phase 2, or land the two phases together.

## Should fix

### S1. mypy often can't see the installed stub

The plan says the wheel and an editable install both carry the stub. Checked with a toy project
using `scikit-build-core` 1.0.3 and the repo's `[tool.scikit-build]` settings:

| install | stub reaches mypy? |
|-|-|
| wheel, or plain editable | yes |
| editable with `-C editable.rebuild=true` (the C++ workflow in `CLAUDE.md`) | no: it lands under `build/<tag>/install/platlib/`, off mypy's path (`import-not-found`) |
| `scripts/bootstrap_qarpx.py` and the CUDA-Q bootstrap | no: they run no CMake install |
| plain editable, after a binding change | stale: CMake installed the committed file before it was regenerated |

So once `mypy.ini` drops its `qarpx` sections, the developers who change bindings either get
import errors or check against an old stub. Point the repo's own mypy at the committed stub, and
keep `qarpx-stubs` for users:

```ini
[mypy]
mypy_path = cpp/libqarpx/python
```

Then check the wheel's stub separately, in the `ci.yml` mypy job, from a temporary directory so
the repo's `mypy.ini` doesn't apply.

### S2. Errors inside the installed stub are invisible

mypy stays quiet about errors inside installed packages, and the committed `.pyi` sits outside
`mypy qarp/ tests/` and is excluded from ruff. So Phase 1's "the stub's own errors gone" is never
checked. S1's `mypy_path` makes them visible. Alternatively, run `python -m mypy.stubtest qarpx`
in CI, which checks the stub against the runtime module and so is an independent oracle.

### S3. The plan's numbers were measured without `-P`

The Design generates the stub with `-P` (private members). The plan's figures reproduce only
without it:

| | plan | with `-P`, as designed |
|-|-|-|
| errors with the stub | 228 in 72 files | 212 in 70 |
| with `check_untyped_defs`, total | 2196 in 274 | 2161 in 272 |
| `qarp/` | 344 in 71 | 329 in 70 |
| tests | 1832 in 202 | 1812 in 201 |

`qarp/_blas_threads.py` and `tests/test_blas_threads.py` are in the Phase 2 list only because the
private bindings were missing. Re-measure with `-P`, and update Why, the Phase 2 lists, the
`mypy.ini` note and the new `to_do.md` entry.

### S4. How precise types get into the stub is undecided

- **`Param | float`.** stubgen ignores `nb::implicitly_convertible`, so `rz(0, 0.5)` runs but fails
  mypy:

  ```python
  b = qx.SimpleBlock(3, "b")
  b.rz(0, 0.5)              # runs today
  # mypy: No overload variant of "rz" of "Block" matches argument types "int", "float"
  ```

  The bulk forms need it too: `b.rz([(0, 0.5)])` gives `expected "tuple[int, Param]"`. The two
  real options are an `nb::sig` on each of about 36 overloads, or a custom type caster that names
  itself `Param | float`. The plan should choose. Sketch, not compiled:

  ```cpp
  .def("rz", nb::overload_cast<uint32_t, Param>(&Block::rz), "qubit"_a, "theta"_a,
       nb::rv_policy::reference,
       nb::sig("def rz(self, qubit: int, theta: Param | float) -> Block"))
  ```

  Applied to `Command.__init__`, `Param | float` also makes the `(GateType, int, int)` overload
  unreachable for mypy. Put the int overloads first.
- **Return types.** 13 bindings return `object`, e.g. `QubitOperator.terms()` and the absorbers'
  `absorb()`. They cause about 85 strict errors. Nullable returns are mistyped too:

  ```python
  c = qx.ConditionalBlock([0], [True], qx.SimpleBlock(1, "then"))
  print(c.else_body)        # None, though the stub types it "-> Block"
  ```

  Extend "precision lives in the bindings" to return types.
- **`build()`.** The binding returns `None` and the Python blocks return `self`. Making the binding
  return the block, with `nb::sig("def build(self) -> typing.Self")`, would ride the same ABI 15
  bump, remove the split's `build` conflict (B3), and fix the six crashes in N1. The plan says
  only "the binding moves to match when the C++ side can".
- **The runtime-agreement table.** "mypy accepts … exactly when the runtime does" can't hold:
  nanobind accepts anything with `__index__` or `__float__`, so `b.h(np.int64(0))` runs but fails
  mypy. State it one way, "mypy accepts ⇒ the runtime accepts", and list the known false
  rejections.

### S5. Fixing the stub's own errors in the bindings would change behaviour

Some of the stub's 20 errors describe real runtime behaviour:

```python
import sympy
from qarp.operators import QubitOperator

op = QubitOperator("Z0")
alias = op
op += sympy.Symbol("a")
print(op is alias)        # False: += falls back to __add__ and rebinds
hash(op)                  # TypeError: unhashable type: 'qarpx.QubitOperator'
```

Adding an in-place `object` overload would change that aliasing, and "fixing" `__hash__ = None`
would make mutable operators hashable. Fix these with `nb::sig` or pattern entries only, such as
`__hash__: None  # type: ignore[assignment]`, and add a test that pins the aliasing.

### S6. Members attached at runtime are invisible to the stub

`qarp` attaches `sparse_matrix` to the operator classes at import (`_qubit_operator.py:46`,
`_fermion_operator.py:35`):

```python
import qarpx as qx

print(hasattr(qx.QubitOperator, "sparse_matrix"))   # False
import qarp.operators
print(hasattr(qx.QubitOperator, "sparse_matrix"))   # True
```

It's the canonical matrix API, with 161 call sites, and causes 13 strict errors in `qarp/` and 52
in tests. A pattern entry would advertise it on a bare `import qarpx`, where it doesn't exist.
`@prepares_known_state` causes 21 more test errors. Bind `sparse_matrix` in C++, or declare it
under `TYPE_CHECKING` as the mixin is, and say which.

### S7. Most of the named bugs are typing artefacts

- **Real:** `_program.py`'s `n_controls` (S8), and `_estimator.py:121` reading `is_built`
  uncalled. An unbuilt raw composite makes `estimate()` raise
  `CompositeBlock::flatten() called before build()`.
- **Artefacts:**
  - The other four `truthy-function` sites (`grover.py`, `amplitude_estimation.py`,
    `amplitude_amplification.py`, `haar_random_block.py`) only ever hold Python blocks.
  - `adapt_vqd.py:163`: mypy infers a list of lists from the other branch's `append`. The branch
    that reaches line 163 appends primitives:

    ```python
    if self.term_gradient == "ADAPT-VQD":
        ...
        term_vector.append(terms)        # a list
    else:
        term_vector.append(comm_alg)     # a primitive: what line 163 passes on
    ```

  - `montecarlo.py:622`: `engine.run()` on a `Sampler` returns a `SamplingDistribution`, which has
    `.items()`. mypy sees only the union of possible results.

Reword Why to list the real bugs, and drop regression tests for the artefacts.

### S8. Test-plan gaps

- **The `n_controls` test as worded passes with the bug present.** Planning returns `None` below
  12 qubits or with fewer than 3 gates in the span, before `n_controls` is read:

  ```python
  import qarpx as qx
  from qarp import _program


  def controlled_x(n_gates: int):
      inner = qx.SimpleBlock(1, "x")
      for _ in range(n_gates):
          inner.x(0)
      inner.build()
      block = qx.ControlledBlock(inner, 1, [True])
      block.build()
      return block


  print(_program.plan(controlled_x(1), 12))   # None: the test as worded never reaches n_controls
  _program.plan(controlled_x(3), 12)
  # AttributeError: 'qarpx.ControlledBlock' object has no attribute 'n_controls'
  ```

  Write the triggering configuration (12 or more qubits, 3 or more gates) into the row.
- **No row checks that the split leaves the runtime classes unchanged**, which is the plan's
  correctness claim for it. Suggested:

  ```python
  def test_type_checking_split_leaves_runtime_classes_unchanged():
      assert _BlockMixin not in SimpleBlock.__mro__
      assert SimpleBlock.__mro__[1] is qx.SimpleBlock
  ```

- **No row checks that an editable install carries the stub** (S1), or that the stub's own errors
  are gone (S2).
- **Oracles deferred.** "Each further bug … named in the PR" names the oracle after implementation,
  against the Gate 1 rule that oracles are named first.

### S9. User impact and the CHANGELOG

The stub turns working raw-`qarpx` code into mypy errors:

```python
import numpy as np
import qarpx as qx

b = qx.SimpleBlock(3, "b")
b.rz(0, 0.5)              # mypy: No overload variant of "rz" ... "int", "float"
b.h(np.int64(0))          # mypy: No overload variant of "h" ... "signedinteger[_64Bit]"
b.rz([(0, 0.5)])          # mypy: expected "tuple[int, Param]"
b.tag = "mine"            # mypy: "SimpleBlock" has no attribute "tag"  (runtime: nb::dynamic_attr)
```

A user's existing `ignore_missing_imports` for `qarpx` no longer hides these, because the stub is
found. The opt-out that works is `follow_imports = skip` with `follow_imports_for_stubs = True`.
The Design's "the stubs change no runtime behaviour", the reason given for having no example, is
also not true (B1, S4). The README roadmap puts this work in a "no public surface change" window,
which it isn't. Add Changed entries with a one-line migration each, for example:

```markdown
### Changed
- `qarpx.Block.is_built` is a read-only property: write `block.is_built`, not `block.is_built()`.
- qarpx ABI 15: rebuild qarpx (`pip install -e .`) before `import qarp` in an existing checkout.
- `qarpx` ships type stubs (`qarpx-stubs`); type-checked code that relied on `qarpx` being `Any`
  may need fixes, or `follow_imports = skip` for `qarpx`.
```

### S10. Missing from Scope

- The emitter and absorber sources compiled into the module
  (`cpp/libqarpx/src/{emit,absorb}/{qiskit,pytket,pennylane,qulacs}_*.cpp`) define their own
  bindings. `absorb()` returns `object` and causes 16 errors in `qarp/absorb/__init__.py`, which is
  on the Phase 2 list.
- `AGENTS.md`: `CLAUDE.md` is a symlink to it, so the diff shows `AGENTS.md`.
- `pyproject.toml`'s `[full-dev]` (B2).
- `tests/test_blocks/test_bridge_cleanup.py` and `test_pauli_exp_builder.py` (B1).
- `qarp/blocks/_primitives/upccd_block.py:67`, newly flagged once the mixin is visible.
- Which `CMakeLists.txt` (`cpp/libqarpx/` or `cpp/libqarpx/python/`). Also give directories for the
  bare file names in the Phase 2 list, such as `test_trotter_block.py` and `functions.py`.

### S11. The wheel check is in the wrong place

`wheels.yml` runs only on tags, weekly, and by hand. Its tests run in cibuildwheel's
`test-command`, which has no mypy. The `ci.yml` mypy job already installs the wheel non-editably
on every PR, so the PEP 561 check belongs there, run from a temporary directory.

### S12. Ordering and existing rules

- §15 says "Update this document, then make the code match" (`qarp_conventions.md:796`), but the
  plan's §15 edit is in Phase 3. Move it to Phase 1.
- `mypy.ini:11-12` says "Do not add strictness flags here piecemeal". The plan adds one flag for one
  tree and only updates the note's numbers. Retire or replace that rule in a Decision.

## Bugs the plan misses

All four were confirmed by running them.

### N1. Six entry points crash on raw blocks

C++ `build()` returns `None`, and these use its return value:

```python
import qarpx as qx
from qarp.blocks import DOSQPEBlock, HadamardTestBlock, QPEBlock, SWAPTestBlock


def raw():
    b = qx.SimpleBlock(1, "raw")
    b.h(0)
    return b


SWAPTestBlock(raw(), raw()).build()
# AttributeError: 'NoneType' object has no attribute 'target_qubits'
```

`HadamardTestBlock(raw(), raw())`, `QPEBlock(raw(), raw(), 2, 1)` and
`DOSQPEBlock(raw(), raw(), n_ancilla=2, n_state=1)` fail the same way, and so do
`generate_states_new_basis` and `VFF` given a raw block. The Python blocks work. Making the
binding's `build()` return the block (S4) fixes all six; otherwise each needs a regression test.

### N2. A conditional with only an `else` branch can't be deep-copied

```python
import copy

import qarpx as qx

cond = qx.conditional([0], [True], else_=lambda b: b.x(0), n_qubits=1)
print(cond.then_body)        # None
copy.deepcopy(cond)
# TypeError: __init__(): incompatible function arguments
```

The `ConditionalBlock` constructor (`bindings.cpp:709`) requires `then_body`, but an else-only
conditional has none. A qarp `CompositeBlock` holding one builds and flattens, but can't be copied.
Make `then_body` nullable in the constructor and the stub, and add a regression test.

### N3. Public attributes shadow C++ gate methods

```python
from qarp.blocks import AmplitudeAmplificationBlock, SimpleBlock

block = AmplitudeAmplificationBlock(SimpleBlock(1), SimpleBlock(1))
block.state_preparation([1, 0])      # the C++ gate method of that name
# TypeError: 'SimpleBlock' object is not callable
```

The block stores its input as `self.state_preparation` (`amplitude_amplification_block.py:111`),
hiding the `qx.SimpleBlock.state_preparation` gate method. `AmplitudeEstimationBlock` does the
same, and `OrbitalRotationBlock` hides `u` (`orbital_rotation_block.py:105`). mypy flags all three.
Renaming is an API change, so plan it explicitly.

### N4. `qx.Command(GateType.Rz, 0, 1)` silently drops the angle

```python
import qarpx as qx

c = qx.Command(qx.GateType.Rz, 0, 1)  # meant: Rz on qubit 0 with angle 1
print(c)                              # Rz [q0, q1]
qx.QarpSimulator().run([c], 2, 10, 1)
# RuntimeError: gate 'Rz' missing param[0]
```

The integers are read as two qubits, and the error surfaces only when the command runs. Reorder the
overloads, or reject a parametric gate with no parameter at construction.

## Minor

- `scripts/generate_stubs.py` should check the ABI stamps first (`qarp._abi`). stubgen imports only
  `qarpx`, so in a shared venv it would quietly write another checkout's surface.
- Python and raw blocks use different keyword names: the sketch's `qubit`/`theta` against the Python
  overrides' `q`/`angle` (`_block.py:1126-1174`). mypy doesn't flag renamed override parameters, so
  the two would drift silently.
- 53 of the 77 `# type: ignore` comments in `qarp/` become unused; remove them in Phase 2.
- `Engine.build(primitives: list[Runnable])` is invariant, causing 9 strict errors; take a
  `Sequence`.
- Running `ruff check` on the stub directly still lints it; set `force-exclude = true`.
- `develop` has moved on to `4093ad5`.
- The header says `Draft` while the index row says 📋 Planned (the template causes this; #13 fixes
  it). The new `to_do.md` entry sits under "Additive, no window needed" but says "after the layout
  move". `AGENTS.md`'s "`[dev]` is ruff, mypy and pre-commit only" becomes false.
- "558 `qx.` references" counts lines, not references.
- The mixin split, `is_built` and `ControlledBlock` are §13 territory, which the plan doesn't cite.

## What checks out

- **Distribution.** A `qarpx.pyi` next to the compiled module isn't found by mypy
  (`import-not-found`); a `qarpx-stubs/__init__.pyi` package is (`builtins.int`). The CMake install
  line works in the wheel and in a plain editable install.
- **Generation.** `-P` adds exactly the private bindings `qarp` uses, and the output is
  byte-identical across runs. The ruff exclusion is needed and works.
- **The plan's specific counts.** 20 errors in the stub, 5 `truthy-function`, 40
  `func-returns-value` and 17 `.symbols` errors all reproduce.
- **The `n_controls` bug is real.**
- **The ABI** is defined in exactly two places (`bindings.cpp:165`, `qarp/_abi.py:9`), and the §15
  bullet the plan extends exists.
- **The split itself is necessary:** nanobind 3.0.1 refuses multiple inheritance, so it has to be
  `TYPE_CHECKING`-only. The base-alias form leaves the runtime classes unchanged.
- **Unaffected:** the CUDA-Q bindings are unconditional, the emit/absorb adapters don't change at
  runtime, and `qarp` has no `py.typed`, so users who only import `qarp` see no change.
