# Make `Engine.batch_run` reuse the prior build by default

**Status:** Landed
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-23
**Tier:** Standard
**Branch:** improvement/batch-run-no-implicit-rebuild
**Green-lit:** 2026-10-28, plan blob 54cb794b9c27f56e88543ae2856efebfe55cad62
**Scope:**
- `qarp/engines/_engine.py`: `batch_run`, `_batch_setup`, `build`,
  `CompiledCircuit`, `run`
- `qarp/engines/_runnable.py`, `qarp/algorithms/_primitives/primitive_algorithm.py`:
  the `built_empty` member
- `qarp/engines/_gradients.py`: the bound-symbol check in `compute_gradients`
- `tests/test_engines/test_compiled_circuits.py`: the new tests
- `tests/test_algorithms/test_primitives/test_shadows/`: the released-shadow test
- the tests that sweep a primitive they never built, which build it first:
  `tests/test_blocks/test_structure.py`, `tests/test_engines/test_cudaq_engine.py`,
  `test_exact_readout.py`, `test_initial_state_engine.py`,
  `test_seed_decorrelation.py`, `test_structured_execution.py`, and three sites
  in `test_compiled_circuits.py`
- `docs/source/tutorial.rst`, `examples/tutorial_03_engines.ipynb`
- `docs/contracts/qarp_conventions.md`: §14, the `batch_run` sentence, the
  binding clause and the planning-cache sentence
- `CHANGELOG.md`, `docs/contributions/to_do.md`, `docs/contributions/README.md`

---

## Why

`Engine.batch_run(primitives, param_sets, n_shots=None, rebuild=True)` compiles
its primitives on every call.  A user who follows the tutorial's advice to build
once and sweep many parameter sets pays the compile twice: once in
`engine.build()` and again, silently, in `batch_run`.  The default dates from
before a `CompiledCircuit` carried its settings stamp, when reusing circuits
across a changed engine was unsafe.  Since `lazy_circuit_derivation_plan.md`,
`build(rebuild=False)` reuses a primitive's circuits only when every one was
compiled under the engine's current `_compile_stamp()` and compiles afresh
otherwise (§14), so reuse is the safe default.  The one in-tree sweep that runs
repeatedly, `MMQCELS._circuit_signal`, already passes `rebuild=False`.

`to_do.md` records the change under *Breaking changes waiting for a minor
release*: default `rebuild=False`, reusing the prior `build()`, and a
`RuntimeError` naming `engine.build` when nothing is built.  0.2 already
changes the `Sampler` return type and the raw-block surface, so the default
flip rides it.

## Design

`batch_run` keeps its template shape, `_batch_setup` then the engine's
`_sweep`; only the default and one check change.

- **Default.** `rebuild: bool = False`.
- **Nothing built is an error, not a compile.** With `rebuild=False`,
  `_batch_setup` raises `RuntimeError` before anything runs when a primitive
  has circuits but none compiled, naming `engine.build(primitives)` and
  `rebuild=True` as the two ways out.  An empty `prim.compiled` alone is not
  the test: a built primitive over a constant operator holds no circuit, so
  `Engine.build` records `prim.built_empty = not prim.compiled` and
  `_batch_setup` raises for `not p.compiled and not p.built_empty`, reading
  the flag rather than deriving circuits (a shadow's `circuits()` drops its
  dataset).  A sweep that compiled on first use
  would hide the cost the default flip exists to expose, and would make a
  second sweep's behaviour depend on whether a first one ran.  `run()` already
  evaluates only what the last `build()` compiled and never compiles;
  `batch_run` joins it.
- **Stale stamp.** A primitive built under other engine settings is recompiled
  by `build(rebuild=False)`, as §14 states; `batch_run` adds no second rule.
  A test pins the decision.
- **`rebuild=True`** compiles every primitive afresh, as today.
- **Binding.** `build(params=)` bakes its values into the circuits, and
  `CompiledCircuit.bound` records them as sorted `(name, value)` pairs.
  `build(rebuild=False)` reuses a primitive's circuits only under an equal
  stamp and an equal binding; `batch_run` binds nothing, so it recompiles a
  bound build.  `run(params)` and `run_gradient(params)` naming a bound
  symbol raise `ValueError`.
- **The engine's primitive set.** `build` and `_batch_setup` share one
  per-primitive compile loop, `_compile_primitives`, which reports whether it
  compiled anything afresh.  `build` sets `_primitives` and bumps the build
  generation; `_batch_setup` leaves `_primitives` alone and bumps the
  generation only after a fresh compile, which invalidates the gradient shift
  plans keyed by index into `_primitives`.
- **Untouched.** `_sweep`, the engines' `_compile_stamp()`.

**§14 edit, deliberate.** After "`build(rebuild=False)` reuses a primitive's
circuits only under an equal stamp" the paragraph gains: "`batch_run` reuses
them by default and raises `RuntimeError` naming `engine.build` for a primitive
that has circuits but none compiled; `rebuild=True` compiles afresh."  The
binding clause and the planning-cache sentence follow from the deviations
below.  The tier table
puts conventions edits in the structural tier; the reviewer decides at the
green-light whether one sentence that restates a docstring needs the sit-down.

**Who sees a change.** `batch_run` on primitives that were never built raises
instead of compiling: build first, or pass `rebuild=True`.  `batch_run` after
`build()` on an unchanged engine no longer recompiles: faster, same results.
`batch_run` after `build()` on an engine whose settings changed recompiles, as
before.  `run()` and `run_gradient` after a sweep evaluate what the last
`build()` built, not the swept list.  `run(params)` naming a symbol
`build(params=)` bound raises instead of dropping the value.  `MMQCELS` is
unchanged.  The tutorial and tutorial 03 build before they sweep.

## API sketch

```python
class Engine:
    def batch_run(
        self,
        primitives: list[Runnable],
        param_sets: Sequence[Mapping],
        n_shots: Optional[Union[int, Shots]] = None,
        rebuild: bool = False,
    ) -> list[list[PrimitiveResult]]:
        """...
        rebuild: False (the default) reuses the circuits a prior ``build()``
            compiled under this engine's settings and raises ``RuntimeError``
            for a primitive that has circuits but none compiled; True compiles
            every primitive afresh.
        """

    def build(self, primitives, params={}, rebuild=True) -> None:
        self._build_generation += 1
        self._primitives = list(primitives)
        self._compile_primitives(self._primitives, _coerce_params(params), rebuild)

    def _compile_primitives(self, primitives, params, rebuild) -> bool:
        """True iff some primitive was compiled afresh."""
        # reuse iff not rebuild and every circuit has this stamp and binding;
        # otherwise compile, then prim.built_empty = not prim.compiled

    def _batch_setup(self, primitives, param_sets, n_shots, rebuild):
        self._pre_run()
        if not rebuild:
            unbuilt = [
                f"primitives[{i}] ({type(p).__name__})"
                for i, p in enumerate(primitives)
                if not p.compiled and not p.built_empty
            ]
            if unbuilt:
                raise RuntimeError(
                    "batch_run(rebuild=False) needs built primitives; not compiled: "
                    f"{', '.join(unbuilt)}.  Call engine.build(primitives) first, "
                    "or pass rebuild=True."
                )
        override = n_shots
        param_sets = [_coerce_params(ps) for ps in param_sets]
        if self._compile_primitives(primitives, {}, rebuild):
            self._build_generation += 1
        ...


@dataclass(frozen=True)
class CompiledCircuit:
    ...
    bound: tuple = ()  # the build(params=) binding, sorted (name, value) pairs


class Runnable(Protocol):
    built_empty: bool  # the last build() compiled no circuit for it
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| `batch_run` on primitives never built raises `RuntimeError` whose message names `engine.build`, and `prim.compiled` is still empty afterwards | the contract: exception type and text; the empty list | `tests/test_engines/test_compiled_circuits.py` |
| After `engine.build([prim])`, the default sweep reuses: the `CompiledCircuit` objects are the same objects before and after, and the planner is not called again | object identity; the `plan_calls` counter fixture | same |
| The reused sweep is right: `⟨Z⟩` of `Ry(θ)|0⟩` at θ ∈ {0, π/2, π} is cos θ | analytic | same |
| `rebuild=True` after a build compiles afresh: new `CompiledCircuit` objects, the same analytic values | identity and the analytic value | same |
| Built, engine settings changed, default sweep: recompiled (new objects), the analytic values hold | identity and the analytic value | same, beside `test_rebuild_false_under_other_settings_compiles_afresh` |
| A built primitive over a constant operator (`compiled == []`) sweeps by default and returns the constant per set | the constant, 1.5 | same |
| A list of one built and two never-built primitives raises naming `primitives[1]` and `primitives[2]` only; the built one's `CompiledCircuit` objects are the same objects afterwards | the contract: the text; object identity | `tests/test_engines/test_compiled_circuits.py` |
| A released `PauliShadow` swept by default raises, keeps the same dataset object, and the dataset still reads the estimate `run()` returned | the contract: object identity; the estimate before `release()` | `tests/test_algorithms/test_primitives/test_shadows/test_base.py` |
| `build(params={"a": 0})` then the default sweep over a ∈ {0, π/2, π} is cos a | analytic | `tests/test_engines/test_compiled_circuits.py` |
| `build(params={"a": 0})` then `build(params={"a": π}, rebuild=False)`: `run()` is −1; an equal binding reuses the same objects | analytic; object identity | same |
| After `build(params={"a": 0})`, `run({"a": π})` and `run_gradient({"a": π})` raise `ValueError` naming `a` | the contract | same |
| `build([a, b])` (Z0 and X0 over `Ry(θ)`), a sweep over `[a]`, then `run({θ})` is `[cos θ, sin θ]` | analytic | same |
| `build([a, b])`, `run_gradient`, a default sweep, `run_gradient` again: no new shift plan, gradient `[−sin θ, cos θ]`; after `batch_run([a], rebuild=True)` the gradient still holds | analytic; the plan counter | same |
| `MMQCELS` passes its suite unchanged | the suite's existing oracles | `tests/test_algorithms/test_composite/test_mmqcels.py` |
| The tutorial snippet and tutorial 03 build before they sweep | docs lint (`tests/test_docs`); nightly `--nbmake` | `docs/source/tutorial.rst`, `examples/tutorial_03_engines.ipynb` |

## Phases

### Phase 1 — Code and tests

- [x] `batch_run` default and the `_batch_setup` check *(2026-10-29)*
- [x] Five new tests in `test_compiled_circuits.py`, covering the first six
      rows of the test table *(2026-10-29)*
- [x] The test call sites that sweep a primitive they never built build it
      first (`rebuild=True` only where the test is about rebuilding) *(2026-10-29)*
- [x] `ruff`, `mypy`, `pytest` green *(2026-10-29)*

### Phase 2 — Docs and records

- [x] `tutorial.rst` and tutorial 03: `engine.build([...])` before `batch_run`,
      and the text "evaluates every set of a built primitive as `run` would"
      *(2026-10-29)*
- [x] §14 sentence *(2026-10-29)*
- [x] `CHANGELOG.md`: Changed, with the two migrations *(2026-10-29)*
- [x] `to_do.md`: the entry leaves *(2026-10-29)*

### Phase 3 — Review fixes

- [x] `built_empty` replaces the `prim.build()` probe; error text names positions *(2026-11-02)*
- [x] `CompiledCircuit.bound`; reuse under an equal binding; `run` /
      `run_gradient` reject a bound symbol *(2026-11-02)*
- [x] `_compile_primitives`; `_batch_setup` keeps `_primitives` *(2026-11-02)*
- [x] The new tests in the test table; the two settings tests pin `structured=True` *(2026-11-02)*
- [x] §14: the binding clause and the planning-cache sentence; `CHANGELOG.md` *(2026-11-02)*
- [x] `ruff`, `mypy`, `pytest` green *(2026-11-02)*

## Deviations log

- **Unbuilt is "circuits but none compiled", not an empty `prim.compiled`.**
  A built `PauliAveraging` over a constant operator compiles no circuit, so
  the green-lit test would have refused a legitimately built primitive with
  advice that cannot help.  `_batch_setup` raises only for a primitive that
  has circuits (how it knows: the `built_empty` entry below); a sixth test
  pins the constant sweep, and
  the §14 sentence, the docstring and the Design section say "has circuits
  but none compiled".
- **Fourteen test sites changed, not twenty.**  The grep behind the count
  included `test_dos_qpe.py`, `test_qpe.py` and the second sweep in
  `test_seed_decorrelation.py`, whose engine had already built the primitive
  (`QPE.build()` and `DOSQPE.build()` call `engine.build`); they reuse under
  the new default and are unchanged, and they leave Scope.
- **The unbuilt check reads a flag, not `prim.build()`.**  Deriving the
  circuits to decide is not side-effect free: `PauliShadow.circuits()`
  redraws its settings and drops the collected dataset, so a raising sweep
  over a released shadow destroyed the data `release()` keeps.
  `Engine.build` sets `prim.built_empty = not prim.compiled` (a `Runnable`
  member, `False` on a fresh primitive and on a rejected build), and
  `_batch_setup` raises for `not p.compiled and not p.built_empty`.
- **Reuse also requires the same `build(params=)` binding.**  `build` bakes
  `params` into the circuits; reuse compared only the settings stamp, so a
  sweep after `build(params={"a": 0})` dropped every swept value of `a`
  (cos a read `[1, 1]` for a ∈ {0, π}).  Develop's default recompiled and was
  right, so the default flip turned this into a regression.
  `CompiledCircuit.bound` holds the binding as sorted `(name, value)` pairs;
  `build(rebuild=False)` reuses only under an equal stamp *and* binding, and
  `batch_run`, which binds nothing, recompiles a bound build.  `run(params)`
  and `run_gradient(params)` naming a symbol `build` bound raise `ValueError`:
  `run()` never compiles, and the value would be dropped.  §14 gains the
  clause.
- **`batch_run` no longer re-targets the engine.**  It called
  `self.build(primitives)`, which replaced the engine's primitive list and
  bumped the build generation: `build([a, b])` then `batch_run([a])` left
  `run()` evaluating `a` alone, and every sweep discarded the cached gradient
  shift plans.  The per-primitive compile loop moves to
  `_compile_primitives`, which `build` and `_batch_setup` share;
  `_batch_setup` keeps `_primitives`, and bumps the generation only when it
  compiled something afresh, since the shift plans are keyed by index into
  `_primitives` and read its circuits.  `run()` after a `batch_run` on a
  fresh engine therefore evaluates nothing until `build()`.  This touches
  `build` and the gradient cache's invalidation, which the Design lists as
  untouched.
- **Error text names positions.**  The `RuntimeError` lists the unbuilt
  primitives as `primitives[i] (Type)`.
- **§14's planning-cache sentence.**  "a rebuild of the same circuit (every
  `batch_run`) never plans twice" is wrong once the default sweep reuses;
  it reads "(a repeated `build()`, `batch_run(rebuild=True)`)".
