# Make `Engine.batch_run` reuse the prior build by default

**Status:** Planned
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-10-23
**Tier:** Standard
**Branch:** improvement/batch-run-no-implicit-rebuild
**Green-lit:**
**Scope:**
- `qarp/engines/_engine.py`: `batch_run` and `_batch_setup`
- `tests/test_engines/test_compiled_circuits.py`: the new tests
- the tests that sweep a primitive they never built, which build it first:
  `tests/test_algorithms/test_composite/test_dos_qpe.py`, `test_qpe.py`,
  `tests/test_blocks/test_structure.py`, `tests/test_engines/test_cudaq_engine.py`,
  `test_exact_readout.py`, `test_initial_state_engine.py`,
  `test_seed_decorrelation.py`, `test_structured_execution.py`, and three sites
  in `test_compiled_circuits.py`
- `docs/source/tutorial.rst`, `examples/tutorial_03_engines.ipynb`
- `docs/contracts/qarp_conventions.md`: one sentence in §14
- `CHANGELOG.md`, `docs/contributions/to_do.md`

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
  has no compiled circuits (`prim.compiled` empty), naming
  `engine.build(primitives)` and `rebuild=True` as the two ways out.  A sweep
  that compiled on first use would hide the cost the default flip exists to
  expose, and would make a second sweep's behaviour depend on whether a first
  one ran.  `run()` already evaluates only what the last `build()` compiled and
  never compiles; `batch_run` joins it.
- **Stale stamp.** A primitive built under other engine settings is recompiled
  by `build(rebuild=False)`, as §14 states; `batch_run` adds no second rule.
  A test pins the decision.
- **`rebuild=True`** compiles every primitive afresh, as today.
- **Untouched.** `_sweep`, the gradient path (it calls `_sweep` directly and
  never builds), `run()`, `build()`, the engines' `_compile_stamp()`.

**§14 edit, deliberate.** After "`build(rebuild=False)` reuses a primitive's
circuits only under an equal stamp" the paragraph gains: "`batch_run` reuses
them by default and raises `RuntimeError` naming `engine.build` for a primitive
with no compiled circuits; `rebuild=True` compiles afresh."  The tier table
puts conventions edits in the structural tier; the reviewer decides at the
green-light whether one sentence that restates a docstring needs the sit-down.

**Who sees a change.** `batch_run` on primitives that were never built raises
instead of compiling: build first, or pass `rebuild=True`.  `batch_run` after
`build()` on an unchanged engine no longer recompiles: faster, same results.
`batch_run` after `build()` on an engine whose settings changed recompiles, as
before.  `MMQCELS` is unchanged.  The tutorial and tutorial 03 build before
they sweep.

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
            for a primitive that has none; True compiles every primitive
            afresh.
        """

    def _batch_setup(self, primitives, param_sets, n_shots, rebuild):
        self._pre_run()
        if not rebuild:
            unbuilt = [type(p).__name__ for p in primitives if not p.compiled]
            if unbuilt:
                raise RuntimeError(
                    f"batch_run(rebuild=False) needs built primitives, but "
                    f"{', '.join(unbuilt)} has no compiled circuits: call "
                    "engine.build(primitives) first, or pass rebuild=True."
                )
        override = n_shots
        param_sets = [_coerce_params(ps) for ps in param_sets]
        self.build(primitives, rebuild=rebuild)
        ...
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| `batch_run` on primitives never built raises `RuntimeError` whose message names `engine.build`, and `prim.compiled` is still empty afterwards | the contract: exception type and text; the empty list | `tests/test_engines/test_compiled_circuits.py` |
| After `engine.build([prim])`, the default sweep reuses: the `CompiledCircuit` objects are the same objects before and after, and the planner is not called again | object identity; the `plan_calls` counter fixture | same |
| The reused sweep is right: `⟨Z⟩` of `Ry(θ)|0⟩` at θ ∈ {0, π/2, π} is cos θ | analytic | same |
| `rebuild=True` after a build compiles afresh: new `CompiledCircuit` objects, the same analytic values | identity and the analytic value | same |
| Built, engine settings changed, default sweep: recompiled (new objects), the analytic values hold | identity and the analytic value | same, beside `test_rebuild_false_under_other_settings_compiles_afresh` |
| `MMQCELS` passes its suite unchanged | the suite's existing oracles | `tests/test_algorithms/test_composite/test_mmqcels.py` |
| The tutorial snippet and tutorial 03 build before they sweep | docs lint (`tests/test_docs`); nightly `--nbmake` | `docs/source/tutorial.rst`, `examples/tutorial_03_engines.ipynb` |

## Phases

### Phase 1 — Code and tests

- [ ] `batch_run` default and the `_batch_setup` check
- [ ] The five new tests in `test_compiled_circuits.py`
- [ ] The twenty test call sites that sweep a primitive they never built build
      it first (`rebuild=True` only where the test is about rebuilding)
- [ ] `ruff`, `mypy`, `pytest` green

### Phase 2 — Docs and records

- [ ] `tutorial.rst` and tutorial 03: `engine.build([...])` before `batch_run`,
      and the text "evaluates every set of a built primitive as `run` would"
- [ ] §14 sentence
- [ ] `CHANGELOG.md`: Changed, with the two migrations
- [ ] `to_do.md`: the entry leaves

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
