# Changelog

All notable changes to OpenQARP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Structured execution.  `Block.statevector` and `QarpEngine`'s sampling
  paths read the block tree and run each piece as the cheapest exact kernel:
  a basis-state permutation as one gather, a small block as one dense
  matrix, a ladder of the same controlled block as one controlled-powers
  kernel, and the rest as fused gates.  A circuit with no such structure
  runs exactly as before.  `structured=False` on either turns it off and
  `QARP_STRUCTURED` sets the process default.  A block may declare its
  permutation with `classical_action` (`ModularMultiplicationBlock` does)
  or its composition with `structure()` and `qarp.blocks.Repeat`
  (`QPEBlock` and `DOSQPEBlock` do); `Block.kernels()` lists what
  `statevector` would run, and `Engine.kernels(primitive)` what `run()`
  and `batch_run` execute for each circuit of a built primitive.  See §14 of the
  conventions and the *Structured execution* section of the configuration
  page.
- `optimization_level` on `Block.statevector` and `QarpEngine`: the
  transpiler level applied to the gate slices after planning, or to the
  whole stream on the gate path.
- `qarp.SamplingDistribution`, the read-only result of `Sampler` and
  `PostSelection.apply`: aligned `outcomes` and `probabilities` arrays,
  `n_bits_measured`, `n_shots`, `probability_of`, `to_dict`, `from_dict`,
  `counts`, `standard_errors`, `marginal`, `to_dense`,
  `parity_expectation`, `top`, `most_likely`, `total_variation`,
  `hellinger_fidelity` and `sample`.
- `QARP_BLAS_THREADS`: `limit`, the default, lowers numpy's and scipy's
  OpenBLAS thread count to qarp's at the first simulation; `pool` also runs
  their parallel jobs on a qarpx-owned pool whose workers sleep as soon as a
  call ends; `native` leaves OpenBLAS untouched (see *Changed*).
- `examples/engines/mwe_structured_execution.ipynb`.
- Type information for the compiled backend.  The wheel ships a stub for
  `qarpx` as the `qarpx-stubs` package, so an editor shows the signatures
  of `import qarpx as qx` and mypy and pyright check calls into it.
- The raw `qarpx` block methods take keyword arguments, with the names the
  Python blocks use (`q`, `angle`, ...).  `cu`'s `gamma` is required on a
  raw block and defaults to 0 on a Python one.
- `qarp.blocks.QarpBlock`, the union of the Python block classes, for
  annotating code that reads members a raw `qx` block lacks (`.symbols`,
  `parameter_map`).  It works with `isinstance`.
- `qx.__with_symengine__`: whether the build has symbolic coefficients.
- A `ConditionalBlock` with only an `else_body`: pass `then_body=None`.

### Changed

- `Sampler` and `PostSelection.apply` return a `SamplingDistribution`
  instead of a `dict`, and `SamplingDictionary` is a read-only `Mapping`
  alias.  `isinstance(result, dict)` and mutating a result no longer work;
  `to_dict()` returns a plain dict.  Keys and probabilities are unchanged.
- A primitive implements `circuits()`, the block trees it runs over its
  inputs as given, in place of `build()`; `PrimitiveAlgorithm.build()`
  builds nothing, and the engine plans a declared structure before building
  it.  A primitive that populated `sub_blocks` in its own `build()` moves
  that code to `circuits()`.
- `QPE` and `DOSQPE` on a `QarpEngine` without a device no longer build
  their controlled-U ladder: the run goes through one controlled-powers
  kernel whatever the ancilla count and `block` stays unbuilt.  Any
  primitive over such a block is planned the same way, `PauliAveraging`
  included.
- `QarpEngine.batch_run` evaluates each parameter set as `run()` does: a
  finite-shot sweep runs a planned circuit's kernels instead of its gates,
  and accepts a `Sampler`'s `initial_state`; it raised before.  Seeded
  counts of a circuit with no plan are unchanged.
- `MirrorTest` accepts an input narrower than its working register and
  places it on the low qubits; it raised before.
- A primitive keeps its compiled circuits as `CompiledCircuit` records in
  `prim.compiled`, each stamped with the engine settings it was compiled
  under; `compiled_circuits` and `_n_qubits_list` are read-only views of
  them.  `QarpEngine` runs a counts circuit that needs no trajectory as a
  program and binds each parameter set in C++, which makes a `batch_run`
  sweep of a narrow, deep circuit about 10% faster.
- OpenBLAS's thread count in numpy and scipy is lowered to qarp's
  (`QARP_NUM_THREADS`) at the first simulation, so a BLAS call no longer
  leaves a worker spinning on every logical CPU against the next simulator
  call (a 16-qubit `statevector` right after a norm: 108 ms to under 8 ms).
  A count set in `OPENBLAS_NUM_THREADS` or in code is kept, and
  `QARP_BLAS_THREADS=native` restores the previous behaviour.
- `is_built` is a read-only property on raw `qx` blocks, as on the Python
  blocks.  `b.is_built()` and `qx.Block.is_built(b)` become `b.is_built`.
- `build()` on a raw `qx` block returns the block instead of `None`.
- A gate method, or `set_commands` with any command, called on a block
  that holds other blocks (composite, controlled, conditional) raises
  `TypeError`; the gate was recorded and never run.  Add it to a block it
  holds instead.
- Three attributes no longer hide gate methods of the same name:
  `AmplitudeAmplificationBlock.state_preparation` and
  `AmplitudeEstimationBlock.state_preparation` are `state_preparation_block`,
  and `OrbitalRotationBlock.u` is `rotation_matrix`.
- `qx.Command` checks a gate's qubit and parameter counts and raises
  `ValueError` on a mismatch; MCZ and Barrier, which take any number of
  qubits, are checked for taking no parameter.  It reads anything integer
  (`__index__`: an `int`, a numpy integer, a `bool`, a sympy `Integer`, a
  0-d integer array) in the second position as a qubit.
  `Command(CX, 0, np.int64(1))` builds the CX; it built a one-qubit CX
  carrying angle 1.  `Command(Rz, 0, 1)` raises; pass the angle as a float
  or a `qx.Param`.
- `qx.Command == other` returns `False` for an object of another type; it
  raised `TypeError`.
- `QubitOperator.substitute` and `free_symbols`, and their
  `FermionOperator` counterparts, exist on every build and raise without
  SymEngine.  Code that tested for the methods reads
  `qx.__with_symengine__`.
- `sparse_matrix` on both operator classes is a compiled method, also on a
  bare `import qarpx`.  It cannot be pickled or read by
  `inspect.signature`, and a non-integer `n_qubits` raises `TypeError`.
- `qx.bravyi_kitaev`, `qx.qubit_operator_coo` and `qx.fermion_operator_coo`
  raise `TypeError` on a non-integer `n_qubits`, as `sparse_matrix` does;
  they raised `RuntimeError: std::bad_cast`.
- `TermwiseHadamardTest` takes its operator blocks as any sequence; a tuple
  raised `TypeError`.
- `Engine.build` takes a `Sequence` of primitives.  A subclass that
  annotates the parameter `list` widens it to `Sequence`.
- The variational algorithms annotate their block parameters with
  `QarpBlock`.  A raw block there raised `AttributeError` and still does;
  wrap it, `CompositeBlock([raw_block])`.  `PauliShadow` takes a
  `QarpBlock` ket and raises `TypeError` on a raw one, with the same
  migration.
- mypy finds the `qarpx` stub, so an `ignore_missing_imports` for `qarpx`
  no longer applies, and a call the stub does not describe (an `np.int64`
  qubit index, an attribute set on a block) is reported.  To opt out, set
  `follow_imports = skip` and `follow_imports_for_stubs = True` for `qarpx`.
- The `qarpx` ABI is 15.  An editable checkout reinstalls after pulling,
  or `import qarp` refuses the old build.

### Removed

- `Engine.prepare_structured_qpe`, `StructuredQPEPlan` and the
  `QarpSimulator.simulate_qpe_structured` and `simulate_dosqpe_structured`
  bindings.  Structured execution covers their case for any block.

### Fixed

- `QPEBlock` and `DOSQPEBlock` place a copy of the eigenstate block, so one
  state block can serve two QPE blocks; sharing it corrupted the first.
- `build(rebuild=False)` and `batch_run(rebuild=False)` reuse a primitive's
  circuits only when they were compiled under the engine's settings.  Reused
  on another engine, a routed circuit ran without its layout and returned
  counts in physical qubit order.
- The default thread count under OpenMP binding (`OMP_PROC_BIND`,
  `OMP_PLACES`) counts the CPUs of all the places; the bound main thread's
  own place gave a count of 1.
- `HaarRandomBlock.depth()` returns the circuit depth; the block's layer
  count, which hid the method, is `n_layers`.  The `depth=` constructor
  argument is unchanged.
- A `CompositeBlock` subclass that calls `add_child` from its own
  `build_vanilla` raises `TypeError` at build; it built an empty block.
  Such a block extends `CompositeBlockBase`.
- `RotosolveOptimizer` reports `nit` as the number of iterations run; it
  was one short.
- The `FH_ham_and_wf` docs say it builds UCCSD; they said doubles only.
- Planning a raw `qx.ControlledBlock` reads its control count; it raised
  `AttributeError` once the span reached three gates.
- `ResourceEstimator.estimate`, `CircuitAdapter` and `CuttingPrimitive.build`
  treat an unbuilt raw block as unbuilt; they took it for built and
  `flatten()` raised.
- `SWAPTestBlock`, `HadamardTestBlock`, `QPEBlock`, `DOSQPEBlock`, `VFF`,
  `CuttingPrimitive` and `generate_states_new_basis` accept raw blocks; they
  used `build()`'s return value, which was `None` on a raw block.
- A `ConditionalBlock` with only an `else` branch can be deep-copied.
- `set_symbols` and `replace_symbols` with string keys update `.symbols`, so
  `parameter_map` no longer counts a bound symbol as free.
- `AdaptVQD.run()` returns the energy as a float, as documented; it
  returned a complex.
- A raw `qx.ControlledBlock` plots as a controlled box; it plotted as a
  plain box.
- `MirrorTest` daggers a raw bra that isn't built; it raised.
- `Reconstructer.reconstruct_filter_qubits` keeps a gate on three or more
  qubits whole: it rebuilt each kept gate from at most two qubits, so a CCZ
  became a CZ carrying an angle.  A gate on three or more qubits across the
  boundary raises, as a two-qubit one does; it was dropped.  The cutters
  still reject gates on more than two qubits.
- `CapabilityError.command` is `None` when no command is attached; reading
  it raised `AttributeError`.

## [0.1.1] - 2026-09-25

### Fixed

- `SynthesizedUnitaryBlock` and the other Quantum Shannon Decomposition
  paths are accurate to ~1e-13 on structured unitaries (DFT, Hadamard,
  phase-decorated targets) up to 7 qubits, where a 6-qubit DFT was far off:
  the cosine–sine decomposition divided by small sines, and the eigenvalue
  and diagonal tolerances merged distinct values.
- Synthesized unitaries are phase-exact.  The global phase is kept down to
  1e-15, so a synthesized block under `ControlledBlock` (for example inside
  QPE) matches the controlled target exactly instead of carrying a spurious
  relative phase.
- The default thread count no longer starts an OpenMP worker on every
  hardware thread, which starved the main thread.  With neither
  `QARP_NUM_THREADS` nor `OMP_NUM_THREADS` set, every layer uses the physical
  cores in the process's CPU affinity, capped one below its logical CPUs, and
  a container CPU limit (any cgroup v1 or v2 quota) lowers it further.  Under
  WSL and some virtual machines the reported topology undercounts physical
  cores; set `QARP_NUM_THREADS` there.

### Changed

- `CITATION.cff` names the arXiv paper as the preferred citation, with the
  Zenodo concept DOI for the software.

## [0.1.0] - 2026-09-14

First public release.  Distributed on PyPI as `openqarp`; the import name
is `qarp`.  Pre-release history from the internal repository is not carried
over; this file records changes from 0.1.0 onward.

[Unreleased]: https://github.com/OpenQARP/openqarp/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/OpenQARP/openqarp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/OpenQARP/openqarp/releases/tag/v0.1.0
