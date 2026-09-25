OpenQARP tutorial
=================

A guided path through the whole stack, in the order you meet it: build a
circuit, say what you want from it, run it, then wrap that in a loop.  We assume
you know quantum computing — this tutorial only teaches you *this package*.

The mental model, one row per section:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Layer
     - Object
     - Role
   * - Circuit
     - ``Block``
     - what the quantum computer *does*
   * - Quantity
     - ``PrimitiveAlgorithm`` (``Sampler``, ``StateVector``, …)
     - what you want to *extract*
   * - Execution
     - ``QarpEngine``
     - compiles once, runs many times
   * - Loop
     - optimizers and composite algorithms (``VQE``, ``QAOA``, …)
     - full workflows

The same material is available as six runnable notebooks,
``examples/tutorial_00_hello_qarp.ipynb`` through
``examples/tutorial_05_algorithm_tour.ipynb``; the sections below follow them
one for one.  For the catalogue of *every* block, primitive and algorithm, the
``mwe_*.ipynb`` notebooks under ``examples/`` are the reference gallery — this
is the guided path.  Worked applications that combine several of them on a real
problem live under ``examples/use_cases/``, named by field rather than by
algorithm.


Hello OpenQARP
--------------

The smallest possible pass through the whole stack: build a Bell state and
sample it.

Circuits are blocks.  A :class:`~qarp.blocks.SimpleBlock` is a flat list of
gates with builder methods (``.h()``, ``.cx()``, ``.rx()``, …).  Declare the
number of qubits up front, add gates, then call ``.build()`` — building
finalises the block so the engine can compile it.

.. code-block:: python

    from qarp.blocks import SimpleBlock

    bell = SimpleBlock(2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([(q, q) for q in range(2)])  # measure qubit q into classical bit q
    bell.build()

A *primitive* pairs a circuit with the quantity you want from it.
:class:`~qarp.algorithms.Sampler` asks for the measured bitstring distribution.

.. code-block:: python

    from qarp.algorithms import Sampler

    sampler = Sampler(ket=bell, n_shots=4000)

:class:`~qarp.engines.QarpEngine` compiles the primitive's circuits
(``.build()``) and simulates them (``.run()``).  Passing ``seed=`` makes the
shots reproducible.

.. code-block:: python

    from qarp.engines import QarpEngine

    engine = QarpEngine(seed=42)
    engine.build([sampler])
    results = engine.run()

    distribution = results[0]
    print(distribution)

You should see a :class:`~qarp.SamplingDistribution` with, up to shot noise,
about 50 % ``(0, 0)`` and 50 % ``(1, 1)`` — a Bell state.  It reads like a
dictionary.  Two things to internalise straight away:

* The keys are **tuples of bits indexed by qubit**: position ``q`` in the tuple
  is qubit ``q``.  OpenQARP is **LSB-first** everywhere (qubit 0 is the
  least-significant bit); see `Bit order`_ below.
* The values are **probabilities** — counts normalised by ``n_shots`` — not raw
  counts.

The result is also stored on the primitive itself, and every object you just
used is an ordinary Python object you can inspect:

.. code-block:: python

    print("result:          ", sampler.result)
    print("n_qubits:        ", bell.n_qubits)
    print("command stream:  ", len(bell.flatten()), "commands")
    print("primitive target:", sampler.target)


Blocks: circuits as composable objects
--------------------------------------

Everything the quantum computer does is a block, and there are two kinds:

* :class:`~qarp.blocks.SimpleBlock` — a flat gate list you fill with builder
  methods.
* :class:`~qarp.blocks.CompositeBlock` — an ordered tree of child blocks, so
  circuits compose like Lego: ``[state-prep, ansatz, readout]``.

On top of these, :mod:`qarp.blocks` ships a library of ready-made blocks
(``HnBlock``, ``HEABlock``, ``ComputationalBasisStateBlock``, ``QFTBlock``,
``TrotterBlock``, …) — see :doc:`blocks` for the full catalogue.

Gate builders and the build lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from sympy import Symbol

    theta = Symbol("theta")

    demo = SimpleBlock(3, name="demo")
    demo.h(0)
    demo.cx(0, 1)
    demo.ry(2, theta)        # parametric gates accept sympy Symbols (or floats)
    demo.rz(2, 0.25)         # angles are ALWAYS in radians
    demo.build()

Before ``.build()`` you add gates, compose and transform.  After it the block is
finalised: ``.symbols`` is populated, and the block can be plotted, flattened,
or handed to an engine.  Calling ``.build()`` again is a no-op.

``.flatten()`` returns the raw command stream — the C++ IR — which is exactly
what will be compiled:

.. code-block:: python

    print("symbols:", demo.symbols)
    for cmd in demo.flatten():
        print(cmd)

Call ``demo.plot()`` to draw the circuit (see :doc:`plotting` for the options).

.. admonition:: Convention 1 — symbols are sorted
   :class: important

   ``block.symbols`` is a tuple sorted by symbol *name*, not by insertion order.
   Never hand-zip a parameter vector against your own gate order — use
   ``block.parameter_map(values)``, which pairs a vector with ``.symbols`` in
   the canonical order for you.

Composition
^^^^^^^^^^^

:class:`~qarp.blocks.CompositeBlock` glues child blocks in order.  The canonical
pattern is ``[state-preparation, ansatz, readout]``:

.. code-block:: python

    from qarp.blocks import (
        CompositeBlock,
        ComputationalBasisStateBlock,
        HEABlock,
        ReadoutBlock,
    )

    prep = ComputationalBasisStateBlock([1, 1, 0, 0])   # X on qubits 0, 1
    hea = HEABlock(4, 2, True, True, True, False)       # n_qubits, n_layers, real, linear, circular, use_cz
    readout = ReadoutBlock(n_qubits=4)                  # measure all qubits, cbit q <- qubit q

    circuit = CompositeBlock([prep, hea, readout]).build()

Each child draws as one box; pass ``decompose_boxes=True`` to ``.plot()`` to see
the gates inside.

There are three ways to measure, and they reduce to one another:

* ``block.measure(q, c)``, or a list of ``(qubit, cbit)`` pairs — the builder
  method on a block you are already filling, as in `Hello OpenQARP`_ above.
* :class:`~qarp.blocks.ReadoutBlock` — a composable child for bulk readout, as
  here.
* :class:`~qarp.blocks.MeasureBlock` — the single-measurement structural
  primitive both of the above reduce to.

For pure sampling the engine reports counts over all qubits even without
explicit measures, but making them explicit keeps the IR honest, and is required
for QASM/QIR export.

Setting parameter values
^^^^^^^^^^^^^^^^^^^^^^^^

Symbolic parameters get values in one of two places:

* **At run time** — pass a ``{symbol: value}`` map to ``engine.run(...)``.  This
  is the usual way inside an optimisation loop.
* **On the block** — ``set_symbols`` returns a **new** block with the values
  recorded; the original is untouched.

``set_symbols`` is *lazy*: the substitution is stored on the new block and
applied when the block is flattened, which is what the engine does when it
compiles.  The symbol *list*, though, updates immediately: a bound symbol drops
out of ``.symbols``, so ``.symbols`` always tells you what is still free.  The
place to see the bound values is the command stream:

.. code-block:: python

    bound = circuit.set_symbols(circuit.parameter_map([0.1] * len(circuit.symbols)))

    print("original:", circuit.flatten()[2:5])   # Ry gates, still symbolic
    print("bound:   ", bound.flatten()[2:5])     # same gates, values applied
    print("still free:", bound.build().symbols)   # () — all eight were bound

.. admonition:: Convention 2 — angles are radians
   :class: important

   Every rotation angle, and every value you substitute for a symbol, is
   interpreted in **radians**.  So ``ry(0, np.pi)`` flips ``|0>`` to ``|1>``:

.. code-block:: python

    flip = SimpleBlock(1)
    flip.ry(0, np.pi)
    flip.measure(0, 0)
    flip.build()

    flip_sampler = Sampler(ket=flip, n_shots=1000)
    flip_engine = QarpEngine(seed=1)
    flip_engine.build([flip_sampler])
    print(flip_engine.run()[0])

Bit order
^^^^^^^^^

.. admonition:: Convention 3 — LSB-first
   :class: important

   Qubit ``q`` is bit ``q``: in a sampled tuple, position ``q`` belongs to qubit
   ``q``, and in an integer basis-state label qubit ``q`` contributes ``2**q``.
   Operator matrices (``op.sparse_matrix()``) use the same convention.

:mod:`qarp.endianness` has the converters — reach for them only at a boundary
with an external MSB-first tool (openfermion, cirq, pennylane).  See
:doc:`endianness` for the full treatment.

.. code-block:: python

    from qarp.endianness import bits_to_label, label_to_bits

    bits = (0, 1, 1)            # qubit 0 = 0, qubit 1 = 1, qubit 2 = 1
    print(bits_to_label(bits))  # 2**1 + 2**2 = 6
    print(label_to_bits(6, 3))


Primitives: from circuits to numbers
------------------------------------

A :class:`~qarp.algorithms.PrimitiveAlgorithm` declares **what you want to
extract** from circuits.  You hand primitives to an engine, and the engine
returns one result per primitive.

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Primitive
     - Returns
     - How
   * - ``Sampler``
     - bitstring distribution
     - simulates and samples shots
   * - ``StateVector``
     - exact ``float`` / ``complex`` scalar
     - exact statevector contraction, no shots

Shot-based *estimators* of the same scalars exist too
(``TermwiseHadamardTest``, ``SWAPTest``, ``PauliAveraging``, …) — same
interface, sampled instead of exact.  See :doc:`algorithms` for the catalogue.

The target is inferred from what you pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Primitives take up to three ingredients — ``bra``, ``operator``, ``ket`` — and
infer the quantity they compute (their ``target``) from which ones you supply:

* ``ket`` only → **sampling**
* ``bra``, ``ket`` → **overlap** :math:`\langle \mathrm{bra}|\mathrm{ket}\rangle`
* ``bra`` = ``ket``, ``operator`` → **expectation value** :math:`\langle \psi|H|\psi\rangle`
* ``bra``, ``operator``, ``ket`` → **transition amplitude** :math:`\langle \mathrm{bra}|H|\mathrm{ket}\rangle`

.. code-block:: python

    from qarp.operators import QubitOperator
    from qarp.blocks import HnBlock
    from qarp.algorithms import StateVector

    psi = HEABlock(3, 2, True, True, True, False).build()
    psi_01 = psi.set_symbols(psi.parameter_map([0.1] * len(psi.symbols))).build()
    plus = HnBlock(n_qubits=3).build()

    H = QubitOperator("Z0 Z1", -1.0) + QubitOperator("X2", 0.5)

    for prim in (
        Sampler(ket=psi_01),
        StateVector(bra=plus, ket=psi_01),
        StateVector(bra=psi_01, operator=H, ket=psi_01),
        StateVector(bra=plus, operator=H, ket=psi_01),
    ):
        prim.build()
        print(f"{type(prim).__name__:12s} -> {prim.target}")

Exact expectation values
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    expval = StateVector(bra=psi, operator=H, ket=psi)   # symbolic: values at run time

    ev_engine = QarpEngine()
    ev_engine.build([expval])

    param_values = psi.parameter_map(np.linspace(0.0, 1.0, len(psi.symbols)))
    print("<psi|H|psi> =", ev_engine.run(param_values)[0])
    print("stored on the primitive too:", expval.result)

The operator is a :class:`~qarp.operators.QubitOperator` with an
openfermion-compatible API.  Its qubit indices are OpenQARP qubit indices, and
``op.sparse_matrix()`` realises it in the same LSB convention as every OpenQARP
statevector — no conversion anywhere.  Only when you bring in an *external*
MSB-first matrix or statevector do you convert; eigenvalues need no conversion,
since they do not depend on bit order.

Overlaps
^^^^^^^^

.. code-block:: python

    overlap = StateVector(bra=plus, ket=psi_01)
    ov_engine = QarpEngine()
    ov_engine.build([overlap])
    print("<+++|psi(0.1)> =", ov_engine.run()[0])

Sampling and shot-based estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Sampler`` returns the measured distribution, and ``n_shots`` set on the
primitive overrides the engine default.  For a *shot-based estimate of an
expectation value* — what real hardware gives you — use
``TermwiseHadamardTest`` or ``PauliAveraging``, which take the same
``bra``/``operator``/``ket`` interface as ``StateVector``:

.. code-block:: python

    from qarp.algorithms import TermwiseHadamardTest

    exact_ev = StateVector(bra=psi_01, operator=H, ket=psi_01)
    estimated_ev = TermwiseHadamardTest(bra=psi_01, operator=H, ket=psi_01, n_shots=20_000)

    shot_engine = QarpEngine(seed=7)
    shot_engine.build([exact_ev, estimated_ev])      # one engine, several primitives
    res = shot_engine.run()
    print("exact:    ", res[0])
    print("estimated:", res[1])

Which to choose: reach for ``StateVector`` while developing or debugging an
algorithm — it is deterministic, fast and differentiable.  Reach for
``Sampler`` or the Hadamard-test family, with a finite ``n_shots`` and a seeded
engine, when you are modelling what hardware would measure.


Engines: compile once, run many
-------------------------------

:class:`~qarp.engines.QarpEngine` is the only execution object you need, and its
API splits the work the way an optimisation loop wants it split:

* ``engine.build(primitives)`` — flatten, transpile and compile every circuit
  once.  Symbols stay symbolic.
* ``engine.run(symbol_map)`` — substitute values and simulate.  Cheap; call it
  thousands of times.

That split is why variational loops are fast: the expensive compilation never
happens inside the loop.

.. code-block:: python

    eng_ansatz = HEABlock(4, 2, True, True, True, False).build()
    eng_H = QubitOperator("Z0 Z1") + QubitOperator("Z2 Z3") + QubitOperator("X0", 0.3)

    eng_expval = StateVector(bra=eng_ansatz, operator=eng_H, ket=eng_ansatz)

    sweep_engine = QarpEngine()
    sweep_engine.build([eng_expval])                  # compile once...

    for angle in (0.0, 0.3, 0.6):                     # ...run many
        params = eng_ansatz.parameter_map([angle] * len(eng_ansatz.symbols))
        print(f"theta={angle:.1f}  <H> = {sweep_engine.run(params)[0]:+.6f}")

Reproducibility
^^^^^^^^^^^^^^^

Shot noise comes from the engine's C++ RNG, not from NumPy.  Seed it at
construction — same seed, same shots:

.. code-block:: python

    for attempt in range(2):
        repeat_sampler = Sampler(ket=bell, n_shots=100)
        repeat_engine = QarpEngine(seed=123)
        repeat_engine.build([repeat_sampler])
        print(repeat_engine.run()[0])

Parameter sweeps with ``batch_run``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For many parameter sets over the same circuits, ``batch_run`` keeps the
simulation loop in C++:

.. code-block:: python

    param_sets = [
        eng_ansatz.parameter_map([t] * len(eng_ansatz.symbols))
        for t in np.linspace(0, 2 * np.pi, 9)
    ]

    batch_engine = QarpEngine()
    batch = batch_engine.batch_run([eng_expval], param_sets)
    for ps, row in zip(param_sets, batch):
        print(f"theta={list(ps.values())[0]:+.3f}  <H> = {row[0]:+.6f}")

Analytic gradients
^^^^^^^^^^^^^^^^^^

For ``StateVector`` expectation values the engine computes exact gradients by
adjoint backpropagation, at a cost of roughly two simulations regardless of the
number of parameters.  Other primitives take the batched parameter-shift rule,
exact for every affine gate angle.  ``run_gradient(params, method=...)`` also
accepts ``"parameter-shift"``, ``"finite-diff"`` and ``"spsa"`` explicitly —
see :doc:`gradients` for the registry and the return contract.

.. code-block:: python

    grad_engine = QarpEngine()
    grad_engine.build([eng_expval])

    grad_params = eng_ansatz.parameter_map([0.4] * len(eng_ansatz.symbols))
    grad = grad_engine.run_gradient(grad_params)[0]
    print("dE/dtheta_k:", np.round(grad, 6))

Devices
^^^^^^^

Everything above ran on an ideal all-to-all simulator.  The engine also accepts
a :class:`~qarp.devices.Device` — qubit count, connectivity, gate set, noise
model — and then routes and rebases circuits during ``build``, and simulates
noise trajectories during ``run``.  That is its own topic: see :doc:`devices`
and :doc:`engines`.


Your own variational loop
-------------------------

Time to assemble the layers.  We find the ground state of a three-qubit
transverse-field Ising model

.. math::

    H = -\sum_{i} Z_i Z_{i+1} - h \sum_i X_i, \qquad h = 0.5

first **by hand** — ansatz, primitive, engine, optimizer — and then with the
built-in ``VQE`` composite, so you can see it is the same machinery.

.. code-block:: python

    n_spins = 3
    field = 0.5
    H_tfim = QubitOperator()
    for i in range(n_spins - 1):
        H_tfim += QubitOperator(f"Z{i} Z{i+1}", -1.0)
    for i in range(n_spins):
        H_tfim += QubitOperator(f"X{i}", -field)

    # Exact reference from the dense matrix (LSB, like everything else).
    exact_energy = np.linalg.eigvalsh(H_tfim.sparse_matrix().toarray())[0]
    print("exact ground energy:", exact_energy)

The four ingredients
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qarp.optimizers import ScipyOptimizer

    tfim_ansatz = HEABlock(n_spins, 2, True, True, True, False).build()
    tfim_expval = StateVector(bra=tfim_ansatz, operator=H_tfim, ket=tfim_ansatz)

    tfim_engine = QarpEngine()
    tfim_engine.build([tfim_expval])        # compile once, before the loop

    print(len(tfim_ansatz.symbols), "parameters")

The objective function
^^^^^^^^^^^^^^^^^^^^^^

An objective is just this: map the parameter vector onto the block's symbols,
run the engine, take the scalar.  ``parameter_map`` is what keeps the vector
aligned with the name-sorted ``.symbols`` (Convention 1).

.. code-block:: python

    evals = []

    def energy(x):
        value = tfim_engine.run(tfim_ansatz.parameter_map(x))[0].real
        evals.append(value)
        return value

Minimise
^^^^^^^^

.. code-block:: python

    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, 2 * np.pi, len(tfim_ansatz.symbols))

    opt = ScipyOptimizer("COBYLA", options={"maxiter": 400})
    result = opt.minimize(energy, x0)

    print(f"VQE energy:   {result.fun:.6f}")
    print(f"exact energy: {exact_energy:.6f}")
    print(f"error:        {result.fun - exact_energy:.2e}   ({len(evals)} evaluations)")

``evals`` now holds the convergence trace, ready to plot against
``exact_energy``.

With gradients
^^^^^^^^^^^^^^

``engine.run_gradient`` plugs straight into gradient-based optimizers (the
composite algorithms below accept the same method names through
``gradient=``):

.. code-block:: python

    def gradient(x):
        return tfim_engine.run_gradient(tfim_ansatz.parameter_map(x))[0]

    result_cg = ScipyOptimizer("CG").minimize(energy, x0, gradient=gradient)
    print(f"CG energy:    {result_cg.fun:.6f}   error {result_cg.fun - exact_energy:.2e}")

The reveal: this *is* ``VQE``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``VQE`` composite wires up exactly the objects you just wired.  It owns the
primitive, engine, optimizer and loop, and exposes the same knobs as
constructor arguments:

.. code-block:: python

    from qarp.algorithms import VQE

    vqe = VQE(
        operator=H_tfim,
        ket=HEABlock(n_spins, 2, True, True, True, False).build(),
        initial_parameters=x0,
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 400}),
        primitive=StateVector(),
        engine=QarpEngine(),
        verbose=False,
    )
    vqe.build()
    vqe_energy, vqe_params = vqe.run()
    print(f"VQE composite: {vqe_energy:.6f}   error {vqe_energy - exact_energy:.2e}")

Swap ``primitive=`` for ``TermwiseHadamardTest(n_shots=...)`` and the same
algorithm runs shot-based; swap ``engine=`` for a noisy device engine and it
runs noisy.  That orthogonality — algorithm × primitive × engine — is the core
design of the package.


The algorithm toolbox
---------------------

You now know the whole stack: blocks → primitives → engine → loop.  Every
composite algorithm in :mod:`qarp.algorithms` is that same pattern,
packaged.  This section runs one end to end, then maps the rest of the toolbox.

One complete run: QAOA on MaxCut
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combinatorial problems skip chemistry entirely, which makes them a compact
end-to-end demo: a six-node MaxCut via QAOA.

.. code-block:: python

    import networkx as nx
    from qarp.algorithms import QAOA

    G = nx.gnm_random_graph(6, 9, seed=7)
    for u, v in G.edges:
        G[u][v]["weight"] = 1

    qaoa = QAOA(
        problem=G,
        n_layers=2,
        use_rzz=False,
        initial_parameters=[0.1] * 4,          # 2 angles per layer
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 150}),
        verbose=False,
    ).build()

    qaoa.run()

    # The optimised ansatz, the optimum and the optimisation record are all on the object.
    print("best energy: ", qaoa.result.fun)
    print("n parameters:", len(qaoa.result.x))

Note the shape: construct with problem plus knobs → ``.build()`` → ``.run()`` →
inspect attributes.  Every algorithm below follows it.

The toolbox
^^^^^^^^^^^

**Ground states (chemistry and spin)**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Algorithm
     - One-liner
     - Notebook
   * - ``VQE``
     - variational ground-state search
     - ``mwe_vqe.ipynb``
   * - ``AdaptVQE``
     - grows the ansatz greedily from an operator pool
     - ``mwe_adapt_vqe_vqd.ipynb``
   * - ``ProjectedVQE``
     - variation after projection: minimise inside a symmetry sector
     - ``mwe_projected_vqe.ipynb``

**Excited states**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Algorithm
     - One-liner
     - Notebook
   * - ``VQD``
     - deflation: penalise overlap with states already found
     - ``mwe_vqd.ipynb``
   * - ``SSVQE``
     - one weighted optimisation, several states
     - ``mwe_ssvqe.ipynb``
   * - ``AdaptVQD``
     - ADAPT ansatz growth for excited states
     - ``mwe_adapt_vqe_vqd.ipynb``
   * - ``QSE``
     - subspace expansion on top of a reference state
     - ``mwe_qse.ipynb``

**Eigenvalue and phase estimation**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Algorithm
     - One-liner
     - Notebook
   * - ``QPE``
     - textbook phase estimation with an ancilla register
     - ``mwe_qpe.ipynb``
   * - ``DOSQPE``
     - density-of-states QPE with a mixed probe
     - ``mwe_dosqpe.ipynb``
   * - ``MMQCELS``
     - multi-modal QCELS eigenvalue fitting
     - ``mwe_mmqcels.ipynb``
   * - ``QMEGS``
     - Gaussian-filter spectral estimation
     - ``mwe_qmegs.ipynb``

**Combinatorial optimisation**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Algorithm
     - One-liner
     - Notebook
   * - ``QAOA``
     - the run above
     - ``mwe_qaoa_max_cut.ipynb``
   * - ``PCE``
     - Pauli-correlation encoding: ~100-node graphs on few qubits
     - ``mwe_pce_algo.ipynb``

**Dynamics and simulation primitives**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Tool
     - One-liner
     - Notebook
   * - ``TrotterBlock``
     - Trotter–Suzuki evolution as a block
     - ``mwe_trotter.ipynb``
   * - qDRIFT
     - randomised compilation of evolution
     - ``mwe_qdrift.ipynb``
   * - Qubitization / QSVT
     - block-encoding based simulation
     - ``mwe_qubitization.ipynb``, ``mwe_qsvt.ipynb``
   * - ``VFF``
     - variational fast-forwarding of dynamics
     - ``mwe_vff.ipynb``
   * - ``MonteCarlo``
     - projector quantum Monte Carlo
     - ``mwe_montecarlo.ipynb``

**Infrastructure**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Topic
     - Notebook
   * - Devices, noise, routing
     - ``mwe_devices.ipynb``, ``mwe_engines.ipynb``
   * - Measurement primitives catalogue
     - ``mwe_primitive_algorithms.ipynb``
   * - Block catalogue
     - ``mwe_blocks.ipynb``
   * - Block factories from operators
     - ``mwe_factories.ipynb``
   * - Circuit cutting
     - ``mwe_circuit_cutting.ipynb``
   * - Graph utilities
     - ``mwe_graphs.ipynb``
   * - Circuit plotting
     - ``mwe_plot.ipynb``
   * - Optimizers
     - ``mwe_optimizers.ipynb``, ``mwe_parameter_optimization.ipynb``

Where you are now
^^^^^^^^^^^^^^^^^

You can read any of those notebooks fluently: find the blocks, find the
primitive, find the engine, find the loop.  When something behaves oddly, check
the three conventions first — name-sorted symbols, radians, LSB-first.
