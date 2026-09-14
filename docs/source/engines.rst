Engines
==============

The engines module is the execution backend of OpenQARP. Every quantum computation flows through an :class:`~qarp.engines.Engine` instance. If a :code:`Device` is given to the engine, it owns the rebase / route / re-rebase compilation pipeline (plus per-gate noise injection); otherwise the engine simulates the circuit as written.

OpenQARP currently provides two qarpx-native engines. The default, always available:

- :class:`~qarp.engines.QarpEngine` — fast statevector / trajectory simulation backed by ``qx.QarpSimulator``.

The second, :class:`~qarp.engines.CudaqEngine`, wraps ``qx.CudaqSimulator`` for GPU
execution (single/multi-GPU statevector or tensor-network backends). It needs both the
``cuda-quantum-cu12`` Python runtime *and* a qarpx build compiled with CUDA-Q support
(``QARP_WITH_CUDAQ=ON`` — the published CPU wheel is not built this way); the
`Installation <installation.html>`_ page has the build steps. A
CPU-only build raises a clear, actionable error from ``CudaqEngine`` — use ``QarpEngine``
for CPU simulation.

To implement a custom engine, subclass :class:`~qarp.engines.Engine`.

QarpEngine
---------------

:class:`~qarp.engines.QarpEngine` is the default qarpx-native engine. It runs
on the C++ ``QarpSimulator`` and supports both the noise-free statevector fast
path and the per-shot trajectory path (the latter is triggered automatically
when a circuit contains mid-circuit measurement, reset, classical control, or
an active ``NoiseModel``).

In the first example we build and run a :class:`QarpEngine` with a single
:class:`~qarp.algorithms.StateVector` primitive computing the expectation value
of a Fermi–Hubbard Hamiltonian:

.. code-block:: python

    from qarp.utils import FH_ham_and_wf
    from qarp.algorithms import StateVector
    from qarp.engines import QarpEngine

    n = 2
    ham, block = FH_ham_and_wf(n, 1.4, 2.31)

    # we set the inputs such that we compute an expectation value
    sv = StateVector(bra=block, operator=ham, ket=block)

    my_engine = QarpEngine()
    my_engine.build([sv])

    params = block.parameter_map([0.1] * len(block.symbols))
    result = my_engine.run(params)

    print("Result:", result[0].real)


In the second example we build the engine with two primitives, mixing an exact
:class:`StateVector` with a shot-based :class:`TermwiseHadamardTest`:

.. code-block:: python

    from qarp.utils import FH_ham_and_wf
    from qarp.algorithms import StateVector, TermwiseHadamardTest
    from qarp.engines import QarpEngine

    n = 2
    ham1, block1 = FH_ham_and_wf(n, 1.4, 2.31)
    ham2, block2 = FH_ham_and_wf(n, 1.6, 2.2)

    sv = StateVector(bra=block1, operator=ham1, ket=block1)
    ht = TermwiseHadamardTest(bra=block2, operator=ham2, ket=block2)

    my_engine = QarpEngine()
    my_engine.build([sv, ht])

    params1 = block1.parameter_map([0.1] * len(block1.symbols))
    params2 = block2.parameter_map([0.1] * len(block2.symbols))
    combined_params = {**params1, **params2}

    result = my_engine.run(combined_params)

    print("Result 1:", result[0].real)
    print("Result 2:", result[1].real)


To make the engine respect device constraints (rebase, route, noise), pass a
:class:`~qarp.devices.Device` to the engine constructor — **not** to the
individual primitives. Below we add an amplitude-damping noise channel on
all 1-qubit gates and compare the noisy ``TermwiseHadamardTest`` against
an exact noise-free ``StateVector``:

.. code-block:: python

    from qarp.utils import FH_ham_and_wf
    from qarp.algorithms import StateVector, TermwiseHadamardTest
    from qarp.engines import QarpEngine
    from qarp.devices import Device, NoiseModel

    n = 2
    noisy_param = 0.001
    ham1, block1 = FH_ham_and_wf(n, 1.4, 2.31)

    # Build a noisy device.  ``.inner`` unwraps the Python NoiseModel wrapper
    # to the underlying qarpx-native qx.NoiseModel that Device expects.
    # +1 qubit: TermwiseHadamardTest's compiled circuit needs an ancilla
    # beyond block1's own register.
    noise_model = NoiseModel.amplitude_damping(noisy_param, gate_set="1q")
    noisy_device = Device(n_qubits=block1.n_qubits + 1, noise_model=noise_model.inner)

    sv = StateVector(bra=block1, operator=ham1, ket=block1)
    ht = TermwiseHadamardTest(bra=block1, operator=ham1, ket=block1, n_shots=1000)

    # Run StateVector on a noise-free engine; TermwiseHadamardTest on a noisy one.
    clean_engine = QarpEngine()
    clean_engine.build([sv])
    noisy_engine = QarpEngine(device=noisy_device)
    noisy_engine.build([ht])

    params = block1.parameter_map([0.1] * len(block1.symbols))
    clean_result = clean_engine.run(params)
    noisy_result = noisy_engine.run(params)

    print("Result noise-free:", clean_result[0].real)
    print("Result noisy:    ", noisy_result[0].real)


Exact evaluation and capability checks
---------------------------------------

Every shot-based primitive (``Sampler``, ``HadamardTest``, ``PauliAveraging``, ...)
also has an exact, infinite-shot limit: pass ``n_shots=qarp.EXACT`` and the engine
computes the Born distribution directly instead of drawing samples, feeding it
through the same estimator the shot-based path uses. This needs simulator
amplitudes, so it shares its capability requirements with :code:`consumes=AMPLITUDES`
primitives like ``StateVector`` — both fail the same way under noise.

.. code-block:: python

    import qarp
    from qarp.utils import FH_ham_and_wf
    from qarp.algorithms import TermwiseHadamardTest, StateVector
    from qarp.engines import QarpEngine

    n = 2
    ham1, block1 = FH_ham_and_wf(n, 1.4, 2.31)

    # n_shots=qarp.EXACT: same primitive, no shot noise.
    ht_exact = TermwiseHadamardTest(bra=block1, operator=ham1, ket=block1, n_shots=qarp.EXACT)
    sv = StateVector(bra=block1, operator=ham1, ket=block1)

    engine = QarpEngine()
    engine.build([ht_exact, sv])

    params = block1.parameter_map([0.1] * len(block1.symbols))
    result = engine.run(params)

    print("HadamardTest (n_shots=qarp.EXACT):", result[0].real)
    print("StateVector (reference):          ", result[1].real)

``Engine.build()`` validates every primitive's declared capabilities against the
engine and device *before* compiling anything, so an incompatible combination
raises :class:`~qarp.errors.CapabilityError` immediately rather than failing deep
inside the simulator. Reusing the noisy device from the example above, asking a
noisy ``QarpEngine`` for exact amplitudes raises:

.. code-block:: python

    noisy_engine = QarpEngine(device=noisy_device)
    noisy_engine.build([ht_exact])
    # CapabilityError: TermwiseHadamardTest requires exact amplitudes
    # (consumes=AMPLITUDES or n_shots=qarp.EXACT), which QarpEngine cannot
    # provide: amplitudes are undefined under noise (each shot is one
    # trajectory of a mixed state). Use a sampling primitive with finite
    # shots, or disable the noise model (engine.noise_model.enabled = False).

``CapabilityError`` also covers a primitive built with a target outside its
``supported_targets``, and a primitive that declares ``supports_exact=False``
run with ``n_shots=qarp.EXACT``. All three checks are cheap: they run once at
``build()``, not per shot.

The same gate guards every algorithm-level statevector fast path (the ADAPT pool
scans, QSE, QITE, MonteCarlo's exact reference distribution, QMEGS's classical
overlaps). Those paths read amplitudes through the engine's own simulator, never
through a fresh one, so a noisy or routed engine refuses with ``CapabilityError``
naming the remedy, disabling the noise model reopens the path on the same engine,
and an engine without a host statevector API is refused rather than idealised.
``CudaqEngine`` caps any device→host statevector transfer at 30 qubits
(``_STATEVECTOR_HOST_QUBIT_CAP``); the cap applies to these fast paths exactly as
it does to an ``OVERLAP`` or ``EXACT`` primitive, while ``EXPECTATION_VALUE`` stays
on-device and is uncapped.

Every simulator backend (``QarpSimulator``, ``CudaqSimulator``)
returns results through the same ``SamplingResult`` contract regardless of
whether the run was exact or shot-based, so ``result[i]`` is handled identically
either way — the primitive's own estimator is what changed, not the code that
reads the result.


Circuit cutting
---------------

Circuit cutting is exposed as the :class:`~qarp.algorithms.CuttingPrimitive`
primitive algorithm and is engine-agnostic. See the cutting documentation for details.

Gradients
---------

Every engine differentiates its built primitives through
``run_gradient(params, method=...)``: the adjoint (``QarpEngine`` only), a
batched per-occurrence parameter shift, finite differences and SPSA, all
returning one array per primitive in the insertion order of ``params``.
``CudaqEngine`` never falls back to a CPU simulator for a gradient.  The
registry, costs and refusals are documented in :doc:`gradients`.
