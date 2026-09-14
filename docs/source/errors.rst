Errors
======

:class:`~qarp.errors.CapabilityError` is OpenQARP's one dedicated exception type: it fires when a
primitive's declared capabilities are incompatible with the engine, device, or circuit it was
built for, and equally when a circuit cannot cross the boundary into an external SDK
(see :doc:`emit_absorb`). It subclasses ``ValueError``, so existing ``except ValueError``
handlers keep working, but code that wants to distinguish "this combination is fundamentally
unsupported" from a generic bad argument should catch it by name.

.. code-block:: python

    from qarp.errors import CapabilityError

Where it is raised
-------------------

Most checks below run at ``build()`` time (or, for a couple of engine-constructor checks, at
construction), **before** compilation or simulation, so a rejected combination costs nothing.
The C++ ``run()``-time guards are a backstop, not the primary gate. Three groups fire later,
and are marked as such in the table: device compilation happens when the circuit is actually
compiled for the device, the parameter-shift checks can only run once a gradient is requested,
and the emit boundary is reached when you call ``to_<sdk>()``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Condition
     - Message (abridged)
   * - Primitive built with a target outside its ``supported_targets``
     - ``{Primitive} built with target {X}, outside its supported_targets {...}``
   * - ``StateVector`` inferred target is ``SAMPLING`` (``ket`` given, no ``bra``/``operator``), see `StateVector versus Sampler`_ below
     - ``StateVector does not serve SAMPLING. ... use Sampler(n_shots=qarp.EXACT)``
   * - A ``consumes=COUNTS`` primitive requests ``n_shots=qarp.EXACT`` but declares ``supports_exact=False``
     - ``{Primitive} declares supports_exact=False (no meaningful ∞-shot limit); run it with a finite n_shots``
   * - ``CuttingPrimitive`` constructed with ``n_shots=qarp.EXACT``. It self-rejects at construction, ahead of the engine check above, see `No infinite-shot limit`_ below
     - ``CuttingPrimitive has no ∞-shot limit (QPD experiment sampling is inherently stochastic); pass a finite n_shots``
   * - A primitive needs exact amplitudes (``consumes=AMPLITUDES`` or ``n_shots=qarp.EXACT``) but the engine's noise model is active
     - ``{Primitive} requires exact amplitudes (consumes=AMPLITUDES or n_shots=qarp.EXACT), which {Engine} cannot provide: amplitudes are undefined under noise ...``
   * - An amplitude-consuming primitive's circuit contains a true mid-circuit operation (Reset, conditioned gate, measure-then-reuse)
     - ``{Primitive} requires exact amplitudes, but its circuit contains a true mid-circuit operation ({cmd}); the evolution is non-deterministic``
   * - An amplitude-consuming primitive's circuit was routed onto a device (logical→physical permutation is not the identity)
     - ``StateVector contracts amplitudes in logical qubit order, but routing permuted the register ... Use a sampling primitive, n_shots=qarp.EXACT, or an architecture-free device``
   * - ``QarpEngine(n_shots=qarp.EXACT)`` constructed with noise already enabled
     - ``QarpEngine(n_shots=qarp.EXACT) with an enabled noise model is self-contradictory: exact amplitudes are undefined under noise``
   * - A structured (noiseless-only) QPE plan is sampled after the engine's noise model was enabled post-``build()``
     - ``This structured QPE plan was prepared without noise, but the engine's noise model is now enabled; rebuild the algorithm ...``
   * - An engine that cannot execute mid-circuit operations at all is given a circuit containing one (independently of whether amplitudes are involved)
     - ``{Engine}: mid-circuit measurement / Reset / classical control is not supported in primitive '{name}'. ... Use QarpEngine for circuits with mid-circuit measurement``
   * - A primitive carrying ``initial_state`` declares ``accepts_initial_state = False`` (every primitive except ``Sampler`` and ``StateVector``), see `Initial State Injection in Primitives`_ below
     - ``{Primitive} does not accept initial_state: its estimator assumes auxiliary qubits start in |0…0⟩, so a full-register seed would silently corrupt the result.  Seed the register via Sampler or StateVector instead.``
   * - A primitive carries ``initial_state`` but the engine's ``supports_initial_state`` is ``False`` (every engine but ``QarpEngine``)
     - ``{Primitive} carries initial_state, but {Engine} cannot seed its register: state injection is a QarpSimulator capability (QarpEngine only).  Encode the state as gates instead (SynthesizedStateBlock).``
   * - A primitive carries ``initial_state`` and the router placed its register (the *initial* logical→physical map is not the identity; a final map moved only by SWAPs is fine — the seed still lands on the right wires)
     - ``{Primitive} carries initial_state in logical qubit order, but the router placed the register (initial_logical_to_physical is not the identity).  Use an architecture-free device.``
   * - A primitive carries ``initial_state`` on a finite-shot sweep, which runs in C++ ``batch_run``
     - ``{Primitive} carries initial_state, but the finite-shot sweep runs in C++ batch_run, which cannot seed the register.  Use n_shots=qarp.EXACT or per-step run() calls.``
   * - **(compile time)** The device cannot run the circuit at all — it does not fit, a gate cannot be rebased onto the device gate set (the diagnostic lists every unreachable gate, sorted), routing fails (disconnected coupling map, bad initial mapping, a 3+-qubit gate reaching the router), or an asymmetric two-qubit gate would run against a directed edge. Raised by the C++ passes themselves, so ``qx.compile_for_device``, ``qx.route``, ``Transpiler.transpile`` and ``Block.optimize`` all raise it directly
     - whatever ``qx.compile_for_device`` reported (``check_fits`` / rebase / route / ``assert_directions``); the offending command rides on ``.command``
   * - **(gradient time)** ``run_gradient`` is asked for a method the engine does not declare (``"adjoint"`` on ``CudaqEngine``, a reserved name such as ``"metric-tensor"``), for ``"adjoint"`` on a primitive it does not cover, or for ``"parameter-shift"`` on a gate angle that is not affine in its symbols or on a primitive whose ``gradient_kind`` is ``"none"``, see `Gradient refusals`_ below
     - ``{Engine} has no 'adjoint' gradient; it declares [...].  QarpEngine provides 'adjoint'.`` / ``{Primitive} declares gradient_kind='none': its value is not a trigonometric polynomial of each circuit's gate angles, so the parameter-shift rule does not apply.  Use method='finite-diff'.``
   * - **(emit time)** A block is emitted to an SDK that cannot represent one of its commands. The emitter's ``validate()`` runs first, so this fires before the SDK is imported; the offending ``Command`` is on the exception's ``command`` attribute
     - the emitter's own incompatibility reason. A *missing* SDK is an ``ImportError`` instead, and parse/internal failures stay ``RuntimeError`` — see :doc:`emit_absorb`
   * - **(absorb time)** An absorber is handed an object belonging to a different SDK
     - ``{Absorber} expected a {sdk} object but received an instance of {module}.{Class} (module root '{root}')``

StateVector versus Sampler
----------------------------

``StateVector`` computes an expectation value, overlap, or transition amplitude directly from
simulator amplitudes and has no measurement-based fallback. Constructing one with only a
``ket`` infers a ``SAMPLING`` target, which ``StateVector`` cannot serve:

.. code-block:: python

    from qarp.blocks import SimpleBlock
    from qarp.algorithms import StateVector
    from qarp.engines import QarpEngine

    block = SimpleBlock(2, name="bell")
    block.h(0)
    block.cx(0, 1)

    sv = StateVector(ket=block)
    engine = QarpEngine()
    engine.build([sv])
    # CapabilityError: StateVector does not serve SAMPLING.  For the exact
    # distribution use Sampler(n_shots=qarp.EXACT) — the Born probabilities
    # |ψ|², phases discarded.  For raw amplitudes use
    # qx.QarpSimulator().statevector(...) directly.

Amplitudes versus noise
--------------------------

Amplitudes are undefined under noise, because each shot of a noisy circuit is one trajectory of a
mixed state, not a pure statevector, so any amplitude-consuming request (``StateVector``, or
a protocol primitive run with ``n_shots=qarp.EXACT``) is rejected against a noisy engine, and
against ``QarpEngine(n_shots=qarp.EXACT)`` constructed with noise already active:

.. code-block:: python

    import qarp
    from qarp.utils import FH_ham_and_wf
    from qarp.algorithms import TermwiseHadamardTest
    from qarp.devices import Device, NoiseModel
    from qarp.engines import QarpEngine

    n = 2
    ham1, block1 = FH_ham_and_wf(n, 1.4, 2.31)

    noise_model = NoiseModel.amplitude_damping(0.001, gate_set="1q")
    noisy_device = Device(n_qubits=block1.n_qubits + 1, noise_model=noise_model.inner)

    ht_exact = TermwiseHadamardTest(bra=block1, operator=ham1, ket=block1, n_shots=qarp.EXACT)
    noisy_engine = QarpEngine(device=noisy_device)
    noisy_engine.build([ht_exact])
    # CapabilityError: TermwiseHadamardTest requires exact amplitudes
    # (consumes=AMPLITUDES or n_shots=qarp.EXACT), which QarpEngine cannot
    # provide: amplitudes are undefined under noise ...

See :doc:`engines` for the full worked example (including the ``n_shots=qarp.EXACT`` success
path on a noise-free engine).

Amplitudes versus routing
----------------------------

``StateVector`` contracts amplitudes in **logical** qubit order. Routing a circuit onto a
``Device`` with an ``architecture`` can permute the logical→physical mapping via inserted
SWAPs, which invalidates that assumption, so ``StateVector`` is rejected on any device that
actually routes the circuit, independently of whether noise is involved at all:

.. code-block:: python

    from qarp.blocks import SimpleBlock
    from qarp.algorithms import StateVector
    from qarp.devices import Device, get_nearest_neighbour_architecture
    from qarp.engines import QarpEngine
    from qarp.operators import QubitOperator

    block = SimpleBlock(4, name="ghz4")
    block.h(0)
    block.cx(0, 3)  # qubits 0 and 3 are not adjacent on a line architecture
    block.build()

    op = QubitOperator("Z0 Z3")
    sv = StateVector(bra=block, operator=op, ket=block)

    # A 4-qubit line: 0-1-2-3.  The cx(0, 3) above forces the router to
    # insert SWAPs, permuting the logical->physical mapping.
    architecture = get_nearest_neighbour_architecture(4, 1)
    routed_device = Device(n_qubits=4, architecture=architecture)

    engine = QarpEngine(device=routed_device)
    engine.build([sv])
    # CapabilityError: StateVector contracts amplitudes in logical qubit
    # order, but routing permuted the register (final_logical_to_physical is
    # not the identity).  Use a sampling primitive, n_shots=qarp.EXACT, or an
    # architecture-free device.

The fix is whichever fits the situation: drop the device for this primitive (run it
unrouted, as in the noise example above), switch to a shot-based primitive, or use an
architecture-free ``Device`` (gate-set rebase only, no routing).

No infinite-shot limit
-------------------------

Some primitives are inherently stochastic and have no meaningful exact limit. ``CuttingPrimitive``'s QPD experiment sampling is the standing example, and reject
``n_shots=qarp.EXACT`` outright, at construction time rather than at ``build()``:

.. code-block:: python

    import qarp
    from qarp.blocks import SimpleBlock
    from qarp.algorithms import CuttingPrimitive
    from qarp.operators import QubitOperator

    block = SimpleBlock(2, name="bell")
    block.h(0)
    block.cx(0, 1)

    CuttingPrimitive(
        ket=block, operator=QubitOperator("Z0 Z1"), n_shots=qarp.EXACT
    )
    # CapabilityError: CuttingPrimitive has no ∞-shot limit (QPD experiment
    # sampling is inherently stochastic); pass a finite n_shots.

Gradient refusals
-----------------

Gradients are a run-time request, so their checks cannot fire at ``build()``: the circuit
is legal and the incompatibility only exists relative to a method.  Three things are
refused with ``CapabilityError`` (see :doc:`gradients` for the full contract):

* a method the engine does not declare — ``"adjoint"`` on ``CudaqEngine``
  (the message names ``QarpEngine``), or a reserved name such as ``"metric-tensor"``;
* ``"adjoint"`` on a primitive it does not cover (anything but a ``StateVector``
  expectation value over a ``QubitOperator``, or a ``StateVector`` overlap);
* ``"parameter-shift"`` on a gate angle that is not affine in its symbols, or on a
  primitive whose ``gradient_kind`` is ``"none"`` — its value is not a trigonometric
  polynomial of the gate angles, so no shift rule applies and ``"finite-diff"`` is the
  way through.

An unknown method name is a plain ``ValueError`` (a typo, not a capability).

.. code-block:: python

    import qarpx as qx
    from qarp.algorithms import StateVector
    from qarp.blocks import SimpleBlock
    from qarp.engines import QarpEngine
    from qarp.errors import CapabilityError

    t = qx.Param.symbol("t")
    block = SimpleBlock(1, name="square")
    block.ry(0, t * t)  # not affine in t
    block.build()

    engine = QarpEngine()
    engine.build([StateVector(ket=block, operator=qx.QubitOperator("Z0"))])
    try:
        engine.run_gradient({"t": 0.3}, method="parameter-shift")
    except CapabilityError as exc:
        print("rejected:", exc)
    grad = engine.run_gradient({"t": 0.3}, method="finite-diff")[0]

Shapes that used to be refused by the two-term rule — a symbol on several gates, a
compound angle such as ``(a + b)/2``, differing coefficients across circuits — are now
differentiated exactly by the per-occurrence rules.

Initial State Injection in Primitives
-------------------------------------

``initial_state`` seeds a primitive's register with a given amplitude vector instead of
starting from ``|0…0⟩``. Only ``Sampler`` and ``StateVector`` accept it (``accepts_initial_state``
is ``True`` on those two and ``False`` on every other primitive), because the protocol
estimators assume their auxiliary qubits start in ``|0…0⟩`` and a full-register seed would
silently corrupt the result rather than fail:

.. code-block:: python

    import numpy as np
    from qarp.utils import FH_ham_and_wf
    from qarp.algorithms import TermwiseHadamardTest
    from qarp.engines import QarpEngine

    ham, block = FH_ham_and_wf(2, 1.4, 2.31)

    ht = TermwiseHadamardTest(bra=block, operator=ham, ket=block)
    ht.initial_state = np.zeros(2**block.n_qubits, dtype=complex)
    ht.initial_state[0] = 1.0

    QarpEngine().build([ht])
    # CapabilityError: TermwiseHadamardTest does not accept initial_state: its
    # estimator assumes auxiliary qubits start in ``|0…0⟩``, so a full-register
    # seed would silently corrupt the result.  Seed the register via Sampler
    # or StateVector instead.

Three further restrictions apply even to ``Sampler`` and ``StateVector``. State injection is
a ``QarpSimulator`` capability, so any other engine (``supports_initial_state = False``) rejects
it and wants the state encoded as gates via ``SynthesizedStateBlock``. Injected amplitudes are
in **logical** qubit order, so a routed circuit rejects them for the same reason
`Amplitudes versus routing`_ rejects ``StateVector``. And a finite-shot parameter sweep runs
in the C++ ``batch_run``, which cannot seed the register, so it wants ``n_shots=qarp.EXACT``
or per-step ``run()`` calls instead.
