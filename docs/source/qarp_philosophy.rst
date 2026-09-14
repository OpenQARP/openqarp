OpenQARP philosophy
===================

The Open Quantum Application Research Package (OpenQARP) is designed around a modular, compositional philosophy for building quantum algorithms. This guide introduces the core concepts and how they work together to create a flexible framework for quantum computing.

The building Blocks philosophy
------------------------------

At its core, OpenQARP treats quantum circuits as composable building blocks. Rather than working directly with low-level gate operations, you work with higher-level abstractions called **Blocks** that encapsulate meaningful quantum operations.

The Block Hierarchy
-------------------

The ``SimpleBlock`` Class
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qarp.blocks.SimpleBlock` is the fundamental leaf unit of circuit construction in OpenQARP — a real subclass of the qarpx ``qx.SimpleBlock`` (direct inheritance via nanobind), so gate methods such as ``self.h(0)`` and ``self.cx(0, 1)`` are inherited natively from C++. Subclass ``SimpleBlock`` for a leaf block and ``CompositeBlockBase`` for one built out of other blocks.

There is no ``Block`` base class to inherit from. ``AnyBlock`` (an alias of the
qarpx ``qx.Block`` type) exists for *annotating* a value that may be any kind of
block — a type, not a base class:

.. code-block:: python

    from qarp.blocks import AnyBlock

    def depth_of(block: AnyBlock) -> int:
        return len(list(block.flatten()))

Every block (``SimpleBlock`` or :class:`~qarp.blocks.CompositeBlock` alike):

- Encapsulates a quantum operation that can be built into a command stream
- Has a ``build()`` method that populates the underlying qarpx C++ command buffer
- Can specify which qubits it targets via ``target_qubits``
- Can be controlled by wrapping it in ``ControlledBlock``
- Can be composed with other blocks

.. code-block:: python

    from qarp.blocks import HEABlock

    # A Hardware-Efficient Ansatz block with 2 layers
    ansatz = HEABlock(
        n_qubits=4,
        n_layers=2,
        real=False,     # Use Ry-Rz rotations (complex ansatz)
        linear=True,    # Linear entanglement pattern
        circular=False,
        use_cz=False,   # Use CNOT gates (not CZ)
    )
    ansatz.build()

    # Access the underlying command stream (list of qarpx Commands)
    commands = ansatz.flatten()

Some of the features of blocks:

- **Lazy building**: Blocks are defined first, then built when needed
- **Symbolic parameters**: Blocks can contain symbolic (variational) parameters
- **Dagger support**: Get the adjoint via ``block.dagger()``
- **Parameter substitution**: Replace symbols with values using ``set_symbols()``

Primitive Blocks
^^^^^^^^^^^^^^^^

OpenQARP provides many pre-built primitive blocks for common quantum operations, amongst them:

**State Preparation**:
  - ``HnBlock``: Applies Hadamard to the target qubits
  - ``ComputationalBasisStateBlock``: Prepares a specific computational basis state
  - ``DickeStateBlock``: Prepares Dicke states
  - ``GHZLikeStateBlock``: Prepares GHZ-like entangled states

**Ansätze (Parameterized Circuits)**:
  - ``HEABlock``: Hardware-Efficient Ansatz with stacked rotation and entangling layers
  - ``UCCBlock``: Unitary Coupled Cluster for quantum chemistry
  - ``QAOABlock``: QAOA mixer and cost layers

**Algorithm-Specific**:
  - ``QPEBlock``: Quantum Phase Estimation circuit
  - ``QFTBlock``: Quantum Fourier Transform
  - ``TrotterBlock``: Trotterized time evolution
  - ``HadamardTestBlock``, ``SWAPTestBlock``: Quantum tests for overlaps

**Operator Blocks**:
  - ``PauliBlock``: Single Pauli string operations
  - ``BlockEncodingBlock``: Block encoding for linear combinations of unitaries

Composing Blocks: The ``CompositeBlock``
----------------------------------------

The :class:`~qarp.blocks.CompositeBlock` is how you combine multiple blocks into larger circuits. It takes a sequence of blocks and stitches them together respecting each block's ``target_qubits``.

.. code-block:: python

    from qarp.blocks import CompositeBlock, HnBlock, HEABlock, ReadoutBlock

    # Create individual blocks
    init = HnBlock(n_qubits=4, target_qubits=[0, 1, 2, 3])
    ansatz = HEABlock(
        n_qubits=4,
        n_layers=2,
        real=False,
        linear=True,
        circular=False,
        use_cz=False,
        target_qubits=[0, 1, 2, 3],
    )
    measure = ReadoutBlock(n_qubits=4, target_qubits=[0, 1, 2, 3])

    # Compose them together
    full_circuit = CompositeBlock(
        blocks=[init, ansatz, measure],
        n_qubits=4,
        name="MyCircuit"
    )
    full_circuit.build()

.. raw:: html

   <figure class="qarp-figure">
     <svg class="qarp-circuit" viewBox="0 0 720 320" role="img"
          aria-label="Four blocks placed on six qubit wires inside one CompositeBlock: block0 on wires 0 and 1, block1 on wires 3 and 4, block2 across all six, block3 on wires 2 and 3.">
       <g class="wires">
         <line x1="30" y1="50" x2="690" y2="50"/><line x1="30" y1="90" x2="690" y2="90"/>
         <line x1="30" y1="130" x2="690" y2="130"/><line x1="30" y1="170" x2="690" y2="170"/>
         <line x1="30" y1="210" x2="690" y2="210"/><line x1="30" y1="250" x2="690" y2="250"/>
       </g>
       <g class="qubits">
         <text x="18" y="54">q0</text><text x="18" y="94">q1</text><text x="18" y="134">q2</text>
         <text x="18" y="174">q3</text><text x="18" y="214">q4</text><text x="18" y="254">q5</text>
       </g>
       <rect class="frame" x="76" y="14" width="512" height="272" rx="16"/>
       <g class="blk"><rect x="110" y="28" width="120" height="84" rx="10"/><text x="170" y="70">block0</text></g>
       <g class="blk"><rect x="110" y="148" width="120" height="84" rx="10"/><text x="170" y="190">block1</text></g>
       <g class="blk"><rect x="272" y="28" width="120" height="244" rx="10"/><text x="332" y="150">block2</text></g>
       <g class="blk"><rect x="434" y="108" width="120" height="84" rx="10"/><text x="494" y="150">block3</text></g>
       <text class="frame-label" x="332" y="308">CompositeBlock</text>
     </svg>
   </figure>

Key rules for composition:

1. **Target qubits matter**: Each block's ``target_qubits`` determines where it acts in the larger circuit
2. **Order matters**: Blocks are applied in sequence
3. **Measurements usually go last**: nothing forbids a ``ReadoutBlock`` in the middle, but
   measuring and then continuing is a *mid-circuit* measurement — it moves the run onto the
   per-shot trajectory path (see below), and any primitive that consumes amplitudes will
   reject the circuit with a :class:`~qarp.errors.CapabilityError` (see :doc:`errors`)

Measurement and Classical Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenQARP exposes three block primitives for the quantum/classical interface:

- :class:`~qarp.blocks.MeasureBlock` — single ``Measure`` command (one qubit, one cbit)
- :class:`~qarp.blocks.ResetBlock` — projective reset of a qubit to ``|0⟩``
- :class:`~qarp.blocks.ConditionalBlock` — wraps another block and only applies it when a classical condition (AND of ``(cbit, value)`` pairs) holds

Convenience ``SimpleBlock`` builder shortcuts (``block.measure(q, c)``, ``block.reset(q)``) are also available for inline use.

End-of-circuit measurement is also still supported via the bulk :class:`~qarp.blocks.ReadoutBlock`:

.. code-block:: python

    from qarp.blocks import ReadoutBlock

    # Measure all 4 qubits
    measurements = ReadoutBlock(n_qubits=4, target_qubits=[0, 1, 2, 3])

Any non-unitary command (``Measure`` with a cbit, ``Reset``, or a conditional gate) puts the simulator on the per-shot trajectory path automatically; circuits without these stay on the fast statevector path.

Executing Quantum Algorithms
----------------------------

OpenQARP separates the *what* (algorithm specification) from the *how* (execution). This is achieved through **Primitive Algorithms**, **Targets**, and **Engines**.

Primitive Algorithms
^^^^^^^^^^^^^^^^^^^^

A :class:`~qarp.algorithms.PrimitiveAlgorithm` defines a fundamental quantum measurement protocol. These are the key operations from which more complex algorithms are built.

**Core primitives include**:

- ``StateVector``: Exact statevector simulation (no sampling)
- ``Sampler``: Shot-based sampling from circuits
- ``PauliAveraging``: Efficient expectation value estimation via Pauli grouping
- ``HadamardTest``: Measures overlaps and matrix elements
- ``SWAPTest``: Alternative method for measuring overlaps

Each primitive takes:

- ``bra``: The bra state block ⟨ψ\|
- ``ket``: The ket state block \|ψ⟩
- ``operator``: The observable to measure (optional)
- ``n_shots``: Number of measurement shots (optional)

Device configuration is now an engine concern, not a primitive concern: pass a
``Device`` to the engine constructor (``QarpEngine(device=...)``), and every
primitive executed by that engine inherits the device's rebase / route / noise
pipeline.

.. code-block:: python

    from qarp.algorithms import StateVector, PauliAveraging
    from qarp.blocks import HEABlock
    from qarp.operators import QubitOperator

    hamiltonian = QubitOperator("Z0 Z1") + QubitOperator("X0")
    ansatz = HEABlock(
        n_qubits=4, n_layers=2, real=False, linear=True, circular=False, use_cz=False
    )

    # Exact statevector computation
    sv_primitive = StateVector(ket=ansatz, operator=hamiltonian)

    # Shot-based Pauli averaging
    pa_primitive = PauliAveraging(ket=ansatz, operator=hamiltonian, n_shots=1000)

The ``Target`` Enum
^^^^^^^^^^^^^^^^^^^

The :class:`~qarp.algorithms.Target` specifies what quantity a primitive algorithm computes:

- ``Target.SAMPLING``: Raw measurement distribution
- ``Target.EXPECTATION_VALUE``: ⟨ψ|O|ψ⟩
- ``Target.OVERLAP``: ⟨φ|ψ⟩
- ``Target.TRANSITION_AMPLITUDE``: ⟨φ|O|ψ⟩

Targets are inferred automatically based on what you provide (bra, ket, operator).

Engines: The Execution Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An :class:`~qarp.engines.Engine` is responsible for actually executing quantum circuits. OpenQARP currently provides two qarpx-native engines:

- :class:`~qarp.engines.QarpEngine`: Fast statevector simulation backed by the ``QarpSimulator``. Optional ``device=`` triggers rebase + route + noise via the unified compilation pipeline.
- :class:`~qarp.engines.CudaqEngine`: GPU execution through ``qx.CudaqSimulator``. Needs the ``cuda-quantum-cu12`` runtime *and* a qarpx build compiled with CUDA-Q support (``QARP_WITH_CUDAQ=ON``); see :doc:`engines`.

The engine abstracts away backend-specific details:

.. code-block:: python

    from qarp.engines import QarpEngine
    from qarp.algorithms import StateVector
    import numpy as np

    # Create an engine
    engine = QarpEngine()

    # Build and run a primitive
    primitive = StateVector(ket=ansatz, operator=hamiltonian)

    engine.build([primitive])

    # Get the symbolic parameters from the ansatz and provide values for them
    # HEABlock creates symbols like ry_0_0, rz_0_0, ry_0_1, rz_0_1, etc.
    params = {symbol: np.random.uniform(0, 2 * np.pi) for symbol in ansatz.symbols}

    result = engine.run(params)

Engines provide two key methods:

- ``build(primitives)``: Prepare the engine with a list of primitives
- ``run(params)``: Execute with given parameter values. Keys may be ``sympy``
  symbols or their names — both are coerced to ``str`` internally.

To draw shots from an arbitrary block, build a :class:`~qarp.algorithms.Sampler` primitive and run it through the engine.

Composite Algorithms
--------------------

:class:`~qarp.algorithms.CompositeAlgorithm` represents higher-level quantum algorithms that orchestrate multiple primitives and potentially classical optimization. These are the algorithms you typically want to run.

**Some available composite algorithms**:

- ``VQE``: Variational Quantum Eigensolver for ground states
- ``VQD``: Variational Quantum Deflation for excited states
- ``QAOA``: Quantum Approximate Optimization Algorithm
- ``PCE``: Pauli Correlation Encoding
- ``QPE``: Quantum Phase Estimation
- ``SSVQE``: Subspace-Search VQE
- ``AdaptVQE``, ``AdaptVQD``: Adaptive ansatz construction
- ``MonteCarlo``: Quantum Monte Carlo methods

Example: VQE
^^^^^^^^^^^^

.. code-block:: python

    from qarp.algorithms import VQE, StateVector
    from qarp.blocks import HEABlock
    from qarp.engines import QarpEngine
    from qarp.optimizers import ScipyOptimizer
    from qarp.operators import QubitOperator

    # Define the Hamiltonian
    hamiltonian = QubitOperator("Z0 Z1") + QubitOperator("X0") + QubitOperator("X1")

    # Create the ansatz
    ansatz = HEABlock(
        n_qubits=2, n_layers=2, real=False, linear=True, circular=False, use_cz=False
    )

    # Set up VQE
    vqe = VQE(
        operator=hamiltonian,
        ket=ansatz,
        primitive=StateVector(),
        optimizer=ScipyOptimizer("COBYLA"),
        engine=QarpEngine(),
    )

    # Build and run
    vqe.build()
    energy, optimal_params = vqe.run()
    print(f"Ground state energy: {energy}")

The Data Flow
-------------

Understanding how data flows through OpenQARP helps clarify the architecture:

.. raw:: html

   <figure class="qarp-figure qarp-flow">
     <section class="qarp-flow-layer">
       <h4>Circuit construction</h4>
       <div class="qarp-flow-chain">
         <code>Block</code><i></i><code>Block</code><i></i><span>…</span><i></i><code>CompositeBlock</code><i></i><em>built circuit</em>
       </div>
       <p class="qarp-flow-note"><code>ReadoutBlock</code> supplies the classical mapping.</p>
     </section>
     <div class="qarp-flow-arrow" aria-hidden="true"></div>
     <section class="qarp-flow-layer">
       <h4>Algorithm layer</h4>
       <div class="qarp-flow-row">
         <code>PrimitiveAlgorithm</code>
         <ul>
           <li>defines the measurement strategy (Hadamard test, Pauli averaging)</li>
           <li>constructs the measurement circuits</li>
           <li>post-processes the results</li>
         </ul>
       </div>
       <div class="qarp-flow-row">
         <code>CompositeAlgorithm</code>
         <ul>
           <li>orchestrates several primitives</li>
           <li>drives the classical optimisation loop</li>
           <li>combines their results</li>
         </ul>
       </div>
     </section>
     <div class="qarp-flow-arrow" aria-hidden="true"></div>
     <section class="qarp-flow-layer">
       <h4>Execution layer</h4>
       <div class="qarp-flow-row">
         <code>Engine</code>
         <ul>
           <li>executes circuits on a backend (QarpEngine, CudaqEngine)</li>
           <li>substitutes parameters</li>
           <li>returns measurement results</li>
         </ul>
       </div>
       <div class="qarp-flow-row">
         <code>Device</code>
         <ul>
           <li>transforms the circuit to the gate set and architecture</li>
           <li>applies the noise model</li>
         </ul>
       </div>
     </section>
   </figure>

Devices and Noise
-----------------

The :class:`~qarp.devices.Device` class represents the target quantum hardware configuration. It is a passive data bundle — no methods beyond ``check_fits``. The compilation pipeline lives on the :class:`~qarp.engines.Engine` (or the standalone helper :func:`qarp.devices.compile_for_device`).

.. code-block:: python

    import qarpx as qx
    from qarp.devices import Device, NoiseModel, Architecture, get_nearest_neighbour_architecture

    # Simple device with just qubit count
    simple_device = Device(n_qubits=10)

    # Device with architecture constraints (linear chain on 4 qubits)
    constrained_device = Device(
        n_qubits=4,
        architecture=Architecture(n_qubits=4, edges=[(0, 1), (1, 2), (2, 3)]),
        gate_set=qx.full_gateset_1q_2q(),
    )

    # Or use a helper for a standard topology (2x2 nearest-neighbour grid)
    grid_device = Device(
        n_qubits=4,
        architecture=get_nearest_neighbour_architecture(xdim=2, ydim=2),
    )

    # Noise model with bit-flip error
    noise_model = NoiseModel.bit_flip(0.01)

    # Device with noise — note ``.inner`` to unwrap to the C++ ``qx.NoiseModel``
    noisy_device = Device(
        n_qubits=10,
        noise_model=noise_model.inner,
    )

The ergonomic ``NoiseModel`` builders are ``depolarizing``, ``pauli``,
``bit_flip``, and ``amplitude_damping``; compose them with ``+``
to build mixed-channel noise models. See :doc:`devices` for their signatures.

Putting It All Together
-----------------------

Here's a complete example showing how all concepts connect:

.. code-block:: python

    from qarp.blocks import CompositeBlock, HnBlock, HEABlock
    from qarp.algorithms import VQE, StateVector
    from qarp.engines import QarpEngine
    from qarp.optimizers import ScipyOptimizer
    from qarp.operators import QubitOperator

    # 1. Define the problem: a simple Hamiltonian
    H = QubitOperator("Z0 Z1", 1.0) + QubitOperator("X0", 0.5) + QubitOperator("X1", 0.5)

    # 2. Build the circuit using blocks
    n_qubits = 2
    initialization = HnBlock(n_qubits=n_qubits, target_qubits=list(range(n_qubits)))
    ansatz = HEABlock(
        n_qubits=n_qubits,
        n_layers=3,
        real=False,
        linear=True,
        circular=False,
        use_cz=False,
        target_qubits=list(range(n_qubits)),
    )

    # Compose into a single state preparation block
    state_prep = CompositeBlock(
        blocks=[initialization, ansatz],
        n_qubits=n_qubits,
        name="StatePreparation"
    )

    # 3. Choose how to measure (primitive algorithm) - using StateVector for exact computation
    primitive = StateVector()

    # 4. Set up the high-level algorithm
    vqe = VQE(
        operator=H,
        ket=state_prep,
        primitive=primitive,
        optimizer=ScipyOptimizer("COBYLA"),
        engine=QarpEngine(),
    )

    # 5. Build and run
    vqe.build()
    energy, optimal_params = vqe.run()
    print(f"Ground state energy: {energy}")

Summary
-------

OpenQARP's architecture can be summarized as:

1. **Blocks** are the building units for quantum circuits
2. **CompositeBlock** composes blocks into larger circuits
3. **ReadoutBlock** bridges quantum to classical information
4. **PrimitiveAlgorithm** defines how to extract information from circuits
5. **Target** specifies what quantity to compute
6. **Engine** executes circuits on a backend
7. **CompositeAlgorithm** orchestrates full quantum algorithms
8. **Device** describes hardware constraints and noise

This separation of concerns makes OpenQARP flexible: you can swap engines, change measurement strategies, or modify ansätze without rewriting your entire algorithm.
