Devices
==========

The devices module characterises a target quantum device. A :code:`Device` object closes the gap between OpenQARP's circuit-construction surface and real quantum hardware, so that the circuits a user writes can be rebased, routed, and (optionally) executed under a noise model that matches the target backend.

A :code:`Device` is a **passive data bundle**: it carries a qubit count (:code:`Device.n_qubits`), an optional gate set (:code:`Device.gate_set`), an optional :code:`Architecture` (coupling map, :code:`Device.architecture`), an optional :code:`NoiseModel` (:code:`Device.noise_model`), and a :code:`directedness` flag. It has no methods that perform compilation; the rebase / route / re-rebase pipeline is owned by the :class:`~qarp.engines.Engine` (or by the standalone helper :func:`qarp.devices.compile_for_device`).

Pass a :code:`Device` to the engine constructor to make every primitive that engine runs respect the device:

.. code-block:: python

    from qarp.engines import QarpEngine
    from qarp.devices import Device

    engine = QarpEngine(device=Device(n_qubits=4))


Number of qubits
-----------------

The number of qubits is mandatory and is used to assert that the circuit fits.
:code:`CuttingPrimitive`, for example, only runs circuit cutting when the
device has fewer qubits than the original circuit requires.

The following example creates a 3-qubit QPU.

.. code-block:: python

    from qarp.devices import Device

    my_device = Device(n_qubits=3)


Gate set
---------

The gate set specifies the available native gates of the target device. It is a
qarpx-native :code:`qx.GateSet` (the same type consumed by :code:`qx.Transpiler`).
When ``gate_set`` is ``None``, no rebase pass is run. When it is set, the engine
runs a rebase pass before routing and (if any SWAPs or H-conjugated CX sequences
are introduced by the router) a second rebase pass after routing.

The following example creates two devices: one with the full built-in gate set,
one restricted to a small set.

.. code-block:: python

    import qarpx as qx
    from qarp.devices import Device

    my_device_full_gates = Device(n_qubits=3, gate_set=qx.full_gateset_1q_2q())

    partial_gate_set = qx.GateSet()
    partial_gate_set.name = "partial"
    partial_gate_set.allowed = {qx.GateType.U, qx.GateType.CX}
    my_device_partial_gates = Device(n_qubits=3, gate_set=partial_gate_set)

Convenience helpers ``qx.full_gateset_1q()``, ``qx.full_gateset_2q()`` and ``qx.full_gateset_1q_2q()`` return the canonical 1q, 2q, and combined sets.


Architecture object
---------------------

The :code:`Architecture` object describes the connectivity of qubits on a quantum device. It is the qarpx-native :code:`qx.Architecture` — a coupling map with constant-time ``is_connected(a, b)`` / ``has_directed_edge(a, b)`` / ``neighbours(q)`` queries and a BFS ``shortest_path(a, b)``. Edges are pairs ``(a, b)`` of physical qubit indices, stored in the order and orientation given (``has_directed_edge`` and the router's tie-breaks read both); assigning ``arch.edges`` replaces the list and rebuilds the indices, rejecting any endpoint ``>= n_qubits``. The constructor also takes the device's ``n_qubits`` and validates it against ``Device.n_qubits``.

The following example creates a chain-structure coupling map on a 3-qubit QPU.

.. code-block:: python

    from qarp.devices import Device, Architecture

    arch = Architecture(n_qubits=3, edges=[(0, 1), (1, 2)])
    my_device = Device(n_qubits=3, architecture=arch)


When a circuit contains a 2-qubit gate between qubits ``(i, j)`` that are not
connected on the architecture, the engine's router inserts SWAPs along the
coupling graph until the gate can be applied directly. Two routers exist:
``qx.RouterKind.Sabre`` — the default — searches for an initial mapping and
then routes with a SABRE front-layer heuristic, while ``qx.RouterKind.Lite``
takes a greedy shortest-path sweep from the identity mapping. ``Sabre`` never
emits more SWAPs than ``Lite``. :func:`qarp.devices.compile_for_device` accepts
a ``router=`` argument to pick one. The second rebase pass then decomposes
those SWAPs into the device's gate set (e.g. ``SWAP → 3 · CX``).

OpenQARP also supports a directedness constraint: a 2-qubit gate may only be
applied with control on one specific endpoint of an edge. When
:code:`Device.directedness` is :code:`True`, the router uses
H-conjugation (``CX(b, a) = H(a) ⊗ H(b) · CX(a, b) · H(a) ⊗ H(b)``) to flip
direction whenever the physical edge runs the wrong way. When
:code:`directedness=False`, edges are treated as bidirectional.

The following example builds a circuit and asks OpenQARP to compile it for an
architecture-and-directedness-constrained device. The standalone helper
:func:`qarp.devices.compile_for_device` returns a ``CompiledCircuit`` —
useful when you want the routed command stream without executing it.

.. code-block:: python

    from qarp.blocks import GHZLikeStateBlock
    from qarp.devices import Device, Architecture, compile_for_device

    block = GHZLikeStateBlock(basis_state=[1, 1, 1]).build()

    arch = Architecture(n_qubits=3, edges=[(1, 0), (1, 2)])
    my_device = Device(n_qubits=3, architecture=arch, directedness=True)

    compiled = compile_for_device(block, my_device)
    # ``compiled.commands`` is the routed command list, on physical wires.
    # ``compiled.initial_logical_to_physical[l]`` is the wire logical qubit ``l``
    # is placed on before the first gate; ``compiled.final_logical_to_physical[l]``
    # is where it sits after the last — the map applied at sampling time.


Some convenient architectures are provided as helpers:

.. code-block:: python

    from qarp.devices import Device, get_all_to_all_architecture, get_nearest_neighbour_architecture

    n_qubits = 3
    arch = get_all_to_all_architecture(n_qubits)
    my_device = Device(n_qubits=n_qubits, architecture=arch)


Use :code:`get_nearest_neighbour_architecture(xdim, ydim)` for a 2D nearest-neighbour grid.


Noise model
------------

The :code:`NoiseModel` object specifies per-gate stochastic noise to apply
after each gate. It is a thin Python wrapper around the qarpx-native
:code:`qx.NoiseModel`, which stores a per-``GateType`` channel array of
``std::function<Channel(const Command&)>`` closures — parametric-capable, so
the same substrate supports angle-dependent noise (e.g. an Rz channel whose
rate depends on the angle).

The primary user API is a set of class-method builders:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Builder
     - Description
   * - ``NoiseModel.depolarizing(p, gate_set='2q')``
     - Uniform depolarising channel.
   * - ``NoiseModel.pauli(p_x, p_y, p_z, gate_set)``
     - Probabilistic Pauli channel with explicit weights.
   * - ``NoiseModel.bit_flip(p, gate_set='1q')``
     - Bit-flip ``PauliChannel(p_x=p)``.
   * - ``NoiseModel.amplitude_damping(p, gate_set='1q')``
     - Kraus amplitude-damping channel.

The ``gate_set`` argument accepts the shorthand strings ``'1q'``, ``'2q'``,
``'all'``, or an explicit ``qx.GateSet`` / list of ``qx.GateType``. Compose
multiple builders with ``+`` (right operand wins on collision). A shorthand or
``qx.GateSet`` is a rebase target, so it admits the ``Measure``, ``Barrier`` and
``GPhase`` markers; those carry no channel and are dropped. Naming one
explicitly in a list is an error.

The following example builds a noise model with amplitude damping plus a
bit-flip channel on all 1-qubit gates, attaches it to a 3-qubit device, and
runs a ``TermwiseHadamardTest`` through :class:`~qarp.engines.QarpEngine`:

.. code-block:: python

    from qarp.operators import QubitOperator

    from qarp.engines import QarpEngine
    from qarp.devices import NoiseModel, Device
    from qarp.blocks import GHZLikeStateBlock
    from qarp.algorithms import TermwiseHadamardTest

    block = GHZLikeStateBlock(basis_state=[1, 1, 1]).build()
    op = QubitOperator("Z0 Z1")

    # Compose multiple channels with ``+``.  Pass ``.inner`` (the underlying
    # qx.NoiseModel) to Device because Device.noise_model is the C++ type.
    noise_model = (
        NoiseModel.amplitude_damping(0.01, gate_set="1q")
        + NoiseModel.bit_flip(0.01, gate_set="1q")
    )
    # +1 qubit: TermwiseHadamardTest's compiled circuit needs an ancilla
    # beyond block's own 3-qubit register.
    my_device = Device(n_qubits=4, noise_model=noise_model.inner)

    engine = QarpEngine(device=my_device)
    meas = TermwiseHadamardTest(bra=block, operator=op, ket=block, n_shots=1000)
    engine.build([meas])
    noisy_result = engine.run({})

    print("Noisy result: ", noisy_result[0].real)


.. note::

   Any active ``NoiseModel`` (``enabled=true`` and at least one channel set)
   automatically puts the simulator on the per-shot **trajectory path** —
   stochastic noise cannot be represented by a single statevector. Expect
   ``~n_shots × suffix_cost`` runtime. The fast statevector path is preserved
   when ``Device.noise_model`` is ``None`` or has ``enabled=False``.

[1] Nielsen, Michael A., and Isaac L. Chuang. Quantum computation and quantum information. Cambridge university press, 2010.
