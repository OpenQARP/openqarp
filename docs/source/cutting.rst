Circuit Cutting
================

Introduction
----------------

Circuit cutting is a technique used in quantum computing to reduce the complexity of large quantum circuits by dividing them into smaller,
more manageable subcircuits. There are three flavours: cutting gates, cutting wires, or a combination of both. After running the
subcircuits, results are combined through classical post-processing to reconstruct the original circuit's output, introducing a sampling
overhead that scales exponentially with the number of cuts.

OpenQARP exposes circuit-cutting techniques so that large circuits can be executed on devices with fewer qubits.

The current implementation lives in :mod:`qarp.cutting`:

 - Automatic cut finder using scipy's differential evolution, with manual override: :class:`~qarp.cutting.EAPartitioning`.
 - Gate-cutting strategy based on KAK decompositions for sub-circuit post-processing: :class:`~qarp.cutting.QPDDecomposition`.
 - End-to-end ``PrimitiveAlgorithm`` integration: :class:`~qarp.algorithms.CuttingPrimitive`.

Additional information on circuit cutting:

  * [1] Hakkaku, Shigeo, Kosuke Mitarai, and Keisuke Fujii. "Sampling-based quasiprobability simulation for fault-tolerant quantum error correction on the surface codes under coherent noise." Physical Review Research 3.4 (2021): 043130.

  * [2] Mitarai, Kosuke, and Keisuke Fujii. "Overhead for simulating a non-local channel with local channels by quasiprobability sampling." Quantum 5 (2021): 388.

  * [3] Schmitt, Lukas, Christophe Piveteau, and David Sutter. "Cutting circuits with multiple two-qubit unitaries." arXiv preprint arXiv:2312.11638 (2023).

  * [4] Mitarai, Kosuke, and Keisuke Fujii. "Constructing a virtual two-qubit gate by sampling single-qubit operations." New Journal of Physics 23.2 (2021): 023021.

  * [5] https://qiskit-extensions.github.io/circuit-knitting-toolbox/

Cutting the circuit
--------------------

When cutting a circuit in OpenQARP, the user has the choice of an automatic strategy, which minimises the number of cuts (and therefore
the post-processing sampling overhead), or a manual strategy, specifying which qubit partitions are produced.

Cuts operate on a flat command stream, not on a circuit object. Build the block, flatten it to commands, and pass that to
:class:`~qarp.cutting.EAPartitioning`:

.. code-block:: python

    from qarp.blocks import CompositeBlock, HnBlock, LinearEntanglingBlock

    n_qubits = 3
    block = CompositeBlock(
        blocks=[
            HnBlock(n_qubits),
            LinearEntanglingBlock(n_qubits, circular=False, use_cz=False),
            LinearEntanglingBlock(n_qubits, circular=False, use_cz=False),
        ]
    ).build()

    commands = list(block.flatten())

We can invoke circuit cutting from OpenQARP as

.. code-block:: python

    from qarp.cutting import EAPartitioning

    cutter = EAPartitioning(
        commands=commands,
        n_qubits=n_qubits,
        max_size_subcircuits=[2, 2],
    )

The ``max_size_subcircuits`` argument bounds the qubit count of each subcircuit. Cutting a three-qubit circuit will not necessarily
produce a strict two/one split because the QPD post-processing may need ancilla qubits — it only guarantees subcircuits smaller than the
original.

To run the evolutionary cut finder:

.. code-block:: python

    cutter_result = cutter.cut()

To fix a specific partition (e.g. ``{0, 1}`` and ``{2}``):

.. code-block:: python

    cutter_result = cutter.cut(manual_setting=[[0, 1], [2]])

You can inspect the cut command stream and the resulting subcircuits via :code:`cutter_result.custom_commands` and
:code:`cutter_result.subcircuits`.

Reconstructing the circuit
--------------------------

From the above example we have two subcircuits resulting from cutting some of the entangling CNOTs.
Cutting an entangling gate involves generating six independent circuits with appropriate post-processing in order to
exactly reproduce the action of the entangling gate. This is the main overhead in circuit cutting: :math:`n_{\rm cuts}` cuts incur a
:math:`6^{n_{\rm cuts}}` post-processing overhead. We advise a prior check that no more than six cuts are being performed before running
the experiments. The experiment circuits come from the KAK decomposition of the two-qubit entangling gate; see Appendix B of Ref. [4]
for a pedagogical introduction.


Expectation values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standalone reconstruction pipeline lives in :class:`~qarp.cutting.QPDDecomposition`. Define a Hamiltonian and ask the decomposer for
its expectation value with respect to the cut circuit:

.. code-block:: python

    from qarp.operators import QubitOperator

    hamiltonian = (
          0.16988452027940318 * QubitOperator("Z0")
        + -0.21886306781219608 * QubitOperator("Z0 Z1 Z2")
        + 0.04544288414432624 * QubitOperator("Y0 Y1 Y2")
        + 0.04544288414432624
    )

    from qarp.cutting import QPDDecomposition

    post_processing = QPDDecomposition(
        cutter_result=cutter_result,
        observable=hamiltonian,
        n_shots=10000,
        verbose=True,
    )

    post_processing.decompose()

Evaluating ``decompose()`` generates all the sub-experiment commands and groups qubit-wise commuting Pauli terms to share measurement
circuits (QWC grouping is on by default).

To run the experiments and reconstruct the expectation value:

.. code-block:: python

    expectation_value = post_processing.compute()
    print("Expectation value:", expectation_value)

The result is exact up to sampling error, which decreases as the number of shots increases. The simulator is :code:`qx.QarpSimulator`;
there is no pluggable backend — all execution is qarpx-native.

Statevectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenQARP can reconstruct, in addition to an expectation value, the statevector probabilities of the original circuit from the cut
experiments. As before, this should be used with caution — extracting statevectors from large circuits is computationally heavy.

The procedure is analogous to expectation-value reconstruction but with an identity observable:

.. code-block:: python

    from qarp.cutting import QPDDecomposition
    from qarp.operators import QubitOperator

    identity_operator = QubitOperator("X0 X0", 1.0)

    post_processing = QPDDecomposition(
        cutter_result=cutter_result,
        observable=identity_operator,
        n_shots=10000,
        verbose=True,
    )

    post_processing.decompose()

The statevector is reconstructed as

.. code-block:: python

    statevector_reconstructed = post_processing.reconstruct_statevector()

The output is a ``dict`` mapping bitstring keys (e.g. ``"010"``) to probabilities (real numbers in :math:`[0, 1]`), not probability
amplitudes; values are subject to sampling error.

CuttingPrimitive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenQARP also exposes circuit cutting as a :class:`~qarp.algorithms.CuttingPrimitive` :class:`~qarp.algorithms.PrimitiveAlgorithm`. Given a
ket block, an observable, and a per-subcircuit qubit cap, the primitive partitions the circuit, generates the QPD experiments, hands them to the engine,
and reconstructs the expectation value in post-processing.

In this part we run the cutting primitive on a 2-qubit GHZ state with a Fermi–Hubbard Hamiltonian, executed via
:class:`~qarp.engines.QarpEngine`:

.. code-block:: python

    from qarp.engines import QarpEngine
    from qarp.algorithms import CuttingPrimitive
    from qarp.blocks import GHZLikeStateBlock
    from qarp.utils import FH_ham_and_wf

    # Build the Fermi-Hubbard Hamiltonian
    n = 2
    ham, _ = FH_ham_and_wf(n, 1.4, 2.31)

    # Build the GHZ circuit state on 4 qubits (the original system size)
    block = GHZLikeStateBlock(basis_state=[1, 1] * n)

    # max_subcircuit_qubits caps how many physical qubits each subcircuit may use.
    meas = CuttingPrimitive(ket=block, operator=ham, max_subcircuit_qubits=2, n_shots=1000)

    engine = QarpEngine()
    engine.build([meas])

    result_engine = engine.run()
    print("Result:", result_engine[0])

The result matches, up to sampling error, the standalone :class:`~qarp.cutting.QPDDecomposition` path:

.. code-block:: python

    from qarp.cutting import EAPartitioning, QPDDecomposition

    commands = list(block.build().flatten())
    autocutter = EAPartitioning(
        commands=commands,
        n_qubits=block.n_qubits,
        max_size_subcircuits=[2, 2],
    )
    result = autocutter.cut()

    post_processing = QPDDecomposition(
        cutter_result=result,
        observable=ham,
        n_shots=10000,
        verbose=False,
    )
    post_processing.decompose()
    expectation_value = post_processing.compute()

The :code:`max_subcircuit_qubits` argument tells the primitive the qubit capacity of each subcircuit; OpenQARP determines the
number of subcircuits needed (``ceil(n_qubits / max_subcircuit_qubits)``) and partitions to fit.
