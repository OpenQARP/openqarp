Factories
==============

The factories module contains 'Factories' - generators for OpenQARP Blocks. The primary factory is the
:class:`PauliBlockFactory`, which generates a list of :class:`PauliBlock` from a given Hamiltonian,
useful for many applications in quantum simulation and quantum chemistry.

To implement a new factory, please extend the :code:`BlockFactory` class with your custom factory.

BlockFactory
---------------

:class:`BlockFactory` is the abstract base class for a factory.


PauliBlockFactory
-----------------

:class:`PauliBlockFactory` generates a list of :class:`PauliBlock` objects from a Hamiltonian. The
input Hamiltonian can be in the form of a :class:`~qarp.operators.QubitOperator` or :class:`~qarp.operators.FermionOperator`
from ``qarp.operators`` (not OpenFermion's — see :doc:`endianness` for why the two are not
interchangeable at this boundary). If a :class:`~qarp.operators.FermionOperator` is provided,
:class:`PauliBlockFactory` will also map it into a qubit operator using, for example, the Jordan Wigner
decomposition.

The following example illustrates how to use :class:`PauliBlockFactory` to generate the corresponding Pauli blocks for some given fermion operators:

.. code-block:: python

    from qarp.operators import FermionOperator
    from qarp.factories import PauliBlockFactory

    annihilate_1 = FermionOperator('1')
    create_1 = FermionOperator('1^')
    annihilate_3 = FermionOperator('3')
    create_3 = FermionOperator('3^')
    annihilate_2 = FermionOperator("2")
    create_4 = FermionOperator("4^")

    # Construct occupation number operators
    num_2 = create_1 * annihilate_1
    num_3 = create_3 * annihilate_3
    # Construct another fermion operator
    op_24 = create_4 * annihilate_2

    fermion_op = num_2 + num_3 + op_24

    # Build PauliBlocks from a FermionOperator
    pauli_factory = PauliBlockFactory()
    pauli_blocks = pauli_factory.create_blocks(hamiltonian=fermion_op)

    # Check which Pauli operators are in the factory
    pauli_strings = pauli_factory.get_pauli_strings(pauli_blocks)

    # Check the coefficients of the Pauli decomposition
    pauli_coeffs = pauli_factory.get_coefficients(pauli_blocks)

Alternatively, a :class:`PauliBlockFactory` can be created from a :class:`QubitOperator`. Optionally, we can
also set the flag :code:`change_basis = True`, so that the blocks returned will be the basis change operators
corresponding to the measurements of the Pauli strings rather than the Pauli strings themselves.

.. code-block:: python

    from qarp.operators import QubitOperator

    op_1 = QubitOperator("X1")
    op_2 = QubitOperator("Y2")
    op_3 = QubitOperator("Z3")

    qubit_operator = op_1 + op_2 + op_3

    # Build PauliBlocks from a QubitOperator
    pauli_factory = PauliBlockFactory()
    pauli_blocks = pauli_factory.create_blocks(hamiltonian=qubit_operator)
    for block in pauli_blocks:
        block.build().plot()

    pauli_blocks = pauli_factory.create_blocks(hamiltonian=qubit_operator, change_basis=True)
    for block in pauli_blocks:
        block.build().plot()

The same :code:`change_basis` flag can be used with :class:`FermionOperator`, which similarly rotates
into the appropriate measurement basis for the Pauli operators.