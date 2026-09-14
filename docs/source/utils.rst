Utility Functions
==================
The utility functions provides a set of scripts and objects aiming at facilitating
the implementation of more structured algorithms and for testing purposes. In the current
release, the utility functions are substantially comprised of a function for generating quanum
circuits and by two Fermi-Hubbard toy model functions.

Random Circuit Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function generates a quantum circuit with a specified number of qubits and gates, 
applying random gates sampled frm a user-specified basis set.  
This function is an useful tool for genereting test circuits, for
experimenting the effect of random circuit in quantum computational pipeline and for 
the generation of initial states in quantum optimization algorithms.

Let's see how to use the function with a basic example.

.. note::
   The function is exported from :code:`qarp.utils`.

The following arguments are accepted as part of the API:

* ``num_gates`` (int): Number of gates in the circuit. Must be a positive integer.
* ``num_qubits`` (int): The number of qubits in the circuit. Must be a positive integer.
* ``gate_set``: List of ``qarpx.GateType`` values to sample from.

The function returns a built :class:`~qarp.blocks.SimpleBlock`.

.. code-block:: python

    from qarp.utils import generate_random_circuit
    import qarpx as qx

    gate_set = [qx.GateType.Rz, qx.GateType.Ry, qx.GateType.Rx]
    num_gates = 6
    num_qubits = 4

    my_block = generate_random_circuit(num_gates, num_qubits, gate_set)

The gate set may contain rotation operators, single-qubit no-parameter gates, two-qubit gates,
and controlled rotations. Note that, if a seed is desired, it can be set in the
``qarp.config.seed`` variable.

You can visualise the block using our in-house visualisation tool:

.. code-block:: python

    from qarp.plotting import plot

    plot(my_block)

Fermi-Hubbard toy models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :code:`qarp.utils` contains two functions for constructing a Fermi-Hubbard Hamiltonian, with a Trotter
block, corresponding to either the UCC Doubles (UCCD) or the UCC Singles and Doubles (UCCSD) ansatz.
UCCD and UCCSD are parametrized ansatz, widely used in chemistry and electronic simulation, entailing doubles
or single and double fermionic exicitation operators. More details about these ansatzes are provided in the block 
documentation.

The functions :code:`FH_ham_and_wf` and :code:`FH_ham_and_wf_singles_and_doubles` generate a Fermi-Hubbard Hamiltonian and a UCCD
Trotterized block, taking the following arguments:

* `n` (int):  Number of site. For each site there are two single particle states 
* `t` (float): Kinetic or hopping energy between site.
* `U` (float): Interaction energy

The function constructs a Fermi Hubbard chain Hamiltonian and then, performing a Jordan-Wigner transformation, maps it to
Pauli matrices. Further, it constructs a UCCD (or UCCSD) block, from the evaluation of the occupation number vector
on a basis state. Further details are provided in the `operators` documentation.

Let's see an example usage.

.. code-block:: python

    from qarp.utils import FH_ham_and_wf, FH_ham_and_wf_singles_and_doubles

    U=2.13 #interaction energy
    t=1 #hopping
    n=2 #number of sites

    FH_ham, UCCD_wfn=FH_ham_and_wf(n,t,U) #returns Hamiltonian and UCCD wfn
    FH_ham,UCCDS_wfn= FH_ham_and_wf_singles_and_doubles(n,t,U) #returns Hamiltonian and UCCDS wfn
