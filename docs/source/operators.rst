Operators
=========

The operators module provides functionality for the encoding and manipulation of operators in OpenQARP. This
covers generic tools such as the linear combination of unitaries, qDRIFT and VUMPO, as well as the fermionic
toolbox: occupation-number vectors, fermion-to-qubit mappings, unitary coupled cluster excitation generators,
model Hamiltonians and electronic-integral utilities.

Linear Combination of Unitaries (LCU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only unitary operators/matrices can be encoded in a quantum circuit. Thus, for a general operator, this needs to be 
written in terms of a basis of unitary operators, establishing the foundation of the linear combination of unitaries. 


While any basis of unitary matrices can be used, it is convenient to use the set of Pauli matrices with the identity
matrix as they are part already of the gateset for universal quantum computation:

.. math::

   A = \sum_i a_i P_i,

where :math:`P_i` is a Pauli string

In OpenQARP, we can perform this decomposition as stated in the previous equation by means of the :code:`LCU` object.

.. code-block:: python

   import numpy as np
   from qarp.operators import LinearCombinationUnitaries

   dim_A, dim_B = 4, 9 
   A = np.random.rand(dim_A, dim_B)
   LCU = LinearCombinationUnitaries(A)
   decomposition = LCU.decomposition()

   for pauli_string, coefficient in decomposition:
      print(pauli_string, coefficient)  # character k of the string acts on qubit k


We observe that the only requirement is that the matrix :math:`A` is two-dimensional. If the dimensions are not *qubit like*, 
that is, a power of two, the :code:`LCU` object will automatically pad the matrix with rows and columns of zeros for the decomposition.

Often, we would like this decomposition to be treated as a QubitOperator object, instead of the plain decomposition, :code:`LCU` allows 
us to write it in that representation as follows 

.. code-block:: python

   import numpy as np
   from qarp.operators import LinearCombinationUnitaries

   dim_A, dim_B = 4, 9 
   A = np.random.rand(dim_A, dim_B)
   LCU = LinearCombinationUnitaries(A)

   A_qo = LCU.to_QubitOperator()

Grouping Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Hamiltonian written as a sum of Pauli terms is rarely measured or exponentiated one term at
a time — terms that commute can share a circuit. A :class:`~qarp.operators.GroupingStrategy`
is a *pure partitioner*: it takes a list of Pauli terms and returns index groups, never touching
coefficients or doing any diagonalisation itself. Three are built in:

- :class:`~qarp.operators.NoGrouping` — one term per group (the termwise behaviour).
- :class:`~qarp.operators.QubitWiseCommuting` (``qubit_wise = True``) — groups terms
  that agree on every shared qubit, so each group diagonalises with a per-qubit basis change
  alone (no entangling gates needed).
- :class:`~qarp.operators.FullyCommuting` (``qubit_wise = False``) — groups terms under
  *general* Pauli commutation, a strict superset of QWC. Diagonalising a general commuting group
  needs an entangling Clifford, but fewer, larger groups mean fewer circuits.

.. code-block:: python

   from qarp.operators import QubitWiseCommuting, FullyCommuting

   # A Jordan-Wigner hopping term (X0 X1 + Y0 Y1, identical support) plus an
   # unrelated Z on qubit 2.
   terms = [
       {0: "X", 1: "X"},
       {0: "Y", 1: "Y"},
       {2: "Z"},
   ]

   print("QWC:            ", QubitWiseCommuting().group(terms, n_qubits=3))
   print("FullyCommuting: ", FullyCommuting().group(terms, n_qubits=3))

``QubitWiseCommuting`` splits the hopping pair into separate groups — ``X0 X1`` and ``Y0 Y1``
disagree on qubits 0 and 1, so they are not qubit-wise commuting even though they do commute in
the general sense. ``FullyCommuting`` keeps all three terms in one group.

**This distinction is not cosmetic.** QWC is a *measurement* criterion — grouping for
simultaneous readout — and using it where general commutation is required silently changes the
physics. A JW hopping generator is exponentiated as one unit precisely because ``X0 X1`` and
``Y0 Y1`` together generate a particle-number- and :math:`S_z`-conserving rotation; splitting
them into separate exponentials (as QWC grouping would) breaks that conservation. This is why
the Trotter family defaults to ``FullyCommuting()``, not ``QubitWiseCommuting()``, even though
the latter looks like the "simpler" grouping.

Each consumer picks the strategy appropriate to what it does with the groups afterwards:

- :class:`~qarp.algorithms.PauliAveraging` — default :class:`~qarp.operators.FullyCommuting`;
  it inserts an entangling Clifford to diagonalise each group, so it can use the coarser grouping.
- :class:`~qarp.blocks.TrotterBlock` / :class:`~qarp.blocks.TrotterAnsatzBlock` — default
  :class:`~qarp.operators.FullyCommuting`, for the symmetry-conservation reason above.
- Circuit cutting (:class:`~qarp.algorithms.CuttingPrimitive`) — default
  :class:`~qarp.operators.QubitWiseCommuting`, and **requires** ``qubit_wise = True``:
  each subcircuit can only apply a per-qubit basis change, not an entangling Clifford, so an
  incompatible strategy raises at construction.

All three consumers accept an explicit ``grouping=`` override:

.. code-block:: python

   from qarp.blocks import SimpleBlock
   from qarp.operators import QubitOperator, QubitWiseCommuting
   from qarp.algorithms import PauliAveraging
   from qarp.engines import QarpEngine
   import qarp

   block = SimpleBlock(2, name="bell")
   block.h(0)
   block.cx(0, 1)
   block.build()

   ham = QubitOperator("X0 X1") + QubitOperator("Y0 Y1")
   pa = PauliAveraging(ket=block, operator=ham, n_shots=qarp.EXACT, grouping=QubitWiseCommuting())

   engine = QarpEngine()
   engine.build([pa])
   result = engine.run()
   print("Expectation value:", result[0].real)

Custom strategies subclass :class:`~qarp.operators.GroupingStrategy` and implement
``group(terms, n_qubits)``. The one hard requirement is that grouping be **order-insensitive**:
permuting the input terms must permute the output partition, not change it — otherwise the same
Hamiltonian written in a different term order would silently produce a different circuit (and,
for symmetry-carrying operators, different physics). The built-in strategies enforce this via a
canonical term ordering before the greedy partitioning pass.

qDRIFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QDRIFT is an algorithm for Hamiltonian simulation, particularly useful for reducing the complexity of the Hamiltonian evolution operator.
Instead of decomposing the Hamiltonian into all its terms deterministically, qDRIFT samples the most relevant terms from the Hamiltonian 
according to their relative weights as a probability distribution. 

A Hamiltonian is written in terms of coefficients and Pauli words as 

.. math::

   H = \sum_i h_i P_i,

and qDRIFT samples from the normalized distribution of :math:`h_i` the corresponding :math:`P_i`, building an approximate reduced Hamiltonian.

In OpenQARP, we can apply qDRIFT to a Hamiltonian in qubit operator form as follows: 

.. code-block:: python

   from qarp.operators import QubitOperator, qDRIFT

   ham = (
      QubitOperator("X0 Y1", 0.5)
      + QubitOperator("Z0 Z1", 1.0)
      + QubitOperator("Y0 X1", 0.3)
      + QubitOperator("X0 X1", 0.2)
    )

   qdrift_ham = qDRIFT(ham, samples=6).qdrift()

where we have sampled from the coefficients distribution six times in this case. 

We also have the option to fix the number of most relevant terms, that is, to always take into account those Pauli words with largest coefficients. In 
that case, we are applying a partially-randomized qDRIFT, whose syntax is as follows: 


.. code-block:: python

   from qarp.operators import QubitOperator, qDRIFT

   op = (
      QubitOperator("X0 Y1", 0.5)
      + QubitOperator("Z0 Z1", 1.0)
      + QubitOperator("Y0 X1", 0.3)
      + QubitOperator("X0 X1", 0.2)
   )

   qdrift_ham_pr = qDRIFT(op, samples=6, ratio=0.5).partially_randomized()

where we have fixed half of the Pauli words (ratio input), in decreasing order of their weight distribution, to be always included; and we sample six times from 
the remaining Pauli words. 


VUMPO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Variational Unitary Matrix Product Operator (VUMPO) algorithm from 
Pollmann *et al.* in **Efficient variational diagonalization of fully many-body
localized Hamiltonians**, *Phys. Rev. B* **94**, 041116(R) (2016),
https://doi.org/10.1103/PhysRevB.94.041116. provides a
tensor network inspired variational method for diagonalizing many body Hamiltonian using shallow, 
brickwork structured quantum circuits
Instead of applying a full unitary decomposition, VUMPO constructs a
layered circuit of nearest neighbour two qubit unitaries that together can correspond to a
unitary Matrix Product Operator (uMPO) ansatz with bond dimension :math:`D`.

Given a Hamiltonian represented as a Matrix Product Operator (MPO)

.. math::

   H = \sum_{\alpha} W^{[1]}_{\alpha_1} W^{[2]}_{\alpha_2} \cdots W^{[L]}_{\alpha_L},

VUMPO aims to find a unitary circuit :math:`U` such that either  
(1) :math:`U^\dagger H U` becomes approximately diagonal (variational diagonalization), or  
(2) :math:`U | \psi_0 \rangle` approximates the ground state of :math:`H` (variational ground‑state mode).

The core idea is to parameterize each two‑qubit brickwork gate via a
skew‑Hermitian generator and apply the exponential map to obtain a unitary tensor.
In **Hamming‑Weight Preserving (HWP)** mode, each gate acts block‑diagonally and
conserves particle number; in **full** mode a general SU(4) parametrization is used.

A VUMPO circuit of depth ``n_layers`` arranges its gates in staggered even–odd
brickwork pattern. For layer ``m`` the gates act on qubit pairs :math:`(n,n+1)`
starting from ``n = m % 2``. Sweeps over these layers can be optimized locally
(DMRG‑style) or globally (VQE‑style) depending on the desired trade‑off between
speed and convergence.

In OpenQARP, constructing and running VUMPO is done directly from an MPO Hamiltonian.
``qubit_operator_to_mpo`` builds that MPO term by term from a ``QubitOperator``,
never forming the dense matrix, with quimb site ``n`` as qarpx qubit ``n``, the
convention ``VUMPOBrickworkBlock`` uses. For simplicity we take the :math:`H_2`
molecule.

.. code-block:: python

   import numpy as np
   from pyscf import gto, scf

   from qarp.operators import VUMPO, JordanWigner, qubit_operator_to_mpo
   from qarp.operators.pyscf import fermion_operator_from_mf
   from qarp.blocks import VUMPOBrickworkBlock

   mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")).run()
   qop_h2 = JordanWigner().encode_operator(fermion_operator_from_mf(mf))

   L = 4
   H_mpo = qubit_operator_to_mpo(qop_h2, L)
   initial_state_h2 = np.array([1] * (2) + [0] * (2))

   vumpo = VUMPO(
       H_mpo=H_mpo,
       initial_state=initial_state_h2,
       n_layers=2,
       n_sweeps=10,     # a cap: the sweep stops when the cost changes by less than tol
       tol=1e-6,
       alpha=0,
       hwp=True,
       mode="diag",
       opt="local",
   )

   params = vumpo.build()

   # The paper's own quality measures, evaluated from the MPO without a dense matrix:
   f = vumpo.energy_variance(params)       # Eq. (4): summed energy variance, >= 0
   r = vumpo.off_diagonal_ratio(params)    # ||U^dag H U - diag||_F / ||H||_F

How the cost is evaluated. The local sweep follows the paper's DMRG-like
protocol: every gate is minimised with the others frozen into its
*environment*, the network contracted column by column with that gate's
insertions left open. One gate visit then costs :math:`O(1)` in :math:`L`
and a sweep :math:`O(L)`; the local minimiser receives the exact gradient
(the cost is a polynomial in the gate matrix, and the chain rule through
``expm`` is one eigendecomposition). The sweep visits columns in order,
all layers at a column before the next, and mirrors that order on odd sweeps.

Depth is the method's limit, not the implementation's: the boundary across a
cut of the doubled network holds :math:`4^{4s}\chi^2` entries for
:math:`s = \lceil n_\mathrm{layers}/2 \rceil` straddling gates and MPO bond
dimension :math:`\chi`. Up to four layers that fits comfortably; deeper
circuits fall back to a searched contraction path over the whole network,
which is fine for short chains (the six-site, eight-layer example) and is the
paper's "exponential scaling of the cost function" otherwise. The
per-evaluation contractions on the environment use ``np.einsum`` rather than
BLAS on purpose: a threaded BLAS call on a 65536-entry tensor is dominated by
thread contention.

Here, the algorithm constructs a variational uMPO with two layers and performs two
left–right / right–left sweeps, optimizing each two qubit gate.  
The resulting parameter vector ``params`` can be used to build tensor networks,
evaluate energies, or reconstruct the unitary circuit.

We can also target purely ground state energies with the ``mode=gs``, which is much cheaper
than full diagonalization, but will only give a good approximation of the ground state energy. 
A full MWE showcasing VUMPO and the corresponding OpenQARP Block, ``VUMPOBrickworkBlock`` is used in, for example, 
Quantum Computing Quantum Monte Carlo, and can be found in the examples folder.

.. note::
   The VUMPO is a generic approach to approximately diagonalizing Hamiltonians, *however* it will only be
   efficient where the number of layers is not too large, and the target Hamiltonian can be represented 
   well by a low bond dimension MPO, e.g. many-body localized systems, where low bond dimension uMPOs can
   efficiently capture quasi‑local structure in eigenstates. As such, it should not be used for generic 
   Hamiltonians where it can take exponential time/memory. An efficient MPO representation/approximation
   should be constructed first.


Complexity
"""""""""""""""""""""""""""""""""""""""""""""""""

The computational cost of the algorithm scales as

.. math::

   \mathcal{O} \left( L D^{5}\chi^{2}d^{4} \right),

where :math:`L` is the system size, :math:`D` the circuit MPO bond dimension, :math:`\chi`
is the bond dimension of the Hamiltonian MPO, and :math:`d` the local physical dimension
(:math:`d=2` for qubits). For a brickwork VUMPO, :math:`D = 4^{\lceil \frac{L}{2}\rceil}`.


Occupation-number vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Single Slater determinants, as they appear in the Fock-space formulation of fermionic structure, are
represented in OpenQARP as plain Python lists of 0s and 1s — an occupation-number vector (ONV, type alias
:code:`Onv = list[int]` in :code:`qarp.operators.onv`). The convention is abab (alpha/beta interleaved)
ordering: spin orbital :math:`2p` is spatial orbital :math:`p` alpha and :math:`2p + 1` its beta partner,
and index :math:`i` of the list corresponds to spin orbital :math:`i` and qubit :math:`i`. For example,
the Hartree-Fock state for dihydrogen in a minimal basis, :math:`|1100\rangle`, is simply

.. code-block:: python

    onv = [1, 1, 0, 0]

The ONV does not know anything about the orbital basis — the user keeps track of the orbital-index
mapping, both in the vector itself and in any fermionic excitations built on top of it.

The :code:`qarp.operators.onv` module provides a handful of helpers for manipulating these lists:

- :code:`onv_from_spatial_occupations(occupations)` - build a spin-orbital ONV from per-spatial-orbital occupations (each 0, 1 or 2). Occupation 1 fills the alpha spin orbital.
- :code:`freeze(onv, indices)` - return a new ONV with the given spin-orbital indices removed.
- :code:`freeze_spatial(onv, spatial_indices)` - return a new ONV with the given spatial orbitals (abab pairs) removed.
- :code:`active_space(onv, active_electrons, active_orbitals)` - return an Aufbau-obeying ONV for an active space carved from the full-space vector. :code:`active_electrons` may also be an :code:`(alpha, beta)` tuple.

.. code-block:: python

    from qarp.operators.onv import active_space, freeze_spatial, onv_from_spatial_occupations

    onv = onv_from_spatial_occupations([2, 2, 0, 0])  # [1, 1, 1, 1, 0, 0, 0, 0]
    cas = active_space(onv, 2, 2)                     # [1, 1, 0, 0]
    valence = freeze_spatial(onv, [0])                # [1, 1, 0, 0, 0, 0]


Mappings
^^^^^^^^^^^^^^^^^^^^^^^^^

We can not work directly with fermion operators in algorithms for quantum computers. We need to translate them to operators that modify qubit states.
These transformations are usually referred to as mappings. In OpenQARP there are three mappings already implemented — Jordan-Wigner,
Bravyi-Kitaev and parity — all importable directly from :code:`qarp.operators`. They share the abstract base class
:code:`qarp.operators.Mapping`, which the user can subclass to implement their own mapping in a similar fashion. Every mapping offers
two methods: :code:`encode`, which maps a :code:`FermionOperator` (or a list of them) to the corresponding :code:`QubitOperator` (or list),
and :code:`encode_state`, which converts an occupation-number vector (a plain abab list, see above) into the mapped computational-basis
bitstring, again as a plain list of 0s and 1s.

Jordan-Wigner mapping
"""""""""""""""""""""""""""""""

The Jordan-Wigner qubit encoding of fermionic operators is arguably the most common one in the literature. We will not
discuss the theory. Rather, we will show how you can map operators and occupation number vectors to their qubit
representation.

If we have a Hamiltonian we have generated and named the
variable ``fermion_hamiltonian``, we can get a QubitOperator object as follows:

.. code-block:: python

    from qarp.operators import FermionOperator, JordanWigner

    fermion_hamiltonian = FermionOperator("1^ 1") + FermionOperator("2^ 2")
    qubit_hamiltonian = JordanWigner().encode_operator(fermion_hamiltonian)

You may also wish to convert a list of operators - for instance, if we have generated a list of
double excitation fermion operators, we can get the corresponding qubit operators with one call to the
``JordanWigner`` object.

An occupation number vector can also be transformed from the occupation-number basis to the Jordan-Wigner basis as follows:

.. code-block:: python

    from qarp.operators import JordanWigner

    onv = [1, 1, 0, 0]
    onv_jw = JordanWigner().encode_state(onv)

For the case of the Jordan-Wigner mapping, however, the expression is identical. That is, the change of basis matrix is an
identity map.

Bravyi-Kitaev mapping
"""""""""""""""""""""""""""""""

Another common mapping used in the community is the Bravyi-Kitaev transformation. Similarly to the Jordan-Wigner mapping,
a QubitOperator object is built from a FermionOperator object, or a list, as follows (reusing
``fermion_hamiltonian`` from above):

.. code-block:: python

    from qarp.operators import BravyiKitaev

    qubit_hamiltonian = BravyiKitaev().encode_operator(fermion_hamiltonian)

The Bravyi-Kitaev image of an operator depends on the register width, so a *list* is mapped
on one width — ``BravyiKitaev(n_qubits=...)`` when given, otherwise the largest
``count_qubits`` over the list — and its members compose (UCC generators against a
Hamiltonian, for instance).  A single operator with the default ``n_qubits=None`` infers its
own width.

Also, occupation number vectors can be transformed to the Bravyi-Kitaev basis

.. code-block:: python

    from qarp.operators import BravyiKitaev

    onv = [1, 1, 0, 0]
    onv_bk = BravyiKitaev().encode_state(onv)

where now the transformation matrix follows the implementation details found in J. Chem. Phys. 137, 224109 (2012).

Parity mapping
"""""""""""""""""""""""""""""""

Alongside Jordan-Wigner and Bravyi-Kitaev, parity is also one of the fermionic-mapping standards. Syntax for the parity mapping is
similar to Jordan-Wigner and Bravyi-Kitaev, but a main difference is that we need to specifically input now the total number of qubits involved.
The reason behind this is because occupation of the orbitals is now stored non-locally. All in all, a QubitOperator object is built
in the parity basis with a maximum number of ``n_qubits`` from a FermionOperator object, or a list, as follows:

.. code-block:: python

    from qarp.operators import Parity

    n_qubits = 4
    qubit_hamiltonian = Parity(n_qubits).encode_operator(fermion_hamiltonian)

Also, a similar syntax follows to transform occupation number vectors to the parity basis

.. code-block:: python

    from qarp.operators import Parity

    onv = [1, 1, 0, 0]
    onv_p = Parity(n_qubits).encode_state(onv)

where the transformation matrix follows the implementation details found in J. Chem. Phys. 137, 224109 (2012).


Unitary coupled cluster generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`qarp.operators.ucc` module provides stateless generator functions for fermionic excitation
pools which, when exponentiated on a quantum circuit, form a Unitary Coupled Cluster (UCC) ansatz of a
certain type. Every generator returns an ``(operators, symbols)`` pair — equal-length lists of
FermionOperators and ``sympy`` symbols, ready to be used with the various :code:`Block` objects. If you
prefer to create your own symbols, simply discard the second element (e.g. take ``[0]`` of the pair).

The available generators are:

- :code:`ucc_singles(onv, ...)` - single excitations.
- :code:`ucc_doubles(onv, ...)` - double excitations; set :code:`paired=True` for spin-paired spatial-to-spatial doubles.
- :code:`ucc_singles_and_doubles(onv, ...)` - the exact concatenation of the two, singles first; the paired flag is called :code:`paired_doubles` here.
- :code:`adjacent_singles(n_spin_orbitals)` - :math:`a_{i+1}^{\dagger} a_{i}` singles over *disjoint* adjacent pairs — ``(0, 1), (2, 3), …``, one per Lipkin doublet, **not** every :math:`(i, i+1)` pair — so ``adjacent_singles(8)`` returns four generators, not seven.
- :code:`spin_adapted_singles(n_spatial_orbitals)` / :code:`spin_adapted_doubles(n_spatial_orbitals)` - singlet spin-adapted generators over spatial-orbital pairs (they commute with :math:`S^2`); reference-free.

Every generator is oriented as the excitation its symbol names: symbol :code:`s_2to4` pairs with the operator
:math:`a_4^{\dagger} a_2` (source → target, abab spin-orbital indices; spin-adapted symbols like
:code:`sas_0to1` use spatial indices). Double symbols group their index pairs, e.g. :code:`d_0to4_1to5`.

Pools that filter on occupations take a reference occupation-number vector (a plain abab list) as their
first argument. To create the excitations for a paired unitary coupled cluster doubles ansatz on
8 spin orbitals,

.. code-block:: python

    from qarp.operators.ucc import ucc_doubles

    onv = [1, 1, 0, 0, 0, 0, 0, 0]
    puccd, symbols = ucc_doubles(onv, paired=True)

``generalised`` pools are reference-free and may instead be sized with
:code:`n_spin_orbitals`. For example, to create a set of generalised unitary coupled cluster operators,

.. code-block:: python

    from qarp.operators.ucc import ucc_singles_and_doubles

    guccsd, symbols = ucc_singles_and_doubles(n_spin_orbitals=8, generalised=True)

If you wish to also include excitations which violate the spin symmetry of the system, this can be achieved by setting
the :code:`spin_conserving` flag in any of the generating methods to :code:`False`. By default the generators return
antihermitized operators :math:`T - T^{\dagger}`; set :code:`antihermitized=False` for the plain coupled-cluster operators.

To construct adjacent singles, such as those used for the Lipkin model, of the form :math:`a_{i+1}^{\dagger} a_{i}`

.. code-block:: python

    from qarp.operators.ucc import adjacent_singles

    singles, symbols = adjacent_singles(8)

In general, and as demonstrated in the former example, OpenQARP offers you the flexibility to build your own excitations tailored to your purposes.


Model Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Builder functions for certain interesting model Hamiltonians may be found in
:code:`qarp.operators.models`. The motivation lies behind a quick execution of widespread
Hamiltonians in the literature for the user's convenience.

Fermi-Hubbard model
"""""""""""""""""""""""""""""""

One of the most common examples used widely in the quantum-computing
community is the Fermi-Hubbard model, whose Hamiltonian is given by

.. math::

   H = -t \sum_{i\sigma} (a^{\dagger}_{i \sigma} a_{i+1, \sigma} + a^{\dagger}_{i+1, \sigma} a_{i, \sigma}) + U\sum_{i} n_{i \uparrow} n_{i \downarrow}

where :math:`(t, U)` are input parameters and :math:`n_{i \sigma} = a^{\dagger}_{i \sigma} a_{i, \sigma}`. Every site :math:`i` has two possible spins :math:`\sigma` and
thus the number of qubits needed for this model is twice the number of sites.

The Fermi-Hubbard model is important in the study of strongly-correlated electronic systems which appear
frequently in condensed-matter physics and material science.

One function covers any hypercubic lattice: ``fermi_hubbard(dims, t, U, V=0.0, periodic=False)``
with ``dims=(n,)`` for a chain, ``(n_x, n_y)`` for 2D, ``(n_x, n_y, n_z)`` for 3D and so on
(sites are indexed first-dimension-fastest; ``periodic=True`` wraps each dimension of size > 2;
``V`` adds the extended-Hubbard nearest-neighbour density-density coupling)

.. code-block:: python

   from qarp.operators import JordanWigner
   from qarp.operators.models import fermi_hubbard

   n_sites, t, U = 4, 1.4, 2.31
   fermionic_FH = fermi_hubbard((n_sites,), t, U)
   qubit_FH = JordanWigner().encode_operator(fermionic_FH)

   # 3 x 2 lattice with periodic boundaries along x
   fermionic_FH_2d = fermi_hubbard((3, 2), t, U, periodic=True)

where we have already written it using fermion operators and the corresponding qubit operators with the Jordan-Wigner encoding.
For this specific example, a brute-force diagonalization approach can be implemented to obtain the ground-state
energy as a benchmark

.. code-block:: python

    from scipy.sparse.linalg import eigsh

    GSE_FH, _ = eigsh(qubit_FH.sparse_matrix(), k=1, which='SA')

Lipkin model
"""""""""""""""""""""""""""""""

A simple yet useful-for-benchmarking algebraic Hamiltonian model is the Lipkin model, given by

.. math::

   H = \frac{t}{2} \sum_{i} Z_i  + \frac{V}{4}\sum_{i,j} (X_i X_j - Y_i Y_j)

where :math:`t,V` are input parameters :math:`X,Y,Z` are the usual Pauli matrices. This model had certain applications back in the day as a simplified shell-model
valence space representation and it has been used widely recently as a benchmark for quantum computing calculations.

OpenQARP offers the functionality to easily implement this, and in fact any other Hamiltonian model of your wish. The
:code:`lipkin` function builds the former Hamiltonian as a qubit operator directly — the model is native to the
qubit basis, so no fermion-to-qubit mapping is involved.

.. code-block:: python

   from qarp.operators.models import lipkin

   # Parameters
   n, t, V = 4, 0.75, 1.0 / 6

   # Lipkin model Hamiltonian
   lm_hamiltonian = lipkin(n, t, V)

We could diagonalize the former Hamiltonian in the qubit operator basis, as we have done before.

Ising and XY models
"""""""""""""""""""""""""""""""

Two further qubit-basis lattice models share the ``dims`` geometry of :code:`fermi_hubbard`
(one qubit per site):

- :code:`transverse_field_ising(dims, j, h_x, h_z=0.0, periodic=False)` builds
  :math:`H = -j \sum_{\langle a,b \rangle} Z_a Z_b - \sum_s h^x_s X_s - \sum_s h^z_s Z_s`.
  The fields accept a scalar or one value per site — passing per-site arrays (e.g. a seeded
  random draw) gives the random-field Ising model.
- :code:`xy_model(dims, j, periodic=False)` builds
  :math:`H = -\frac{j}{2} \sum_{\langle a,b \rangle} (X_a X_b + Y_a Y_b)` — on a chain this
  is exactly the Jordan-Wigner image of free-fermion hopping, and it is the hardcore
  Bose-Hubbard model.

.. code-block:: python

   import numpy as np
   from qarp.operators.models import transverse_field_ising, xy_model

   tfim = transverse_field_ising((3, 3), j=1.0, h_x=0.5)
   rfim = transverse_field_ising((6,), j=1.0, h_x=0.4, h_z=np.random.default_rng(7).normal(size=6))
   xy = xy_model((6,), j=1.0, periodic=True)


Electronic integrals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic, chemistry-free operator construction lives in :code:`qarp.operators`:

- :code:`fermion_operator_from_tensor(tensor, threshold=1e-12)` - build the k-body :code:`FermionOperator` of any 2k-index spin-orbital tensor: :math:`T[p_1..p_k, q_1..q_k]` contributes :math:`T \, a^\dagger_{p_1} \dots a^\dagger_{p_k} a_{q_1} \dots a_{q_k}` (openfermion :code:`InteractionOperator` convention — first k indices create, last k annihilate). Works for 1-, 2-, 3-, 4-body (any k); no tensor symmetry is assumed.
- :code:`rotate_tensor(u, tensor)` - change of single-particle basis for any 2k-index coefficient tensor: creator slots contract with :math:`u^*` and annihilator slots with :math:`u` (the rank-2k generalization of :math:`u^\dagger h u`, valid for complex unitaries; a rectangular :math:`u` projects into a truncated mode basis). For real orthogonal :math:`u` every slot transforms identically, so the same function rotates chemists'-notation integral tensors.
- :code:`orbital_rotation_matrix(kappa)`, :code:`orbital_rotation_generator(parameters, n)`, :code:`orbital_rotation_parameters(kappa)` - the standard parameterisation of that :math:`u` for orbital optimisation: :math:`U = e^{-\kappa}` for an anti-Hermitian generator :math:`\kappa` (new orbitals are the columns, :math:`C_\text{new} = C_\text{old} U`), and the packing of a real skew-symmetric :math:`\kappa` into its :math:`n(n-1)/2` strict-lower-triangle entries — the flat real vector an optimiser drives. Module-qualified (:code:`qarp.operators`); the optimiser and the energy are yours.

The :code:`qarp.operators.integrals` module is the restricted-chemistry layer on top of it:

- :code:`spatial_to_spin_orbital(tensor)` - expand a 2k-index spatial-orbital tensor to abab spin orbitals; adjacent index pairs share a spin (the chemists' :math:`(pq|rs)` pairing for two-body tensors), and every spin assignment carries the same block (restricted).
- :code:`spin_blocks_to_spin_orbital(blocks)` - the spin-resolved counterpart: a dict of per-spin-pattern blocks — :code:`{"a": h_alpha, "b": h_beta}`, :code:`{"aa": ..., "ab": ..., "ba": ..., "bb": ...}` — places each block on its own sublattice combination; missing patterns are zero blocks.

- :code:`spin_orbital_integrals_to_fermion_operator(constant, one_electron, two_electron, threshold=1e-12)` - the spin-orbital-level entry point (e.g. for FCIDUMP-style data): :math:`h_{pq} a^\dagger_p a_q` plus chemists'-notation :math:`H_2 = \frac{1}{2}\sum (pq|rs)\, a^\dagger_p a^\dagger_r a_s a_q` over spin orbitals. The spatial wrappers below delegate here after their spin unpack.
- :code:`restricted_integrals_to_fermion_operator(constant, one_electron, two_electron, threshold=1e-12)` - build a :code:`FermionOperator` (abab spin-orbital ordering) from a scalar term (e.g. nuclear repulsion), an :math:`N \times N` one-electron integral matrix and an :math:`N \times N \times N \times N` two-electron integral tensor in chemists' notation. Terms with coefficients below the threshold are omitted.
- :code:`unrestricted_integrals_to_fermion_operator(constant, (h_alpha, h_beta), (g_aa, g_ab, g_bb), threshold=1e-12)` - the spin-resolved counterpart. The two-electron triple follows pyscf's UHF block order; the :math:`(\beta\beta|\alpha\alpha)` block is derived by particle exchange, so only the three independent blocks are passed.
- :code:`active_space_integrals(constant, one_electron, two_electron, n_electrons, active_electrons, active_orbitals)` - driver-agnostic reduction of full-space restricted integrals to an active-space set. The lowest :math:`(n_{\text{electrons}} - n_{\text{active}})/2` spatial orbitals are treated as doubly occupied core and folded into the returned constant and effective one-electron matrix.
- :code:`unrestricted_active_space_integrals(constant, (h_alpha, h_beta), (g_aa, g_ab, g_bb), (n_alpha, n_beta), (active_alpha, active_beta), active_orbitals)` - the spin-resolved reduction (pyscf :code:`UCASCI.get_h1eff` convention): each channel freezes its lowest :math:`n_\sigma - n^{\text{active}}_\sigma` orbitals; Coulomb folds in from every core electron, exchange from same-spin cores only.

.. note::
    The :code:`integrals` functions assume chemists' notation, that is, that the two-electron integrals are of the form :math:`(11|22)`, and integrals obtained under a restricted formalism. Spin-orbital tensors in any other convention can go through :code:`fermion_operator_from_tensor` directly.

OpenQARP does not wrap any classical chemistry driver: the contract is plain numpy integral arrays,
obtained from whichever package you prefer. With pyscf, the arrays are three lines from a converged
mean-field object —

.. code-block:: python

    from pyscf import ao2mo, gto, scf
    from qarp.operators.integrals import restricted_integrals_to_fermion_operator

    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")).run()
    constant = mf.energy_nuc()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mf.mol, mf.mo_coeff, aosym="s1").reshape([h1.shape[0]] * 4)
    fop = restricted_integrals_to_fermion_operator(constant, h1, h2)

— and that recipe ships as :mod:`qarp.operators.pyscf`, so the same thing, plus the reference
determinant, is

.. code-block:: python

    from qarp.operators.pyscf import fermion_operator_from_mf, onv_from_mf

    fop = fermion_operator_from_mf(mf)
    onv = onv_from_mf(mf)  # abab: [1, 1, 0, 0]

pyscf stays optional: :mod:`qarp.operators.pyscf` imports without it and only reaches for it inside
each call (``pip install pyscf`` when you need it), and nothing in it runs SCF — the mean-field object
is yours, converged how you like. The resulting :code:`FermionOperator` can then be mapped to a qubit
operator with any of the mappings above, e.g. ``JordanWigner().encode_operator(fop)``. To work in an
active space instead of the full orbital space, reduce the integrals first and carve the matching
reference ONV — :func:`~qarp.operators.pyscf.active_space_from_mf` does both:

.. code-block:: python

    from qarp.operators.integrals import restricted_integrals_to_fermion_operator
    from qarp.operators.pyscf import active_space_from_mf

    integrals, onv_cas = active_space_from_mf(mf, active_electrons=2, active_orbitals=2)
    fop_cas = restricted_integrals_to_fermion_operator(*integrals)

A correlated reference — a pyscf FCI or CASCI vector rather than the single determinant — goes
through :func:`~qarp.operators.pyscf.onv_coefficients_from_civec` ``(civec, n_orbitals, nelec,
threshold=1e-12, n_core=0)``, which turns the ``(n_alpha_strings, n_beta_strings)`` array that
``fci.FCI(mf).kernel()`` / ``mcscf.CASCI(...).ci`` return over ``n_orbitals`` *spatial* orbitals
into the abab ONV-coefficient dict :code:`MultiONVStateBlock` takes (see :doc:`blocks`); ``n_core``
prepends that many doubly occupied core orbitals so a CASCI vector can be placed on the full
register. pyscf's determinant is :math:`(\alpha\text{ string})(\beta\text{ string})|\text{vac}\rangle`,
all α creation operators to the left of the β ones, whereas qarp's (``MultiONVStateBlock``,
``FermionOperator``) is ascending spin-orbital index, which interleaves the two — so each
determinant picks up the sign :math:`(-1)^{\#\{(p \in \text{occ}_\beta,\, q \in \text{occ}_\alpha) : q > p\}}`
from moving the β operators into place. The CASCI energy oracle in
``tests/test_operators/test_pyscf_helpers.py`` confirms it: the converted vector loaded by
``MultiONVStateBlock`` is an eigenvector of the mapped Hamiltonian at pyscf's energy, and dropping
the sign is not.

.. code-block:: python

    from pyscf import mcscf
    from qarp.operators.pyscf import onv_coefficients_from_civec

    casci = mcscf.CASCI(mf, ncas=2, nelecas=2)
    casci.kernel()
    coefficients = onv_coefficients_from_civec(casci.ci, 2, casci.nelecas, n_core=casci.ncore)

Orbital optimisation is the same integrals, rotated: parameterise the rotation with the
helpers above, let any optimiser drive the packed vector, and read the optimised
coefficients back as :code:`mf.mo_coeff @ U` —

.. code-block:: python

    import numpy as np
    from scipy.optimize import minimize

    from qarp.operators import rotate_tensor
    from qarp.operators.pyscf import integrals_from_mf
    from qarp.operators import orbital_rotation_generator, orbital_rotation_matrix

    constant, h1, h2 = integrals_from_mf(mf)
    n = h1.shape[0]
    occupied = range(mf.mol.nelectron // 2)

    def energy(x):  # any functional of the rotated integrals: mean-field here, VQE, a solver
        u = orbital_rotation_matrix(orbital_rotation_generator(x, n))
        a, b = rotate_tensor(u, h1), rotate_tensor(u, h2)
        one = 2 * sum(a[i, i] for i in occupied)
        two = sum(2 * b[i, i, j, j] - b[i, j, j, i] for i in occupied for j in occupied)
        return constant + one + two

    result = minimize(energy, np.zeros(n * (n - 1) // 2))
    mo_optimised = mf.mo_coeff @ orbital_rotation_matrix(orbital_rotation_generator(result.x, n))

(``examples/operators/mwe_orbital_rotation.ipynb`` runs this loop on LiH and recovers the RHF
energy from perturbed orbitals.)

