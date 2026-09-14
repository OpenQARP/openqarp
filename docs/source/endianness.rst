Endianness
==========

qarp/qarpx index statevectors and operator matrices **LSB-first**: qubit ``q`` carries bit
``q`` of the amplitude index, i.e. amplitude index :math:`i = \sum_q 2^q \cdot b_q`. Qubit 0
is the *least*-significant bit. This is the opposite convention from openfermion, cirq,
pennylane, and pytket, which are all MSB-first (qubit 0 = *most*-significant bit).

This mismatch is a recurring source of real bugs: code that crosses the boundary without
converting doesn't raise — it silently computes against the *wrong qubit*. This page is the
one place that convention, and the helpers for crossing it correctly, are documented; see
also ``qarp_conventions.md`` §1, the authoritative source this page summarises.

Two different things are both called "MSB/LSB", and mixing them up is its own trap:

- **Binary notation** (``0b110``, ``bin(x)``, ``format(x, "b")``) is written MSB-first: the
  *rightmost* digit is qubit 0.
- **Bitstring containers** — sampler keys, occupation-number vectors, basis-state lists — are
  LSB-first: the *first element* is qubit 0.

``6 = 0b110`` is the tuple ``(0, 1, 1)``, not ``(1, 1, 0)``. Converting between the two
requires a reversal; ``int("".join(...), 2)`` or ``bin(...)`` silently produce the wrong one.

``label_to_bits`` / ``bits_to_label``
----------------------------------------

:func:`~qarp.endianness.label_to_bits` and :func:`~qarp.endianness.bits_to_label` convert
between an integer basis-state label and its LSB-ordered bit list — the container
convention used throughout qarp (``ComputationalBasisStateBlock``, walker labels,
``generate_states_new_basis``):

.. code-block:: python

    from qarp.endianness import label_to_bits, bits_to_label

    bits = label_to_bits(6, n_qubits=3)
    print(bits)             # [0, 1, 1]  — qubit 0 is bit 0, not the MSB digit of 0b110
    print(bin(6))           # '0b110' — MSB-first notation, for comparison only

    print(bits_to_label([0, 1, 1]))   # 6, the inverse conversion

Both accept string bits too (``label_to_bits("6", 3)``, ``bits_to_label(["0", "1", "1"])``),
matching how bits are often carried around as characters.

The openfermion MSB boundary
-------------------------------

Operator matrices computed *inside* qarp never need conversion: ``QubitOperator.sparse_matrix()``
/ ``FermionOperator.sparse_matrix()`` realise directly in the qarpx LSB convention, ready to
contract with qarpx statevectors and unitaries.

The boundary only exists at :func:`qarp.operators.compat.get_sparse_operator` — the
**openfermion-interop surface** (needs the ``openqarp[openfermion]`` extra), kept MSB-ordered
on purpose so it matches what
``openfermion.linalg.get_sparse_operator`` would produce (useful when comparing directly
against an openfermion computation). Contracting its output with a qarpx statevector or
another LSB matrix requires bit-reversing it first, with
:func:`~qarp.endianness.msb_to_lsb_matrix` / :func:`~qarp.endianness.lsb_to_msb_matrix`
(bit reversal is an involution, so both names are the same permutation — pick whichever
documents the direction at the call site).

.. code-block:: python

    import numpy as np
    from qarp.operators import QubitOperator
    from qarp.operators.compat import get_sparse_operator
    from qarp.endianness import msb_to_lsb_matrix

    x0 = QubitOperator("X0")

    # |q1 q0> = |10>, i.e. qubit 0 = 0, qubit 1 = 1 -> LSB index 2.
    state = np.array([0, 0, 1, 0], dtype=complex)

    lsb_mat = x0.sparse_matrix(n_qubits=2).toarray()
    msb_mat = get_sparse_operator(x0, n_qubits=2).toarray()

    # X0 should flip qubit 0: |10> (index 2) -> |11> (index 3).
    print("sparse_matrix() directly:            ", lsb_mat @ state)
    print("get_sparse_operator(), uncorrected:  ", msb_mat @ state)
    print("get_sparse_operator(), bit-reversed: ", msb_to_lsb_matrix(msb_mat) @ state)

The uncorrected MSB matrix does not raise, and does not merely round differently — it flips
a completely different qubit (landing on ``|00⟩`` instead of ``|11⟩``, since "qubit 0" means
opposite ends of the tensor product in the two conventions). Only the bit-reversed version
agrees with ``sparse_matrix()``'s direct LSB computation. This is the shape every bug in this
class takes: no exception, a plausible-looking wrong number.

:func:`~qarp.endianness.msb_to_lsb_statevector` (aliased as ``lsb_to_msb_statevector``, same
permutation) does the equivalent conversion for statevectors, e.g. when comparing a
diagonalised ``get_sparse_operator`` eigenvector against a qarpx-simulated statevector.

Rule of thumb
---------------

- Computing or contracting entirely inside qarp/qarpx (``sparse_matrix()``, qarpx
  statevectors, qarpx unitaries): nothing to convert, LSB throughout.
- Crossing into or out of openfermion (``get_sparse_operator``, an openfermion
  ``QubitOperator``/``FermionOperator`` fed a qarpx result, or vice versa): bit-reverse with
  the helpers above.
- Printing or parsing a label as binary text (``bin()``, ``format()``, an f-string):
  MSB-first, and not interchangeable with an LSB bit-tuple without :func:`label_to_bits` /
  :func:`bits_to_label`.
