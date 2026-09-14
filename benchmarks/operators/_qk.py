"""qiskit stack (SparsePauliOp, Rust-backed): builders and canonical views.

SparsePauliOp addition concatenates term lists without combining duplicates,
so the accumulate/product kernels include the .simplify() a qiskit user needs
for the equivalent result — timing it is part of the comparison, not overhead.
Batch construction (from_sparse_list) is the SparsePauliOp idiom and is used
wherever the other stacks accumulate into a dict.  Labels are MSB-left
strings; entries() converts back to the canonical (qubit, letter) form.
"""

from qiskit.quantum_info import SparsePauliOp

from benchmarks.operators import inputs

N_QUBITS = inputs.N_QUBITS


def _sparse_term(factors, coeff):
    letters = "".join(letter for _, letter in factors)
    indices = [qubit for qubit, _ in factors]
    return (letters, indices, coeff)


def prepare_terms(specs):
    return [_sparse_term(factors, coeff) for factors, coeff in specs]


def construct_one(item, n_qubits=N_QUBITS):
    return SparsePauliOp.from_sparse_list([item], num_qubits=n_qubits)


def accumulate(prepared, n_qubits=N_QUBITS):
    return SparsePauliOp.from_sparse_list(prepared, num_qubits=n_qubits).simplify()


def qubit_op(specs, n_qubits=N_QUBITS):
    return accumulate(prepare_terms(specs), n_qubits)


def hold(spec_stream, n_qubits=24):
    # The materialized term list is part of the batch-construction idiom and
    # of this stack's memory footprint (noted in the table).
    prepared = [_sparse_term(factors, coeff) for factors, coeff in spec_stream]
    return SparsePauliOp.from_sparse_list(prepared, num_qubits=n_qubits).simplify()


def entries(op):
    for label, coeff in op.to_list():
        factors = tuple(
            (qubit, letter) for qubit, letter in enumerate(reversed(label)) if letter != "I"
        )
        yield factors, complex(coeff)


def product(a, b):
    return (a @ b).simplify()


def commutator(a, b):
    return ((a @ b) - (b @ a)).simplify()


def hc(op):
    return op.adjoint()


def to_sparse(op, n_qubits):
    del n_qubits  # width is carried by the operator itself
    return op.to_matrix(sparse=True)
