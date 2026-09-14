"""qarp/qarpx stack: builders and canonical views for the operators track."""

import sympy

from benchmarks.operators import inputs
from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.functions import bravyi_kitaev as _bk
from qarp.operators.functions import hermitian_conjugated as _hc
from qarp.operators.functions import jordan_wigner as _jw
from qarp.operators.functions import parity_transform as _parity


def prepare_terms(specs):
    return [(inputs.term_string(factors), coeff) for factors, coeff in specs]


def construct_one(item):
    return QubitOperator(item[0], item[1])


def accumulate(prepared, n_qubits=None):
    del n_qubits  # width is implicit in the term dict
    op = QubitOperator()
    for term, coeff in prepared:
        op += QubitOperator(term, coeff)
    return op


def qubit_op(specs, n_qubits=None):
    return accumulate(prepare_terms(specs), n_qubits)


def hold(spec_stream, n_qubits=None):
    del n_qubits
    op = QubitOperator()
    for factors, coeff in spec_stream:
        op += QubitOperator(inputs.term_string(factors), coeff)
    return op


def entries(op):
    return ((tuple(term), complex(coeff)) for term, coeff in op.terms.items())


def fermion_op(specs):
    op = FermionOperator()
    for ops, coeff in specs:
        op += FermionOperator(inputs.ladder_string(ops), coeff)
    return op


def jordan_wigner(op):
    return _jw(op)


def bravyi_kitaev(op, n_qubits):
    return _bk(op, n_qubits)


def parity_transform(op, n_qubits):
    return _parity(op, n_qubits)


def product(a, b):
    return a * b


def commutator(a, b):
    return a * b - b * a


def hc(op):
    return _hc(op)


def to_sparse(op, n_qubits):
    return op.sparse_matrix(n_qubits)


def symbolic_op(sym_specs):
    op = QubitOperator()
    for factors, coeff, name in sym_specs:
        op += QubitOperator(inputs.term_string(factors), coeff * sympy.Symbol(name))
    return op


def substituted_entries(op):
    return entries(op.substitute(dict(inputs.SYMBOL_VALUES)))
