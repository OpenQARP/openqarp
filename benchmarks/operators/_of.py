"""openfermion stack (the §18 oracle): builders and canonical views."""

import openfermion
import sympy

from benchmarks.operators import inputs


def prepare_terms(specs):
    return [(inputs.term_string(factors), coeff) for factors, coeff in specs]


def construct_one(item):
    return openfermion.QubitOperator(item[0], item[1])


def accumulate(prepared, n_qubits=None):
    del n_qubits  # width is implicit in the term dict
    op = openfermion.QubitOperator()
    for term, coeff in prepared:
        op += openfermion.QubitOperator(term, coeff)
    return op


def qubit_op(specs, n_qubits=None):
    return accumulate(prepare_terms(specs), n_qubits)


def hold(spec_stream, n_qubits=None):
    del n_qubits
    op = openfermion.QubitOperator()
    for factors, coeff in spec_stream:
        op += openfermion.QubitOperator(inputs.term_string(factors), coeff)
    return op


def entries(op):
    return ((tuple(term), complex(coeff)) for term, coeff in op.terms.items())


def fermion_op(specs):
    op = openfermion.FermionOperator()
    for ops, coeff in specs:
        op += openfermion.FermionOperator(inputs.ladder_string(ops), coeff)
    return op


def jordan_wigner(op):
    return openfermion.jordan_wigner(op)


def bravyi_kitaev(op, n_qubits):
    return openfermion.bravyi_kitaev(op, n_qubits=n_qubits)


def parity_transform(op, n_qubits):
    """openfermion ships no parity mapping; this is the hand-rolled reference
    over openfermion operators from tests/test_operators/
    test_mappings_vs_openfermion.py, kept verbatim as the oracle column."""
    Q = openfermion.QubitOperator

    def ladder(orbital, creation):
        qop = Q("X0 X0", 1.0 / 2)
        for k in range(orbital + 1, n_qubits):
            qop *= Q((k, "X"), 1.0)
        sign = -1j if creation else 1j
        if orbital > 0:
            qop *= Q((orbital - 1, "Z"), 1.0) * Q((orbital, "X"), 1.0) + Q((orbital, "Y"), sign)
        else:
            qop *= Q((orbital, "X"), 1.0) + Q((orbital, "Y"), sign)
        return qop

    qop = Q()
    for term, coefficient in op.terms.items():
        qop_term = Q("X0 X0", coefficient)
        for orb_num, action in term:
            qop_term *= ladder(orb_num, action == 1)
        qop += qop_term
    return qop


def product(a, b):
    return a * b


def commutator(a, b):
    return a * b - b * a


def hc(op):
    return openfermion.hermitian_conjugated(op)


def to_sparse(op, n_qubits):
    return openfermion.get_sparse_operator(op, n_qubits=n_qubits)


def symbolic_op(sym_specs):
    op = openfermion.QubitOperator()
    for factors, coeff, name in sym_specs:
        op += openfermion.QubitOperator(inputs.term_string(factors), coeff * sympy.Symbol(name))
    return op


def substituted_entries(op):
    values = {sympy.Symbol(k): v for k, v in inputs.SYMBOL_VALUES.items()}
    return (
        (tuple(term), complex(sympy.sympify(coeff).subs(values)))
        for term, coeff in op.terms.items()
    )
