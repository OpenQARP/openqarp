"""pennylane stack (PauliSentence arithmetic): builders and canonical views.

Per-term PauliSentence addition is quadratic, so the accumulate idiom here is
dict accumulation keyed by PauliWord, closed into one PauliSentence — the
same dict semantics openfermion/qarp use natively.
"""

from pennylane import fermi
from pennylane.pauli import PauliSentence, PauliWord

_ = fermi  # re-exported for the mapping kernels below


def prepare_terms(specs):
    return [(dict(factors), coeff) for factors, coeff in specs]


def construct_one(item):
    return PauliSentence({PauliWord(item[0]): item[1]})


def accumulate(prepared, n_qubits=None):
    del n_qubits  # width is implicit in the wire labels
    terms: dict = {}
    for factors, coeff in prepared:
        w = PauliWord(factors)
        terms[w] = terms.get(w, 0.0) + coeff
    return PauliSentence(terms)


def qubit_op(specs, n_qubits=None):
    return accumulate(prepare_terms(specs), n_qubits)


def hold(spec_stream, n_qubits=None):
    del n_qubits
    terms: dict = {}
    for factors, coeff in spec_stream:
        w = PauliWord(dict(factors))
        terms[w] = terms.get(w, 0.0) + coeff
    return PauliSentence(terms)


def entries(op):
    for w, coeff in op.items():
        factors = tuple(sorted((int(wire), letter) for wire, letter in w.items()))
        yield factors, complex(coeff)


def fermion_op(specs):
    terms: dict = {}
    for ops, coeff in specs:
        w = fermi.FermiWord(
            {(i, mode): "+" if action else "-" for i, (mode, action) in enumerate(ops)}
        )
        terms[w] = terms.get(w, 0.0) + coeff
    return fermi.FermiSentence(terms)


def jordan_wigner(op):
    return fermi.jordan_wigner(op, ps=True)


def bravyi_kitaev(op, n_qubits):
    return fermi.bravyi_kitaev(op, n_qubits, ps=True)


def parity_transform(op, n_qubits):
    return fermi.parity_transform(op, n_qubits, ps=True)


def product(a, b):
    return a @ b


def commutator(a, b):
    return (a @ b) - (b @ a)


def hc(op):
    """Pauli words are self-adjoint: the conjugate is coefficient conjugation."""
    return PauliSentence({w: complex(coeff).conjugate() for w, coeff in op.items()})


def to_sparse(op, n_qubits):
    return op.to_mat(wire_order=list(range(n_qubits)), format="csr")
