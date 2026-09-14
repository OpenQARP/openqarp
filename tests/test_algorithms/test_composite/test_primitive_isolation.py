"""B14: composite algorithms take a private copy of a caller-supplied
primitive — ``build()`` rebinds ket/bra/operator/measured_qubits on it, so a
primitive shared across two algorithms must not cross-corrupt, and the
caller's object must never be mutated.
"""

from sympy import Symbol

from qarp.algorithms import QPE, VQA, Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.operators import QubitOperator


def _ry_ket(sym):
    b = SimpleBlock(1)
    b.ry(0, Symbol(sym))
    return b


def test_shared_primitive_across_vqas_is_isolated():
    sv = StateVector()
    v1 = VQA(QubitOperator("Z0"), _ry_ket("a"), "v1", primitive=sv)
    v2 = VQA(QubitOperator("X0"), _ry_ket("b"), "v2", primitive=sv)
    v1.build()
    v2.build()
    assert v1.primitive is not v2.primitive
    assert v1.primitive.operator != v2.primitive.operator
    # The caller's object is never rebound.
    assert sv.ket is None
    assert sv.operator is None


def test_qpe_leaves_caller_primitive_untouched():
    sampler = Sampler(n_shots=100)
    u = SimpleBlock(1)
    u.p(0, Symbol("t"))  # parametric U → generic path → build() rebinds
    u.build()
    prep = SimpleBlock(1)
    qpe = QPE(state=prep, unitary=u, n_ancilla=2, primitive=sampler).build()
    assert qpe.primitive is not sampler
    assert qpe.primitive.ket is not None
    assert sampler.ket is None
    assert sampler.measured_qubits is None
