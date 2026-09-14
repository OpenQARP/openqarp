# Local dependencies
import numpy as np

# External dependencies
from qarp.operators import BravyiKitaev, FermionOperator, JordanWigner, Parity, QubitOperator
from qarp.operators.functions import hermitian_conjugated


def test_JW_encode_state():
    onv = [1, 1, 1, 1, 0, 0, 0, 0]
    assert JordanWigner().encode_state(onv) == onv
    assert JordanWigner().encode_state(onv) == onv


def test_JW_encode_one_elec():
    fop = FermionOperator("2^ 1")
    qop = JordanWigner().encode_operator(fop)
    terms = [
        (((1, "Y"), (2, "X")), 0.25j),
        (((1, "X"), (2, "X")), 0.25),
        (((1, "Y"), (2, "Y")), 0.25),
        (((1, "X"), (2, "Y")), -0.25j),
    ]
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16

    fop = FermionOperator("1^ 5")
    qop = JordanWigner().encode_operator(fop)
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    terms = [
        (((1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")), -0.25j),
        (((1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")), (0.25 + 0j)),
        (((1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")), (0.25 + 0j)),
        (((1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")), 0.25j),
    ]
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_JW_encode_two_elec():
    fop = FermionOperator("2^ 1 5^ 3")
    qop = JordanWigner().encode_operator(fop)
    terms = [
        (((1, "Y"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), -1 / 16),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), 0.0625j),
        (((1, "Y"), (2, "X"), (3, "Y"), (4, "Z"), (5, "Y")), 0.0625j),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "Y")), 0.0625),
        (((1, "X"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), 0.0625j),
        (((1, "X"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), 0.0625),
        (((1, "X"), (2, "X"), (3, "Y"), (4, "Z"), (5, "Y")), 0.0625),
        (((1, "X"), (2, "X"), (3, "X"), (4, "Z"), (5, "Y")), -0.0625j),
        (((1, "Y"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), 0.0625j),
        (((1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), 0.0625),
        (((1, "Y"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "Y")), 0.0625),
        (((1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "Y")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), 0.0625),
        (((1, "X"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "Y")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "X"), (4, "Z"), (5, "Y")), -0.0625),
    ]
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_JW_encode_list_one_elec():
    fops = [FermionOperator("2^ 1"), FermionOperator("1^ 5")]
    terms = [
        (((1, "Y"), (2, "X")), 0.25j),
        (((1, "X"), (2, "X")), 0.25),
        (((1, "Y"), (2, "Y")), 0.25),
        (((1, "X"), (2, "Y")), -0.25j),
    ] + [
        (((1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")), -0.25j),
        (((1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")), (0.25 + 0j)),
        (((1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")), (0.25 + 0j)),
        (((1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")), 0.25j),
    ]
    qops = JordanWigner().encode_operator(fops)
    qop_terms = [term for term in qops[0].terms.items()] + [term for term in qops[1].terms.items()]  # type: ignore

    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_JW_encode_list_two_elec():
    fops = [FermionOperator("2^ 1 5^ 3"), FermionOperator("2^ 0 3^ 1")]
    qops = JordanWigner().encode_operator(fops)
    terms = [
        (((1, "Y"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), -1 / 16),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), 0.0625j),
        (((1, "Y"), (2, "X"), (3, "Y"), (4, "Z"), (5, "Y")), 0.0625j),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "Y")), 0.0625),
        (((1, "X"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), 0.0625j),
        (((1, "X"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), 0.0625),
        (((1, "X"), (2, "X"), (3, "Y"), (4, "Z"), (5, "Y")), 0.0625),
        (((1, "X"), (2, "X"), (3, "X"), (4, "Z"), (5, "Y")), -0.0625j),
        (((1, "Y"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), 0.0625j),
        (((1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), 0.0625),
        (((1, "Y"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "Y")), 0.0625),
        (((1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "Y")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), 0.0625),
        (((1, "X"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "Y")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "X"), (4, "Z"), (5, "Y")), -0.0625),
    ] + [
        (((0, "Y"), (1, "X"), (2, "Y"), (3, "X")), 0.0625),
        (((0, "Y"), (1, "Y"), (2, "Y"), (3, "X")), 0.0625j),
        (((0, "Y"), (1, "X"), (2, "Y"), (3, "Y")), -0.0625j),
        (((0, "Y"), (1, "Y"), (2, "Y"), (3, "Y")), 0.0625),
        (((0, "X"), (1, "X"), (2, "Y"), (3, "X")), -0.0625j),
        (((0, "X"), (1, "Y"), (2, "Y"), (3, "X")), 0.0625),
        (((0, "X"), (1, "X"), (2, "Y"), (3, "Y")), -0.0625),
        (((0, "X"), (1, "Y"), (2, "Y"), (3, "Y")), -0.0625j),
        (((0, "Y"), (1, "X"), (2, "X"), (3, "X")), 0.0625j),
        (((0, "Y"), (1, "Y"), (2, "X"), (3, "X")), -0.0625),
        (((0, "Y"), (1, "X"), (2, "X"), (3, "Y")), 0.0625),
        (((0, "Y"), (1, "Y"), (2, "X"), (3, "Y")), 0.0625j),
        (((0, "X"), (1, "X"), (2, "X"), (3, "X")), 0.0625),
        (((0, "X"), (1, "Y"), (2, "X"), (3, "X")), 0.0625j),
        (((0, "X"), (1, "X"), (2, "X"), (3, "Y")), -0.0625j),
        (((0, "X"), (1, "Y"), (2, "X"), (3, "Y")), 0.0625),
    ]
    qop_terms = [term for term in qops[0].terms.items()] + [term for term in qops[1].terms.items()]  # type: ignore

    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_JW_conjugated():
    a, b = np.random.randint(0, 20), np.random.randint(0, 20)
    fop1 = FermionOperator(str(a) + "^ " + str(a))
    fop1 -= hermitian_conjugated(fop1)
    fop2 = FermionOperator(str(b) + "^ " + str(b))
    fop2 -= hermitian_conjugated(fop2)

    assert JordanWigner().encode_operator(fop1) == -JordanWigner().encode_operator(fop2)
    assert JordanWigner().encode_operator(fop1) == -JordanWigner().encode_operator(fop2)


def test_JW_properties():
    a = np.random.randint(0, 20)
    fops = [
        FermionOperator("0^"),
        FermionOperator("0"),
        FermionOperator(str(a) + "^ " + str(a) + "^ "),
        FermionOperator(str(a) + " " + str(a)),
        FermionOperator(str(a) + "^ " + str(a)),
        FermionOperator(str(a) + "^ " + str(a + 1)),
    ]

    qop_terms = [
        QubitOperator((0, "X"), 0.5) - QubitOperator((0, "Y"), 0.5j),
        QubitOperator((0, "X"), 0.5) + QubitOperator((0, "Y"), 0.5j),
        QubitOperator((0, "X"), 0.0),
        QubitOperator((0, "X"), 0.0),
        (QubitOperator((a, "X"), 0.5) - QubitOperator((a, "Y"), 0.5j))
        * (QubitOperator((a, "X"), 0.5) + QubitOperator((a, "Y"), 0.5j)),
        (QubitOperator((a, "X"), 0.5) - QubitOperator((a, "Y"), 0.5j))
        * QubitOperator((a, "Z"), 1.0)
        * (QubitOperator((a + 1, "X"), 0.5) + QubitOperator((a + 1, "Y"), 0.5j)),
    ]

    qops = JordanWigner().encode_operator(fops)
    qops = JordanWigner().encode_operator(fops)

    for i_op in range(len(fops)):
        assert qops[i_op] == qop_terms[i_op]


def test_BK_encode_state():
    onv = [1, 1, 1, 1, 0, 0, 0, 0]
    assert BravyiKitaev().encode_state(onv) == [1, 0, 1, 0, 0, 0, 0, 0]
    assert BravyiKitaev().encode_state(onv) == [1, 0, 1, 0, 0, 0, 0, 0]


def test_BK_encode_one_elec():
    fop = FermionOperator("2^ 1")
    qop = BravyiKitaev().encode_operator(fop)
    qop = BravyiKitaev().encode_operator(fop)
    terms = [
        (((0, "Z"), (1, "Y"), (2, "X")), 0.25j),
        (((1, "X"), (2, "X")), 0.25),
        (((0, "Z"), (1, "Y"), (2, "Y")), 0.25),
        (((1, "X"), (2, "Y")), -0.25j),
    ]
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16

    fop = FermionOperator("1^ 5")
    qop = BravyiKitaev().encode_operator(fop)
    qop = BravyiKitaev().encode_operator(fop)
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    terms = [
        (((0, "Z"), (1, "X"), (3, "Y"), (4, "Z"), (5, "X")), -0.25j),
        (((0, "Z"), (1, "X"), (3, "Y"), (5, "Y")), (0.25 + 0j)),
        (((1, "Y"), (3, "Y"), (4, "Z"), (5, "X")), (-0.25 + 0j)),
        (((1, "Y"), (3, "Y"), (5, "Y")), -0.25j),
    ]
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_BK_encode_two_elec():
    fop = FermionOperator("2^ 1 5^ 3")
    qop = BravyiKitaev().encode_operator(fop)
    qop = BravyiKitaev().encode_operator(fop)
    terms = [
        (((0, "Z"), (1, "X"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), (-0.0625 + 0j)),
        (((0, "Z"), (1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), 0.0625j),
        (((0, "Z"), (1, "X"), (2, "Y"), (3, "Y"), (5, "Y")), 0.0625j),
        (((0, "Z"), (1, "Y"), (2, "X"), (3, "X"), (5, "Y")), (0.0625 + 0j)),
        (((1, "Y"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), -0.0625j),
        (((1, "X"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), (0.0625 + 0j)),
        (((1, "Y"), (2, "Y"), (3, "Y"), (5, "Y")), (-0.0625 + 0j)),
        (((1, "X"), (2, "X"), (3, "X"), (5, "Y")), -0.0625j),
        (((0, "Z"), (1, "X"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), -0.0625j),
        (((0, "Z"), (1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), (0.0625 + 0j)),
        (((0, "Z"), (1, "X"), (2, "X"), (3, "Y"), (5, "Y")), (-0.0625 + 0j)),
        (((0, "Z"), (1, "Y"), (2, "Y"), (3, "X"), (5, "Y")), -0.0625j),
        (((1, "Y"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), (0.0625 + 0j)),
        (((1, "X"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), -0.0625j),
        (((1, "Y"), (2, "X"), (3, "Y"), (5, "Y")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "X"), (5, "Y")), (-0.0625 + 0j)),
    ]
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_BK_encode_list_one_elec():
    fops = [FermionOperator("2^ 1"), FermionOperator("1^ 5")]
    terms = [
        (((0, "Z"), (1, "Y"), (2, "X")), 0.25j),
        (((1, "X"), (2, "X")), 0.25),
        (((0, "Z"), (1, "Y"), (2, "Y")), 0.25),
        (((1, "X"), (2, "Y")), -0.25j),
    ] + [
        (((0, "Z"), (1, "X"), (3, "Y"), (4, "Z"), (5, "X")), -0.25j),
        (((0, "Z"), (1, "X"), (3, "Y"), (5, "Y")), (0.25 + 0j)),
        (((1, "Y"), (3, "Y"), (4, "Z"), (5, "X")), (-0.25 + 0j)),
        (((1, "Y"), (3, "Y"), (5, "Y")), -0.25j),
    ]
    qops = BravyiKitaev().encode_operator(fops)
    qops = BravyiKitaev().encode_operator(fops)
    qop_terms = [term for term in qops[0].terms.items()] + [term for term in qops[1].terms.items()]  # type: ignore

    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_BK_encode_list_two_elec():
    fops = [FermionOperator("2^ 1 5^ 3"), FermionOperator("2^ 0 3^ 1")]
    qops = BravyiKitaev().encode_operator(fops)
    qops = BravyiKitaev().encode_operator(fops)
    terms = [
        (((0, "Z"), (1, "X"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), (-0.0625 + 0j)),
        (((0, "Z"), (1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), 0.0625j),
        (((0, "Z"), (1, "X"), (2, "Y"), (3, "Y"), (5, "Y")), 0.0625j),
        (((0, "Z"), (1, "Y"), (2, "X"), (3, "X"), (5, "Y")), (0.0625 + 0j)),
        (((1, "Y"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "X")), -0.0625j),
        (((1, "X"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")), (0.0625 + 0j)),
        (((1, "Y"), (2, "Y"), (3, "Y"), (5, "Y")), (-0.0625 + 0j)),
        (((1, "X"), (2, "X"), (3, "X"), (5, "Y")), -0.0625j),
        (((0, "Z"), (1, "X"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), -0.0625j),
        (((0, "Z"), (1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), (0.0625 + 0j)),
        (((0, "Z"), (1, "X"), (2, "X"), (3, "Y"), (5, "Y")), (-0.0625 + 0j)),
        (((0, "Z"), (1, "Y"), (2, "Y"), (3, "X"), (5, "Y")), -0.0625j),
        (((1, "Y"), (2, "X"), (3, "Y"), (4, "Z"), (5, "X")), (0.0625 + 0j)),
        (((1, "X"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")), -0.0625j),
        (((1, "Y"), (2, "X"), (3, "Y"), (5, "Y")), -0.0625j),
        (((1, "X"), (2, "Y"), (3, "X"), (5, "Y")), (-0.0625 + 0j)),
    ] + [
        (((0, "Y"), (2, "Y")), (0.0625 + 0j)),
        (((0, "X"), (1, "Z"), (2, "Y")), -0.0625j),
        (((0, "Y"), (1, "Z"), (2, "X"), (3, "Z")), 0.0625j),
        (((0, "X"), (2, "X"), (3, "Z")), (0.0625 + 0j)),
        (((0, "X"), (2, "Y")), -0.0625j),
        (((0, "Y"), (1, "Z"), (2, "Y")), (0.0625 + 0j)),
        (((0, "X"), (1, "Z"), (2, "X"), (3, "Z")), (0.0625 + 0j)),
        (((0, "Y"), (2, "X"), (3, "Z")), 0.0625j),
        (((0, "Y"), (2, "X")), 0.0625j),
        (((0, "X"), (1, "Z"), (2, "X")), (0.0625 + 0j)),
        (((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")), (0.0625 + 0j)),
        (((0, "X"), (2, "Y"), (3, "Z")), -0.0625j),
        (((0, "X"), (2, "X")), (0.0625 + 0j)),
        (((0, "Y"), (1, "Z"), (2, "X")), 0.0625j),
        (((0, "X"), (1, "Z"), (2, "Y"), (3, "Z")), -0.0625j),
        (((0, "Y"), (2, "Y"), (3, "Z")), (0.0625 + 0j)),
    ]
    qop_terms = [term for term in qops[0].terms.items()] + [term for term in qops[1].terms.items()]  # type: ignore

    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_BK_conjugated():
    a, b = np.random.randint(0, 20), np.random.randint(0, 20)
    fop1 = FermionOperator(str(a) + "^ " + str(a))
    fop1 -= hermitian_conjugated(fop1)
    fop2 = FermionOperator(str(b) + "^ " + str(b))
    fop2 -= hermitian_conjugated(fop2)

    assert BravyiKitaev().encode_operator(fop1) == -BravyiKitaev().encode_operator(fop2)
    assert BravyiKitaev().encode_operator(fop1) == -BravyiKitaev().encode_operator(fop2)


def test_BK_properties():
    a = np.random.randint(0, 20)
    fops = [
        FermionOperator("0^"),
        FermionOperator("0"),
        FermionOperator(str(a) + "^ " + str(a) + "^ "),
        FermionOperator(str(a) + " " + str(a)),
    ]

    qop_terms = [
        QubitOperator((0, "X"), 0.5) - QubitOperator((0, "Y"), 0.5j),
        QubitOperator((0, "X"), 0.5) + QubitOperator((0, "Y"), 0.5j),
        QubitOperator((0, "X"), 0.0),
        QubitOperator((0, "X"), 0.0),
    ]

    # The expected images are each operator's own-width BK form; a list maps
    # on one shared width (P1.17), so encode one at a time here.
    qops = [BravyiKitaev().encode_operator(fop) for fop in fops]

    for i_op in range(len(fops)):
        assert qops[i_op] == qop_terms[i_op]

    if (a % 2) == 0:
        fop = FermionOperator(str(a) + "^ " + str(a))
        qop_term = QubitOperator("X0 X0", 0.5) - QubitOperator((a, "Z"), 0.5)
        qop = BravyiKitaev().encode_operator(fop)
        qop = BravyiKitaev().encode_operator(fop)
        assert qop == qop_term


def test_P_encode_state():
    onv = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert Parity(4).encode_state(onv) == [1, 0, 1, 0, 1, 1, 1, 1, 1, 1]


def test_P_encode_one_elec():
    fop = FermionOperator("2^ 1")
    qop = Parity(3).encode_operator(fop)
    qop = Parity(3).encode_operator(fop)
    terms = [
        (((0, "Z"), (1, "Y")), 0.25j),
        (((1, "X"),), (0.25 + 0j)),
        (((0, "Z"), (1, "X"), (2, "Z")), (-0.25 + 0j)),
        (((1, "Y"), (2, "Z")), -0.25j),
    ]
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16

    fop = FermionOperator("1^ 5")
    qop = Parity(6).encode_operator(fop)
    qop = Parity(6).encode_operator(fop)
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    terms = [
        (((0, "Z"), (1, "X"), (2, "X"), (3, "X"), (4, "Y")), -0.25j),
        (((0, "Z"), (1, "X"), (2, "X"), (3, "X"), (4, "X"), (5, "Z")), (-0.25 + 0j)),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "Y")), (-0.25 + 0j)),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "X"), (5, "Z")), 0.25j),
    ]
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_P_encode_two_elec():
    fop = FermionOperator("2^ 1 5^ 3")
    qop = Parity(6).encode_operator(fop)
    qop = Parity(6).encode_operator(fop)
    terms = [
        (((0, "Z"), (1, "Y"), (2, "Z"), (3, "X"), (4, "Y")), (-0.0625 + 0j)),
        (((0, "Z"), (1, "Y"), (3, "Y"), (4, "Y")), -0.0625j),
        (((0, "Z"), (1, "Y"), (2, "Z"), (3, "X"), (4, "X"), (5, "Z")), -0.0625j),
        (((0, "Z"), (1, "Y"), (3, "Y"), (4, "X"), (5, "Z")), (0.0625 + 0j)),
        (((1, "X"), (2, "Z"), (3, "X"), (4, "Y")), 0.0625j),
        (((1, "X"), (3, "Y"), (4, "Y")), (-0.0625 + 0j)),
        (((1, "X"), (2, "Z"), (3, "X"), (4, "X"), (5, "Z")), (-0.0625 + 0j)),
        (((1, "X"), (3, "Y"), (4, "X"), (5, "Z")), -0.0625j),
        (((0, "Z"), (1, "X"), (3, "X"), (4, "Y")), -0.0625j),
        (((0, "Z"), (1, "X"), (2, "Z"), (3, "Y"), (4, "Y")), (0.0625 + 0j)),
        (((0, "Z"), (1, "X"), (3, "X"), (4, "X"), (5, "Z")), (0.0625 + 0j)),
        (((0, "Z"), (1, "X"), (2, "Z"), (3, "Y"), (4, "X"), (5, "Z")), 0.0625j),
        (((1, "Y"), (3, "X"), (4, "Y")), (0.0625 + 0j)),
        (((1, "Y"), (2, "Z"), (3, "Y"), (4, "Y")), 0.0625j),
        (((1, "Y"), (3, "X"), (4, "X"), (5, "Z")), 0.0625j),
        (((1, "Y"), (2, "Z"), (3, "Y"), (4, "X"), (5, "Z")), (-0.0625 + 0j)),
    ]
    qop_terms = [term for term in qop.terms.items()]  # type: ignore
    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_P_encode_list_one_elec():
    fops = [FermionOperator("2^ 1"), FermionOperator("1^ 5")]
    terms = [
        (((0, "Z"), (1, "Y")), 0.25j),
        (((1, "X"),), (0.25 + 0j)),
        (((0, "Z"), (1, "X"), (2, "Z")), (-0.25 + 0j)),
        (((1, "Y"), (2, "Z")), -0.25j),
    ] + [
        (((0, "Z"), (1, "X"), (2, "X"), (3, "X"), (4, "Y")), -0.25j),
        (((0, "Z"), (1, "X"), (2, "X"), (3, "X"), (4, "X"), (5, "Z")), (-0.25 + 0j)),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "Y")), (-0.25 + 0j)),
        (((1, "Y"), (2, "X"), (3, "X"), (4, "X"), (5, "Z")), 0.25j),
    ]
    qops = Parity(6).encode_operator(fops)
    qops = Parity(6).encode_operator(fops)
    qop_terms = [term for term in qops[0].terms.items()] + [term for term in qops[1].terms.items()]  # type: ignore

    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_P_encode_list_two_elec():
    fops = [FermionOperator("2^ 1 5^ 3"), FermionOperator("2^ 0 3^ 1")]
    qops = Parity(6).encode_operator(fops)
    qops = Parity(6).encode_operator(fops)
    terms = [
        (((0, "Z"), (1, "Y"), (2, "Z"), (3, "X"), (4, "Y")), (-0.0625 + 0j)),
        (((0, "Z"), (1, "Y"), (3, "Y"), (4, "Y")), -0.0625j),
        (((0, "Z"), (1, "Y"), (2, "Z"), (3, "X"), (4, "X"), (5, "Z")), -0.0625j),
        (((0, "Z"), (1, "Y"), (3, "Y"), (4, "X"), (5, "Z")), (0.0625 + 0j)),
        (((1, "X"), (2, "Z"), (3, "X"), (4, "Y")), 0.0625j),
        (((1, "X"), (3, "Y"), (4, "Y")), (-0.0625 + 0j)),
        (((1, "X"), (2, "Z"), (3, "X"), (4, "X"), (5, "Z")), (-0.0625 + 0j)),
        (((1, "X"), (3, "Y"), (4, "X"), (5, "Z")), -0.0625j),
        (((0, "Z"), (1, "X"), (3, "X"), (4, "Y")), -0.0625j),
        (((0, "Z"), (1, "X"), (2, "Z"), (3, "Y"), (4, "Y")), (0.0625 + 0j)),
        (((0, "Z"), (1, "X"), (3, "X"), (4, "X"), (5, "Z")), (0.0625 + 0j)),
        (((0, "Z"), (1, "X"), (2, "Z"), (3, "Y"), (4, "X"), (5, "Z")), 0.0625j),
        (((1, "Y"), (3, "X"), (4, "Y")), (0.0625 + 0j)),
        (((1, "Y"), (2, "Z"), (3, "Y"), (4, "Y")), 0.0625j),
        (((1, "Y"), (3, "X"), (4, "X"), (5, "Z")), 0.0625j),
        (((1, "Y"), (2, "Z"), (3, "Y"), (4, "X"), (5, "Z")), (-0.0625 + 0j)),
    ] + [
        (((0, "Y"), (1, "Z"), (2, "Y")), (0.0625 + 0j)),
        (((0, "X"), (2, "Y")), -0.0625j),
        (((0, "Y"), (1, "Z"), (2, "X"), (3, "Z")), 0.0625j),
        (((0, "X"), (2, "X"), (3, "Z")), (0.0625 + 0j)),
        (((0, "X"), (1, "Z"), (2, "Y")), -0.0625j),
        (((0, "Y"), (2, "Y")), (0.0625 + 0j)),
        (((0, "X"), (1, "Z"), (2, "X"), (3, "Z")), (0.0625 + 0j)),
        (((0, "Y"), (2, "X"), (3, "Z")), 0.0625j),
        (((0, "Y"), (2, "X")), 0.0625j),
        (((0, "X"), (1, "Z"), (2, "X")), (0.0625 + 0j)),
        (((0, "Y"), (2, "Y"), (3, "Z")), (0.0625 + 0j)),
        (((0, "X"), (1, "Z"), (2, "Y"), (3, "Z")), -0.0625j),
        (((0, "X"), (2, "X")), (0.0625 + 0j)),
        (((0, "Y"), (1, "Z"), (2, "X")), 0.0625j),
        (((0, "X"), (2, "Y"), (3, "Z")), -0.0625j),
        (((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")), (0.0625 + 0j)),
    ]
    qop_terms = [term for term in qops[0].terms.items()] + [term for term in qops[1].terms.items()]  # type: ignore

    for idx, term in enumerate(terms):
        assert qop_terms[idx][0] == term[0]
        assert abs(qop_terms[idx][1] - term[1]) < 1e-16


def test_P_conjugated():
    a, b = np.random.randint(0, 20), np.random.randint(0, 20)
    fop1 = FermionOperator(str(a) + "^ " + str(a))
    fop1 -= hermitian_conjugated(fop1)
    fop2 = FermionOperator(str(b) + "^ " + str(b))
    fop2 -= hermitian_conjugated(fop2)

    assert Parity(a).encode_operator(fop1) == -Parity(b).encode_operator(fop2)
    assert Parity(a).encode_operator(fop1) == -Parity(b).encode_operator(fop2)


def test_P_properties():
    a = np.random.randint(0, 20)
    fops = [
        FermionOperator("0^"),
        FermionOperator("0"),
        FermionOperator(str(a) + "^ " + str(a) + "^ "),
        FermionOperator(str(a) + " " + str(a)),
        FermionOperator(str(a) + "^ " + str(a)),
    ]

    qopX1 = QubitOperator("X0 X0", 1.0 / 2)
    qopX2 = QubitOperator("X0 X0", 1.0)
    qopY = QubitOperator("Y0", 1j / 2)
    for orb in range(a + 1):
        qopX1 *= QubitOperator((orb, "X"))
        if orb > 0:
            qopX2 *= QubitOperator((orb, "X"))

        # qopY *= QubitOperator((orb, "Y"))

    if a != 0:
        qopZ = QubitOperator("X0 X0", 1.0 / 2) - QubitOperator(((a - 1, "Z"), (a, "Z")), 1.0 / 2)
    else:
        qopZ = QubitOperator("X0 X0", 1.0 / 2) - QubitOperator(((a, "Z")), 1.0 / 2)

    qop_terms = [
        qopX1 - qopY * qopX2,
        qopX1 + qopY * qopX2,
        QubitOperator((0, "X"), 0.0),
        QubitOperator((0, "X"), 0.0),
        qopZ,
    ]

    qops = Parity(a + 1).encode_operator(fops)
    qops = Parity(a + 1).encode_operator(fops)

    for i_op in range(len(fops)):
        assert qops[i_op] == qop_terms[i_op]


def test_encode_rejects_unrecognised_types():
    import pytest

    for mapping in (JordanWigner(), BravyiKitaev(), Parity(4)):
        with pytest.raises(RuntimeError, match="Unrecognised type"):
            mapping.encode_operator("not an operator")
