import os

import networkx as nx
import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from qarp import config
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    MappedONVStateBlock,
    UCCBlock,
    UPCCDBlock,
)
from qarp.engines import Engine
from qarp.operators import FermionOperator, JordanWigner
from tests.operator_test_utils import eigenspectrum

config.seed = 1234


class _StubEngine(Engine):
    """Base-default engine: opts into nothing.  ``_sweep`` is the only
    abstractmethod; ``_compile_one``/``_dispatch_one`` have base bodies and
    ``_validate_primitive`` runs before either, so every base-contract
    rejection is reachable through ``build()``."""

    def __init__(self) -> None:
        self._seed = None
        self._primitives = []
        self._l2p_per_primitive = []

    def _sweep(self, *args, **kwargs):
        raise NotImplementedError


class CountRzModeler:
    """Test-local ``ResourceModeler``: one T per Rz.  Exercises the seam
    (``estimate(modeler=)``, ``with_model``, ``provenance.modeler``), not a
    cost model.  A ``Custom`` nulls the count as §19 requires."""

    name = "count_rz:t=1"

    def model(self, commands, vector):
        import qarpx as qx

        n_rz = sum(1 for c in commands if c.gate == qx.GateType.Rz)
        n_opaque = sum(1 for c in commands if c.gate == qx.GateType.Custom)
        extras = {"n_rz": float(n_rz)}
        if n_opaque:
            extras["n_custom_opaque"] = float(n_opaque)
            return vector.with_model(modeler=self.name, t_count_modeled=None, extras=extras)
        return vector.with_model(modeler=self.name, t_count_modeled=float(n_rz), extras=extras)


# derandomize: identical inputs every run, so a failure is always reproducible
# and never lands as an intermittent red.  deadline=None: statevector
# simulation routinely exceeds Hypothesis' 200 ms per-example default, which
# it would otherwise report as a failure.
settings.register_profile(
    "gate",
    max_examples=25,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=500,
    derandomize=False,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "gate"))


def pytest_addoption(parser):
    parser.addoption(
        "--leakdiag",
        action="store_true",
        help="report qarpx objects still alive at session end (see tests/leakdiag.py)",
    )


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--leakdiag"):
        from tests.leakdiag import report

        report(printer=lambda s: print(s))  # noqa: T201 — diagnostic output


@pytest.fixture
def h2_ev():
    terms = [
        ((), 0.7199689944489797),
        (((0, 1), (0, 0)), -1.25633907300325),
        (((2, 1), (2, 0)), -0.47189600728114184),
        (((1, 1), (1, 0)), -1.25633907300325),
        (((3, 1), (3, 0)), -0.47189600728114184),
        (((2, 1), (0, 1), (0, 0), (2, 0)), 0.4836505304710652),
        (((3, 1), (1, 1), (1, 0), (3, 0)), 0.4836505304710652),
        (((1, 1), (0, 1), (0, 0), (1, 0)), 0.6757101548035163),
        (((2, 1), (1, 1), (1, 0), (2, 0)), 0.6645817302552967),
        (((1, 1), (0, 1), (2, 0), (3, 0)), 0.18093119978423133),
        (((2, 1), (1, 1), (0, 0), (3, 0)), -0.1809311997842314),
        (((3, 1), (0, 1), (1, 0), (2, 0)), -0.1809311997842314),
        (((3, 1), (2, 1), (0, 0), (1, 0)), 0.18093119978423136),
        (((3, 1), (0, 1), (0, 0), (3, 0)), 0.6645817302552967),
        (((3, 1), (2, 1), (2, 0), (3, 0)), 0.6985737227320175),
    ]
    fham = FermionOperator()
    for term, coeff in terms:
        fham += FermionOperator(term, coeff)
    qop = JordanWigner().encode_operator(fham)
    onv = [1, 1, 0, 0]
    ref = MappedONVStateBlock(occupation_number_vector=onv, mapping=JordanWigner())
    ucc = UCCBlock(occupation_number_vector=onv, mapping=JordanWigner(), singles=True, doubles=True)
    wfn = CompositeBlock([ref, ucc], 4)
    return (wfn, qop)


@pytest.fixture
def h2_ev_upccd():
    terms = [
        ((), 0.7199689944489797),
        (((0, 1), (0, 0)), -1.25633907300325),
        (((2, 1), (2, 0)), -0.47189600728114184),
        (((1, 1), (1, 0)), -1.25633907300325),
        (((3, 1), (3, 0)), -0.47189600728114184),
        (((2, 1), (0, 1), (0, 0), (2, 0)), 0.4836505304710652),
        (((3, 1), (1, 1), (1, 0), (3, 0)), 0.4836505304710652),
        (((1, 1), (0, 1), (0, 0), (1, 0)), 0.6757101548035163),
        (((2, 1), (1, 1), (1, 0), (2, 0)), 0.6645817302552967),
        (((1, 1), (0, 1), (2, 0), (3, 0)), 0.18093119978423133),
        (((2, 1), (1, 1), (0, 0), (3, 0)), -0.1809311997842314),
        (((3, 1), (0, 1), (1, 0), (2, 0)), -0.1809311997842314),
        (((3, 1), (2, 1), (0, 0), (1, 0)), 0.18093119978423136),
        (((3, 1), (0, 1), (0, 0), (3, 0)), 0.6645817302552967),
        (((3, 1), (2, 1), (2, 0), (3, 0)), 0.6985737227320175),
    ]
    fham = FermionOperator()
    for term, coeff in terms:
        fham += FermionOperator(term, coeff)
    qop = JordanWigner().encode_operator(fham)
    onv = [1, 1, 0, 0]
    upcc = UPCCDBlock(basis_state=onv).build()
    ref = ComputationalBasisStateBlock([1, 0, 1, 0])
    symbolic_ansatz = CompositeBlock([ref, upcc]).build()
    ansatz = symbolic_ansatz.set_symbols({upcc.symbols[0]: -0.11278283})
    return (ansatz, qop)


@pytest.fixture()
def scaled_h2_hamiltonian_jw(grid_shift=0.0):
    terms = [
        ((), 0.7199689944489797),
        (((0, 1), (0, 0)), -1.25633907300325),
        (((2, 1), (2, 0)), -0.47189600728114184),
        (((1, 1), (1, 0)), -1.25633907300325),
        (((3, 1), (3, 0)), -0.47189600728114184),
        (((2, 1), (0, 1), (0, 0), (2, 0)), 0.4836505304710652),
        (((3, 1), (1, 1), (1, 0), (3, 0)), 0.4836505304710652),
        (((1, 1), (0, 1), (0, 0), (1, 0)), 0.6757101548035163),
        (((2, 1), (1, 1), (1, 0), (2, 0)), 0.6645817302552967),
        (((1, 1), (0, 1), (2, 0), (3, 0)), 0.18093119978423133),
        (((2, 1), (1, 1), (0, 0), (3, 0)), -0.1809311997842314),
        (((3, 1), (0, 1), (1, 0), (2, 0)), -0.1809311997842314),
        (((3, 1), (2, 1), (0, 0), (1, 0)), 0.18093119978423136),
        (((3, 1), (0, 1), (0, 0), (3, 0)), 0.6645817302552967),
        (((3, 1), (2, 1), (2, 0), (3, 0)), 0.6985737227320175),
    ]
    fham = FermionOperator()
    for term, coeff in terms:
        fham += FermionOperator(term, coeff)
    ham = JordanWigner().encode_operator(fham)
    eigs = eigenspectrum(ham, 4)
    ham = (ham - np.min(eigs)) / (np.max(eigs) - np.min(eigs)) * (1 - grid_shift)
    return ham, eigenspectrum(ham, 4)


@pytest.fixture
def h2_uccsd_ovlp(h2_ev):
    onv = [1, 1, 0, 0]
    ref = MappedONVStateBlock(occupation_number_vector=onv, mapping=JordanWigner())
    ucc = UCCBlock(occupation_number_vector=onv, mapping=JordanWigner(), singles=True, doubles=True)
    bra = CompositeBlock([ref, ucc], 4).build()
    ket = bra.refresh_symbols("_1")
    return (bra, ket)


def qft_unitary_matrix(nqubits):
    """A function for building the unitary matrix associated to the QFT

    Args:
        nqubits: number of qubits in the circuit
    """
    Nq = 2**nqubits
    qft_mat = np.ones((Nq, Nq), dtype=complex)
    w = np.exp(2.0 * np.pi * 1j / Nq)
    c = 1
    for k in range(1, Nq):
        power_seq = [np.mod(c * j, Nq) for j in range(1, Nq)]
        qft_mat[1:, k] = [w**p for p in power_seq]
        c += 1
    qft_mat /= np.sqrt(Nq)
    return qft_mat


def generate_toy_graph():
    edges = [(0, 1, 2.0), (2, 3, 2.0)]
    solutions = [
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
    ]

    graph = nx.Graph()
    for u, v, coeff in edges:
        graph.add_edge(u, v, weight=coeff)

    assert graph.number_of_edges() == 2
    assert graph.number_of_nodes() == 4

    return graph, solutions
