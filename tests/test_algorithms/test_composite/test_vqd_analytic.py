"""VQD pyscf-free: contracts, deflation arithmetic, and the analytic
spectrum of H = Z0 + 2·Z1 (eigenvalues {−3, −1, 1, 3}, product RY ansatz).

The penalty tests drive `objective` with a stubbed engine returning known
scalars, so the asserted values are hand-computed: ev + Σ wᵢ·|⟨ψᵢ|ψ⟩|² for
amplitude-returning primitives, ev + Σ wᵢ·pᵢ for the probability-returning
ones (SWAPTest, MirrorTest, TermwiseSWAPTest).
"""

import numpy as np
import pytest
from sympy import Symbol

from qarp.algorithms import VQD, MirrorTest, StateVector, SWAPTest, TermwiseSWAPTest
from qarp.blocks import SimpleBlock
from qarp.operators import QubitOperator
from qarp.optimizers import ScipyOptimizer


def _ham():
    return QubitOperator("Z0") + QubitOperator("Z1", 2.0)


def _ry_kets(n_states):
    kets = []
    for i in range(n_states):
        b = SimpleBlock(2)
        b.ry(0, Symbol(f"a{i}"))
        b.ry(1, Symbol(f"b{i}"))
        kets.append(b.build())
    return kets


def _exact_spectrum():
    # Independent oracle: numpy-built diagonal H = Z⊗I + 2·I⊗Z (spectrum is
    # ordering-invariant), never qarp's own matrix path.
    z = np.diag([1.0, -1.0])
    eye = np.eye(2)
    h = np.kron(eye, z) + 2.0 * np.kron(z, eye)
    return np.sort(np.linalg.eigvalsh(h))


def test_vqd_weights_length_mismatch_raises():
    with pytest.raises(RuntimeError, match="number of weights"):
        VQD(_ham(), _ry_kets(2), weights=[1.0, 2.0])


def test_vqd_defaults_zero_parameters_and_statevector_primitive():
    vqd = VQD(_ham(), _ry_kets(2), weights=[5.0])
    assert isinstance(vqd.primitive, StateVector)
    assert vqd.initial_parameters == [[0.0, 0.0], [0.0, 0.0]]


def test_vqd_optimal_parameters_before_run_raises():
    vqd = VQD(_ham(), _ry_kets(2), weights=[5.0])
    with pytest.raises(RuntimeError, match="call run"):
        vqd.optimal_parameters


def test_vqd_gradient_finds_two_lowest_eigenvalues():
    exact = _exact_spectrum()
    vqd = VQD(
        _ham(),
        _ry_kets(2),
        weights=[10.0],
        initial_parameters=[[0.3, 0.2], [0.2, 0.3]],
        gradient=True,
        verbose=False,
    )
    vqd.build()
    energies, state_parameters = vqd.run()

    assert abs(np.real(energies[0]) - exact[0]) < 1e-5  # −3
    assert abs(np.real(energies[1]) - exact[1]) < 1e-5  # −1
    # Optimized states: |11⟩ then |q0=0, q1=1⟩ ⇒ (a0, b0) → (π, π),
    # (a1, b1) → (0, π); check via the order-proof dicts.
    assert abs(abs(state_parameters[0][Symbol("a0")]) - np.pi) < 1e-3
    assert abs(abs(state_parameters[0][Symbol("b0")]) - np.pi) < 1e-3
    assert abs(state_parameters[1][Symbol("a1")]) < 1e-3
    assert abs(abs(state_parameters[1][Symbol("b1")]) - np.pi) < 1e-3


class _StubEngine:
    def __init__(self, results):
        self.results = results

    def run(self, params=None):
        return self.results


def test_vqd_objective_penalty_squares_amplitude_overlaps():
    vqd = VQD(_ham(), _ry_kets(2), weights=[0.5], primitive=StateVector())
    vqd.iter = 1
    vqd.engine = _StubEngine([(-2.0 + 0j), (0.6 + 0.8j)])
    # |0.6 + 0.8i|² = 1 ⇒ −2 + 0.5·1 = −1.5
    assert vqd.objective([0.1, 0.2]) == pytest.approx(-1.5)
    assert vqd.energies[1] == -2.0 + 0j


# Constructed inside the test — a qarpx-backed primitive in a parametrize list
# stays alive to interpreter shutdown and is reported as a leaked instance.
@pytest.mark.parametrize("primitive_name", ["SWAPTest", "MirrorTest", "TermwiseSWAPTest"])
def test_vqd_objective_probability_overlaps_pass_through(primitive_name):
    """Every probability-returning overlap primitive passes |⟨ψᵢ|ψ⟩|² through.

    Recognising only SWAPTest squared the penalty a second time for the other
    two, giving −2 + 0.5·0.36² = −1.9352 against the correct −1.82.
    """
    ket = _ry_kets(2)[0]
    primitive = {
        "SWAPTest": lambda: SWAPTest(),
        "MirrorTest": lambda: MirrorTest(),
        "TermwiseSWAPTest": lambda: TermwiseSWAPTest(bra=ket, ket=ket),
    }[primitive_name]()

    vqd = VQD(_ham(), _ry_kets(2), weights=[0.5], primitive=primitive)
    vqd.iter = 1
    vqd.engine = _StubEngine([(-2.0 + 0j), 0.36])
    # run() is already |⟨ψᵢ|ψ⟩|² ⇒ −2 + 0.5·0.36 = −1.82
    assert vqd.objective([0.1, 0.2]) == pytest.approx(-1.82)


def test_vqd_verbose_build_gradient_banner(capsys):
    vqd = VQD(_ham(), _ry_kets(2), weights=[1.0], gradient=True, verbose=True)
    vqd.build()
    assert "\tGradient: analytic (default) via " in capsys.readouterr().out


def test_vqd_verbose_build_and_run(capsys):
    vqd = VQD(
        _ham(),
        _ry_kets(2),
        weights=[1.0],
        initial_parameters=[[0.3, 0.2], [0.2, 0.3]],
        verbose=True,
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 5}),
    )
    vqd.build()
    vqd.run()
    out = capsys.readouterr().out

    assert "VQD Build:" in out
    assert "\tGradient: No analytic gradients." in out
    assert "VQD Run:" in out
    # One iteration table per deflation macroiteration, with the Cost column
    # carrying the accumulated overlap penalty.
    assert "Macroiteration 0 Run:" in out
    assert "Macroiteration 1 Run:" in out
    assert "\t\t\t\tCost" in out


@pytest.mark.parametrize("imaginary", [False, True])
def test_vqd_gradient_with_hadamard_test_overlap_matches_finite_differences(imaginary):
    """A HadamardTest overlap returns the amplitude (its real part, or both
    parts) and the engine differentiates exactly that, while VQD's penalty is
    the squared modulus.  Without the chain rule in ``objective_gradient`` the
    gradient was −0.840656 against −1.024940 (real) / −1.043731 (complex)
    from central FD of VQD's own objective."""
    import qarpx as qx
    from qarp import EXACT
    from qarp.algorithms import HadamardTest

    zb = SimpleBlock(1, name="Zb")
    zb.z(0)
    zb.build()
    k1 = SimpleBlock(1)
    k1.ry(0, qx.Param.symbol("a"))
    k1.build()
    k2 = SimpleBlock(1)
    k2.ry(0, qx.Param.symbol("b"))
    k2.rz(0, 0.3)
    k2.build()
    vqd = VQD(
        zb,
        [k1, k2],
        weights=[2.0],
        gradient=True,
        primitive=HadamardTest(n_shots=EXACT, real=True, imaginary=imaginary),
        initial_parameters=[[0.3], [0.7]],
    )
    vqd.build()
    vqd.state_parameters = [{k1.symbols[0]: 0.3}]
    vqd.iter = 1
    vqd._build_iteration()
    x = np.array([0.7])
    h = 1e-6
    g = vqd.objective_gradient(x)
    fd = (vqd.objective(x + h) - vqd.objective(x - h)) / (2 * h)
    assert abs(fd) > 1e-3
    assert g[0] == pytest.approx(fd, abs=1e-6)


def test_vqd_rejects_a_wrong_length_initial_parameters_list():
    """One entry per ket.  Too few used to truncate, leaving later kets
    unseeded and surfacing as an IndexError inside ``iterate()``."""
    kets = _ry_kets(2)
    with pytest.raises(ValueError, match="1 entries for 2 kets"):
        VQD(_ham(), kets, weights=[1.0], initial_parameters=[[0.3, 0.2]])
    with pytest.raises(ValueError, match="3 entries for 2 kets"):
        VQD(_ham(), kets, weights=[1.0], initial_parameters=[[0.3, 0.2], [0.7, 0.1], [0.1, 0.4]])

    vqd = VQD(_ham(), kets, weights=[1.0], initial_parameters=[[0.3, 0.2], [0.7, 0.1]])
    assert len(vqd.initial_parameters) == 2


@pytest.mark.parametrize(
    ("primitive_name", "expected"),
    [
        ("StateVector_overlap", True),
        ("StateVector_expectation", False),
        ("SWAPTest", True),
        ("MirrorTest", True),
        ("TermwiseSWAPTest", True),
        ("HadamardTest", False),
    ],
)
def test_gradient_is_squared_overlap_covers_both_routes(primitive_name, expected):
    """The predicate behind the deflation chain rule, pinned per primitive.

    Two disjoint routes reach ``∂|o|²``: a StateVector OVERLAP declares
    ``gradient_kind == "squared_overlap"`` (run() gives the amplitude, the
    engine differentiates the modulus squared), while SWAPTest/MirrorTest/
    TermwiseSWAPTest declare ``returns_probability`` (run() is already |o|²).
    Missing either route is a penalty that is |o|⁴ or misses the chain rule.
    """
    from qarp.algorithms import HadamardTest
    from qarp.algorithms._composite.vqd import _gradient_is_squared_overlap

    k0, k1 = _ry_kets(2)

    def _state_vector_overlap():
        prim = StateVector(bra=k0, ket=k1)
        prim.build()
        return prim

    def _state_vector_expectation():
        prim = StateVector(ket=k0, operator=_ham())
        prim.build()
        return prim

    primitive = {
        "StateVector_overlap": _state_vector_overlap,
        "StateVector_expectation": _state_vector_expectation,
        "SWAPTest": SWAPTest,
        "MirrorTest": MirrorTest,
        "TermwiseSWAPTest": lambda: TermwiseSWAPTest(bra=k0, ket=k1),
        "HadamardTest": HadamardTest,
    }[primitive_name]()

    assert _gradient_is_squared_overlap(primitive) is expected


@pytest.mark.parametrize(
    ("primitive_name", "value"),
    [("StateVector", 0.6 - 0.5j), ("SWAPTest", 0.61)],
)
def test_squared_overlap_is_the_squared_modulus_not_the_real_part(primitive_name, value):
    """The shared deflation term used by both VQD and ADAPT-VQD.

    For o = 0.6 − 0.5i: |o|² = 0.36 + 0.25 = 0.61, while Re(o²) = 0.36 − 0.25
    = 0.11 — the value ADAPT-VQD's objective used to add.  A probability-returning
    primitive hands 0.61 over directly and must not be squared again.
    """
    from qarp.algorithms._composite.vqd import squared_overlap

    primitive = {"StateVector": StateVector, "SWAPTest": SWAPTest}[primitive_name]()
    assert squared_overlap(primitive, value) == pytest.approx(0.61)
    assert np.real(np.asarray(value) ** 2) != pytest.approx(0.61)


@pytest.mark.parametrize("primitive_name", ["SWAPTest", "MirrorTest"])
def test_vqd_gradient_with_a_probability_overlap_matches_finite_differences(primitive_name):
    """The other half of the routing claim, checked numerically.

    ``_gradient_is_squared_overlap`` asserts that a probability-returning
    primitive's ``run_gradient`` is already ``∂|⟨bra|ket⟩|²``, so
    ``squared_overlap_gradient`` passes it through untouched.  The predicate
    table pins what we believe; central FD of VQD's own objective is what shows
    the belief is true.  Applying the amplitude chain rule here instead would
    scale the penalty term by 2·|o|² and miss the FD by ~0.4.
    """
    import qarpx as qx
    from qarp import EXACT

    zb = SimpleBlock(1, name="Zb")
    zb.z(0)
    zb.build()
    k1 = SimpleBlock(1)
    k1.ry(0, qx.Param.symbol("a"))
    k1.build()
    k2 = SimpleBlock(1)
    k2.ry(0, qx.Param.symbol("b"))
    k2.rz(0, 0.3)
    k2.build()

    primitive = {"SWAPTest": SWAPTest, "MirrorTest": MirrorTest}[primitive_name](n_shots=EXACT)
    vqd = VQD(
        zb,
        [k1, k2],
        weights=[2.0],
        gradient=True,
        primitive=primitive,
        initial_parameters=[[0.3], [0.7]],
    )
    vqd.build()
    vqd.state_parameters = [{k1.symbols[0]: 0.3}]
    vqd.iter = 1
    vqd._build_iteration()

    x = np.array([0.7])
    h = 1e-6
    g = vqd.objective_gradient(x)
    fd = (vqd.objective(x + h) - vqd.objective(x - h)) / (2 * h)
    assert abs(fd) > 1e-3
    assert g[0] == pytest.approx(fd, abs=1e-6)


# Every OVERLAP-capable primitive, with the value its run() returns.  A new one
# defaults to returns_probability = False and would be squared a second time by
# the deflation penalty, which is the bug this table exists to make loud.
_OVERLAP_RETURNS_PROBABILITY = {
    "SWAPTest": True,
    "MirrorTest": True,
    "TermwiseSWAPTest": True,
    "HadamardTest": False,  # amplitude
    "TermwiseHadamardTest": False,  # amplitude (one operator-less HadamardTest child)
    "StateVector": False,  # amplitude; the engine differentiates |o|² separately
}


def test_every_overlap_primitive_declares_whether_it_returns_a_probability():
    """Completeness guard, in the style of the block symbols contract.

    ``returns_probability`` is a class flag with a permissive default, so an
    overlap primitive that forgets it is not a failure anywhere — it is a
    silently quartic deflation penalty.  Add the new class to the table above
    with the value its ``run()`` actually returns.
    """
    import qarp.algorithms as algorithms
    from qarp.algorithms import PrimitiveAlgorithm
    from qarp.algorithms._primitives.target import Target

    found = {
        name: obj.returns_probability
        for name in dir(algorithms)
        if isinstance(obj := getattr(algorithms, name), type)
        and issubclass(obj, PrimitiveAlgorithm)
        and Target.OVERLAP in obj.supported_targets
    }
    assert found.keys() == _OVERLAP_RETURNS_PROBABILITY.keys(), (
        "overlap primitives changed; update _OVERLAP_RETURNS_PROBABILITY with "
        f"the value each run() returns. Missing: {found.keys() - _OVERLAP_RETURNS_PROBABILITY.keys()}, "
        f"stale: {_OVERLAP_RETURNS_PROBABILITY.keys() - found.keys()}"
    )
    assert found == _OVERLAP_RETURNS_PROBABILITY


def test_vqd_gradient_with_a_termwise_hadamard_overlap_matches_finite_differences():
    """``TermwiseHadamardTest`` as VQD's primitive: the overlap copies get no
    operator, which the primitive now reads as the bare ``<bra|ket>`` (identity
    default).  Same 1-qubit problem as the ``HadamardTest`` test above, so the
    value is a second oracle: both amplitude-returning primitives must give the
    same objective gradient, −1.043731 in the complex case."""
    import qarpx as qx
    from qarp import EXACT
    from qarp.algorithms import TermwiseHadamardTest

    k1 = SimpleBlock(1)
    k1.ry(0, qx.Param.symbol("a"))
    k1.build()
    k2 = SimpleBlock(1)
    k2.ry(0, qx.Param.symbol("b"))
    k2.rz(0, 0.3)
    k2.build()
    vqd = VQD(
        QubitOperator("Z0"),
        [k1, k2],
        weights=[2.0],
        gradient=True,
        primitive=TermwiseHadamardTest(n_shots=EXACT, real=True, imaginary=True),
        initial_parameters=[[0.3], [0.7]],
    )
    vqd.build()
    vqd.state_parameters = [{k1.symbols[0]: 0.3}]
    vqd.iter = 1
    vqd._build_iteration()

    x = np.array([0.7])
    h = 1e-6
    g = vqd.objective_gradient(x)
    fd = (vqd.objective(x + h) - vqd.objective(x - h)) / (2 * h)
    assert g[0] == pytest.approx(fd, abs=1e-6)
    assert g[0] == pytest.approx(-1.043731, abs=1e-6)
