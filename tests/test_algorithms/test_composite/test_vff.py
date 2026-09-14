import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")
from qarp.operators.compat import from_openfermion


class hamiltonians:
    """Adapter: openfermion's reference fermi_hubbard builder, converted to
    the qarpx FermionOperator at the boundary."""

    @staticmethod
    def fermi_hubbard(*args, **kwargs):
        return from_openfermion(openfermion.hamiltonians.fermi_hubbard(*args, **kwargs))


from scipy.optimize import OptimizeResult

from qarp.algorithms import VFF
from qarp.blocks import (
    ComputationalBasisStateBlock,
    HEABlock,
    HnBlock,
    SPABlock,
)
from qarp.operators import JordanWigner
from qarp.optimizers import AdamOptimizer


def test_unitary_vff():

    U = HnBlock(n_qubits=2)
    hea = HEABlock(
        n_qubits=2, n_layers=2, real=True, linear=False, circular=True, use_cz=True
    ).build()
    opt = AdamOptimizer({"maxiter": 10, "lr": 0.1})
    vff = VFF(U, hea, optimizer=opt).build()
    res = vff.run()

    assert type(res) is OptimizeResult

    # Order-proof result surface: ansatz symbols first, then the diagonal
    # layer's (documented concatenation — symbols-ordering contract).
    opt_params = vff.optimal_parameters
    assert set(opt_params) == set(hea.symbols) | set(vff.D.symbols)
    assert list(opt_params.values()) == list(res.x)


def test_hermitian_exact_vff():

    HH = hamiltonians.fermi_hubbard(
        1,
        1,
        -0.9,
        0.05,
        chemical_potential=0.1,
        magnetic_field=0.0,
        periodic=True,
        spinless=True,
        particle_hole_symmetry=False,
    )
    qop = JordanWigner().encode_operator(HH)
    hea = SPABlock(n_qubits=2, n_layers=2, real=False, linear=False, circular=True).build()
    opt = AdamOptimizer({"maxiter": 10, "lr": 0.1})
    vff = VFF(qop, hea, t_time=1, optimizer=opt, use_trotter=False).build()
    res = vff.run()

    assert type(res) is OptimizeResult


def test_hermitian_pauli_vff():

    HH = hamiltonians.fermi_hubbard(
        1,
        1,
        -0.9,
        0.05,
        chemical_potential=0.1,
        magnetic_field=0.0,
        periodic=True,
        spinless=True,
        particle_hole_symmetry=False,
    )
    qop = JordanWigner().encode_operator(HH)
    hea = SPABlock(n_qubits=2, n_layers=2, real=False, linear=False, circular=True).build()
    opt = AdamOptimizer({"maxiter": 10, "lr": 0.1})
    vff = VFF(qop, hea, t_time=1, optimizer=opt, use_trotter=True).build()
    res = vff.run()

    assert type(res) is OptimizeResult


def test_vff_parameterless_ansatz_rejected():
    """A parameterless ansatz must fail loudly at construction — on both
    the default and the explicit initial_parameters path."""
    U = HnBlock(n_qubits=2)
    with pytest.raises(ValueError, match="no parameters"):
        VFF(U, ComputationalBasisStateBlock([0, 0]))
    with pytest.raises(ValueError, match="no parameters"):
        VFF(U, ComputationalBasisStateBlock([0, 0]), initial_parameters=np.zeros(2))


def test_vff_rejects_a_wrong_length_initial_parameters_vector():
    """The positional surface is ansatz symbols + one Rz per qubit; a short
    vector used to truncate the parameter map silently instead of failing."""
    U = HnBlock(n_qubits=2)
    hea = HEABlock(
        n_qubits=2, n_layers=2, real=True, linear=False, circular=True, use_cz=True
    ).build()
    n_parameters = len(hea.symbols) + 2

    with pytest.raises(ValueError, match=f"{n_parameters} symbols"):
        VFF(U, hea, initial_parameters=np.zeros(n_parameters - 1))
    with pytest.raises(ValueError, match=f"{n_parameters} symbols"):
        VFF(U, hea, initial_parameters=np.zeros(n_parameters + 1))

    # The exact length is accepted and reaches the optimizer unchanged.
    vff = VFF(U, hea, initial_parameters=np.zeros(n_parameters))
    assert vff.initial_parameters.shape == (n_parameters,)


# ── P1.6: verbose adds no quantum work; wrong-length vectors fail loudly ────


def _engine_run_count(verbose):
    U = HnBlock(n_qubits=2)
    hea = HEABlock(
        n_qubits=2, n_layers=1, real=True, linear=False, circular=True, use_cz=True
    ).build()
    opt = AdamOptimizer({"maxiter": 5, "lr": 0.1})
    vff = VFF(U, hea, optimizer=opt, verbose=verbose).build()
    calls = {"n": 0}
    real_run = vff.engine.run

    def counting(*a, **k):
        calls["n"] += 1
        return real_run(*a, **k)

    vff.engine.run = counting
    vff.initial_parameters = np.zeros(len(vff._all_symbols))
    vff.run()
    return calls["n"]


def test_verbose_callback_does_not_rerun_the_engine(capsys):
    """VFF's callback reads the cached objective.  The only extra evaluations
    under verbose are the optimizer's own — GradientDescentOptimizer
    evaluates once per iteration whenever a callback is supplied (its scipy
    contract) — so the difference is exactly maxiter, not 2x."""
    quiet = _engine_run_count(verbose=False)
    loud = _engine_run_count(verbose=True)
    assert loud == quiet + 5
    assert "Loss value" in capsys.readouterr().out


def test_verbose_callback_survives_firing_before_the_first_evaluation(capsys):
    """The cache the callback reads is written by ``objective`` and initialised
    to None in ``CompositeAlgorithm.__init__``; an optimizer that reports before
    evaluating must print None, not raise AttributeError."""
    U = HnBlock(n_qubits=2)
    hea = HEABlock(
        n_qubits=2, n_layers=1, real=True, linear=False, circular=True, use_cz=True
    ).build()

    class CallbackFirst(AdamOptimizer):
        def minimize(self, fun, x0, callback=None, **kwargs):
            callback(OptimizeResult(x=x0))
            return super().minimize(fun, x0, callback=callback, **kwargs)

    vff = VFF(U, hea, optimizer=CallbackFirst({"maxiter": 1, "lr": 0.1}), verbose=True).build()
    vff.initial_parameters = np.zeros(len(vff._all_symbols))
    vff.run()
    assert "Loss value: None" in capsys.readouterr().out


def test_wrong_length_parameter_vector_raises():
    U = HnBlock(n_qubits=2)
    hea = HEABlock(
        n_qubits=2, n_layers=1, real=True, linear=False, circular=True, use_cz=True
    ).build()
    vff = VFF(U, hea, optimizer=AdamOptimizer({"maxiter": 2, "lr": 0.1})).build()
    with pytest.raises(ValueError):
        vff.objective(np.zeros(len(vff._all_symbols) - 1))
