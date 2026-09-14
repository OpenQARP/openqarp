"""VQA base class: defaults, contracts, MPI parameter seeding, and the
analytic single-qubit minimization ⟨Z⟩ = cos θ (min −1 at θ = π under the
exp(−iθP/2) convention).
"""

import numpy as np
import pytest
from sympy import Symbol

from qarp.algorithms import VQA, Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.operators import QubitOperator
from qarp.optimizers import ScipyOptimizer


def _ry_ket():
    b = SimpleBlock(1)
    b.ry(0, Symbol("a"))
    return b


def test_defaults_gradient_off():
    vqa = VQA(QubitOperator("Z0"), _ry_ket(), "t")
    assert isinstance(vqa.primitive, StateVector)
    assert vqa.optimizer.method == "COBYLA"


def test_defaults_gradient_on():
    vqa = VQA(QubitOperator("Z0"), _ry_ket(), "t", gradient=True)
    assert vqa.optimizer.method == "CG"


def test_gradient_with_sampling_primitive_uses_parameter_shift():
    """Formerly refused at construction ("backprop-gradient"); the gradient
    registry resolves a COUNTS primitive to the batched parameter shift."""
    from qarp import EXACT
    from qarp.algorithms import PauliAveraging

    vqa = VQA(
        QubitOperator("Z0"),
        _ry_ket(),
        "t",
        gradient=True,
        initial_parameters=[0.7],
        primitive=PauliAveraging(n_shots=EXACT),
    )
    assert vqa.gradient_method == "default"
    vqa.build()
    assert vqa.engine.resolve_gradient_method(vqa.primitive, "default") == "parameter-shift"
    (g,) = vqa.engine.run_gradient(vqa.ket.parameter_map([0.7]))
    assert g[0] == pytest.approx(-np.sin(0.7), abs=1e-8)  # ⟨Z⟩ = cos t


def test_gradient_method_string_is_validated_at_construction():
    with pytest.raises(ValueError, match="unknown gradient method"):
        VQA(QubitOperator("Z0"), _ry_ket(), "t", gradient="parameter_shift")
    vqa = VQA(QubitOperator("Z0"), _ry_ket(), "t", gradient="finite-diff")
    assert vqa.gradient is True and vqa.gradient_method == "finite-diff"
    assert vqa.optimizer.method == "CG"


def test_gradient_with_sampler_is_refused_at_gradient_time():
    from qarp.errors import CapabilityError

    vqa = VQA(QubitOperator("Z0"), _ry_ket(), "t", gradient=True, primitive=Sampler(n_shots=100))
    vqa.build()
    with pytest.raises(CapabilityError, match="gradient_kind='none'"):
        vqa.engine.run_gradient(vqa.ket.parameter_map([0.3]))


def test_access_before_run_raises():
    vqa = VQA(QubitOperator("Z0"), _ry_ket(), "t", initial_parameters=[0.1])
    with pytest.raises(RuntimeError, match="call run"):
        vqa.optimal_parameters
    with pytest.raises(RuntimeError, match="call run"):
        vqa.get_final_state_block()


def test_vqa_minimizes_pauli_z():
    vqa = VQA(
        QubitOperator("Z0"),
        _ry_ket(),
        "tiny",
        initial_parameters=[0.5],
        optimizer=ScipyOptimizer("COBYLA"),
    )
    vqa.build()
    energy, x = vqa.run()
    assert abs(energy - (-1.0)) < 1e-6
    assert abs(x[0] - np.pi) < 1e-3
    # Order-proof surface agrees with the raw vector.
    assert vqa.optimal_parameters == {Symbol("a"): x[0]}


def test_vqa_verbose_build_no_gradient_banner(capsys):
    vqa = VQA(QubitOperator("Z0"), _ry_ket(), "tiny", initial_parameters=[0.5], verbose=True)
    vqa.build()
    assert "\tGradient: No analytic gradients." in capsys.readouterr().out


def test_vqa_verbose_build_and_run(capsys):
    vqa = VQA(
        QubitOperator("Z0"),
        _ry_ket(),
        "tiny",
        initial_parameters=[0.5],
        gradient=True,
        verbose=True,
        optimizer=ScipyOptimizer("CG", options={"maxiter": 3}),
    )
    vqa.build()
    energy, _ = vqa.run()
    out = capsys.readouterr().out

    assert "tiny Build:" in out
    assert "\tGradient: analytic (default) via " in out
    # Iteration table: header with the |grad| column, then rows.
    assert "tiny Run:" in out
    assert "\t\tIteration\t\tEnergy\t\t\t\t  dE\t\t\t\t|step|\t\t\t\t|grad|" in out
    # verbose controls printing only — history needs save_energy_history.
    assert vqa.energy_history == []
    assert energy is not None


def test_energy_history_default_off():
    vqa = VQA(
        QubitOperator("Z0"),
        _ry_ket(),
        "tiny",
        initial_parameters=[0.5],
        optimizer=ScipyOptimizer("COBYLA"),
    )
    vqa.build()
    vqa.run()
    assert vqa.energy_history == []


def test_save_energy_history_independent_of_verbose(capsys):
    vqa = VQA(
        QubitOperator("Z0"),
        _ry_ket(),
        "tiny",
        initial_parameters=[0.5],
        optimizer=ScipyOptimizer("COBYLA"),
        save_energy_history=True,
        verbose=False,
    )
    vqa.suppress_success_message = True
    vqa.build()
    energy, _ = vqa.run()

    assert len(vqa.energy_history) >= 1
    assert abs(vqa.energy_history[-1] - energy) < 1e-6
    # Quiet run stays quiet: no iteration table on stdout.
    assert "Run:" not in capsys.readouterr().out


def test_vqe_threads_save_energy_history():
    from qarp.algorithms import VQE

    vqe = VQE(
        QubitOperator("Z0"),
        _ry_ket(),
        initial_parameters=[0.5],
        optimizer=ScipyOptimizer("COBYLA"),
        save_energy_history=True,
    )
    vqe.suppress_success_message = True
    vqe.build()
    energy, _ = vqe.run()
    assert len(vqe.energy_history) >= 1
    assert abs(vqe.energy_history[-1] - energy) < 1e-6
