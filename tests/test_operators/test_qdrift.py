import random

import numpy as np

from qarp.operators import QubitOperator, qDRIFT


def op_A_is_in_op_B(op_a: QubitOperator, op_b: QubitOperator, tol=1e-8) -> bool:
    """
    Check if all terms in op_A are present in op_B with matching coefficients.
    """
    for term, coeff_a in op_a.terms.items():
        coeff_b = op_b.terms.get(term)
        if coeff_b is None or abs(coeff_a - coeff_b) > tol:
            return False
    return True


def test_qubit_operator():

    op = QubitOperator("X0 Y1", 0.5) + QubitOperator("Z0 Z1", 1.0) + QubitOperator("Y0 X1", 0.3)
    qdrift_ham = qDRIFT(op, samples=1).qdrift()

    assert isinstance(qdrift_ham, QubitOperator)


def test_qdrift():

    op = (
        QubitOperator("X0 Y1", 0.5)
        + QubitOperator("Z0 Z1", 1.0)
        + QubitOperator("Y0 X1", 0.3)
        + QubitOperator("X0 X1", 0.2)
    )
    qdrift_ham = qDRIFT(op, samples=6).qdrift()

    assert op_A_is_in_op_B(qdrift_ham, op)


def test_partially_randomized_no_sampling():

    op = (
        QubitOperator("X0 Y1", 0.5)
        + QubitOperator("Z0 Z1", 1.0)
        + QubitOperator("Y0 X1", 0.3)
        + QubitOperator("X0 X1", 0.2)
    )
    qdrift_ham_pr = qDRIFT(
        op, samples=0, ratio=0.5
    ).partially_randomized()  # ratio of 0.5 and no samples, thus we always take 2 largest terms

    qdrift_ham_pr_bm = QubitOperator("X0 Y1", 0.5) + QubitOperator("Z0 Z1", 1.0)

    assert qdrift_ham_pr == qdrift_ham_pr_bm


def test_partially_randomized_with_sampling():

    op = (
        QubitOperator("X0 Y1", 0.5)
        + QubitOperator("Z0 Z1", 1.0)
        + QubitOperator("Y0 X1", 0.3)
        + QubitOperator("X0 X1", 0.2)
    )
    qdrift_ham_pr = qDRIFT(op, samples=8, ratio=0.5).partially_randomized()

    assert op_A_is_in_op_B(qdrift_ham_pr, op)


def test_qdrift_seed_reproduces_fully_randomized_operator():
    op = QubitOperator("X0", 0.5) + QubitOperator("Y0", 0.3) + QubitOperator("Z0", 0.2)

    first = qDRIFT(op, samples=4, seed=17).qdrift()
    second = qDRIFT(op, samples=4, seed=17).qdrift()

    assert first == second


def test_qdrift_seed_reproduces_partially_randomized_operator():
    op = (
        QubitOperator("X0", 0.4)
        + QubitOperator("Y0", 0.3)
        + QubitOperator("Z0", 0.2)
        + QubitOperator("X1", 0.1)
    )

    first = qDRIFT(op, samples=3, ratio=0.5, seed=23).partially_randomized()
    second = qDRIFT(op, samples=3, ratio=0.5, seed=23).partially_randomized()

    assert first == second


def test_qdrift_does_not_mutate_process_wide_random_state():
    op = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.5)
    random.seed(101)
    np.random.seed(202)
    expected_python = random.random()
    expected_numpy = np.random.random()

    random.seed(101)
    np.random.seed(202)
    qDRIFT(op, samples=5, seed=303).qdrift()

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


# =============================================================================
# Contracts, verbose paths, exact-recovery invariants
# =============================================================================
import pytest

from qarp.operators import qDRIFT as _qDRIFT


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"H": "Z0", "samples": 1}, TypeError, "must be a QubitOperator"),
        ({"samples": -1}, ValueError, "positive integer"),
        ({"samples": 1.5}, ValueError, "positive integer"),
        ({"samples": 1, "ratio": 2.0}, ValueError, "between 0 and 1"),
        ({"samples": 1, "ratio": 1}, ValueError, "between 0 and 1"),  # int rejected
    ],
)
def test_constructor_contracts(kwargs, exc, match):
    # Fresh dict: setdefault would mutate the parametrize-retained dict and
    # pin a qarpx object for the whole session (nanobind leak report).
    kwargs = {"H": QubitOperator("Z0"), **kwargs}
    with pytest.raises(exc, match=match):
        _qDRIFT(**kwargs)


def test_zero_norm_hamiltonian_raises():
    with pytest.raises(ValueError, match="zero norm"):
        _qDRIFT(QubitOperator(), samples=1).qdrift()


def test_qdrift_zero_samples_raises():
    with pytest.raises(ValueError, match="at least 1 sample"):
        _qDRIFT(QubitOperator("Z0"), samples=0).qdrift()


def test_partially_randomized_requires_ratio():
    with pytest.raises(ValueError, match="ratio parameter required"):
        _qDRIFT(QubitOperator("Z0"), samples=1).partially_randomized()


def test_sampled_terms_keep_original_coefficients():
    # lambda * (c/lambda) = c: any kept term must carry its exact input
    # coefficient; with enough samples on 2 terms both are kept, so the
    # output IS the input.
    op = QubitOperator("X0", 0.5) + QubitOperator("Z0 Z1", 1.5)
    out = _qDRIFT(op, samples=200, seed=7).qdrift()
    assert out == op


def test_partially_randomized_ratio_one_is_exact(capsys):
    op = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3) + QubitOperator("Y0", 0.2)
    out = _qDRIFT(op, samples=5, ratio=1.0, verbose=True).partially_randomized()
    assert out == op
    text = capsys.readouterr().out
    assert "Deterministic terms: 3/3" in text
    assert "No random terms to sample" in text


def test_partially_randomized_zero_samples_verbose(capsys):
    op = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3)
    _qDRIFT(op, samples=0, ratio=0.5, verbose=True).partially_randomized()
    assert "No sampling (samples=0" in capsys.readouterr().out


def test_qdrift_verbose_reports_sampling(capsys):
    op = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3)
    _qDRIFT(op, samples=10, seed=3, verbose=True).qdrift()
    out = capsys.readouterr().out
    assert "unique terms out of 10 samples" in out
    assert "Original Hamiltonian had 2 terms" in out


def test_partially_randomized_with_sampling_verbose(capsys):
    op = (
        QubitOperator("X0", 0.5)
        + QubitOperator("Z0", 0.3)
        + QubitOperator("Y0", 0.2)
        + QubitOperator("X1", 0.1)
    )
    out = _qDRIFT(op, samples=50, ratio=0.5, seed=5, verbose=True).partially_randomized()
    # Deterministic top-2 exact; sampled leftovers keep exact coefficients.
    assert op_A_is_in_op_B(out, op)
    text = capsys.readouterr().out
    assert "Deterministic terms: 2/4" in text
    assert "Sampled" in text
