"""Phase 4a of the pipeline property suite: gradient consistency.

Every registry method (adjoint, per-occurrence parameter shift, batched
finite differences, and the ``"default"`` policy) must equal central finite differences of the very objective the engine
evaluates — the independent numeric method is the oracle.  Sweeps the whole
parameterized-block registry × optimization levels, generalising the
HEA-specific incident test
(tests/test_engines/test_gradient_order.py::
test_hea_ansatz_gradients_survive_optimization, which stays): the O1
rotation-merge bug folded two symbols into one gate, killing exactly this
agreement.

``UNWRAPPABLE`` (deferred-buffer symbolic-time) blocks are excluded: they
flatten to 0–1 commands before ``set_time`` materialises them, so
engine-level substitution — the path differentiated here — is not their
execution model.
"""

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qarp.algorithms import StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from tests.strategies import FACTORIES, UNWRAPPABLE, random_pauli_sum, unitary_seeds

pytestmark = pytest.mark.property

GRAD_BLOCKS = tuple(n for n in sorted(FACTORIES) if n not in UNWRAPPABLE)

FD_STEP = 1e-6
ATOL = 1e-5


def _symbolic_ket(name: str, opt_level: int):
    block = FACTORIES[name]()
    block.build()
    return block.optimize(level=opt_level)


def _seeded_params(ket, values_seed: int) -> dict:
    rng = np.random.default_rng(values_seed)
    values = [float(v) for v in rng.uniform(-np.pi, np.pi, size=len(ket.symbols))]
    return ket.parameter_map(values)


QOP_METHODS = ("default", "adjoint", "parameter-shift", "finite-diff")
BLOCK_METHODS = ("default", "parameter-shift", "finite-diff")  # adjoint needs a QubitOperator


def _assert_gradient_matches_fd(engine, ket, params, method: str = "default") -> None:
    grad = np.real(engine.run_gradient(params, method=method)[0])
    assert grad.shape == (len(ket.symbols),)
    for i, sym in enumerate(ket.symbols):
        up, dn = dict(params), dict(params)
        up[sym] = params[sym] + FD_STEP
        dn[sym] = params[sym] - FD_STEP
        fd = (np.real(engine.run(up)[0]) - np.real(engine.run(dn)[0])) / (2 * FD_STEP)
        assert abs(grad[i] - fd) < ATOL, f"∂/∂{sym}: analytic {grad[i]} vs FD {fd}"


@example(name="HEABlock", opt_level=2, values_seed=5, method="default")  # the O1 merge incident
@example(name="UCCBlock-generalised", opt_level=2, values_seed=7, method="adjoint")
@example(name="UCCBlock-generalised", opt_level=2, values_seed=7, method="parameter-shift")
@example(name="GivensBlock", opt_level=1, values_seed=0, method="finite-diff")
@example(name="TrotterAnsatzBlock", opt_level=0, values_seed=0, method="parameter-shift")  # −2·x
@given(
    name=st.sampled_from(GRAD_BLOCKS),
    opt_level=st.sampled_from([0, 1, 2]),
    values_seed=unitary_seeds,
    method=st.sampled_from(QOP_METHODS),
)
def test_qubit_operator_gradient_matches_finite_differences(name, opt_level, values_seed, method):
    """Every registry method on a QubitOperator target — ``"default"`` resolves
    to the adjoint here, so the explicit ``"parameter-shift"`` draw is what
    exercises the per-occurrence rules on O1-merged circuits."""
    ket = _symbolic_ket(name, opt_level)
    op = random_pauli_sum(ket.n_qubits, np.random.default_rng(values_seed))
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=op)])
    _assert_gradient_matches_fd(engine, ket, _seeded_params(ket, values_seed), method)


@example(name="TrotterAnsatzBlock", values_seed=0, method="default")  # −2·x: the zero find
@example(name="UCCBlock", values_seed=7, method="parameter-shift")  # multi-occurrence
@example(name="HEABlock", values_seed=5, method="finite-diff")
@given(
    name=st.sampled_from(GRAD_BLOCKS),
    values_seed=unitary_seeds,
    method=st.sampled_from(BLOCK_METHODS),
)
def test_parameter_shift_gradient_matches_fd(name, values_seed, method):
    """A Block observable forces the parameter-shift branch (adjoint needs a
    QubitOperator).  The per-occurrence generator-spectrum rules make the
    shift exact for every affine angle shape the registry produces — shared
    symbols, scaled angles, compound angles from the O1 merge — so the former
    "or rejects" disjunction is gone.  Found on the first nightly run of the
    disjunctive form: the bare ±π/2 symbol shift returned an identically ZERO
    gradient for TrotterAnsatzBlock's −2·x angles (the shifted evaluations
    sat one full gate period apart)."""
    ket = _symbolic_ket(name, 0)
    obs = SimpleBlock(ket.n_qubits, name="Zobs")
    obs.z(0)
    obs.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=obs)])
    _assert_gradient_matches_fd(engine, ket, _seeded_params(ket, values_seed), method)


def test_parameter_shift_multi_occurrence_symbol_matches_fd():
    """Ad-hoc record of the case that found the reject branch: a UCC symbol
    drives several exponentials.  The single-occurrence rule refused it
    (``CapabilityError``, "appears N times"); per-occurrence shifting sums the
    contributions and matches finite differences.  Kept alongside the
    property test per the never-delete-the-finding rule."""
    ket = _symbolic_ket("UCCBlock", 0)
    obs = SimpleBlock(ket.n_qubits, name="Zobs")
    obs.z(0)
    obs.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=obs)])
    _assert_gradient_matches_fd(engine, ket, _seeded_params(ket, 0))
