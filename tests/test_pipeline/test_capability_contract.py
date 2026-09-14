"""Phase 1 of the pipeline property suite: the capability contract.

Any drawn pipeline configuration — block × primitive × engine × device ×
optimization level × shots × noise, deliberately unfiltered for legality —
either produces a sane result or raises ``qarp.errors.CapabilityError`` with an
actionable message.  Never a wrong-shaped value, a bare ``RuntimeError``, a
nanobind ``TypeError``, or a non-finite number.  The invariant is the oracle:
it pins the unified rejection taxonomy (architecture review S-D) permanently.

Hand-written counterparts pinning *specific known* illegal combinations live
in tests/test_engines/test_capability_validation.py; this sweep searches the
composition space around them and never replaces them.
"""

import numpy as np
import pytest
from hypothesis import example, given

from qarp.algorithms import StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator
from tests.strategies import ENGINES, PRIMITIVES, PipelineConfig, build_pipeline, pipeline_configs


def _assert_actionable(err: CapabilityError) -> None:
    # Actionability floor: a real sentence, not a bare class name or code.
    msg = str(err)
    assert len(msg) >= 20, f"CapabilityError message too terse to act on: {msg!r}"


def _assert_sane_result(cfg: PipelineConfig, magnitude_bound, result) -> None:
    if cfg.primitive == "Sampler":
        assert isinstance(result, dict), f"Sampler returned {type(result).__name__}"
        total = 0.0
        for key, weight in result.items():
            assert isinstance(key, tuple) and all(bit in (0, 1) for bit in key), (
                f"malformed sampler key {key!r}"
            )
            assert np.isfinite(weight) and weight >= 0.0
            total += weight
        assert abs(total - 1.0) < 1e-6, f"sampler weights sum to {total}"
        return
    z = complex(result)
    assert np.isfinite(z.real) and np.isfinite(z.imag), f"non-finite result {result!r}"
    assert abs(z) <= magnitude_bound + 1e-6, (
        f"|{z}| exceeds the analytic ceiling {magnitude_bound} for {cfg.primitive}"
    )


def _exercise(cfg: PipelineConfig) -> None:
    """The contract disjunction: run-to-sane-result or CapabilityError."""
    try:
        built = build_pipeline(cfg)
        built.engine.build([built.primitive])
        results = built.engine.run()
    except CapabilityError as err:
        _assert_actionable(err)
        return
    assert len(results) == 1
    _assert_sane_result(cfg, built.magnitude_bound, results[0])


@pytest.mark.property
@example(
    # Nightly find 2026-08-21: device wider than ket AND operator (1-qubit
    # Trotter ket on a 2-qubit grid) crashed StateVector's diagonal
    # contraction with a numpy broadcast ValueError (stale-width diagonal).
    cfg=PipelineConfig(
        block_name="TrotterBlock-symbolic-time",
        primitive="StateVector",
        engine="qarp",
        device="grid",
        opt_level=0,
        shots=None,
        noise=None,
        values_seed=346,
    )
)
@given(cfg=pipeline_configs())
def test_config_runs_or_raises_capability_error(cfg):
    _exercise(cfg)


def test_wide_device_expectation_matches_analytic_value():
    """Ad-hoc record of the sweep find pinned above (§18 oracle: after
    ``Rx(θ)|0⟩``, ``⟨Z⟩ = cos θ``): a device wider than ket and operator
    leaves the padding qubits in |0⟩, so the value — not a crash, not a
    rejection — is the contract."""
    theta = 0.7
    b = SimpleBlock(1)
    b.rx(0, theta)
    b.build()
    sv = StateVector(ket=b, operator=QubitOperator("Z0"))
    eng = QarpEngine(n_qubits=2)
    eng.build([sv])
    assert abs(eng.run()[0] - np.cos(theta)) < 1e-12


# ── Deterministic backbone ───────────────────────────────────────────────
# The derandomized 25-example gate slice cannot promise per-axis coverage;
# this fixed grid exercises every primitive × engine × shots cell every run
# (specs only in the parametrize list — never constructed objects).


@pytest.mark.property
@pytest.mark.parametrize("shots", [None, 200], ids=["EXACT", "200shots"])
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_axis_backbone(primitive, engine, shots):
    _exercise(
        PipelineConfig(
            block_name="SimpleBlock-fallback",
            primitive=primitive,
            engine=engine,
            device=None,
            opt_level=0,
            shots=shots,
            noise=None,
            values_seed=7,
        )
    )
