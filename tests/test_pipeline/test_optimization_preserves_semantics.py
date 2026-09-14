"""Phase 2a of the pipeline property suite: optimization-level invariance.

For any registry block, any primitive, and any bound parameter values, the
result at O0, O1 and O2 is the same number (§16 EQ-2 lifted to Python: the
optimizer passes and the unoptimized path are independently implemented, so
agreement is the oracle — a self-consistent wrong answer cannot satisfy it).

The sweep runs in the deterministic slice — QarpEngine, EXACT readout, no
noise, no device — where every configuration must RUN: a ``CapabilityError``
here is a failure, not an accepted branch (the device axis belongs to the
routing property next door).
"""

import pytest
from hypothesis import example, given

from tests.strategies import (
    PipelineConfig,
    pipeline_configs,
    results_allclose,
    run_pipeline,
)

pytestmark = pytest.mark.property

_LEVELS = (0, 1, 2)


def _cfg(block_name: str, primitive: str, values_seed: int) -> PipelineConfig:
    return PipelineConfig(
        block_name=block_name,
        primitive=primitive,
        engine="qarp",
        device=None,
        opt_level=0,
        shots=None,
        noise=None,
        values_seed=values_seed,
    )


@example(cfg=_cfg("GivensBlock", "SWAPTest", 0))  # CSWAP-bearing circuit
@example(cfg=_cfg("UCCBlock", "StateVector", 7))  # deepest registry block
@example(cfg=_cfg("TrotterAnsatzBlock", "PauliAveraging", 3))  # grouped estimator
@example(cfg=_cfg("RandomCircuit", "StateVector", 11))  # merge-rich dense gates
@given(
    cfg=pipeline_configs(
        engines=("qarp",),
        devices=(None,),
        opt_levels=(0,),
        shots_axis=(None,),
        noise_axis=(None,),
    )
)
def test_expectation_is_opt_level_invariant(cfg):
    results = [run_pipeline(cfg._replace(opt_level=lvl)) for lvl in _LEVELS]
    for lvl, res in zip(_LEVELS[1:], results[1:], strict=True):
        assert results_allclose(results[0], res), (
            f"O0 vs O{lvl} disagree for {cfg}: {results[0]!r} != {res!r}"
        )
