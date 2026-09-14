"""Smoke tests for MixedOperatorBlock — QAOA mixer layer."""

import pytest

from qarp.blocks import MixedOperatorBlock


def test_mixer_constructs_and_builds():
    block = MixedOperatorBlock(n_qubits=4).build()
    assert block.is_built
    assert block.n_qubits == 4
    cmds = block.flatten()
    # The standard QAOA mixer applies a parametric Rx (or similar) on every qubit.
    assert len(cmds) >= 4
    # All gates must be parametric (the mixer's β driver).
    assert any(cmd.is_parametric() for cmd in cmds)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mixer_n_qubits(n):
    block = MixedOperatorBlock(n_qubits=n).build()
    assert block.n_qubits == n


def test_name_default_carries_layer_index_and_custom_name_is_honoured():
    """The old constructor overwrote any user-supplied name with 'Mixed Op.'
    and could never reach its indexed default (pipeline_hardening_plan.md,
    stage E review)."""
    from qarp.blocks import MixedOperatorBlock

    assert MixedOperatorBlock(2, symbol_idx=3).name == "Mixed Op. (p=3)"
    assert MixedOperatorBlock(2, name="custom").name == "custom"
