"""``qx.SamplingResult`` is constructible from Python.

Counts arriving from a remote service must be wrapped in the one result
type §14 mandates; until this constructor existed only the simulators could
produce one.  Oracles are the contract (field echo, validation) and an
analytic bit permutation for the reindex row.
"""

import pytest

import qarpx as qx


def test_constructor_exposes_fields():
    r = qx.SamplingResult(2, {0: 5, 3: 5}, 10)
    assert r.n_qubits == 2
    assert r.n_shots == 10
    assert dict(r.counts) == {0: 5, 3: 5}
    assert r.n_cbits == 0
    assert list(r.cbit_history) == []
    assert list(r.statevector) == []
    assert r.probability(3) == pytest.approx(0.5)
    assert r.probability(1) == 0.0


def test_constructor_passes_n_cbits_through():
    assert qx.SamplingResult(2, {1: 3}, 3, n_cbits=2).n_cbits == 2


def test_constructor_rejects_count_sum_mismatch():
    with pytest.raises(ValueError, match="n_shots"):
        qx.SamplingResult(2, {0: 5, 3: 4}, 10)


def test_constructor_rejects_key_outside_register():
    with pytest.raises(ValueError, match="n_qubits"):
        qx.SamplingResult(2, {4: 10}, 10)


def test_reindex_constructed():
    # Logical qubit 0 sits on physical qubit 1: physical key 0b01 (qubit 0 set)
    # is logical key 0b10 after the permutation.
    physical = qx.SamplingResult(2, {1: 10}, 10)
    logical = qx.reindex_sampling_result(physical, [1, 0])
    assert dict(logical.counts) == {2: 10}
    assert logical.n_shots == 10
