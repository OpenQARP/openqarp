"""Unit tests for the ``PrimitiveAlgorithm`` ABC.

These exercise the pure-Python logic of the base class — ``infer_target()``
target detection from the ``{ket, bra, operator}`` inputs (identity-based,
``is``) and the default ``run_from_amplitudes`` behaviour.  The C++ core is not
involved: blocks are used only as distinct/identical sentinel objects.
"""

import copy

import pytest

from qarp.algorithms import PrimitiveAlgorithm, Target
from qarp.blocks import ComputationalBasisStateBlock


class _ConcretePrimitive(PrimitiveAlgorithm):
    """Minimal concrete subclass so the ABC can be instantiated."""

    def build(self):
        return self

    def run(self, results):
        pass


def _ket():
    return ComputationalBasisStateBlock(basis_state=[0])


def _operator():
    return ComputationalBasisStateBlock(basis_state=[1])


# ── infer_target: target detection ──────────────────────────────────────


def test_infer_target_ket_only_is_sampling():
    ket = _ket()
    prim = _ConcretePrimitive(ket=ket)
    assert prim.infer_target() == Target.SAMPLING


def test_infer_target_ket_plus_operator_defaults_bra_to_ket():
    """ket + operator, bra=None → EXPECTATION_VALUE and bra defaults to ket."""
    ket = _ket()
    prim = _ConcretePrimitive(ket=ket, operator=_operator())
    assert prim.infer_target() == Target.EXPECTATION_VALUE
    assert prim.bra is ket


def test_infer_target_bra_is_ket_plus_operator_is_expectation_value():
    ket = _ket()
    prim = _ConcretePrimitive(ket=ket, bra=ket, operator=_operator())
    assert prim.infer_target() == Target.EXPECTATION_VALUE


def test_infer_target_distinct_bra_plus_operator_is_transition_amplitude():
    prim = _ConcretePrimitive(ket=_ket(), bra=_ket(), operator=_operator())
    assert prim.infer_target() == Target.TRANSITION_AMPLITUDE


def test_infer_target_distinct_bra_no_operator_is_overlap():
    prim = _ConcretePrimitive(ket=_ket(), bra=_ket())
    assert prim.infer_target() == Target.OVERLAP


# ── infer_target: the bra ≔ ket default is live, not sticky ─────────────


def test_infer_target_redefaults_bra_after_deepcopy_ket_rebind():
    """Rebinding only ket on a copy must keep EXPECTATION_VALUE — the
    defaulted bra follows the current ket instead of latching the old one."""
    prim = _ConcretePrimitive(ket=_ket(), operator=_operator())
    prim.infer_target()
    clone = copy.deepcopy(prim)
    clone.ket = _ket()
    assert clone.infer_target() == Target.EXPECTATION_VALUE
    assert clone.bra is clone.ket


def test_infer_target_explicit_bra_survives_ket_rebind():
    bra = _ket()
    prim = _ConcretePrimitive(ket=_ket(), bra=bra, operator=_operator())
    prim.infer_target()
    prim.ket = _ket()
    assert prim.infer_target() == Target.TRANSITION_AMPLITUDE
    assert prim.bra is bra


def test_infer_target_bra_none_reset_restores_live_default():
    ket = _ket()
    prim = _ConcretePrimitive(ket=ket, bra=_ket(), operator=_operator())
    prim.infer_target()
    prim.bra = None
    assert prim.infer_target() == Target.EXPECTATION_VALUE
    assert prim.bra is ket


def test_infer_target_operator_removal_restores_sampling_while_defaulted():
    """With bra never explicitly assigned, dropping the operator re-infers
    SAMPLING — the materialized default does not linger as a stale bra."""
    prim = _ConcretePrimitive(ket=_ket(), operator=_operator())
    prim.infer_target()
    prim.operator = None
    assert prim.infer_target() == Target.SAMPLING
    assert prim.bra is None


# ── infer_target: error paths ───────────────────────────────────────────


def test_infer_target_missing_ket_raises():
    prim = _ConcretePrimitive(ket=None)
    with pytest.raises(RuntimeError, match="ket must be provided"):
        prim.infer_target()


def test_infer_target_bra_is_ket_without_operator_raises():
    ket = _ket()
    prim = _ConcretePrimitive(ket=ket, bra=ket)
    with pytest.raises(RuntimeError, match="operator must be provided"):
        prim.infer_target()


# ── infer_target: return/store consistency ──────────────────────────────


def test_infer_target_returns_value_it_stores():
    prim = _ConcretePrimitive(ket=_ket(), operator=_operator())
    returned = prim.infer_target()
    assert returned is prim.target


# ── run_from_amplitudes default ─────────────────────────────────────────────────


def test_run_from_amplitudes_default_raises_not_implemented():
    prim = _ConcretePrimitive(ket=_ket())
    with pytest.raises(NotImplementedError):
        prim.run_from_amplitudes([])


# ── canonical-class identity ────────────────────────────────────────────


def test_canonical_primitive_algorithm_module():
    assert PrimitiveAlgorithm.__module__ == "qarp.algorithms._primitives.primitive_algorithm"
