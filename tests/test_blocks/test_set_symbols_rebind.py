"""``set_symbols``: a later binding of an already-bound symbol wins, and the
pending queue stays one dict deep (pipeline_hardening_plan.md P1.7).

Oracles are the analytic gate matrices of §2.
"""

import numpy as np
from sympy import Symbol

from qarp.blocks import SimpleBlock


def _rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _u(theta, phi, lam):
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
        ]
    )


def test_later_binding_replaces_earlier():
    a = Symbol("a")
    blk = SimpleBlock(1)
    blk.rz(0, a)
    blk.build()
    bound = blk.set_symbols({a: 1.0}).set_symbols({a: 2.0}).set_symbols({a: 3.0})
    assert [str(c) for c in bound.flatten()] == ["Rz(3) [q0]"]
    np.testing.assert_allclose(bound.build().unitary_matrix(), _rz(3.0), atol=1e-12)


def test_queue_stays_one_dict_deep():
    a = Symbol("a")
    blk = SimpleBlock(1)
    blk.rz(0, a)
    blk.build()
    cur = blk
    for i in range(50):
        cur = cur.set_symbols({a: float(i)})
    assert len(cur._pending_substitutions) == 1
    assert [str(c) for c in cur.flatten()] == ["Rz(49) [q0]"]


def test_multi_symbol_param_still_binds_all_or_nothing():
    """A U gate's (phi+lam)/2 param spans two symbols; binding them in two
    separate calls must still substitute, and equal the analytic U."""
    theta, phi, lam = Symbol("theta"), Symbol("phi"), Symbol("lam")
    blk = SimpleBlock(1)
    blk.u(0, theta, phi, lam)
    blk.build()
    bound = blk.set_symbols({theta: 0.3}).set_symbols({phi: 0.7}).set_symbols({lam: 1.1})
    assert bound.free_symbols() == []
    np.testing.assert_allclose(bound.build().unitary_matrix(), _u(0.3, 0.7, 1.1), atol=1e-12)


def test_rebinding_one_of_several_symbols_keeps_the_others():
    a, b = Symbol("a"), Symbol("b")
    blk = SimpleBlock(1)
    blk.rz(0, a)
    blk.rz(0, b)
    blk.build()
    bound = blk.set_symbols({a: 0.2, b: 0.5}).set_symbols({a: 0.9})
    np.testing.assert_allclose(bound.build().unitary_matrix(), _rz(0.5) @ _rz(0.9), atol=1e-12)
