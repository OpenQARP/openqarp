"""Shared fixtures for the SDK absorber suites (each SDK is importorskip'd
in its own module; nothing here imports one)."""

from qarp.blocks import SimpleBlock


def feedforward_block(with_else: bool):
    """H(0); c0 = M(0); if c0 == 1: X(1) GPhase(0.3) [else: Y(1)] — all on one
    classical bit so the branch outcome is analytic per shot.  The ``GPhase``
    inside the branch pins the ``emit_absorb.rst`` claim that it round-trips;
    it sits last in the arm because pytket's ``get_commands()`` is DAG order
    and a qubit-less ``Phase`` op has no wire to pin its position."""
    from qarp.blocks import CompositeBlock, ConditionalBlock

    meas = SimpleBlock(2)
    meas.h(0)
    meas.measure(0, 0)
    meas.build()
    then_body = SimpleBlock(2)
    then_body.x(1)
    then_body.gphase(0.3)
    then_body.build()
    else_body = None
    if with_else:
        else_body = SimpleBlock(2)
        else_body.y(1)
        else_body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=then_body, else_body=else_body)
    cond.target_cbits = [0]
    comp = CompositeBlock([meas, cond], 2)
    comp.build()
    return comp
