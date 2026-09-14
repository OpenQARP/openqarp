"""Multi-controlled single-qubit rotations and flips, built from native gates.

Shared by the state-preparation blocks.  Every entry point takes a ``path`` —
a list of ``(qubit, bit)`` pairs — and acts only on the branch where each
listed qubit holds its bit; ``_x_sandwich`` maps a 0-bit onto ``mcx``'s
all-ones control convention.
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Recursive Barenco doubling: with V such that V(angle/2)**2 == U(angle),
#
#   C^k(U)[c_1..c_{k-1}, c_k; t] =
#       CV(c_k, t) . MCX(c_1..c_{k-1} -> c_k) . CV^-1(c_k, t)
#       . MCX(c_1..c_{k-1} -> c_k) . C^{k-1}(V)[c_1..c_{k-1}; t]
# ---------------------------------------------------------------------------


def _apply_mc_rotation(
    block, controls: List[int], target: int, angle: float, single, controlled
) -> None:
    if not controls:
        single(block, target, angle)
        return
    if len(controls) == 1:
        controlled(block, controls[0], target, angle)
        return
    *rest, last = controls
    controlled(block, last, target, angle / 2)
    block.mcx(*rest, last)
    controlled(block, last, target, -angle / 2)
    block.mcx(*rest, last)
    _apply_mc_rotation(block, rest, target, angle / 2, single, controlled)


def _ry_single(block, q: int, angle: float) -> None:
    block.ry(q, angle)


def _ry_controlled(block, c: int, t: int, angle: float) -> None:
    block.cry(c, t, angle)


def _rz_single(block, q: int, angle: float) -> None:
    block.rz(q, angle)


def _rz_controlled(block, c: int, t: int, angle: float) -> None:
    block.crz(c, t, angle)


def _p_single(block, q: int, angle: float) -> None:
    block.p(q, angle)


def _p_controlled(block, c: int, t: int, angle: float) -> None:
    block.cp(c, t, angle)


def _x_sandwich(block, path: List[Tuple[int, int]]) -> List[int]:
    """Flip every qubit in ``path`` whose fixed bit is 0, matching ``mcx``'s
    all-ones control convention; caller restores with the returned list."""
    zero_qubits = [q for q, bit in path if bit == 0]
    if zero_qubits:
        block.x(zero_qubits)
    return zero_qubits


def _apply_mc_ry(block, path: List[Tuple[int, int]], target: int, angle: float) -> None:
    """``Ry(angle)`` on ``target``, on the branch matching ``path``."""
    zero_qubits = _x_sandwich(block, path)
    _apply_mc_rotation(block, [q for q, _ in path], target, angle, _ry_single, _ry_controlled)
    if zero_qubits:
        block.x(zero_qubits)


def _apply_mc_rz(block, path: List[Tuple[int, int]], target: int, angle: float) -> None:
    """``Rz(angle)`` on ``target``, on the branch matching ``path``."""
    zero_qubits = _x_sandwich(block, path)
    _apply_mc_rotation(block, [q for q, _ in path], target, angle, _rz_single, _rz_controlled)
    if zero_qubits:
        block.x(zero_qubits)


def _apply_mc_phase(block, path: List[Tuple[int, int]], angle: float) -> None:
    """Multiply the single basis state matching ``path`` (all n qubits fixed)
    by ``exp(i * angle)``, using one of ``path``'s own qubits as the phase
    gate's target (a controlled-phase needs a target distinct from its
    controls; here that target is simply the last address qubit, since a
    completed address has no free qubit left to spare)."""
    if not path:
        block.gphase(angle)
        return
    controls = [q for q, _ in path[:-1]]
    target = path[-1][0]
    zero_qubits = _x_sandwich(block, path)
    _apply_mc_rotation(block, controls, target, angle, _p_single, _p_controlled)
    if zero_qubits:
        block.x(zero_qubits)


def _apply_mcx_path(block, path: List[Tuple[int, int]], target: int) -> None:
    """Flip ``target`` iff every qubit in ``path`` matches its required bit —
    the X-sandwich + native ``mcx`` pattern, degenerating to a plain ``X``
    when ``path`` is empty."""
    if not path:
        block.x(target)
        return
    zero_qubits = _x_sandwich(block, path)
    block.mcx(*[q for q, _ in path], target)
    if zero_qubits:
        block.x(zero_qubits)
