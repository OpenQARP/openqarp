"""Approximate Rz → Clifford+T synthesis (Ross–Selinger, via pygridsynth).

Deliberately OUTSIDE the transpiler: DAG/rebase passes are unitary-exact by
contract (qarp_conventions §16 EQ-2), so an ε-approximation can be neither a
decomposition rule nor an optimization pass.  It runs as an explicit
post-rebase stage with a declared tolerance — per rotation, not a global
budget, the usual convention of gridsynth-based compilation stacks.

Input must be Clifford+T+Rz format (Clifford + T + Rz + meta,
``qx.clifford_t_rz_gateset()``) — the intermediate that rebase already emits.
Output is phase-exact (§13): each Rz folds to ``T^j · Rz(θ_r)`` with
``θ_r = remainder(θ, π/4)``; zero residual is free, otherwise gridsynth
approximates ``Rz(θ_r)`` over {H, S, T, X} within ``epsilon`` in operator
norm.  The fold's ``e^{-ijπ/8}``, gridsynth's W gates (= ``e^{iπ/4}``) and
its circuit phase accumulate into one trailing GPhase.

The fold is π/4, not π/2: ``Rz(π/4)`` **is** ``T`` up to phase, so folding
only to Clifford would hand gridsynth angles with exact 1-gate forms and
pay O(log 1/ε) T gates plus ε error for them.  Multi-controlled lowering
emits exactly those angles (``decompose_cp`` halves down from π), so the
distinction is not academic.

Needs the optional ``pygridsynth`` dependency (``pip install
"openqarp[cliffordt]"``) — the reference Python implementation of
Ross–Selinger gridsynth.
"""

import math
from functools import lru_cache
from typing import Sequence

import qarpx as qx

_QUARTER_PI = math.pi / 4
_FOLD_TOL = 1e-12  # residual below this is an exact T power, not a rotation

_LETTER = {"H": qx.GateType.H, "S": qx.GateType.S, "T": qx.GateType.T, "X": qx.GateType.X}

# clifford_t_rz_gateset minus Rz: everything that may pass through unsynthesized.
_PASSTHROUGH = frozenset(
    {
        qx.GateType.X,
        qx.GateType.Y,
        qx.GateType.Z,
        qx.GateType.H,
        qx.GateType.S,
        qx.GateType.Sdg,
        qx.GateType.T,
        qx.GateType.Tdg,
        qx.GateType.CX,
        qx.GateType.CZ,
        qx.GateType.Measure,
        qx.GateType.Barrier,
        qx.GateType.GPhase,
    }
)


def _gphase_command(gamma: float) -> "qx.Command":
    # Command has no qubit-less constructor binding; mint through a scratch
    # block.  Lazy import: qarp.engines imports qarp.resources at import
    # time, and qarp.blocks must stay out of that chain.
    from qarp.blocks import SimpleBlock

    b = SimpleBlock(1)
    b.gphase(gamma)
    return next(iter(b.commands()))


@lru_cache(maxsize=None)
def _rz_sequence(theta_r: float, epsilon: float) -> "tuple[str, float]":
    """Gridsynth letters (W stripped into the phase) for one folded residual.

    Cached: a QPE ladder repeats a handful of distinct angles thousands of
    times, so synthesis cost is per distinct (θ_r, ε), not per gate.
    """
    try:
        import mpmath
        from pygridsynth.gridsynth import gridsynth_circuit
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            'Rz → Clifford+T synthesis needs pygridsynth: pip install "openqarp[cliffordt]"'
        ) from exc

    # mpf(float) is exact — the double IS the target angle; bare floats only
    # trigger pygridsynth's decimal-literal precision warning.
    circ = gridsynth_circuit(mpmath.mpf(theta_r), mpmath.mpf(epsilon))  # phase-exact mode
    letters = circ.to_simple_str()
    phase = float(circ.phase) + (math.pi / 4) * letters.count("W")
    return letters.replace("W", ""), phase


def synthesize_clifford_t(
    commands: "Sequence[qx.Command]", *, epsilon: float
) -> "list[qx.Command]":
    """Replace every Rz in a Clifford+T+Rz-format stream by a Clifford+T sequence.

    Each output sequence is within ``epsilon`` of its exact rotation in
    operator norm, global phase included; all other gates of the set pass through
    untouched, so the stream's unitary is preserved to ``m·ε`` for ``m``
    synthesized rotations.  Raises ``ValueError`` on symbolic angles or on
    gates outside the Clifford+T+Rz gate set (rebase first).
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

    out: list[qx.Command] = []
    gamma = 0.0
    for cmd in commands:
        if cmd.gate != qx.GateType.Rz:
            if cmd.gate not in _PASSTHROUGH:
                raise ValueError(
                    "synthesize_clifford_t expects Clifford+T+Rz-format input (rebase "
                    f"to qx.clifford_t_rz_gateset() first); got {qx.gate_name(cmd.gate)}"
                )
            out.append(cmd)
            continue

        param = cmd.params[0]
        if not param.is_concrete():
            raise ValueError(
                "synthesize_clifford_t needs bound angles: substitute "
                "parameters before synthesizing."
            )
        theta = param.value()
        qubit = cmd.qubits[0]

        # Exact Clifford+T fold: Rz(θ) = e^{-ijπ/8} · T^j · Rz(θ_r), j ∈ ℤ.
        # T^8 = I and T^2 = S, so T^(j mod 8) is emitted as S^a·T^b — at most
        # four gates, and never an approximation.
        residual = math.remainder(theta, _QUARTER_PI)
        j = round((theta - residual) / _QUARTER_PI)
        turns = j % 8
        out.extend(qx.Command(qx.GateType.S, qubit) for _ in range(turns // 2))
        if turns % 2:
            out.append(qx.Command(qx.GateType.T, qubit))
        gamma -= j * math.pi / 8

        if abs(residual) >= _FOLD_TOL:
            letters, phase = _rz_sequence(residual, epsilon)
            out.extend(qx.Command(_LETTER[ch], qubit) for ch in letters)
            gamma += phase

    gamma = math.remainder(gamma, 2 * math.pi)
    if abs(gamma) > _FOLD_TOL:
        out.append(_gphase_command(gamma))
    return out
