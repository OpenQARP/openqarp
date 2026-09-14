"""User-facing :class:`NoiseModel` wrapper around the C++ ``qx.NoiseModel``.

A thin Python class exposing ergonomic builders (depolarizing, pauli,
thermal_relaxation, ...) that delegates storage + sampling to the unified
C++ ``qx.NoiseModel`` consumed by ``QarpSimulator``.

Custom Python ``Command -> Channel`` closures are deliberately not
supported — Python callables in a per-shot-per-gate hot loop would erase
the unification's performance benefit.  Power users wanting parametric
channels should add them in C++ alongside the built-in channels.
"""

from typing import Iterable, List, Optional, Union

import qarpx as qx

GateSetLike = Union[str, "qx.GateSet", Iterable["qx.GateType"]]


def _resolve_gates(gate_set: Optional[GateSetLike], default_string: str) -> List["qx.GateType"]:
    """Normalise the various accepted forms into a flat list of GateType."""
    if gate_set is None:
        gate_set = default_string

    if isinstance(gate_set, str):
        s = gate_set.lower()
        if s in ("1q", "single", "single_qubit"):
            return _gateset_to_list(qx.full_gateset_1q())
        if s in ("2q", "two", "two_qubit"):
            return _gateset_to_list(qx.full_gateset_2q())
        if s in ("all", "1q+2q", "1q_2q"):
            return _gateset_to_list(qx.full_gateset_1q_2q())
        raise ValueError(f"NoiseModel: unknown gate_set shorthand '{gate_set}'")

    if isinstance(gate_set, qx.GateSet):
        return _gateset_to_list(gate_set)

    return [_coerce_gate(g) for g in gate_set]


def _coerce_gate(g):
    if isinstance(g, qx.GateType):
        return g
    if isinstance(g, str):
        try:
            return getattr(qx.GateType, g)
        except AttributeError as exc:
            raise ValueError(f"NoiseModel: unknown GateType '{g}'") from exc
    raise TypeError(f"NoiseModel: cannot coerce {g!r} to GateType")


# qx.GateSet exposes `contains(g)` but not an iterable view; brute-force by
# scanning the enum.  Cheap (~40 entries).
# A GateSet is a rebase target, so it admits the non-physical markers
# (Measure/Barrier/GPhase) that rebasing must not drop; they can carry no
# channel, so drop them here instead of failing the factories' arity check.
def _gateset_to_list(gs: "qx.GateSet") -> List["qx.GateType"]:
    out: List["qx.GateType"] = []
    for name in dir(qx.GateType):
        if name.startswith("_") or name == "name":
            continue
        try:
            g = getattr(qx.GateType, name)
        except AttributeError:
            continue
        if isinstance(g, qx.GateType) and gs.contains(g) and qx.gate_is_physical(g):
            out.append(g)
    return out


class NoiseModel:
    """Per-gate noise model attached to a :class:`qarp.devices.Device`.

    Ergonomic builders (:meth:`depolarizing`, :meth:`pauli`,
    :meth:`bit_flip`, :meth:`amplitude_damping`) cover the common cases.
    Compose with the ``+`` operator (right operand wins on collision).

    The underlying C++ object is reachable via :attr:`inner` for callers
    that need to pass it straight into ``qx.QarpSimulator``.

    Args:
        _inner: existing ``qx.NoiseModel`` to wrap (internal use).  End
            users construct via the named class methods.
    """

    def __init__(self, _inner: Optional["qx.NoiseModel"] = None):
        self._inner: "qx.NoiseModel" = _inner if _inner is not None else qx.NoiseModel()

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def depolarizing(cls, p: float, gate_set: Optional[GateSetLike] = "2q") -> "NoiseModel":
        """Depolarizing channel of strength ``p``.

        1q gates receive ``PauliChannel(p/3, p/3, p/3)``; 2q gates receive
        the uniform 2q depolarizing (each of the 15 non-identity Paulis
        with weight ``p/15``).
        """
        gates = _resolve_gates(gate_set, "2q")
        return cls(qx.make_depolarizing(p, gates))

    @classmethod
    def pauli(
        cls,
        p_x: float = 0.0,
        p_y: float = 0.0,
        p_z: float = 0.0,
        gate_set: Optional[GateSetLike] = "1q",
    ) -> "NoiseModel":
        """1q Pauli channel with explicit X/Y/Z weights."""
        gates = _resolve_gates(gate_set, "1q")
        return cls(qx.make_pauli(p_x, p_y, p_z, gates))

    @classmethod
    def bit_flip(cls, p: float, gate_set: Optional[GateSetLike] = "1q") -> "NoiseModel":
        """``PauliChannel(p, 0, 0)`` — flip the qubit with probability p."""
        gates = _resolve_gates(gate_set, "1q")
        return cls(qx.make_bit_flip(p, gates))

    @classmethod
    def amplitude_damping(cls, p: float, gate_set: Optional[GateSetLike] = "1q") -> "NoiseModel":
        """Amplitude damping (T1-like relaxation) of strength ``p``."""
        gates = _resolve_gates(gate_set, "1q")
        return cls(qx.make_amplitude_damping(p, gates))

    # ── Properties / composition ────────────────────────────────────────────

    @property
    def inner(self) -> "qx.NoiseModel":
        """Underlying C++ ``qx.NoiseModel`` for direct simulator use."""
        return self._inner

    @property
    def enabled(self) -> bool:
        return self._inner.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._inner.enabled = value

    def has_channel(self, gate: "qx.GateType") -> bool:
        return self._inner.has_channel(gate)

    def has_any_channel(self) -> bool:
        return self._inner.has_any_channel()

    def __add__(self, other: "NoiseModel") -> "NoiseModel":
        return NoiseModel(self._inner + other._inner)

    def __iadd__(self, other: "NoiseModel") -> "NoiseModel":
        self._inner += other._inner
        return self
