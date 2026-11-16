"""The resource-vector contract: frozen, versioned, wire-stable.

``ResourceVector`` is the schema external consumers (resource-estimation
frameworks, surrogate training) code against.  ``to_dict()`` is the wire format; it is
pinned by a golden test and must never change silently — additive changes
bump ``SCHEMA_VERSION``.

Two semantic rules the schema encodes:

* ``None`` is not ``0``.  ``swap_count=None`` means "routing has not
  happened", never "no SWAPs"; ``t_count=None`` means "not expressible as a
  count at this stage", never "zero T gates".
* Counted and modeled quantities never mix.  ``t_count`` only ever comes
  from gates present in the circuit; ``t_count_modeled`` / ``extras`` are
  filled exclusively by a :class:`~qarp.resources.ResourceModeler`
  and carry a ``provenance.modeler`` tag saying which one.

This module may import only stdlib — everything in ``qarp.resources``
depends on it.
"""

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

# Wire-format version consumers pin against: every change is additive and
# bumps this number, which never moves without a migration.
SCHEMA_VERSION = 1


class Stage(str, Enum):
    """Compilation stage a :class:`ResourceVector` snapshot describes."""

    LOGICAL = "logical"  # as-authored (block.flatten()), no transpilation
    OPTIMIZED = "optimized"  # rebased + peephole-optimized to the gate set
    ROUTED = "routed"  # after routing, PRE-rebase: the only stage with SWAPs
    TARGET = "target"  # routed then rebased back to the target gate set
    SYNTHESIZED = "synthesized"  # Rz → Clifford+T at declared ε (approximate)


@dataclass(frozen=True)
class Provenance:
    """What produced the numbers — makes every vector self-describing."""

    stage: Stage
    gateset: str | None = None  # e.g. "clifford_t", "clifford_t_rz"
    opt_level: str | None = None  # "O0" | "O1" | "O2"
    router: str | None = None  # "Sabre" | "Lite"; ROUTED/TARGET only
    device: str | None = None  # user-supplied device label
    modeler: str | None = None  # e.g. "my_modeler:eps=1e-10"
    synthesis: str | None = None  # e.g. "gridsynth:eps=1e-10"; SYNTHESIZED only


@dataclass(frozen=True)
class ResourceVector:
    """One stage snapshot of a circuit's resources.

    ``n_qubits`` is the logical width before routing and the physical device
    width at ROUTED/TARGET.  ``n_gates`` counts unitary operations only
    (barriers, global phases, branch markers, measurements and resets are
    excluded).  The arity buckets partition it exactly:
    ``n_1q + n_2q + n_3q_plus == n_gates`` at every stage.  ``swap_count`` is
    router-inserted overhead, defined only at ROUTED (user-authored SWAPs
    appear in ``op_histogram`` and ``n_2q`` at every stage).
    """

    n_qubits: int
    depth: int
    n_gates: int
    n_1q: int
    n_2q: int
    n_3q_plus: int  # gates on 3+ qubits (CCX, CSWAP, wide MCZ); completes the partition
    n_measurements: int
    n_resets: int
    t_count: int | None  # counted T+Tdg; None while any parametric/opaque unitary remains
    swap_count: int | None  # None unless stage is ROUTED
    op_histogram: dict[str, int]  # count_ops() minus pseudo-ops (Barrier/GPhase/markers)
    provenance: Provenance
    t_count_modeled: float | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Wire format (schema pinned by test_vector.py's golden test)."""
        d = asdict(self)
        d["provenance"]["stage"] = self.provenance.stage.value
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResourceVector":
        d = dict(d)
        version = d.pop("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"resource vector schema_version {version} != supported {SCHEMA_VERSION}"
            )
        prov = dict(d.pop("provenance"))
        prov["stage"] = Stage(prov["stage"])
        return cls(provenance=Provenance(**prov), **d)

    def with_model(
        self,
        *,
        modeler: str,
        t_count_modeled: float | None,
        extras: dict[str, float],
    ) -> "ResourceVector":
        """Copy with modeler-filled fields set (the only sanctioned mutation)."""
        return replace(
            self,
            t_count_modeled=t_count_modeled,
            extras=extras,
            provenance=replace(self.provenance, modeler=modeler),
        )
