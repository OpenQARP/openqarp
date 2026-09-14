"""Modelers: cost models filling the *modeled* fields of a resource vector.

A :class:`ResourceModeler` never touches counted fields — it returns a copy
of the vector with ``t_count_modeled`` / ``extras`` set and a
``provenance.modeler`` tag naming itself (§19: counted and modeled never
mix).  A ``Custom`` command in the stream must null ``t_count_modeled``
rather than undercount: rotations fused inside it are invisible.

This is an extension point.  No modeler ships in the tree; engines may
advertise one via ``Engine.resource_modeler()``, and ``estimate(modeler=…)``
accepts any object satisfying the protocol.

This module may import only ``qarpx`` and :mod:`qarp.resources` —
``qarp.engines`` imports it.
"""

from typing import Protocol, Sequence

import qarpx as qx

from ._vector import ResourceVector


class ResourceModeler(Protocol):
    """Fills modeled fields of a final-stage :class:`ResourceVector`."""

    @property
    def name(self) -> str: ...

    def model(self, commands: "Sequence[qx.Command]", vector: ResourceVector) -> ResourceVector: ...
