"""``Runnable`` — the structural contract between engines and what they execute.

Engines depend on this protocol only, never on ``qarp.algorithms`` (the
primitive base class there satisfies it structurally, PEP 544 — no
registration, no import in either direction).  Dependency rule, lint-enforced by
``tests/test_engines/test_layering.py``: ``composite → primitives →
engines-see-only-Runnable``.
"""

from typing import Any, Optional, Protocol, Union

from .._types import Consumes, Shots, Target


class Runnable(Protocol):
    """Members engines actually touch on an executable item, grouped by what
    they describe: the quantity, the circuits, the estimator.  A new member
    joins one group.

    ``Target`` and ``Consumes`` are importable: they live in ``qarp._types``,
    below both layers, and engines compare them by identity.  ``operator``
    stays ``Any``: ``QubitOperator`` lives above this layer, so engines
    duck-type it — importing it here would recreate the engines⇄algorithms
    cycle this protocol removes.
    """

    # The quantity.
    target: Target
    operator: Any
    supported_targets: frozenset
    # The circuits.
    sub_blocks: list
    compiled_circuits: list[list]
    _n_qubits_list: list[int]
    # None, or LSB-indexed ket-seeding amplitudes (ndarray — typed Any for the
    # same layering reason as operator).
    initial_state: Any
    n_shots: Optional[Union[int, Shots]]
    # The estimator.
    consumes: Consumes
    supports_exact: bool
    supports_backprop_gradient: bool
    # Linearity class of run() per compiled circuit — see _gradients.GRADIENT_KINDS.
    gradient_kind: str

    def build(self) -> "Runnable": ...

    def run(self, results: list) -> Any: ...

    def run_from_amplitudes(self, compiled_circuits: list, simulator=None) -> Any: ...
