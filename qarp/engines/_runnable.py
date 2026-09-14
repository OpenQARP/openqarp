"""``Runnable`` — the structural contract between engines and what they execute.

Engines depend on this protocol only, never on ``qarp.algorithms`` (the
primitive base class there satisfies it structurally, PEP 544 — no
registration, no import in either direction).  Dependency rule, lint-enforced by
``tests/test_engines/test_layering.py``: ``composite → primitives →
engines-see-only-Runnable``.
"""

from typing import Any, Optional, Protocol, Union

from .._types import Consumes, Shots


class Runnable(Protocol):
    """Members engines actually touch on an executable item.

    ``target`` / ``operator`` are typed ``Any``: their concrete types
    (``Target``, ``QubitOperator``) live above this layer, so engines match
    ``target.name`` by string and duck-type the rest — importing them here
    would recreate the engines⇄algorithms cycle this protocol removes.
    ``Consumes`` is importable: it lives in ``qarp._types``, below both layers.
    """

    sub_blocks: list
    compiled_circuits: list[list]
    _n_qubits_list: list[int]
    n_shots: Optional[Union[int, Shots]]
    # None, or LSB-indexed ket-seeding amplitudes (ndarray — typed Any for the
    # same layering reason as target/operator).
    initial_state: Any
    supported_targets: frozenset
    target: Any
    operator: Any
    consumes: Consumes
    supports_exact: bool
    supports_backprop_gradient: bool
    # Linearity class of run() per compiled circuit — see _gradients.GRADIENT_KINDS.
    gradient_kind: str

    def build(self) -> "Runnable": ...

    def run(self, results: list) -> Any: ...

    def run_from_amplitudes(self, compiled_circuits: list, simulator=None) -> Any: ...
