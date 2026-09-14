"""Devices: architectures, noise models and the compile-for-device pipeline.
Public depth: flat.  Submodules are private.
"""

from ._architecture import (
    Architecture,
    get_nearest_neighbour_architecture,
    get_all_to_all_architecture,
)
from ._device import Device
from ._noise_model import NoiseModel

import qarpx as _qx


def compile_for_device(*args, router=None):
    """Run check_fits → rebase → route → rebase against a Device.

    Two call forms:

        compile_for_device(block, device) -> CompiledCircuit
        compile_for_device(commands, n_qubits, device) -> CompiledCircuit

    The block form is a thin wrapper around the commands form: it calls
    ``block.flatten()`` and reads ``block.n_qubits`` for the user.

    Args:
        router: Optional ``qx.RouterKind`` selecting the routing
            implementation (default ``Sabre``;
            ``qx.RouterKind.Lite`` is the simpler greedy sweep).
    """
    router = _qx.RouterKind.Sabre if router is None else router
    if len(args) == 2 and isinstance(args[0], _qx.Block):
        block, device = args
        return _qx.compile_for_device(block.flatten(), block.n_qubits, device, router)
    return _qx.compile_for_device(*args, router)


__all__ = [
    "Architecture",
    "Device",
    "NoiseModel",
    "compile_for_device",
    "get_all_to_all_architecture",
    "get_nearest_neighbour_architecture",
]
