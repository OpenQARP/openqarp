"""Architecture helpers.

Returns a ``qx.Architecture`` describing device qubit connectivity.  Pass
the result to :class:`qarp.devices.Device` as the ``architecture`` argument.
The Engine consumes it via the qarpx-native router during ``build()`` —
``Sabre`` by default, ``Lite`` for the greedy shortest-path sweep.
"""

import qarpx as qx

Architecture = qx.Architecture


def get_nearest_neighbour_architecture(xdim: int, ydim: int) -> "qx.Architecture":
    """Nearest-neighbour grid connectivity (undirected).

    Args:
        xdim: Number of qubits per row.
        ydim: Number of qubits per column.

    Returns:
        A ``qx.Architecture`` for the requested grid.
    """
    return qx.nearest_neighbour_architecture(xdim, ydim)


def get_all_to_all_architecture(n_qubits: int) -> "qx.Architecture":
    """All-to-all connectivity (undirected)."""
    return qx.all_to_all_architecture(n_qubits)
