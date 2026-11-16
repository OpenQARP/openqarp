"""Thin Python re-export of the C++ ``qx.Device``.

Constructor::

    Device(n_qubits)
    Device(n_qubits, architecture=..., noise_model=..., gate_set=..., directedness=...)

``Device`` is data-only.  Compilation happens via
``qarp.devices.compile_for_device(block, device)`` (or the lower-level
``compile_for_device(commands, n_qubits, device)``); the Engine reads
these fields and assembles the rebase → route → simulate pipeline.
"""

import qarpx as qx

Device = qx.Device
