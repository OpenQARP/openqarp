"""qarp adapter: `compile_for_device` (check_fits -> rebase -> route -> rebase).

Router defaults to Sabre, the value the router benchmark selected; `ROUTER`
lets a shim pin Lite for the head-to-head against it.
"""

from benchmarks.compilation import architectures

LABEL = "qarpx"
ROUTER = "Sabre"

# qarp's rebase to qulacs_gateset emits exactly {H, Rx, Ry, Rz, CX, SWAP} on
# these fixtures — the same basis every other stack is asked to target.
_NAMES = {"h": "h", "rx": "rx", "ry": "ry", "rz": "rz", "cx": "cx", "cnot": "cx", "swap": "swap"}


def _device(topology: str, n: int):
    import qarpx as qx

    arch = qx.Architecture(n, architectures.edges(topology, n), topology)
    return qx.Device(n, architecture=arch, gate_set=qx.qulacs_gateset())


def _block(n: int, ops: list):
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(n)
    for op in ops:
        if op[0] == "h":
            block.h(op[1])
        elif op[0] == "cx":
            block.cx(op[1], op[2])
        elif op[0] == "swap":
            block.swap(op[1], op[2])
        else:
            getattr(block, op[0])(op[1], op[2])
    block.build()
    return block


def prepare(n: int, ops: list, topology: str):
    return _block(n, ops), _device(topology, n)


def compile_circuit(n: int, prepared):
    import qarpx as qx
    from qarp.devices import compile_for_device

    block, device = prepared
    router = getattr(qx.RouterKind, ROUTER)
    return compile_for_device(block, device, router=router)


def to_ops(n: int, compiled) -> list:
    out = []
    for command in compiled.commands:
        name = _NAMES.get(str(command.gate).split(".")[-1].lower())
        if name is None:
            raise ValueError(f"gate outside the target basis: {command.gate}")
        qubits = list(command.qubits)
        params = [float(p.evaluate({})) for p in command.params]
        if name in architectures.TWO_QUBIT:
            out.append((name, qubits[0], qubits[1]))
        elif name == "h":
            out.append(("h", qubits[0]))
        else:
            out.append((name, qubits[0], params[0]))
    return out


def layout_of(compiled) -> list:
    return list(compiled.final_logical_to_physical)


def initial_layout_of(compiled) -> list:
    """The router's placement before the first gate (§14)."""
    return list(compiled.initial_logical_to_physical)
