"""Capability coverage: which stack ships which capability, resolved from live APIs.

NB: this file must not be named coverage.py -- that shadows the installed
`coverage` package for every dependency imported below, and the failures surface
as silent "--" cells rather than as errors.

Companion to run.py.  run.py measures how many lines an algorithm costs; this
measures whether a stack offers it at all.  Every cell is decided by importing a
candidate symbol, never by assertion -- the candidates are in this file, so a
wrong one is a visible bug rather than an invisible claim.

A "-" means *not found at any probed path*.  That is weaker than "does not
exist": it is an invitation to add the path and re-run.

    python benchmarks/code_volume/capabilities.py
"""

import importlib
import sys

STACKS = ("qarp", "qulacs", "cirq", "qiskit_raw", "pennylane", "qiskit_nature")

# capability -> stack -> candidate "module:attribute" paths (empty list = none known)
CAPABILITIES = {
    "noise model": {
        "qarp": ["qarp.devices:NoiseModel"],
        "qulacs": ["qulacs.gate:DepolarizingNoise"],
        "cirq": ["cirq:depolarize"],
        "qiskit_raw": ["qiskit_aer.noise:NoiseModel"],
        "pennylane": ["pennylane:DepolarizingChannel"],
        "qiskit_nature": ["qiskit_aer.noise:NoiseModel"],
    },
    "mid-circuit measurement": {
        "qarp": ["qarp.blocks:MeasureBlock"],
        "qulacs": ["qulacs.gate:Measurement"],
        "cirq": ["cirq:measure"],
        "qiskit_raw": ["qiskit.circuit:Measure"],
        "pennylane": ["pennylane:measure"],
        "qiskit_nature": ["qiskit.circuit:Measure"],
    },
    "device transpilation / routing": {
        "qarp": ["qarp.devices:compile_for_device"],
        "qulacs": [],
        "cirq": ["cirq:optimize_for_target_gateset"],
        "qiskit_raw": ["qiskit:transpile"],
        "pennylane": ["pennylane.transforms:transpile"],
        "qiskit_nature": ["qiskit:transpile"],
    },
    "circuit cutting": {
        "qarp": ["qarp.cutting:AutoCutFinder"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": ["qiskit_addon_cutting:partition_problem"],
        "pennylane": ["pennylane.qcut:cut_circuit"],
        "qiskit_nature": ["qiskit_addon_cutting:partition_problem"],
    },
    "fermion->qubit mappings": {
        "qarp": ["qarp.operators:BravyiKitaev"],
        "qulacs": [],
        "cirq": ["openfermion:bravyi_kitaev"],
        "qiskit_raw": ["openfermion:bravyi_kitaev"],
        "pennylane": ["pennylane:bravyi_kitaev"],
        "qiskit_nature": ["qiskit_nature.second_q.mappers:BravyiKitaevMapper"],
    },
    "classical shadows": {
        "qarp": ["qarp.algorithms:ShadowEstimator"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:ClassicalShadow"],
        "qiskit_nature": [],
    },
    "QSP / QSVT": {
        "qarp": ["qarp.blocks:QSVTBlock"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:qsvt"],
        "qiskit_nature": [],
    },
    "resource estimation": {
        "qarp": ["qarp.resources:ResourceEstimator"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:specs"],
        "qiskit_nature": [],
    },
    "VQE": {
        "qarp": ["qarp.algorithms:VQE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": ["qiskit_algorithms:VQE"],
    },
    "QAOA": {
        "qarp": ["qarp.algorithms:QAOA"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane.qaoa:maxcut"],
        "qiskit_nature": ["qiskit_algorithms:QAOA"],
    },
    "QPE": {
        "qarp": ["qarp.algorithms:QPE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:QuantumPhaseEstimation"],
        "qiskit_nature": ["qiskit_algorithms:PhaseEstimation"],
    },
    "ADAPT-VQE": {
        "qarp": ["qarp.algorithms:AdaptVQE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:AdaptiveOptimizer"],
        "qiskit_nature": ["qiskit_algorithms:AdaptVQE"],
    },
    "VQD (excited states)": {
        "qarp": ["qarp.algorithms:VQD"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": ["qiskit_algorithms:VQD"],
    },
    "SS-VQE": {
        "qarp": ["qarp.algorithms:SSVQE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": [],
    },
    "quantum subspace expansion": {
        "qarp": ["qarp.algorithms:QSE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": [],
    },
    "imaginary-time evolution": {
        "qarp": ["qarp.algorithms:QITE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": ["qiskit_algorithms:ImaginaryTimeEvolver"],
    },
    "density-of-states QPE": {
        "qarp": ["qarp.algorithms:DOSQPE"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": [],
    },
    "analytic gradients": {
        "qarp": ["qarp.engines:QarpEngine.run_gradient"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:grad"],
        "qiskit_nature": ["qiskit_algorithms.gradients:ParamShiftEstimatorGradient"],
    },
    "autodiff / ML-framework interface": {
        "qarp": [],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": ["pennylane:QNode"],
        "qiskit_nature": [],
    },
    "pulse-level control": {
        "qarp": [],
        "qulacs": [],
        "cirq": ["cirq:Duration"],
        "qiskit_raw": ["qiskit.pulse:Schedule"],
        "pennylane": ["pennylane.pulse:ParametrizedHamiltonian"],
        "qiskit_nature": ["qiskit.pulse:Schedule"],
    },
    "tensor-network simulation": {
        "qarp": ["qarp.operators:VUMPO"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": ["qiskit_aer:AerSimulator"],
        "pennylane": ["pennylane.devices.default_tensor:DefaultTensor"],
        "qiskit_nature": ["qiskit_aer:AerSimulator"],
    },
    "QMEGS / MMQCELS": {
        "qarp": ["qarp.algorithms:QMEGS", "qarp.algorithms:MMQCELS"],
        "qulacs": [],
        "cirq": [],
        "qiskit_raw": [],
        "pennylane": [],
        "qiskit_nature": [],
    },
}


def resolve(path):
    """True when `module:attribute` imports and the attribute exists."""
    module_name, _, attribute = path.partition(":")
    try:
        target = importlib.import_module(module_name)
    except Exception:
        return False
    for part in filter(None, attribute.split(".")):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def found(capability, stack):
    """The first candidate path that resolves, or None."""
    return next((p for p in CAPABILITIES[capability][stack] if resolve(p)), None)


def table():
    header = ["capability", *STACKS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for capability in CAPABILITIES:
        cells = [capability]
        for stack in STACKS:
            cells.append("yes" if found(capability, stack) else "--")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    print(table())
    print()
    print("Resolved symbols:")
    for capability in CAPABILITIES:
        for stack in STACKS:
            path = found(capability, stack)
            if path:
                print(f"  {capability:32s} {stack:15s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
