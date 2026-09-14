import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import CVQRAMStateBlock


def test_state_preparation():
    # CV-QRAM layout: q_u0[0], q_u1[1], anc[2..n], q_mem[n+1..2n]
    # After the circuit, all ancilla/q_u are |0>, state is in q_mem.
    # Index for a memory bitstring b: sum(b[j] << (n+1+j) for j in range(n))

    set_datasets = [
        {
            (0, 0, 0): np.sqrt(0.5),
            (1, 0, 0): np.sqrt(0.5),
        },
        {
            (0, 1, 0): np.sqrt(0.25),
            (1, 0, 1): np.sqrt(0.25),
            (1, 1, 0): np.sqrt(0.25),
            (0, 0, 0): np.sqrt(0.25),
        },
        {
            (1, 1, 1): np.sqrt(1 / 3),
            (0, 0, 0): np.sqrt(1 / 3),
            (1, 0, 0): np.sqrt(1 / 3),
        },
    ]

    sim = qx.QarpSimulator()
    transpiler = qx.Transpiler(qx.qulacs_gateset())

    for dataset in set_datasets:
        block = CVQRAMStateBlock(dataset)
        block.build()
        n = block._data_n_qubits

        compiled = transpiler.transpile_and_optimize(block.flatten())
        sv = sim.statevector(compiled, block.n_qubits)

        for bitstring, amplitude in dataset.items():
            idx = sum(b << (n + 1 + j) for j, b in enumerate(bitstring))
            prob = abs(sv[idx]) ** 2
            assert abs(prob - amplitude**2) < 1e-6, (
                f"bitstring={bitstring}: expected prob={amplitude**2:.6f}, got {prob:.6f}"
            )


# The normalisation check must be two-sided (10.1): an under-normalized dataset
# passed the old one-sided check and left the missing mass on the ancillas,
# contradicting ancilla_postselection=None.
def test_under_normalized_dataset_is_rejected():
    with pytest.raises(ValueError, match="equal to 1"):
        CVQRAMStateBlock({(1, 0, 0): 0.5, (0, 1, 0): 0.5})


def test_over_normalized_dataset_is_rejected():
    with pytest.raises(ValueError, match="equal to 1"):
        CVQRAMStateBlock({(1, 0, 0): 0.8, (0, 1, 0): 0.8})


def test_normalized_dataset_with_negative_amplitude_builds():
    dataset = {(1, 0, 0): 0.6, (0, 1, 0): -0.8}
    block = CVQRAMStateBlock(dataset)
    block.build()
    n = block._data_n_qubits
    sv = qx.QarpSimulator().statevector(block.flatten(), block.n_qubits)
    for bitstring, amplitude in dataset.items():
        idx = sum(b << (n + 1 + j) for j, b in enumerate(bitstring))
        assert abs(sv[idx]) ** 2 == pytest.approx(amplitude**2, abs=1e-10)
