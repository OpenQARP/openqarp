import numpy as np

from .._block import SimpleBlock
from .._prepares_known_state import prepares_known_state


@prepares_known_state
class CVQRAMStateBlock(SimpleBlock):
    def __init__(
        self,
        dataset: dict[tuple, float],
        target_qubits=None,
        name: str = "CVQRAM",
    ):
        """An object for constructing CV-QRAM state preparation method [1]. By using state preparation
        methods we are able to load a classical dataset into a quantum state and initialize a given
        wavefunction. Please, check CVO_QRAM for an optimized version of this method.
        Preparation of an n-qubit state requires additional n+1 ancilla qubits (2n+1 qubits in total).

        [1] de Veras, T. M., da Silva, L. D., & da Silva, A. J. (2022). Double sparse quantum
            state preparation. Quantum Information Processing, 21(6), 204.

        Args:
            dataset: a dictionary specifying the data to be loaded with tuples as keys and amplitudes as values.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.dataset = dataset
        self._data_n_qubits = self._verify_dataset(self.dataset)
        n = self._data_n_qubits
        total_qubits = 2 * n + 1
        if target_qubits is None:
            target_qubits = list(range(total_qubits))
        super().__init__(
            n_qubits=total_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        if len(target_qubits) < total_qubits:
            raise ValueError("Not enough qubits for this method.")

    def _verify_dataset(self, dataset: dict[tuple, float]) -> int:
        # Two-sided: an under-normalized dataset would leave the ancillas away
        # from |0> with the missing mass, contradicting ancilla_postselection=None.
        if abs(sum(abs(amp) ** 2 for amp in dataset.values()) - 1.0) > 1e-8:
            raise ValueError("The sum of squared amplitude moduli should be equal to 1")
        size_tuples = [len(i) for i, _ in dataset.items()]
        if len(set(size_tuples)) != 1:
            raise ValueError("All the tuples in the dataset should have the same size")
        n_qubits = size_tuples[0]
        return n_qubits

    def build_vanilla(self):
        assert self.dataset, "No dataset was provided to the state preparation object."
        n = self._data_n_qubits
        # Layout: anc_u0[0], anc_u1[1], anc[2..n], q_mem[n+1..2n]  → 2n+1 total
        q_u0 = 0
        q_u1 = 1
        q_anc = list(range(2, n + 1))  # indices 2..n  (n-1 ancillas)
        q_mem = list(range(n + 1, 2 * n + 1))  # indices n+1..2n

        self.x(q_u1)
        norm = 1.0
        for instance, amplitude in self.dataset.items():
            _load_binary(self, instance, q_u1, q_mem)
            _load_amplitude_cn_u(self, amplitude, norm, q_u0, q_u1, q_anc, q_mem, n)
            _load_binary(self, instance, q_u1, q_mem)
            norm -= abs(amplitude**2)

    @staticmethod
    def _calculate_angles(norm, amplitude):
        alpha, beta, phi = 0, 0, 0
        phase = abs(amplitude**2)
        if (norm - phase) < 0:
            norm = phase
        cos_value = np.sqrt((norm - phase) / norm)
        if cos_value > 1:
            cos_value = 1
        elif cos_value < -1:
            cos_value = -1
        alpha = 2 * np.arccos(cos_value)
        beta = np.arccos(-amplitude.real / np.sqrt(abs(amplitude**2)))
        if amplitude.imag < 0:
            beta = 2 * np.pi - beta
        phi = -beta
        return alpha, beta, phi

    @property
    def state_qubits(self) -> tuple[int, ...]:
        """The memory register — the ancillas carry no part of the state."""
        return tuple(range(self._data_n_qubits + 1, 2 * self._data_n_qubits + 1))

    def target_statevector(self) -> np.ndarray:
        """The dataset amplitudes, indexed LSB over the memory register.

        The ancillas are restored to ``|0⟩`` deterministically, so
        ``ancilla_postselection`` stays ``None`` and the block is control-safe
        — provided the caller reads ``state_qubits`` rather than ``n_qubits``.
        ``validate_amplification_blocks`` is the documented consumer that still
        reads ``n_qubits`` (a deferred contract limit), so this block does
        not yet fit amplitude amplification without padding the oracle.
        """
        psi = np.zeros(2**self._data_n_qubits, dtype=complex)
        for instance, amplitude in self.dataset.items():
            psi[sum(int(bit) << i for i, bit in enumerate(instance))] = amplitude
        return psi


def _load_binary(block, instance, q_u1_idx: int, q_mem) -> None:
    for q_idx, q_bin in enumerate(instance):
        if q_bin == 1:
            block.cx(q_u1_idx, q_mem[q_idx])
        else:
            block.x(q_mem[q_idx])


def _load_amplitude_cn_u(
    block, amplitude, norm, q_u0_idx: int, q_u1_idx: int, q_anc, q_mem, n: int
) -> None:
    alpha, beta, phi = CVQRAMStateBlock._calculate_angles(norm, complex(amplitude))
    # Nielsen & Chuang p.184 multi-controlled unitary
    block.ccx(q_mem[0], q_mem[1], q_anc[0])
    for j in range(2, n):
        block.ccx(q_mem[j], q_anc[j - 2], q_anc[j - 1])
    block.cx(q_anc[n - 2], q_u0_idx)
    block.cu(q_u0_idx, q_u1_idx, alpha, beta, phi, 0.0)
    block.cx(q_anc[n - 2], q_u0_idx)
    for j in range(n - 1, 2 - 1, -1):
        block.ccx(q_mem[j], q_anc[j - 2], q_anc[j - 1])
    block.ccx(q_mem[0], q_mem[1], q_anc[0])
