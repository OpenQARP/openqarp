import numpy as np

from .._block import SimpleBlock
from .._prepares_known_state import prepares_known_state
from .cv_qram_state_block import CVQRAMStateBlock


@prepares_known_state
class CVOQRAMStateBlock(SimpleBlock):
    def __init__(
        self,
        dataset: dict[tuple, float],
        target_qubits=None,
        name: str = "CVOQRAM",
    ):
        """An object for constructing CVO-QRAM state preparation method [1]. By using state preparation
        methods we are able to load a classical dataset into a quantum state and initialize a given
        wavefunction. This method is an optimized version of CV-QRAM in which the number of 2-qubit gates
        are drastically reduced. Preparation of an n-qubit state requires additional n ancilla qubits (2n qubits in total).

        [1] de Veras, T. M., da Silva, L. D., & da Silva, A. J. (2022). Double sparse quantum
            state preparation. Quantum Information Processing, 21(6), 204.

        Args:
            dataset: a dictionary specifying the data to be loaded with tuples as keys and amplitudes as values.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.dataset = dataset
        self._data_n_qubits = self._verify_dataset(self.dataset)
        n = self._data_n_qubits
        total_qubits = 2 * n
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
        n = self._data_n_qubits
        n_instances = len(self.dataset)
        # Layout: anc[0..n-2], anc_u[n-1], q_mem[n..2n-1]  → 2n total
        q_anc = list(range(0, n - 1))  # indices 0..n-2
        q_u = n - 1  # index n-1
        q_mem = list(range(n, 2 * n))  # indices n..2n-1

        self.x(q_u)
        norm = 1.0

        for idx_instance, (instance, amplitude) in enumerate(sorted(self.dataset.items())):
            ones = [i for i in range(len(instance)) if instance[i] == 1]
            t = len(ones)

            for one in ones:
                self.cx(q_u, q_mem[one])

            alpha, beta, phi = CVQRAMStateBlock._calculate_angles(norm, amplitude)

            if t > 1:
                # Nielsen & Chuang p.184 multi-controlled unitary
                self.ccx(q_mem[ones[0]], q_mem[ones[1]], q_anc[0])
                for j in range(2, t):
                    self.ccx(q_mem[ones[j]], q_anc[j - 2], q_anc[j - 1])
                self.cu(q_anc[t - 2], q_u, alpha, beta, phi, 0.0)
                for j in range(t - 1, 2 - 1, -1):
                    self.ccx(q_mem[ones[j]], q_anc[j - 2], q_anc[j - 1])
                self.ccx(q_mem[ones[0]], q_mem[ones[1]], q_anc[0])
            elif t == 1:
                self.cu(q_mem[ones[0]], q_u, alpha, beta, phi, 0.0)
            elif t == 0:
                # U3(theta, phi, lambda) = Rz(lambda) Ry(theta) Rz(phi)
                self.rz(q_u, phi)
                self.ry(q_u, alpha)
                self.rz(q_u, beta)

            if idx_instance < n_instances:
                for one in ones:
                    self.cx(q_u, q_mem[one])

            norm -= abs(amplitude**2)

    @property
    def state_qubits(self) -> tuple[int, ...]:
        """The memory register — the ancillas carry no part of the state."""
        return tuple(range(self._data_n_qubits, 2 * self._data_n_qubits))

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
