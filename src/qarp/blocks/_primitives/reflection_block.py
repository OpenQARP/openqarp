import math

from .._block import SimpleBlock


class ReflectionBlock(SimpleBlock):
    r"""Reflection about ``|0…0⟩``: ``2|0⟩⟨0| - I`` on n qubits.

    Built as ``-1 · X^n · MCZ · X^n``: the X-sandwich turns the MCZ's
    "phase −1 on ``|11…1⟩``" into "phase −1 on ``|00…0⟩``", and the global
    GPhase(π) flips the overall sign so the ``|0⟩`` subspace gets +1 (and
    everything else gets −1 — i.e. ``2|0⟩⟨0| − I``).

    Args:
        n_qubits: number of qubits of the circuit block
        name: name of the block
    """

    def __init__(self, n_qubits: int, name: str = "Reflection"):
        super().__init__(n_qubits, name=name)

    def build_vanilla(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("ReflectionBlock requires n_qubits >= 1")
        if self.n_qubits == 1:
            # 2|0⟩⟨0| − I = diag(+1, −1) = Z.  MCZ requires ≥2 qubits, so
            # short-circuit the X-sandwich and emit Z directly.
            self.z(0)
            return
        # Global phase of −1 (= e^{iπ}) flips the sign on the whole register.
        self.gphase(math.pi)
        all_q = list(range(self.n_qubits))
        # X-sandwich the MCZ — bulk emit each X layer.
        self.x(all_q)
        # MCZ takes [c0, …, c_{n-1}, t]: the last qubit is the target,
        # all preceding are controls.  For the symmetric "phase -1 on |11…1⟩"
        # interpretation we just pass all qubits.
        self.mcz(all_q)
        self.x(all_q)
