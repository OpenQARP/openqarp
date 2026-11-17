from typing import Dict, List, Optional, Union

import numpy as np

from .._block import SimpleBlock


def _coefficient_phase(coefficient: complex, tol: float = 1e-10) -> float:
    """The angle ``arg(c)`` a PauliBlock carries in its circuit as a GPhase gate.

    A coefficient splits as ``c = |c| * exp(i*arg(c))``; only the unit-modulus
    factor ``exp(i*arg(c))`` is representable by a unitary.  Guards float noise on
    the angle, and folds angles within ``tol`` of ``-pi`` (numpy's result for a
    negative real whose imaginary part is ``-0.0``, e.g. from
    ``hermitian_conjugated``) onto ``+pi``, so two blocks for the same operator
    store the same value and compare equal.

    The magnitude is tested against exact zero, not a tolerance: ``arg`` is
    scale-free, so a small coefficient still has a well-defined direction, and
    under ``ControlledBlock`` — which keeps the direction and discards ``|c|`` —
    dropping it would be an O(1) error.  Only ``-0.0`` needs the guard at all
    (``np.angle`` returns ``pi`` for it, which would put a ``GPhase(pi)`` on the
    zero operator).  A coefficient that is genuinely rounding dust therefore
    carries its noise angle; deciding that requires the other terms' scale, which
    a single block cannot see.
    """
    if coefficient == 0:  # the zero operator (incl. -0.0), not a phase
        return 0.0
    angle = float(np.angle(coefficient))
    if angle <= -np.pi + tol:  # -pi and +pi are the same turn; always store +pi
        angle += 2 * np.pi
    return angle if abs(angle) > tol else 0.0


class PauliBlock(SimpleBlock):
    def __init__(
        self,
        pauli_string: Union[Dict[int, str], str],
        coefficient: Optional[complex] = None,
        phase: Optional[float] = None,
        n_qubits: Optional[int] = None,
        change_basis: bool = False,
        measure: bool = False,
        target_qubits: Optional[List[int]] = None,
        name: str = "Pauli",
    ):
        """A Block representing a single Pauli string from a Hamiltonian.

        Args:
            pauli_string: Either a dictionary mapping qubit indices to Pauli operators ('X', 'Y', 'Z')
                        or a string of Pauli operators (e.g., "XYZ")
            coefficient: Complex coefficient for the Pauli string (cannot be used with phase)
            phase: Global phase for the circuit in radians (cannot be used with coefficient)
            n_qubits: Total number of qubits in the system (only needed for dict format)
            change_basis: If True, applies basis change gates for measurement (X->H, Y->SdgH, Z->Identity)
            measure: If True, adds measurements at the end of the circuit
            target_qubits: The target qubits the underlying block will act on when added to a circuit
            name: Name for the block

        Raises:
            ValueError: If both coefficient and phase are provided, or if invalid inputs are given

        Note:
            When built from ``coefficient=`` (and ``change_basis=False``), a
            coefficient splits as ``c = |c| * exp(i*arg(c))``.  The circuit
            carries the unit-modulus factor ``exp(i*arg(c))`` as a ``GPhase``
            gate when non-zero, readable as ``self.phase``; the magnitude ``|c|``
            is not representable by a unitary and must be applied classically by
            the consumer.  ``self.coefficient`` keeps the full complex value.
            Under ``ControlledBlock`` this overall phase becomes a relative
            phase, so it is physically observable there, and the angle is kept
            for any non-zero ``c`` however small — only an exactly zero
            coefficient carries none.  (On the ``phase=`` branch
            ``self.coefficient`` is ``None``; both attributes reflect
            construction only and are not updated by ``dagger()``.)
        """
        self._validate_inputs(pauli_string, coefficient, phase, n_qubits)

        # Compute n_qubits before super().__init__() so SimpleBlock allocates the right size.
        if isinstance(pauli_string, dict):
            computed_n_qubits = n_qubits or (max(pauli_string.keys()) + 1 if pauli_string else 1)
        else:
            computed_n_qubits = len(pauli_string)

        super().__init__(
            n_qubits=computed_n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

        self.pauli_string: Union[Dict[int, str], str] = pauli_string
        self.string_format = isinstance(pauli_string, str)

        self.change_basis = change_basis
        # `measure_at_end` is the bool flag — distinct from `self.measure(q, c)`,
        # which is the inherited gate-builder method.
        self.measure_at_end = measure

        if coefficient is not None:
            self.coefficient: Optional[complex] = coefficient
            # A change_basis circuit is a measurement-basis rotation, not
            # e^{i arg c}*P, so a coefficient phase has no meaning on it: the
            # contract is scoped to change_basis=False.
            self.phase = 0.0 if change_basis else _coefficient_phase(coefficient)
        elif phase is not None:
            self.coefficient = None
            self.phase = phase
        else:
            self.coefficient = 1
            self.phase = 0.0

    def _validate_inputs(
        self,
        pauli_string: Union[Dict[int, str], str],
        coefficient: Optional[complex],
        phase: Optional[float],
        n_qubits: Optional[int],
    ) -> None:
        """Validate all input parameters."""
        if coefficient is not None and phase is not None:
            raise ValueError("Cannot specify both coefficient and phase. Use one or the other.")

        if not isinstance(pauli_string, (dict, str)):
            raise ValueError("pauli_string must be either a dictionary or a string")

        valid_paulis = {"I", "X", "Y", "Z"}

        if isinstance(pauli_string, str):
            if not pauli_string:
                raise ValueError("pauli_string cannot be an empty string")

            invalid_paulis = set(pauli_string) - valid_paulis
            if invalid_paulis:
                raise ValueError(f"Invalid Pauli operators: {invalid_paulis}. Use I, X, Y, Z.")

            if n_qubits is not None and n_qubits != len(pauli_string):
                raise ValueError(
                    f"n_qubits ({n_qubits}) must match string length ({len(pauli_string)})"
                )

        elif isinstance(pauli_string, dict):
            if not pauli_string and n_qubits is None:
                raise ValueError("For empty pauli_string dict, n_qubits must be specified")

            invalid_paulis = set(pauli_string.values()) - valid_paulis
            if invalid_paulis:
                raise ValueError(f"Invalid Pauli operators: {invalid_paulis}. Use I, X, Y, Z.")

            if any(not isinstance(idx, int) or idx < 0 for idx in pauli_string.keys()):
                raise ValueError("Qubit indices must be non-negative integers")

    def build_vanilla(self) -> None:
        if isinstance(self.pauli_string, str) and self.string_format:
            for i, pauli_op in enumerate(self.pauli_string):
                self._add_pauli_operation(i, pauli_op)
        elif isinstance(self.pauli_string, dict):
            for qubit_idx, pauli_op in self.pauli_string.items():
                self._add_pauli_operation(qubit_idx, pauli_op)

        if self.phase != 0.0:
            self.gphase(self.phase)

        if self.measure_at_end:
            for q in range(self.n_qubits):
                self.measure(q, q)

    def _add_pauli_operation(self, qubit_idx: int, pauli_op: str) -> None:
        """Add the appropriate Pauli operation to the circuit."""
        if self.change_basis:
            if pauli_op == "X":
                self.h(qubit_idx)
            elif pauli_op == "Y":
                self.sdg(qubit_idx)
                self.h(qubit_idx)
            # Z and I: no basis-change gates needed
        else:
            if pauli_op == "X":
                self.x(qubit_idx)
            elif pauli_op == "Y":
                self.y(qubit_idx)
            elif pauli_op == "Z":
                self.z(qubit_idx)
            # I: no gate
