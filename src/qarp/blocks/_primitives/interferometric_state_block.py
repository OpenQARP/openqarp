"""Blocks for preparing and measuring an interferometric transition state.

``InterferometricStateBlock`` prepares the measurement-free state

    (|0> |phi> + |1> |psi>) / sqrt(2),

with the requested ancilla quadrature encoded in the transition circuit.  The
separate ``InterferometricMeasurementBlock`` adds local basis rotations and
terminal measurements for a qubit-wise commuting Pauli group.  Keeping these
two concerns separate makes the state reusable for other primitives (for
example, custom observables, amplitude estimation, or tomography) while
retaining the grouped transition-measurement convenience wrapper.
"""

from typing import Dict, Optional

import qarpx as qx

from .._block import AnyBlock, CompositeBlockBase, SimpleBlock
from .hadamard_test_block import HadamardTestBlock
from .identity_block import IdentityBlock


def _validate_states(bra: AnyBlock, ket: AnyBlock) -> None:
    """Validate the state-preparation blocks shared by both public classes."""
    if not isinstance(bra, qx.Block):
        raise TypeError("bra must be a Block instance")
    if not isinstance(ket, qx.Block):
        raise TypeError("ket must be a Block instance")
    if bra.n_qubits != ket.n_qubits:
        raise ValueError(
            f"bra ({bra.n_qubits} qubits) and ket ({ket.n_qubits} qubits) "
            "must act on the same number of qubits"
        )


class InterferometricStateBlock(CompositeBlockBase):
    """Prepare a measurement-free interferometric transition state.

    Qubit 0 is the interferometric ancilla and qubits ``1..n`` are the state
    register.  The branch synthesis is delegated to :class:`HadamardTestBlock`,
    including its branch-sharing optimization for similar state-preparation
    circuits.  No basis rotations or measurements are appended, so the block
    can be embedded in a larger circuit or measured by a caller-specific
    primitive.
    """

    def __init__(
        self,
        bra: AnyBlock,
        ket: AnyBlock,
        estimate_imaginary: bool = False,
        name: str = "InterferometricState",
    ):
        _validate_states(bra, ket)

        super().__init__(n_qubits=1 + ket.n_qubits, name=name)
        self.bra = bra
        self.ket = ket
        self.estimate_imaginary = estimate_imaginary

    def build_vanilla(self) -> None:
        full_qubits = list(range(self.n_qubits))

        # Leave all qubits, including the ancilla, unmeasured.  The selected
        # quadrature is prepared by HadamardTestBlock itself.
        transition = HadamardTestBlock(
            state=IdentityBlock(self.ket.n_qubits),
            unitary=self.ket,
            unitary_dagger=self.bra,
            estimate_imaginary=self.estimate_imaginary,
            measure=False,
            name="TransitionInterferometer",
        )
        transition.target_qubits = full_qubits
        self.add_wired_child(transition)

    def __repr__(self) -> str:
        part = "imaginary" if self.estimate_imaginary else "real"
        return f"InterferometricStateBlock(estimate_{part}=True)"


class InterferometricMeasurementBlock(CompositeBlockBase):
    """Measure an interferometric transition state in a QWC Pauli basis.

    This wrapper first prepares :class:`InterferometricStateBlock`, then
    rotates state-register qubits into the requested local ``X``, ``Y`` or
    ``Z`` basis and measures the ancilla and all data qubits.  ``basis`` maps
    state-register qubits (indexed from zero, excluding the ancilla) to Pauli
    letters; omitted qubits remain in the computational Z basis.
    """

    def __init__(
        self,
        bra: AnyBlock,
        ket: AnyBlock,
        basis: Optional[Dict[int, str]] = None,
        estimate_imaginary: bool = False,
        name: str = "InterferometricMeasurement",
    ):
        _validate_states(bra, ket)

        n_state = ket.n_qubits
        normalized_basis = dict(basis or {})
        for qubit, pauli in normalized_basis.items():
            if not isinstance(qubit, int) or qubit < 0 or qubit >= n_state:
                raise ValueError(f"basis qubit {qubit!r} is outside the state register")
            if pauli not in {"X", "Y", "Z"}:
                raise ValueError(f"invalid QWC basis {pauli!r}; expected 'X', 'Y' or 'Z'")

        super().__init__(n_qubits=1 + n_state, name=name)
        self.bra = bra
        self.ket = ket
        self.basis = normalized_basis
        self.estimate_imaginary = estimate_imaginary

    def build_vanilla(self) -> None:
        full_qubits = list(range(self.n_qubits))

        state = InterferometricStateBlock(
            bra=self.bra,
            ket=self.ket,
            estimate_imaginary=self.estimate_imaginary,
        )
        state.target_qubits = full_qubits
        self.add_wired_child(state)

        # QWC groups need only local basis changes: X -> H and
        # Y -> Sdg followed by H.  The state-register offset is one because
        # the ancilla occupies qubit 0.
        basis_change = SimpleBlock(self.n_qubits, name="QWCBasisChange")
        for qubit, pauli in sorted(self.basis.items()):
            state_qubit = qubit + 1
            if pauli == "X":
                basis_change.h(state_qubit)
            elif pauli == "Y":
                basis_change.sdg(state_qubit)
                basis_change.h(state_qubit)
        basis_change.target_qubits = full_qubits
        self.add_wired_child(basis_change)

        # Measure ancilla into cbit 0 and data qubit q+1 into cbit q+1.  Thus
        # the engine's integer outcome has the layout expected by the grouped
        # transition post-processor: ancilla is bit 0 and data is outcome >> 1.
        measure = SimpleBlock(self.n_qubits, name="TransitionMeasure")
        measure.measure([(qubit, qubit) for qubit in full_qubits])
        measure.target_qubits = full_qubits
        self.add_wired_child(measure)

        self.n_cbits = self.n_qubits

    def __repr__(self) -> str:
        part = "imaginary" if self.estimate_imaginary else "real"
        return f"InterferometricMeasurementBlock(estimate_{part}=True, basis={self.basis!r})"
