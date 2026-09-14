from typing import List, Optional, Union

from sympy import Symbol

import qarpx as qx

from .._block import SimpleBlock, as_param

# GateType → (Block method name, takes_params)
# Only gates covered by the Block builder API are listed.
_GATE_BUILDER = {
    qx.GateType.H: ("h", False),
    qx.GateType.X: ("x", False),
    qx.GateType.Y: ("y", False),
    qx.GateType.Z: ("z", False),
    qx.GateType.S: ("s", False),
    qx.GateType.Sdg: ("sdg", False),
    qx.GateType.T: ("t", False),
    qx.GateType.Tdg: ("tdg", False),
    qx.GateType.Rx: ("rx", True),
    qx.GateType.Ry: ("ry", True),
    qx.GateType.Rz: ("rz", True),
    qx.GateType.P: ("p", True),
    qx.GateType.U: ("u", True),
    qx.GateType.CX: ("cx", False),
    qx.GateType.CY: ("cy", False),
    qx.GateType.CZ: ("cz", False),
    qx.GateType.SWAP: ("swap", False),
    qx.GateType.CCX: ("ccx", False),
    qx.GateType.RZZ: ("rzz", True),
}


_to_param = as_param  # the single §13 coercion; linear expressions now accepted here too


class LayerBlock(SimpleBlock):
    def __init__(
        self,
        gate_type: qx.GateType,
        n_qubits: int,
        qubit_indices: Optional[List[int]] = None,
        overlapping: int = 0,
        periodic_boundary: bool = False,
        parameters: Optional[List[Union[float, Symbol]]] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "LayerBlock",
    ):
        """Constructs a layer of gates with a specified GateType.

        Args:
            gate_type: The qarpx GateType for the gates in this layer.
            n_qubits: The number of qubits to include.
            qubit_indices: Optional list of specific qubit indices to apply gates to.
            overlapping: The number of qubits that overlap between consecutive gates (default 0).
            periodic_boundary: Whether to use periodic boundary conditions (default False).
            parameters: Parameters for parametric gates. Either one set of params reused for
                        all gates, or a flattened list of params for each gate individually.
                        Angles are in radians.
            target_qubits: The target qubits this block acts on when added to a larger block.
            name: The name of the block.
        """
        super().__init__(n_qubits, target_qubits, name=name)

        self.gate_type = gate_type
        if qubit_indices is None:
            qubit_indices = list(range(self.n_qubits))
        self.qubit_indices = qubit_indices
        self.overlapping = overlapping
        self.periodic_boundary = periodic_boundary
        self.parameters = parameters if parameters is not None else []
        self.gate_arity = qx.gate_num_qubits(gate_type)
        self.params_per_gate = qx.gate_num_params(gate_type)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not self.qubit_indices:
            raise ValueError("qubit_indices cannot be empty")
        if any(q >= self.n_qubits for q in self.qubit_indices):
            raise ValueError("All qubit indices must be less than n_qubits")
        if self.periodic_boundary and len(self.qubit_indices) % self.gate_arity > self.overlapping:
            raise ValueError(
                "Periodic boundaries are not compatible with the gate arity and the overlapping"
            )

        max_overlap = self.gate_arity - 1
        if (
            not isinstance(self.overlapping, int)
            or self.overlapping < 0
            or self.overlapping > max_overlap
        ):
            raise ValueError(
                f"overlapping must be an integer between 0 and {max_overlap} "
                f"for {self.gate_arity}-qubit gates"
            )

        if self.params_per_gate > 0:
            if not self.parameters:
                raise ValueError(
                    f"Parameters are required for {self.gate_type}. "
                    f"Expected either {self.params_per_gate} parameter(s) for all gates "
                    f"or {self._count_gates() * self.params_per_gate} parameter(s) total."
                )

            n_gates = self._count_gates()
            if (
                len(self.parameters) != self.params_per_gate
                and len(self.parameters) != n_gates * self.params_per_gate
            ):
                raise ValueError(
                    f"Number of parameters must be either {self.params_per_gate} "
                    f"(same params for all {n_gates} gates) "
                    f"or {n_gates * self.params_per_gate} (individual params per gate), "
                    f"but got {len(self.parameters)}"
                )

        if self.gate_type not in _GATE_BUILDER:
            raise ValueError(
                f"Unsupported gate type: {self.gate_type}. "
                f"Supported gates: {list(_GATE_BUILDER.keys())}"
            )

    def _count_gates(self) -> int:
        """Count the number of gates that will be applied in the layer."""
        indices = self.qubit_indices.copy()
        gate_count = 0

        if self.gate_arity == 1:
            gate_count = len([q for q in indices if q < self.n_qubits])

        elif self.gate_arity == 2:
            if self.periodic_boundary:
                if len(indices) % 2 == 1 or self.overlapping == 1:
                    indices.append(indices[0])

            step = self.gate_arity - self.overlapping
            for i in range(0, len(indices) - 1, step):
                if i + 1 < len(indices):
                    if indices[i] < self.n_qubits and indices[i + 1] < self.n_qubits:
                        gate_count += 1

        elif self.gate_arity == 3:
            if self.periodic_boundary:
                remainder = len(indices) % 3
                if remainder == 1 or self.overlapping == 2:
                    indices.extend([indices[0], indices[1]])
                elif remainder == 2 or self.overlapping == 1:
                    indices.append(indices[0])

            step = self.gate_arity - self.overlapping
            for i in range(0, len(indices) - 2, step):
                if i + 2 < len(indices):
                    if all(q < self.n_qubits for q in indices[i : i + 3]):
                        gate_count += 1

        return gate_count

    def _update_parameters_attr(self, symbol_map) -> None:
        # ``parameters`` mirrors constructor input; remap it alongside the
        # C++ command-buffer transform (the base hook is a no-op).
        if self.parameters:
            self.parameters = [symbol_map.get(p, p) for p in self.parameters]

    def _get_gate_params(self, gate_index: int) -> list:
        """Return the parameters for a specific gate, converted to qx.Param."""
        if not self.parameters or self.params_per_gate == 0:
            return []

        if len(self.parameters) == self.params_per_gate:
            raw = self.parameters
        else:
            start = gate_index * self.params_per_gate
            raw = self.parameters[start : start + self.params_per_gate]

        return [_to_param(v) for v in raw]

    def _apply_gate(self, qubits: List[int], params: list) -> None:
        """Call the correct Block builder method for this layer's gate type."""
        method_name, takes_params = _GATE_BUILDER[self.gate_type]
        method = getattr(self, method_name)
        if takes_params:
            method(*qubits, *params)
        else:
            method(*qubits)

    def build_vanilla(self) -> None:
        gate_count = 0

        if self.gate_arity == 1:
            for q in self.qubit_indices:
                if q < self.n_qubits:
                    self._apply_gate([q], self._get_gate_params(gate_count))
                    gate_count += 1

        elif self.gate_arity == 2:
            indices = self.qubit_indices.copy()
            if self.periodic_boundary:
                if len(indices) % 2 == 1 or self.overlapping == 1:
                    indices.append(indices[0])

            step = self.gate_arity - self.overlapping
            for i in range(0, len(indices) - 1, step):
                if i + 1 < len(indices):
                    q0, q1 = indices[i], indices[i + 1]
                    if q0 < self.n_qubits and q1 < self.n_qubits:
                        self._apply_gate([q0, q1], self._get_gate_params(gate_count))
                        gate_count += 1

        elif self.gate_arity == 3:
            indices = self.qubit_indices.copy()
            if self.periodic_boundary:
                remainder = len(indices) % 3
                if remainder == 1 or self.overlapping == 2:
                    indices.extend([indices[0], indices[1]])
                elif remainder == 2 or self.overlapping == 1:
                    indices.append(indices[0])

            step = self.gate_arity - self.overlapping
            for i in range(0, len(indices) - 2, step):
                if i + 2 < len(indices):
                    q0, q1, q2 = indices[i], indices[i + 1], indices[i + 2]
                    if all(q < self.n_qubits for q in [q0, q1, q2]):
                        self._apply_gate([q0, q1, q2], self._get_gate_params(gate_count))
                        gate_count += 1

    @property
    def arity(self) -> int:
        """Return the arity of the gate type."""
        return self.gate_arity

    def __repr__(self) -> str:
        return (
            f"LayerBlock(gate_type={self.gate_type}, n_qubits={self.n_qubits}, "
            f"arity={self.gate_arity}, periodic_boundary={self.periodic_boundary}, "
            f"overlapping={self.overlapping})"
        )
