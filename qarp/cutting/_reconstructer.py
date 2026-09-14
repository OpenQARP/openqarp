"""Circuit reconstruction utilities operating on lists of qx.Command."""

import qarpx as qx

from ._internals import _cmd, _CutMarker, _measure_cmd


class Reconstructer:
    def __init__(self, commands: list, n_qubits: int):
        """
        Args:
            commands: Flat list of qx.Command objects (from block.build().flatten()).
            n_qubits: Number of qubits in the circuit.
        """
        if not commands:
            raise ValueError("The circuit is empty")
        non_barrier = [
            c for c in commands if isinstance(c, qx.Command) and c.gate != qx.GateType.Barrier
        ]
        if not non_barrier:
            raise ValueError("The circuit is empty")
        self.commands = list(commands)
        self.n_qubits = n_qubits

    def reconstruct_delete_gates(
        self,
        to_delete: list,
    ) -> tuple[list, int]:
        """Remove all 2q gates between qubit pairs listed in to_delete.

        Args:
            to_delete: List of [q0, q1] pairs; gates on these pairs are removed.

        Returns:
            (new_commands, n_cuts): pruned command list and number of removed gates.
        """
        new_commands = []
        n_cuts = 0
        for cmd in self.commands:
            if isinstance(cmd, qx.Command) and len(cmd.qubits) == 2:
                q0, q1 = cmd.qubits[0], cmd.qubits[1]
                if [q0, q1] in to_delete or [q1, q0] in to_delete:
                    n_cuts += 1
                    continue
            elif isinstance(cmd, qx.Command) and len(cmd.qubits) > 2:
                raise RuntimeError("More than 2q gates are not yet supported.")
            new_commands.append(cmd)
        return new_commands, n_cuts

    def reconstruct_swap_gates_by_customs(
        self,
        to_delete: list,
    ) -> tuple[list, list, int, dict]:
        """Replace 2q gates at cut positions with _CutMarker sentinels.

        Returns two variants of the cut circuit: a 1q-representation (each
        cut gate replaced by two single-qubit _CutMarkers at [q0] and [q1])
        used for qubit filtering, and a 2q-representation (one marker spanning
        [q0, q1]) used for the decomposition iterator.

        Args:
            to_delete: List of [q0, q1] pairs to cut.

        Returns:
            A ``(cut_cmds_1q, cut_cmds_2q, n_cuts, cut_names)`` tuple:

            - ``cut_cmds_1q``: command list where each cut 2q gate is replaced
              by two single-qubit ``_CutMarker`` objects (one per qubit).
            - ``cut_cmds_2q``: command list where each cut 2q gate is replaced
              by one two-qubit ``_CutMarker`` object (used in
              ``_iterate_experiment``).
            - ``n_cuts``: number of cuts performed.
            - ``cut_names``: dict ``{name: (gate_type_str, params)}`` for each cut.
        """
        cut_cmds_1q = []
        cut_cmds_2q = []
        n_cuts = 0
        cut_names: dict = {}

        for cmd in self.commands:
            if isinstance(cmd, qx.Command) and len(cmd.qubits) == 2:
                q0, q1 = cmd.qubits[0], cmd.qubits[1]
                if [q0, q1] in to_delete or [q1, q0] in to_delete:
                    n_cuts += 1
                    name = f"cut_{n_cuts}"
                    gate_type_str = cmd.gate.name  # e.g. "CX", "RZZ"
                    params = list(cmd.params)
                    cut_names[name] = (gate_type_str, params)
                    marker = _CutMarker(name, [q0, q1], gate_type_str, params)
                    # 1q representation: two single-qubit markers
                    cut_cmds_1q.append(_CutMarker(name, [q0], gate_type_str, params))
                    cut_cmds_1q.append(_CutMarker(name, [q1], gate_type_str, params))
                    # 2q representation: one marker spanning both qubits
                    cut_cmds_2q.append(marker)
                    continue
            elif isinstance(cmd, qx.Command) and len(cmd.qubits) > 2:
                raise RuntimeError("More than 2q gates are not yet supported.")
            cut_cmds_1q.append(cmd)
            cut_cmds_2q.append(cmd)

        return cut_cmds_1q, cut_cmds_2q, n_cuts, cut_names

    @staticmethod
    def reconstruct_filter_qubits(commands: list, n_qubits: int, qubits: list) -> tuple[list, int]:
        """Filter a command list to only include commands acting on qubits in the subset.

        Qubit indices are remapped to be 0-based local indices within the subset.
        QPD Measure commands are kept and their cbit indices are re-assigned
        sequentially (0, 1, …) in the order they are encountered.

        Args:
            commands: Command list (may contain _CutMarker objects — they are skipped).
            n_qubits: Total qubit count (used for validation).
            qubits: Subset of qubit indices to include.

        Returns:
            (filtered_commands, local_n_qubits)
        """
        qubit_set = set(qubits)
        qubit_map = {gq: lq for lq, gq in enumerate(sorted(qubits))}
        new_commands = []
        qpd_cbit_counter = 0

        if n_qubits > len(qubits):
            # Validate: no command spans across the boundary
            for cmd in commands:
                if not isinstance(cmd, qx.Command):
                    continue
                if len(cmd.qubits) == 2:
                    q0, q1 = cmd.qubits[0], cmd.qubits[1]
                    in_set = q0 in qubit_set, q1 in qubit_set
                    if in_set[0] != in_set[1]:
                        raise RuntimeError(
                            f"2q gate on ({q0},{q1}) straddles subcircuit boundary. "
                            "Cut the circuit before filtering."
                        )

        for cmd in commands:
            if not isinstance(cmd, qx.Command):
                continue  # skip _CutMarker sentinels
            cmd_qubits = set(cmd.qubits)
            if not cmd_qubits.issubset(qubit_set):
                continue
            local_qubits = [qubit_map[q] for q in cmd.qubits]
            if cmd.gate == qx.GateType.Measure:
                # Remap qubit to local; assign next sequential cbit
                new_commands.append(_measure_cmd(local_qubits[0], qpd_cbit_counter))
                qpd_cbit_counter += 1
            else:
                new_commands.append(
                    _cmd(cmd.gate, *local_qubits, *(cmd.params if cmd.params else []))
                )

        local_n_qubits = len(qubits)
        return new_commands, local_n_qubits

    @staticmethod
    def remove_measure_gates(commands: list) -> tuple[list, list]:
        """Remove all Measure commands, returning them and the rest separately.

        Returns:
            (filtered_commands, measured_qubit_indices)
        """
        new_commands = []
        measured_qubits = []
        for cmd in commands:
            if isinstance(cmd, qx.Command) and cmd.gate == qx.GateType.Measure:
                measured_qubits.append(cmd.qubits[0])
            else:
                new_commands.append(cmd)
        return new_commands, measured_qubits
