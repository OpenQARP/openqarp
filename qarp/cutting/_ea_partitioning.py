"""Evolutionary-algorithm-based automatic circuit cut finder."""

from itertools import permutations
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import differential_evolution

import qarpx as qx

from ._auto_cut_finder import AutoCutFinder, CutterResult
from ._reconstructer import Reconstructer


def _register_unique_connectivities(commands: list) -> list:
    """Return unique (q0, q1) qubit pairs connected by 2q gates."""
    pairs = []
    for cmd in commands:
        if isinstance(cmd, qx.Command) and len(cmd.qubits) == 2 and cmd.gate != qx.GateType.Measure:
            pairs.append((cmd.qubits[0], cmd.qubits[1]))
    return list(set(pairs))


def _n_subcircuits(commands: list, n_qubits: int) -> tuple[int, list, list]:
    """Enumerate separable connected components in the circuit connectivity graph.

    Returns:
        (n_components, sizes_of_components, list_of_qubit_sets)
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_qubits))
    for cmd in commands:
        if isinstance(cmd, qx.Command) and len(cmd.qubits) == 2 and cmd.gate != qx.GateType.Measure:
            g.add_edge(cmd.qubits[0], cmd.qubits[1])
    components = list(nx.connected_components(g))
    sizes = [len(c) for c in components]
    return len(components), sizes, [set(c) for c in components]


def find_valid_combination(subgroups: list, max_sizes: list):
    """Check whether subgroups fit into max_sizes buckets (any permutation).

    Returns:
        (valid: bool, solution: list[list[int]])
    """
    if len(subgroups) < len(max_sizes):
        return False, []
    if len(subgroups) == len(max_sizes):
        for perm in permutations(subgroups):
            if all(perm[i] <= max_sizes[i] for i in range(len(subgroups))):
                return True, [[i] for i in perm]
    subgroups.sort(reverse=True)
    data_structures: list = [[] for _ in max_sizes]
    remaining = max_sizes[:]
    for number in subgroups:
        placed = False
        for i in range(len(data_structures)):
            if remaining[i] >= number:
                data_structures[i].append(number)
                remaining[i] -= number
                placed = True
                break
        if not placed:
            return False, []
    return True, data_structures


def convert_sol_perm_to_idx(sol_perm_idx: list, subgroups: list) -> list:
    """Translate size-based solution to index-based solution."""
    idxs_used = []
    sol_final = []
    for sol in sol_perm_idx:
        aux = []
        for size in sol:
            for idx in np.where(np.array(subgroups) == size)[0]:
                if idx not in idxs_used:
                    break
            aux.append(idx)
            idxs_used.append(idx)
        sol_final.append(aux)
    return sol_final


class EAPartitioning(AutoCutFinder):
    def __init__(
        self,
        commands: list,
        n_qubits: int,
        max_size_subcircuits: list,
        verbose: bool = True,
    ):
        """Automatic cut finder using scipy differential evolution.

        Args:
            commands: Flat qx.Command list (from block.build().flatten(),
                      with measurements and barriers already removed).
            n_qubits: Number of qubits in the circuit.
            max_size_subcircuits: Maximum qubit count per subcircuit, e.g. [2, 2].
            verbose: Print progress.
        """
        super().__init__(
            commands=commands,
            n_qubits=n_qubits,
            max_size_subcircuits=max_size_subcircuits,
            penalization_term=n_qubits * 10,
            verbose=verbose,
        )
        self.variables = _register_unique_connectivities(self.commands)
        self.n_vars = len(self.variables)
        assert self.n_vars >= 1, "Circuit is empty or has no entangling gates."
        self.reconstructer = Reconstructer(self.commands, n_qubits)
        self.n_cuts = 0

    def cut(self, manual_setting: Optional[list] = None) -> CutterResult:
        """Partition the circuit, returning a CutterResult.

        Args:
            manual_setting: If given, a list of qubit-index lists (one per
                subcircuit) that specifies the partition manually.
        """
        if manual_setting:
            flatten_setting = [x for xs in manual_setting for x in xs]
            assert set(flatten_setting) == set(range(self.n_qubits)), (
                "manual_setting must cover all qubit indices exactly once."
            )
            assert len(flatten_setting) == self.n_qubits, (
                "manual_setting qubit indices are invalid."
            )
            self.n_subcircuits = len(manual_setting)
            self.max_size_subcircuits = [len(s) for s in manual_setting]

            variables_to_remove = []
            for var in self.variables:
                q0, q1 = var[0], var[1]
                for subcircuit in manual_setting:
                    if q0 in subcircuit and q1 in subcircuit:
                        break
                    if q0 in subcircuit and q1 not in subcircuit:
                        variables_to_remove.append(list(var))
                        break

            cut_cmds_1q, cut_cmds_2q, self.n_cuts, self.cut_names = (
                self.reconstructer.reconstruct_swap_gates_by_customs(variables_to_remove)
            )
            self._cut_cmds_2q = cut_cmds_2q
            self._cut_cmds_1q = cut_cmds_1q
            self._subcircuit_qubits = manual_setting

            self._subcircuits_cmds = []
            for subcircuit_qubits in manual_setting:
                filtered_cmds, local_n = Reconstructer.reconstruct_filter_qubits(
                    cut_cmds_1q, self.n_qubits, subcircuit_qubits
                )
                self._subcircuits_cmds.append((filtered_cmds, local_n))

        else:
            if not self.max_size_subcircuits:
                raise AssertionError(
                    "max_size_subcircuits must be set when manual_setting is not provided."
                )

            def f(sol_float):
                sol = np.round(sol_float).astype(int)
                variables_to_remove = [
                    list(self.variables[i]) for i in range(self.n_vars) if sol[i] == 0
                ]
                new_cmds, n_cuts_performed = self.reconstructer.reconstruct_delete_gates(
                    variables_to_remove
                )
                n_subcs, size_subcs, _ = _n_subcircuits(new_cmds, self.n_qubits)
                valid, _ = find_valid_combination(size_subcs, self.max_size_subcircuits)
                if valid:
                    return float(n_cuts_performed)
                result = abs(self.n_subcircuits - n_subcs) * self.penalization_term
                if result == 0.0:
                    diff_sizes = [
                        max(0, size_subcs[i] - self.max_size_subcircuits[i]) for i in range(n_subcs)
                    ]
                    result = sum(diff_sizes) * self.penalization_term
                return float(result)

            bounds = [(0, 1)] * self.n_vars
            result = differential_evolution(
                f,
                bounds,
                strategy="best1bin",
                maxiter=5 * self.n_vars,
                popsize=5 * self.n_vars,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
            )
            result_bin = np.round(result.x).astype(int)
            variables_to_remove = [
                list(self.variables[i]) for i in range(self.n_vars) if result_bin[i] == 0
            ]
            new_cmds, _ = self.reconstructer.reconstruct_delete_gates(variables_to_remove)
            _, size_subcs, comps = _n_subcircuits(new_cmds, self.n_qubits)
            _, sol_valid = find_valid_combination(size_subcs, self.max_size_subcircuits)
            sol_final = convert_sol_perm_to_idx(sol_valid, size_subcs)

            cut_cmds_1q, cut_cmds_2q, self.n_cuts, self.cut_names = (
                self.reconstructer.reconstruct_swap_gates_by_customs(variables_to_remove)
            )
            self._cut_cmds_2q = cut_cmds_2q
            self._cut_cmds_1q = cut_cmds_1q

            # comps[i] already contains real qubit indices from the networkx
            # graph — no node-index ÷2 conversion.
            self._subcircuit_qubits = []
            self._subcircuits_cmds = []
            for sol in sol_final:
                qubit_set = set()
                for i in sol:
                    qubit_set |= comps[i]
                qubits_sorted = sorted(qubit_set)
                self._subcircuit_qubits.append(qubits_sorted)
                filtered_cmds, local_n = Reconstructer.reconstruct_filter_qubits(
                    cut_cmds_1q, self.n_qubits, qubits_sorted
                )
                self._subcircuits_cmds.append((filtered_cmds, local_n))

        self._cut_performed = True
        return CutterResult(self)
