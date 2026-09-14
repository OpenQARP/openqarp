"""Quasi-Probability Decomposition engine.

Three optimizations over the naïve implementation:

1. **Streaming experiment generation** – experiments are produced depth-first
   one at a time via ``_iter_experiments()``.  Peak memory is O(n_cuts × n_gates)
   instead of O(6^n_cuts × n_gates).  Both the engine path
   (``build_experiment_blocks_streaming``) and the standalone path
   (``decompose`` + ``compute``) use streaming — ``self.experiments`` and
   ``self.jobs`` are never materialised.

2. **QWC grouping per subcircuit** – observable terms that are qubit-wise
   commuting are measured in a single circuit (same approach as
   ``PauliAveraging``).  Only ONE block is built per QWC group per
   (experiment × subcircuit) instead of one per term.  For a Hamiltonian with
   k QWC groups this reduces the block count by n_terms/k.

3. **Experiment subsampling with automatic strategy selection** – set
   ``experiment_fraction`` (a value in ``(0, 1]``) to run only that fraction
   of the 6^n_cuts experiments.  The sampling strategy is chosen automatically:
   ``"top_k"`` for parametric gates (RZZ, CRx/y/z — skewed coefficients) and
   ``"uniform"`` for non-parametric gates (CX — equal-magnitude coefficients).
   Useful when n_cuts ≥ 4 and approximate results are acceptable.
"""

import itertools
import math
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
from sympy import Float

import qarpx as qx
from qarp.operators import QubitOperator

from ..operators._grouping import GroupingStrategy, QubitWiseCommuting, group_basis
from ._auto_cut_finder import CutterResult, check_cut_budget
from ._decompositions import (
    decompose_crx,
    decompose_cry,
    decompose_crz,
    decompose_cx,
    decompose_cy,
    decompose_cz,
    decompose_rzz,
)
from ._internals import _cmd, _count_qpd_measures, _CutMarker, _measure_cmd, _param_to_value
from ._post_processing import PostProcessing
from ._reconstructer import Reconstructer

parser: Dict[str, Callable] = {
    "CX": decompose_cx,
    "CY": decompose_cy,
    "CZ": decompose_cz,
    "CRx": decompose_crx,
    "CRy": decompose_cry,
    "CRz": decompose_crz,
    "RZZ": decompose_rzz,
}


def _post_process_shot(qpd_bit_values: list) -> int:
    """Return +1 if an even number of QPD bits are 1, else -1."""
    return (-1) ** sum(qpd_bit_values)


class QPDDecomposition(PostProcessing):
    def __init__(
        self,
        cutter_result: CutterResult,
        observable: QubitOperator,
        verbose: bool = True,
        n_shots: int = 1024,
        experiment_fraction: Optional[float] = None,
        rng_seed: Optional[int] = None,
        shot_seed: Optional[int] = None,
        force_max_number_cuts: bool = False,
        grouping: Optional[GroupingStrategy] = None,
    ) -> None:
        """
        Args:
            cutter_result: Output of EAPartitioning.cut().
            observable: Hamiltonian (``qarp.operators.QubitOperator``).
            verbose: Print progress.
            n_shots: Shots per sub-experiment.
            experiment_fraction: Fraction of the 6^n_cuts experiments to run,
                as a value in ``(0, 1]``.  ``None`` (default) runs all
                experiments.  Setting this to, e.g., ``0.1`` runs 10% of
                experiments.

                The **sampling strategy** is chosen automatically based on the
                gate types at the cut locations:

                - **Any parametric cut (RZZ, CRx, CRy, CRz)** → ``"top_k"``:
                  experiments are ranked by |coefficient| and the highest-weight
                  ones are selected.  Biased, but error is bounded by the total
                  weight of the dropped experiments — typically small for
                  small-angle gates where one or two branches dominate.

                - **All non-parametric cuts (CX only)** → ``"uniform"``:
                  experiments are selected uniformly at random with a correction
                  factor that keeps the estimator unbiased.

                The chosen strategy is accessible via the read-only
                ``sampling_strategy`` property after construction.
            rng_seed: Seed for the experiment-selection RNG used in the
                ``"uniform"`` strategy.  Ignored for ``"top_k"`` (deterministic).
                Independent of the shot-simulation seed (``shot_seed``).
            shot_seed: Seed for the standalone shot simulation
                (``compute()`` / ``reconstruct_statevector()``); each
                sub-experiment draws a derived stream so results are
                decorrelated.  None → nondeterministic.
            force_max_number_cuts: Skip the ``config.max_number_of_cuts``
                guard (6^n_cuts experiments).
            grouping: Observable-term partitioning strategy; ``None`` →
                ``QubitWiseCommuting()``.  Must have ``qubit_wise=True`` —
                measurement bases here are per-qubit rotations on each
                subcircuit, so entangling-Clifford groupings cannot apply.
        """
        super().__init__(cutter_result=cutter_result, observable=observable, verbose=verbose)
        assert n_shots > 0
        self.grouping: GroupingStrategy = grouping if grouping is not None else QubitWiseCommuting()
        if not self.grouping.qubit_wise:
            raise ValueError(
                f"circuit cutting requires a qubit-wise grouping strategy (per-qubit "
                f"measurement bases); got {self.grouping!r}"
            )
        if experiment_fraction is not None:
            if not (0.0 < experiment_fraction <= 1.0):
                raise ValueError(
                    f"experiment_fraction must be in (0, 1], got {experiment_fraction}"
                )
        self.n_shots = n_shots
        self.experiment_fraction = experiment_fraction
        self._rng_seed = rng_seed
        self._shot_seed = shot_seed
        check_cut_budget(self.n_cuts, force=force_max_number_cuts)

        # Derive max_experiments and auto-select strategy from gate types
        total = 6**self.n_cuts
        if experiment_fraction is not None and experiment_fraction < 1.0:
            self._max_experiments: Optional[int] = max(1, int(total * experiment_fraction))
            self._sampling_strategy: str = self._auto_sampling_strategy()
        else:
            self._max_experiments = None
            self._sampling_strategy = "uniform"  # irrelevant when running all
        self.results_shots: dict = {}
        self._reconstruction_metadata: list = []
        self.coefficients: list = []

        # Cache full-circuit qubit count for _iterate_experiment calls
        self._circuit_n_qubits = max(
            (
                max(c.qubits) + 1
                for c in self.custom_commands
                if isinstance(c, qx.Command) and c.qubits
            ),
            default=1,
        )

    # ── Strategy auto-selection ───────────────────────────────────────────

    _PARAMETRIC_GATES = frozenset({"RZZ", "CRx", "CRy", "CRz"})

    def _auto_sampling_strategy(self) -> str:
        """Return ``'top_k'`` if any cut marker is a parametric gate, else ``'uniform'``.

        Parametric gates (RZZ, CRx, CRy, CRz) produce skewed coefficient
        distributions where a few experiments dominate — top-k selection is
        more accurate.  Non-parametric gates (CX etc.) have equal-magnitude
        coefficients, so uniform random sampling is unbiased and preferred.
        """
        for cmd in self.custom_commands:
            if isinstance(cmd, _CutMarker) and cmd.gate_type_str in self._PARAMETRIC_GATES:
                return "top_k"
        return "uniform"

    @property
    def sampling_strategy(self) -> str:
        """Read-only: the strategy that will be used for subsampling.

        ``'top_k'`` if any cut is parametric (auto-selected), ``'uniform'``
        otherwise.  Always ``'uniform'`` when no subsampling is active.
        """
        return self._sampling_strategy

    # ── Core expand step ─────────────────────────────────────────────────

    def _iterate_experiment(
        self, exp_commands: list, n_qubits: int, coeff
    ) -> Tuple[list, np.ndarray]:
        """Expand the first _CutMarker in exp_commands into 6 experiments."""
        prefix: list = []
        experiments: list = []
        coefficients = np.array([])
        flag_found_cut = False
        next_qpd_cbit = 0

        for cmd in exp_commands:
            if isinstance(cmd, _CutMarker) and not flag_found_cut:
                idx = np.full(6, next_qpd_cbit, dtype=int)
                gate_str = cmd.gate_type_str
                if cmd.params:
                    param_val = _param_to_value(cmd.params[0])
                    experiments, coefficients = parser[gate_str](
                        list(prefix), n_qubits, cmd.qubits, idx, param_val
                    )
                else:
                    experiments, coefficients = parser[gate_str](
                        list(prefix), n_qubits, cmd.qubits, idx
                    )
                flag_found_cut = True
            elif flag_found_cut:
                for exp_cmd_list in experiments:
                    exp_cmd_list.append(cmd)
            else:
                if isinstance(cmd, qx.Command):
                    prefix.append(cmd)
                    if cmd.gate == qx.GateType.Measure:
                        next_qpd_cbit += 1

        if not flag_found_cut:
            return [exp_commands], np.array([coeff])
        return experiments, coeff * coefficients

    # ── Random-access single experiment (Plan 3) ─────────────────────────

    def _generate_single_experiment(self, k_exp: int) -> Tuple[list, float]:
        """Generate experiment k_exp directly using base-6 index decoding.

        O(n_cuts × n_gates) time and memory — the 5 unused branches at each
        cut level are discarded immediately.
        """
        branch_indices = []
        remainder = k_exp
        for _ in range(self.n_cuts):
            branch_indices.append(remainder % 6)
            remainder //= 6

        current_cmds = list(self.custom_commands)
        current_coeff = 1.0
        for cut_level in range(self.n_cuts):
            exps, coeffs = self._iterate_experiment(
                list(current_cmds), self._circuit_n_qubits, current_coeff
            )
            idx = branch_indices[cut_level]
            current_cmds = exps[idx]
            current_coeff = float(coeffs[idx])
            del exps  # discard the 5 unused branches immediately
        return current_cmds, current_coeff

    # ── Coefficient enumeration (top_k strategy) ─────────────────────────

    def _compute_all_coefficients(self) -> np.ndarray:
        """Return all 6^n_cuts QPD coefficients without generating command lists.

        Exploits the fact that QPD coefficients at each cut depend only on the
        gate type and its parameters — not on the circuit content before the
        cut.  The full coefficient of experiment k is the product of one entry
        per cut level, selected by the base-6 digits of k.

        Returns:
            np.ndarray of shape (6^n_cuts,) with dtype float64.
            coefficients[k] matches the coefficient yielded by
            _generate_single_experiment(k) and _iter_experiments().

        Raises:
            ValueError: If any coefficient is a sympy symbolic expression
                (i.e. the circuit has un-substituted parameters).  Pass a
                symbol_map via compute() or substitute before calling this.
        """
        # Scan custom_commands to extract the base coefficient array (6 values)
        # for each cut marker, in the order they appear in the circuit.
        cut_base_coeffs: list = []
        prefix: list = []
        next_qpd_cbit = 0

        for cmd in self.custom_commands:
            if isinstance(cmd, _CutMarker):
                idx = np.full(6, next_qpd_cbit, dtype=int)
                if cmd.params:
                    param_val = _param_to_value(cmd.params[0])
                    _, raw_coeffs = parser[cmd.gate_type_str](
                        list(prefix), self._circuit_n_qubits, cmd.qubits, idx, param_val
                    )
                else:
                    _, raw_coeffs = parser[cmd.gate_type_str](
                        list(prefix), self._circuit_n_qubits, cmd.qubits, idx
                    )
                try:
                    cut_base_coeffs.append(np.array([float(c) for c in raw_coeffs]))
                except (TypeError, AttributeError) as exc:
                    raise ValueError(
                        "_compute_all_coefficients() requires concrete (numeric) "
                        "gate parameters.  The circuit contains symbolic parameters "
                        "that have not been substituted.  Call compute(symbol_map=…) "
                        "instead, or substitute symbols before ranking."
                    ) from exc
            elif isinstance(cmd, qx.Command):
                prefix.append(cmd)
                if cmd.gate == qx.GateType.Measure:
                    next_qpd_cbit += 1

        # Vectorised product: for each experiment index k, coefficients[k] =
        # ∏_i  cut_base_coeffs[i][digit_i(k)]  where digit_i(k) = (k // 6^i) % 6
        total = 6**self.n_cuts
        indices = np.arange(total, dtype=np.int64)
        all_coeffs = np.ones(total)
        for i, level_coeffs in enumerate(cut_base_coeffs):
            branches = (indices // (6**i)) % 6
            all_coeffs *= level_coeffs[branches]

        return all_coeffs

    # ── Streaming generator (Plans 1 + 3) ────────────────────────────────

    def _iter_experiments(self) -> Iterator[Tuple[list, float]]:
        """Yield (command_list, coefficient) one at a time.

        Behaviour depends on max_experiments and sampling_strategy:

        - max_experiments is None or >= 6^k:
            Yields all 6^n_cuts experiments depth-first (no subsampling).

        - sampling_strategy="uniform" (default):
            Randomly selects max_experiments indices without replacement.
            Each coefficient is scaled by 6^k / max_experiments so the
            estimator is **unbiased** (variance increases, bias = 0).

        - sampling_strategy="top_k":
            Ranks all 6^k experiments by |coefficient| and selects the top
            max_experiments.  Coefficients are used as-is (no scaling).
            The estimator is **biased** by Σᵢ∉S cᵢ eᵢ, but the bias is
            bounded by Σᵢ∉S |cᵢ| — small when low-weight experiments are
            dropped.  Most effective for parametric gates (RZZ) where a
            few branches carry most of the weight.

        Peak memory: O(n_cuts × n_gates) in all cases.
        """
        total = 6**self.n_cuts

        if self._max_experiments is not None and self._max_experiments < total:
            if self._sampling_strategy == "top_k":
                # Rank all experiments by |coeff| — O(6^k), no simulation needed
                all_coeffs = self._compute_all_coefficients()
                selected = np.argsort(-np.abs(all_coeffs))[: self._max_experiments]
                selected = np.sort(selected)  # depth-first order for efficiency
                if self.verbose:
                    dropped_weight = np.sum(np.abs(all_coeffs)) - np.sum(
                        np.abs(all_coeffs[selected])
                    )
                    print(
                        f"top_k: running {len(selected)} of {total} experiments "
                        f"(dropped |coeff| weight = {dropped_weight:.4f})"
                    )
                for k_exp in selected:
                    exp_cmds, coeff = self._generate_single_experiment(int(k_exp))
                    yield exp_cmds, coeff  # original coefficient, no correction
            else:  # "uniform"
                rng = np.random.default_rng(self._rng_seed)
                selected = np.sort(rng.choice(total, size=self._max_experiments, replace=False))
                correction = total / self._max_experiments
                for k_exp in selected:
                    exp_cmds, coeff = self._generate_single_experiment(int(k_exp))
                    yield exp_cmds, coeff * correction
        else:
            n_qubits = self._circuit_n_qubits

            def _expand(cmds, coeff, cuts_left):
                if cuts_left == 0:
                    yield cmds, coeff
                    return
                exps, coeffs = self._iterate_experiment(list(cmds), n_qubits, coeff)
                for exp, c in zip(exps, coeffs, strict=True):
                    yield from _expand(exp, float(c), cuts_left - 1)

            yield from _expand(list(self.custom_commands), 1.0, self.n_cuts)

    # ── Observable splitting + QWC grouping ──────────────────────────────

    def _split_observable(self) -> None:
        """Split observable terms across subcircuits and compute QWC groups.

        Uses the injected qubit-wise grouping strategy — same technique as
        ``PauliAveraging(grouping=QubitWiseCommuting())`` — so all terms that
        can share a single measurement circuit are grouped together.  One
        block is built per group per (experiment × subcircuit) instead of one
        per observable term.
        """
        self.sub_observables: dict = {str(i): [] for i in range(self.n_subcircuits)}
        self.sub_observables_coeffs = []
        self.n_valid_obs_terms = 0

        for term, coeff in self.observable.terms.items():
            if not term:
                continue
            aux_dict: dict = {str(i): [] for i in range(self.n_subcircuits)}
            for q, pauli in term:
                sub_idx = self._locate_qubit_idx_subcircuit(q)
                aux_dict[str(sub_idx)].append((q, pauli))
            for i in range(self.n_subcircuits):
                self.sub_observables[str(i)].append(aux_dict[str(i)])
            self.sub_observables_coeffs.append(coeff)
            self.n_valid_obs_terms += 1

        self.sub_observable_qwc_groups: dict = {}
        self.sub_observable_qwc_bases: dict = {}
        for i_sub in range(self.n_subcircuits):
            terms = self.sub_observables[str(i_sub)]
            pauli_dicts = [{q: p for q, p in term} for term in terms]
            groups = self.grouping.group(pauli_dicts, self._circuit_n_qubits)
            bases = [group_basis(grp, pauli_dicts) for grp in groups]
            self.sub_observable_qwc_groups[str(i_sub)] = groups
            self.sub_observable_qwc_bases[str(i_sub)] = bases

    def _locate_qubit_idx_subcircuit(self, global_qubit: int) -> int:
        for i in range(self.n_subcircuits):
            if global_qubit in self.idxs[i]:
                return i
        raise RuntimeError(f"Qubit {global_qubit} not found in any subcircuit.")

    # ── Block building ────────────────────────────────────────────────────

    def _build_group_experiment_block(
        self,
        sub_cmds: list,
        local_n: int,
        n_qpd: int,
        group_basis_local: dict,
        global_to_local: dict,
        term_indices: list,
        i_sub: int,
        name: str = "exp",
    ) -> Tuple:
        """Build ONE SimpleBlock for a full QWC group.

        All terms in the group share the same basis-rotation + measure circuit.
        Post-processing uses bitmask parity to extract each term's contribution
        from the single shared measurement result.
        """
        basis_cmds = []
        for local_q, pauli in group_basis_local.items():
            if pauli == "X":
                basis_cmds.append(_cmd(qx.GateType.H, local_q))
            elif pauli == "Y":
                basis_cmds.append(_cmd(qx.GateType.Rx, local_q, math.pi / 2))

        obs_cbit_indices = list(range(n_qpd, n_qpd + local_n))
        obs_measure_cmds = [_measure_cmd(q, n_qpd + q) for q in range(local_n)]

        group_local_masks = []
        for term_idx in term_indices:
            obs_term = self.sub_observables[str(i_sub)][term_idx]
            mask = 0
            for q, _ in obs_term:
                lq = global_to_local.get(q, q)
                mask |= 1 << lq
            group_local_masks.append(mask)

        all_commands = list(sub_cmds) + basis_cmds + obs_measure_cmds
        block = qx.SimpleBlock(local_n, name)
        block.set_commands(all_commands)
        block.n_cbits = n_qpd + local_n
        # Raw qx block: single C++ flag; mark_built() is the qarp-block spelling.
        block.set_built(True)

        return block, list(range(n_qpd)), obs_cbit_indices, group_local_masks

    # ── Engine path ───────────────────────────────────────────────────────

    def build_experiment_blocks_streaming(self) -> Tuple[list, list]:
        """Stream experiments, build QWC-grouped sub-blocks for the engine.

        Does NOT populate self.experiments or self.jobs.
        """
        sub_blocks: list = []
        metadata: list = []
        streaming_coefficients: list = []
        circuit_n_qubits = self._circuit_n_qubits

        for k_exp, (exp_cmds, coeff) in enumerate(self._iter_experiments()):
            streaming_coefficients.append(coeff)
            for i_sub in range(self.n_subcircuits):
                sub_cmds, local_n = Reconstructer.reconstruct_filter_qubits(
                    exp_cmds, circuit_n_qubits, self.idxs[i_sub]
                )
                n_qpd = _count_qpd_measures(sub_cmds)
                global_to_local = {gq: lq for lq, gq in enumerate(sorted(self.idxs[i_sub]))}
                groups = self.sub_observable_qwc_groups[str(i_sub)]
                bases_global = self.sub_observable_qwc_bases[str(i_sub)]

                for grp_idx, (term_indices, basis_global) in enumerate(
                    zip(groups, bases_global, strict=True)
                ):
                    basis_local = {global_to_local.get(q, q): p for q, p in basis_global.items()}
                    name = f"exp{k_exp}_s{i_sub}_g{grp_idx}"
                    block, qpd_cbits, obs_cbits, local_masks = self._build_group_experiment_block(
                        sub_cmds,
                        local_n,
                        n_qpd,
                        basis_local,
                        global_to_local,
                        term_indices,
                        i_sub,
                        name,
                    )
                    sub_blocks.append(block)
                    metadata.append(
                        (k_exp, i_sub, grp_idx, term_indices, qpd_cbits, obs_cbits, local_masks)
                    )

        self.coefficients = streaming_coefficients
        self._reconstruction_metadata = metadata
        return sub_blocks, metadata

    # ── Reconstruction ────────────────────────────────────────────────────

    def _reconstruct_expectation_value_from_shots(
        self,
        shots_data: dict,
        symbol_map: Optional[dict] = None,
    ) -> np.ndarray:
        """Reconstruct per-term expectation values.

        shots_data: {str(i_sub): [{term_idx: ev}, …]} indexed by [i_sub][i_exp].
        """
        symbol_map = symbol_map or {}
        reconstruction_exp_val = np.zeros(self.n_valid_obs_terms)

        def subs(val):
            if isinstance(val, (float, int, complex, Float)):
                return float(val)
            return float(val.subs(symbol_map))

        for idx_coeff, coeff in enumerate(self.coefficients):
            exp_vals = np.ones(self.n_valid_obs_terms)
            for i in range(self.n_valid_obs_terms):
                for n_circ in range(self.n_subcircuits):
                    ev = shots_data[str(n_circ)][idx_coeff].get(i, 1.0)
                    exp_vals[i] *= ev
            reconstruction_exp_val += subs(coeff) * exp_vals

        return reconstruction_exp_val

    # ── Standalone compute path (Plans 1 + 3) ────────────────────────────

    def decompose(self) -> None:
        """Prepare observable metadata. No experiments are materialised.

        self.coefficients is populated lazily on the first compute() call.
        """
        self._split_observable()
        n_total = 6**self.n_cuts
        if self.verbose:
            if self._max_experiments is not None:
                pct = 100.0 * self._max_experiments / n_total
                print(
                    f"Total experiments: {n_total}  |  "
                    f"Running: {self._max_experiments} ({pct:.0f}%)  |  "
                    f"Strategy: {self._sampling_strategy}"
                )
            else:
                print(f"Total experiments: {n_total}")

    @property
    def n_experiments(self) -> int:
        """Number of experiments that will actually be executed."""
        total = 6**self.n_cuts
        if self._max_experiments is not None and self._max_experiments < total:
            return self._max_experiments
        return total

    def _shot_seed_for(self, ordinal: int) -> Optional[int]:
        # Prime stride mirrors Engine._circuit_seed: sub-experiments must not
        # share a random tape (correlated shot noise breaks the QPD sum).
        if self._shot_seed is None:
            return None
        return (self._shot_seed + 100_003 * ordinal) % 2**32

    def compute(
        self,
        parallelize: bool = False,  # noqa: ARG002 — kept for API compat
        symbol_map: Optional[dict] = None,
        **kwargs,  # accept the base PostProcessing.compute(**kwargs) surface
    ) -> float:
        """Execute experiments with QarpSimulator and return the expectation value.

        Streams experiments one at a time (Plan 1).  Peak memory is
        O(n_cuts × n_gates + n_subcircuits × local_n_qubits).
        """
        sim = qx.QarpSimulator()
        run_ordinal = 0

        circuit_n_qubits = self._circuit_n_qubits
        results_shots: dict = {str(i): [] for i in range(self.n_subcircuits)}
        self.coefficients = []

        for exp_cmds, coeff in self._iter_experiments():
            self.coefficients.append(coeff)

            for i_sub in range(self.n_subcircuits):
                sub_cmds, local_n = Reconstructer.reconstruct_filter_qubits(
                    exp_cmds, circuit_n_qubits, self.idxs[i_sub]
                )
                if symbol_map:
                    sym_str = {str(k): float(v) for k, v in symbol_map.items()}
                    sub_cmds = [
                        c.substitute(sym_str) if isinstance(c, qx.Command) else c for c in sub_cmds
                    ]

                n_qpd = _count_qpd_measures(sub_cmds)
                global_to_local = {gq: lq for lq, gq in enumerate(sorted(self.idxs[i_sub]))}
                per_term_ev: dict = {}
                groups = self.sub_observable_qwc_groups[str(i_sub)]
                bases_global = self.sub_observable_qwc_bases[str(i_sub)]

                for term_indices, basis_global in zip(groups, bases_global, strict=True):
                    basis_local = {global_to_local.get(q, q): p for q, p in basis_global.items()}
                    block, qpd_cbits, obs_cbits, local_masks = self._build_group_experiment_block(
                        sub_cmds,
                        local_n,
                        n_qpd,
                        basis_local,
                        global_to_local,
                        term_indices,
                        i_sub,
                    )
                    result = sim.run(
                        block.flatten(),
                        local_n,
                        self.n_shots,
                        seed=self._shot_seed_for(run_ordinal),
                    )
                    run_ordinal += 1

                    if len(result.cbit_history) == 0:
                        if qpd_cbits or obs_cbits:
                            # Recorded measurements with no per-shot register:
                            # "all contributions are 1" would be silently wrong.
                            raise RuntimeError(
                                "QPDDecomposition: experiment expects classical "
                                f"bits (qpd={list(qpd_cbits)}, obs={list(obs_cbits)}) "
                                "but the SamplingResult has an empty cbit_history."
                            )
                        for term_idx in term_indices:
                            per_term_ev[term_idx] = 1.0
                    else:
                        accum = {idx: 0.0 for idx in term_indices}
                        for shot_cbits in result.cbit_history:
                            qpd_vals = [int(shot_cbits[i]) for i in qpd_cbits]
                            obs_bits_int = sum(
                                int(shot_cbits[obs_cbits[k]]) << k for k in range(local_n)
                            )
                            qpd_factor = _post_process_shot(qpd_vals)
                            for term_idx, mask in zip(term_indices, local_masks, strict=True):
                                pauli_factor = (-1) ** bin(obs_bits_int & mask).count("1")
                                accum[term_idx] += qpd_factor * pauli_factor
                        n = result.n_shots
                        for term_idx in term_indices:
                            per_term_ev[term_idx] = accum[term_idx] / n

                results_shots[str(i_sub)].append(per_term_ev)

            del exp_cmds  # discard immediately — O(1) working memory per experiment

        self.results_shots = results_shots
        if self.verbose:
            print(f"Running experiments… ({len(self.coefficients)} done)")
        exp_vals = self._reconstruct_expectation_value_from_shots(results_shots, symbol_map)
        self.reconstruction_exp_val = exp_vals
        return float(
            np.real(np.dot(exp_vals, self.sub_observables_coeffs) + self.observable.constant)
        )

    # ── overhead ─────────────────────────────────────────────────────────

    def overhead(self) -> float:
        return 6**self.n_cuts

    # ── Statevector reconstruction (Plan 2) ──────────────────────────────

    def initialize_qubit_dict(self, n_qubits: int) -> dict:
        basis_states = ["".join(map(str, bs)) for bs in itertools.product([0, 1], repeat=n_qubits)]
        return {s: 0.0 for s in basis_states}

    def normalize_SV_probs(self, SV: dict) -> dict:
        normalized = {k: (0 if v < 0 else v) for k, v in SV.items()}
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v / total for k, v in normalized.items()}
        return normalized

    def reconstruct_statevector(self, parallelize: bool = False) -> dict:
        """Reconstruct the statevector via shot simulation.

        Plan 2: replaces itertools.product with incremental subcircuit folding.
        Peak memory is O(2^n_full_qubits) instead of O(outcomes^n_subcircuits).
        Streams experiments (Plan 1) — self.jobs is never required.
        """
        sim = qx.QarpSimulator()
        run_ordinal = 0
        n_full = max(q for qs in self.idxs for q in qs) + 1
        SV_rec = self.initialize_qubit_dict(n_full)
        circuit_n_qubits = self._circuit_n_qubits

        for exp_cmds, coeff in self._iter_experiments():
            # ── Run all subcircuits for this experiment ──────────────────
            sub_results: dict = {}
            for i_sub in range(self.n_subcircuits):
                sub_cmds, local_n = Reconstructer.reconstruct_filter_qubits(
                    exp_cmds, circuit_n_qubits, self.idxs[i_sub]
                )
                n_qpd = _count_qpd_measures(sub_cmds)
                meas_cmds = list(sub_cmds) + [_measure_cmd(q, n_qpd + q) for q in range(local_n)]
                b = qx.SimpleBlock(local_n, "sv")
                b.set_commands(meas_cmds)
                b.n_cbits = n_qpd + local_n
                # Raw qx block: single C++ flag; mark_built() is the qarp-block spelling.
                b.set_built(True)

                result = sim.run(
                    b.flatten(), local_n, self.n_shots, seed=self._shot_seed_for(run_ordinal)
                )
                run_ordinal += 1
                counts: dict = {}
                for shot_cbits in result.cbit_history:
                    qpd_bits = tuple(int(shot_cbits[i]) for i in range(n_qpd))
                    qubit_bits = tuple(int(shot_cbits[n_qpd + q]) for q in range(local_n))
                    key = qubit_bits + qpd_bits
                    counts[key] = counts.get(key, 0) + 1
                sub_results[i_sub] = counts

            # ── Incremental cross-product — no itertools.product (Plan 2) ─
            # partial[assignments_tuple] = cumulative_weight
            # assignments_tuple = ((global_q0, bit0), (global_q1, bit1), …)
            partial: dict = {(): 1.0}
            for i_sub in range(self.n_subcircuits):
                sub_qubits = self.idxs[i_sub]
                local_n = len(sub_qubits)
                next_partial: dict = {}
                for prev_assignments, prev_weight in partial.items():
                    for key, count in sub_results[i_sub].items():
                        qubit_bits = key[:local_n]
                        qpd_bits = key[local_n:]
                        qpd_fac = _post_process_shot(list(qpd_bits))
                        prob = count / self.n_shots
                        new_assignments = prev_assignments + tuple(
                            (sub_qubits[k], qubit_bits[k]) for k in range(local_n)
                        )
                        new_weight = prev_weight * qpd_fac * prob
                        next_partial[new_assignments] = (
                            next_partial.get(new_assignments, 0.0) + new_weight
                        )
                partial = next_partial

            for assignments, weight in partial.items():
                basis = ["0"] * n_full
                for gq, bit_val in assignments:
                    basis[gq] = str(bit_val)
                SV_rec["".join(basis)] += float(coeff) * weight

            del exp_cmds, sub_results, partial  # O(1) working memory

        return self.normalize_SV_probs(SV_rec)
