"""CuttingPrimitive: PrimitiveAlgorithm for QPD circuit cutting.

build() uses streaming experiment generation (O(n_cuts × n_gates) peak memory)
and QWC-grouped sub-blocks (one block per QWC group per experiment × subcircuit
instead of one per observable term).

run() post-processes engine SamplingResults using bitmask parity — one result
feeds all terms in a QWC group simultaneously.

Set max_experiments to run a random subset of the 6^n_cuts experiments for an
approximate (but unbiased) result at reduced cost.
"""

from math import ceil
from typing import Optional, Self

import numpy as np

import qarpx as qx
from qarp._types import Shots
from qarp.algorithms._primitives.primitive_algorithm import PrimitiveAlgorithm
from qarp.algorithms._primitives.target import Target
from qarp.blocks import AnyBlock
from qarp.cutting import EAPartitioning, QPDDecomposition, Reconstructer, check_cut_budget
from qarp.cutting._qpd_decomposition import _post_process_shot
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator


class CuttingPrimitive(PrimitiveAlgorithm):
    """PrimitiveAlgorithm that evaluates expectation values via QPD circuit cutting.

    Works with any OpenQARP engine without modification.
    """

    supported_targets = frozenset({Target.EXPECTATION_VALUE})
    # QPD experiment sampling is inherently stochastic — no ∞-shot limit.
    supports_exact = False
    # The QPD reconstruction is linear in each fragment's expectation value taken
    # separately — all a per-occurrence shift needs (products across fragments are fine).
    gradient_kind = "expectation"

    def __init__(
        self,
        ket: Optional[AnyBlock] = None,
        operator: Optional[QubitOperator] = None,
        max_subcircuit_qubits: Optional[int] = None,
        n_shots: int = 5000,
        force_max_number_cuts: bool = False,
        experiment_fraction: Optional[float] = None,
        rng_seed: Optional[int] = None,
    ):
        """
        Args:
            ket: Quantum state block to measure.
            operator: Observable (``qarp.operators.QubitOperator``).
            max_subcircuit_qubits: Capacity cap per subcircuit (qubit count
                each partition must fit into); defaults to 256.
            n_shots: Shots per sub-experiment.
            force_max_number_cuts: Skip the config.max_number_of_cuts guard.
            experiment_fraction: Fraction of the 6^n_cuts experiments to run,
                as a value in ``(0, 1]``.  ``None`` (default) runs all
                experiments.  The sampling strategy is selected automatically:
                ``"top_k"`` for parametric cut gates (RZZ etc.) and
                ``"uniform"`` for non-parametric gates (CX).  Inspect
                ``prim._qpd.sampling_strategy`` after ``build()`` to see which
                was chosen.
            rng_seed: Seed for the experiment-selection RNG (``"uniform"``
                mode only).
        """
        if n_shots is Shots.EXACT:
            # supports_exact = False; EXACT would also slip past the
            # `n_shots is not None` assert in build().
            raise CapabilityError(
                "CuttingPrimitive has no ∞-shot limit (QPD experiment sampling "
                "is inherently stochastic); pass a finite n_shots."
            )
        super().__init__(
            ket=ket,
            operator=operator,
            n_shots=n_shots,
            target=Target.EXPECTATION_VALUE,
        )
        self.max_subcircuit_qubits = max_subcircuit_qubits or 256
        self.force_max_number_cuts = force_max_number_cuts
        self.experiment_fraction = experiment_fraction
        self.rng_seed = rng_seed

    def build(self) -> Self:
        """Partition the circuit and build all measurement sub-blocks.

        Uses streaming experiment generation and QWC grouping so that
        sub_blocks contains n_experiments × n_subcircuits × n_qwc_groups
        blocks (instead of × n_obs_terms).
        """
        if self.ket is None:
            raise RuntimeError("ket must be set before calling build().")
        if self.operator is None:
            raise RuntimeError("operator must be set before calling build().")

        # 1. Flatten the ket block, strip barriers and measurements
        built_ket = self.ket.build() if not self.ket.is_built else self.ket
        raw_commands = list(built_ket.flatten())
        n_qubits = built_ket.n_qubits

        rec = Reconstructer(raw_commands, n_qubits)
        commands, _ = rec.remove_measure_gates(
            [c for c in raw_commands if c.gate != qx.GateType.Barrier]
        )

        # 2. Partition into subcircuits
        max_n = self.max_subcircuit_qubits
        n_subcircuits = ceil(n_qubits / max_n)
        max_sizes = [max_n] * n_subcircuits

        self._cutter = EAPartitioning(commands, n_qubits, max_sizes, verbose=False)
        self._cutter_result = self._cutter.cut()

        check_cut_budget(self._cutter_result.n_cuts, force=self.force_max_number_cuts)

        # 3. QPD decomposition — stream experiments + QWC block building
        # CuttingPrimitive always carries an int n_shots (defaulted in __init__),
        # though the base attribute is typed Optional[Union[int, Shots]].
        assert isinstance(self.n_shots, int)  # CuttingPrimitive rejects Shots.EXACT in __init__
        self._qpd = QPDDecomposition(
            self._cutter_result,
            observable=self.operator,
            verbose=False,
            n_shots=self.n_shots,
            experiment_fraction=self.experiment_fraction,
            rng_seed=self.rng_seed,
            force_max_number_cuts=self.force_max_number_cuts,
        )
        # Split observable and compute QWC groups (does NOT materialise experiments)
        self._qpd._split_observable()
        # Build blocks streaming: yields one experiment at a time, builds QWC groups
        self.sub_blocks, self._metadata = self._qpd.build_experiment_blocks_streaming()
        return self

    def run(self, results: list) -> float:
        """Reconstruct the expectation value from engine sampling results.

        Args:
            results: list[qx.SamplingResult] in the same order as sub_blocks.

        Returns:
            Reconstructed expectation value (float).
        """
        n_exps = len(self._qpd.coefficients)

        # shots_data[str(i_sub)][i_exp][term_idx] = ev
        shots_data: dict = {
            str(i): [{} for _ in range(n_exps)] for i in range(self._qpd.n_subcircuits)
        }

        for result, (
            k_exp,
            i_sub,
            _grp_idx,
            term_indices,
            qpd_cbits,
            obs_cbits,
            local_masks,
        ) in zip(results, self._metadata, strict=True):
            local_n = len(obs_cbits)

            if len(result.cbit_history) == 0:
                if qpd_cbits or obs_cbits:
                    # The experiment recorded measurements but the backend
                    # returned no per-shot register — treating that as "all
                    # contributions are 1" would be silently wrong.
                    raise RuntimeError(
                        "CuttingPrimitive: experiment expects classical bits "
                        f"(qpd={list(qpd_cbits)}, obs={list(obs_cbits)}) but the "
                        "SamplingResult has an empty cbit_history — the engine's "
                        "simulator does not fill the per-shot classical register."
                    )
                # No measurements recorded — all contributions are 1
                for term_idx in term_indices:
                    shots_data[str(i_sub)][k_exp][term_idx] = 1.0
            else:
                accum = {term_idx: 0.0 for term_idx in term_indices}
                for shot_cbits in result.cbit_history:
                    qpd_vals = [int(shot_cbits[i]) for i in qpd_cbits]
                    obs_bits_int = sum(int(shot_cbits[obs_cbits[k]]) << k for k in range(local_n))
                    qpd_factor = _post_process_shot(qpd_vals)
                    for term_idx, mask in zip(term_indices, local_masks, strict=True):
                        pauli_factor = (-1) ** bin(obs_bits_int & mask).count("1")
                        accum[term_idx] += qpd_factor * pauli_factor
                n = result.n_shots
                for term_idx in term_indices:
                    shots_data[str(i_sub)][k_exp][term_idx] = accum[term_idx] / n

        exp_vals = self._qpd._reconstruct_expectation_value_from_shots(shots_data)
        return float(
            np.real(
                np.dot(exp_vals, self._qpd.sub_observables_coeffs) + self._qpd.observable.constant
            )
        )
