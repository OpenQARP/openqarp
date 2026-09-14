import itertools
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from qarp.operators import QubitOperator

from ...operators import LinearCombinationUnitaries
from .._block import CompositeBlockBase
from .._state_preparation.synthesized_state_block import SynthesizedStateBlock
from .select_block import SelectBlock, _infer_target_size


class BlockEncodingBlock(CompositeBlockBase):
    r"""Pattern B composite: block-encode an operator ``A`` via LCU.

    Decomposes ``A = Σ_i c_i U_i`` into Pauli strings and assembles the
    standard ``Prep† · Select · Prep`` LCU circuit.  The block-encoded
    operator on the ``|0…0⟩_anc`` subspace is ``A / λ`` where
    ``λ = Σ_i |c_i|`` (stored as ``self.lambda_norm``).

    ``Prep`` loads real amplitudes ``√(|c_i|/λ)`` on the ancilla register;
    each LCU phase ``φ_i = arg(c_i)`` is carried by the corresponding
    ``SelectBlock`` entry and lowered through the multi-controlled ``GPhase``
    decomposition.
    """

    def __init__(
        self,
        A: Optional[Union[np.ndarray, QubitOperator]] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "BlockEncoding",
        *,
        coefficients: Optional[Sequence[complex]] = None,
        unitaries: Optional[Sequence[Any]] = None,
    ):
        """Args:
        A: Operator to block-encode, either as an `np.ndarray` (general,
            read in the qarpx LSB basis — row/column bit ``k`` ↔ target
            qubit ``k`` — so the encoded block is ``A`` itself, no
            bit-reversal) or a `QubitOperator` (sparse Pauli sum).  Mutually
            exclusive with explicit ``coefficients``/``unitaries`` input.
        target_qubits, name: standard Block kwargs.
        coefficients: LCU coefficients ``c_i`` (keyword-only; requires
            ``unitaries``).
        unitaries: LCU unitaries ``U_i`` (keyword-only; requires
            ``coefficients``).  Entries may be any unitary accepted by
            ``SelectBlock``: a Block, ``(phase, Block)``,
            ``(phase, pauli_string)``, or ``(phase, pauli_dict)``.
        """
        if (unitaries is None) != (coefficients is None):
            raise TypeError("Explicit LCU input requires both coefficients and unitaries.")

        self.A = A

        # LCU decomposition lives in __init__ so that consumers (Qubitization,
        # QSVT) can read self.unitaries and self.lambda_norm before .build().
        if unitaries is not None and coefficients is not None:
            if A is not None:
                raise TypeError(
                    "Pass either an operator A or explicit coefficients/unitaries, not both."
                )
            coeffs, select_unitaries, raw_coeffs = self._preprocess_explicit_lcu(
                coefficients=coefficients,
                unitaries=unitaries,
            )
        else:
            if isinstance(A, np.ndarray):
                dec = LinearCombinationUnitaries(A).decomposition()
            elif isinstance(A, QubitOperator):
                dec = self._qubit_operator_decomposition(A)
            else:
                raise TypeError(
                    "Expected a np.ndarray or QubitOperator in BlockEncoding, "
                    "or provide both coefficients and unitaries for explicit LCU input."
                )
            coeffs, select_unitaries, raw_coeffs = self._preprocess_decomposition(dec)

        self.coefficients = coeffs
        self.lcu_coefficients = raw_coeffs
        self.unitaries = select_unitaries
        self.lambda_norm = float(np.sum(self.coefficients))

        if self.lambda_norm <= 0.0:
            raise ValueError("LCU coefficients cannot all be zero.")

        num_unitaries = len(self.unitaries)
        self.num_controls = max(1, int(np.ceil(np.log2(num_unitaries))))
        target_size = _infer_target_size(self.unitaries, context="LCU")
        num_qubits = self.num_controls + target_size

        super().__init__(
            n_qubits=num_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def lambda_factor(self) -> float:
        """Compatibility shim: returns ``self.lambda_norm``."""
        return self.lambda_norm

    def build_vanilla(self) -> None:
        n_anc = self.num_controls
        N_anc = 2**n_anc

        sqrt_norm = np.sqrt(np.asarray(self.coefficients) / self.lambda_norm)

        # Align amplitudes with SelectBlock's `itertools.product([False, True],
        # repeat=n_anc)` enumeration: entry index i corresponds to control
        # tuple `t = product[i]`, where `t[k]` is the value of qubit `k` (LSB
        # convention).  So the ancilla integer for entry i is
        # `sum_k int(t[k]) << k`.
        ctrl_tuples = list(itertools.product([False, True], repeat=n_anc))
        amps = np.zeros(N_anc, dtype=complex)
        for i in range(len(self.unitaries)):
            anc_int = sum(int(b) << k for k, b in enumerate(ctrl_tuples[i]))
            amps[anc_int] = sqrt_norm[i]

        # 1. Prep on the ancilla register (real amplitudes; phases live in Select).
        prep = SynthesizedStateBlock(
            n_qubits=n_anc,
            amplitudes=list(amps),
            name="Prep",
        )
        prep.build()
        prep.target_qubits = list(range(n_anc))
        self.add_child(prep)

        # 2. Select on all qubits — each entry carries its LCU phase.
        select = SelectBlock(self.unitaries, n_anc, name="Select")
        select.target_qubits = list(range(self.n_qubits))
        self.add_wired_child(select)

        # 3. Prep† on the ancilla register.  Lazy Python-wrapper dagger
        # (deepcopy + flag): `add_child` materialises it into concrete
        # daggered commands, keeping the child deepcopy-safe.
        prep_dag = prep.dagger()
        prep_dag.target_qubits = list(range(n_anc))
        self.add_child(prep_dag)

    @staticmethod
    def _qubit_operator_decomposition(qop: QubitOperator) -> List[tuple]:
        """Convert a QubitOperator into a `[(pauli_string, coeff)]` list
        (character k of each string acts on qubit k)."""
        if not qop.terms:
            return []

        n_qubits = max((i for term in qop.terms for i, _ in term), default=-1) + 1
        n_qubits = max(1, n_qubits)

        decomposition: List[tuple] = []
        for term, coeff in qop.terms.items():
            string = ["I"] * n_qubits
            for index, pauli in term:
                string[index] = pauli
            decomposition.append(("".join(string), coeff))
        return decomposition

    @staticmethod
    def _preprocess_decomposition(decomposition) -> tuple:
        """Split ``[(unitary_spec, coeff)]`` into magnitudes and Select entries."""
        if not decomposition:
            raise ValueError("LCU decomposition cannot be empty.")

        coeffs: List[float] = []
        raw_coeffs: List[complex] = []
        unitaries: List[Any] = []

        for unitary, coeff in decomposition:
            c = complex(coeff)
            raw_coeffs.append(c)
            coeffs.append(float(abs(c)))
            unitaries.append((float(np.angle(c)), unitary))

        return coeffs, unitaries, raw_coeffs

    @staticmethod
    def _preprocess_explicit_lcu(
        *,
        coefficients: Sequence[complex],
        unitaries: Sequence[Any],
    ) -> tuple:
        if not unitaries:
            raise ValueError("unitaries list cannot be empty.")

        coeff_array = np.asarray(coefficients, dtype=complex)
        if coeff_array.ndim != 1:
            raise ValueError("coefficients must be a one-dimensional sequence.")
        if len(coeff_array) != len(unitaries):
            raise ValueError(
                "coefficients and unitaries must have the same length: "
                f"{len(coeff_array)} != {len(unitaries)}."
            )
        if not np.all(np.isfinite(coeff_array.real)) or not np.all(np.isfinite(coeff_array.imag)):
            raise ValueError("coefficients must be finite.")

        coeffs: List[float] = []
        raw_coeffs: List[complex] = []
        select_unitaries: List[Any] = []

        for coeff, unitary in zip(coeff_array, unitaries, strict=True):
            c = complex(coeff)
            raw_coeffs.append(c)
            coeffs.append(float(abs(c)))
            select_unitaries.append(
                BlockEncodingBlock._add_phase_to_select_entry(
                    unitary,
                    float(np.angle(c)),
                )
            )

        return coeffs, select_unitaries, raw_coeffs

    @staticmethod
    def _add_phase_to_select_entry(entry: Any, phase: float) -> Any:
        """Fold a coefficient phase into a SelectBlock-compatible entry."""
        if isinstance(entry, tuple) and len(entry) == 2:
            existing_phase, unitary = entry
            return (float(existing_phase) + phase, unitary)
        return (phase, entry)
