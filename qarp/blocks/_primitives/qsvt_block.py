from typing import List, Optional, Union

import numpy as np

import qarpx as qx
from qarp.operators import QubitOperator

from .._block import CompositeBlockBase, SimpleBlock
from .block_encoding_block import BlockEncodingBlock


class QSVTBlock(CompositeBlockBase):
    r"""Pattern B composite: Quantum Singular Value Transformation of an
    operator ``A`` driven by a sequence of projector-controlled-phase angles.

    For a polynomial of degree ``d = len(P_angles) - 1`` (per the QSVT
    convention), the assembled circuit is:

    Even ``d``::

        Π_φ₀ · BE† · Π_φ₁ · BE · Π_φ₂ · BE† · … · Π_φ_d

    Odd ``d``::

        Π_φ₀ · BE · Π_φ₁ · BE† · Π_φ₂ · BE · … · BE · Π_φ_d

    where ``BE`` block-encodes ``A / λ`` and ``Π_φ`` is a
    ``ProjectedControlPhaseBlock`` rotating the LCU-control subspace.

    Inputs are validated to be square / Hermitian / real-block-encoding (the
    standard QSVT preconditions).
    """

    def __init__(
        self,
        A: Union[np.ndarray, QubitOperator],
        P_angles: List[float],
        target_qubits: Optional[List[int]] = None,
        name: str = "QSVT",
    ):
        # ── Input validation: QSVTBlock requires a square, Hermitian, real A ──
        if isinstance(A, np.ndarray):
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise NotImplementedError(
                    "Current QSVT implementation only accepts square matrices/ops"
                )
            if not np.allclose(A, A.conj().T):
                raise NotImplementedError(
                    "Current QSVT implementation only accepts Hermitian matrices/ops"
                )
        elif not isinstance(A, QubitOperator):
            raise TypeError("Expected np.ndarray or QubitOperator in QSVTBlock")

        # Build a BE up-front to get the lambda and full-unitary matrix for
        # the real-block-encoding check; we discard the built circuit and
        # rebuild fresh BE / BE† inside ``build_vanilla``.
        be_probe = BlockEncodingBlock(A)
        be_probe.build()
        be_FU = np.array(qx.QarpSimulator().unitary_matrix(be_probe.flatten(), be_probe.n_qubits))
        lambda_factor = be_probe.lambda_factor()

        if np.linalg.norm(np.imag(be_FU)) > 1e-8:
            raise NotImplementedError("Current QSVT implementation only accepts real matrices/ops")
        if np.abs(lambda_factor - 1) > 1e-8:
            print(
                "Warning: lambda in QSVT is not 1. It is proceeded with A -> A / lambda. "
                f"lambda = {lambda_factor}"
            )

        # Stash inputs (post-rescaling) for ``build_vanilla``.
        self.A = A / lambda_factor if isinstance(A, np.ndarray) else A
        self.optimal_angles = list(P_angles)
        self.nqubits = be_probe.n_qubits

        # Determine the target-subspace dimension for the PCP projectors.
        if isinstance(self.A, np.ndarray):
            self.Adim = self.A.shape[0]
        else:
            # max qubit index in any term of A; if A is a constant, default to 1.
            max_q = max((i for term in self.A.terms for i, _ in term), default=-1)
            self.Adim = 2 ** (1 + max_q)

        super().__init__(
            n_qubits=self.nqubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        n_full = self.n_qubits
        full_targets = list(range(n_full))
        # ``Adim`` is the user-facing matrix dimension (may be non-power-of-2
        # for ndarray inputs).  The BE pads to ``2^ceil(log2(Adim))`` target
        # qubits, so the ancilla register has the matching size.
        n_anc = n_full - int(np.ceil(np.log2(self.Adim)))
        N_anc = 2**n_anc
        N_full = 2**n_full

        # Helper to add a "QSVT projector": phase ``e^{+iφ}`` on
        # ``ancilla = |0…0⟩`` indices, ``e^{-iφ}`` on the complement.  In
        # qarpx LSB convention, ``ancilla = 0`` indices are strided
        # (`i % 2^n_anc == 0`), so we cannot reuse ``ProjectedControlPhaseBlock``
        # — its "first ``dim`` states" semantics matches MSB layouts only.
        def _add_pcp(k: int) -> None:
            phi = self.optimal_angles[k]
            plus = np.exp(1j * phi)
            minus = np.exp(-1j * phi)
            diag = [plus if (i % N_anc == 0) else minus for i in range(N_full)]
            pcp = SimpleBlock(n_full, name=f"PCP_{k}")
            pcp.diagonal_unitary(diag)
            pcp.target_qubits = full_targets
            self.add_wired_child(pcp)

        # Helper to add a fresh BE (or BE†) on the full register.
        def _add_be(dagger: bool = False) -> None:
            be = BlockEncodingBlock(self.A)
            be.build()
            if dagger:
                # Lazy Python-wrapper dagger (deepcopy + flag): `add_child`
                # materialises it into concrete daggered commands, keeping
                # the child deepcopy-safe.
                be_use = be.dagger()
            else:
                be_use = be
            be_use.target_qubits = full_targets
            self.add_child(be_use)

        d = len(self.optimal_angles) - 1
        if d % 2 == 0:
            # Even degree: Π · BE† · Π · BE · Π · BE† · … · Π
            for k in range(d // 2):
                _add_pcp(2 * k)
                _add_be(dagger=True)
                _add_pcp(2 * k + 1)
                _add_be(dagger=False)
            _add_pcp(d)
        else:
            # Odd degree: Π · BE · Π · BE† · … · BE · Π
            _add_pcp(0)
            for k in range((d - 1) // 2):
                _add_be(dagger=True)
                _add_pcp(2 * k + 1)
                _add_be(dagger=False)
                _add_pcp(2 * k + 2)
            _add_be(dagger=False)
            _add_pcp(d)
