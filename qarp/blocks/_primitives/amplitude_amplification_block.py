"""Phase-exact amplitude-amplification iterate."""

from copy import deepcopy
from typing import List, Optional

import qarpx as qx

from .._block import AnyBlock, CompositeBlockBase
from .reflection_block import ReflectionBlock


def validate_amplification_blocks(state_preparation: AnyBlock, oracle: AnyBlock) -> int:
    """Validate the shared structural contract and return its register width.

    Shared by :class:`AmplitudeAmplificationBlock` and the composite
    amplitude-amplification algorithm, which must reject identical inputs.
    """
    if not isinstance(state_preparation, qx.Block):
        raise TypeError("state_preparation must be a Block instance")
    if not isinstance(oracle, qx.Block):
        raise TypeError("oracle must be a Block instance")
    if state_preparation.n_qubits < 1:
        raise ValueError("state_preparation must act on at least one qubit")
    if oracle.n_qubits < 1:
        raise ValueError("oracle must act on at least one qubit")
    if state_preparation.n_qubits != oracle.n_qubits:
        raise ValueError("state_preparation and oracle must act on the same number of qubits")
    # The iterate reflects about |0…0> on the *whole* register, ancillas included;
    # a postselected prep only holds its state on one ancilla branch.
    if getattr(state_preparation, "ancilla_postselection", None) is not None:
        raise ValueError(
            "amplitude amplification needs a deterministic state preparation "
            "(ancilla_postselection is None): the reflection about |0...0> would "
            f"act on the ancillas of {type(state_preparation).__name__}"
        )
    return state_preparation.n_qubits


def validate_power(power: int) -> int:
    """Validate and return a non-negative amplification power (rejects ``bool``)."""
    if isinstance(power, bool) or not isinstance(power, int):
        raise TypeError("power must be an integer")
    if power < 0:
        raise ValueError("power must be non-negative")
    return power


class AmplitudeAmplificationBlock(CompositeBlockBase):
    r"""The phase-exact amplitude-amplification iterate.

    For a state-preparation unitary ``A`` and a good-state phase oracle
    ``O_good = I - 2 Pi_good``, this block implements exactly

    ``Q = A R0 A_dagger O_good``,

    where :class:`ReflectionBlock` supplies
    ``R0 = 2|0...0><0...0| - I``.  Consequently, the circuit-time child order
    is ``O_good``, ``A_dagger``, ``R0``, ``A``.

    The oracle contract is mathematical: ``oracle`` must be a unitary block
    with the stated phase convention.  The constructor checks its type and
    width but deliberately does not build a dense matrix to prove its
    semantics — that proof is exponential in the register width, so the
    phase convention is the caller's promise.  In particular, a raw
    ``ReflectionBlock`` about the good subspace implements
    ``2 Pi_good - I = -(I - 2 Pi_good)`` — the exact *negative* of a
    good-state oracle.  To use one as an oracle, compose it with a
    ``gphase(pi)`` block to restore the sign.  Getting this wrong is
    invisible in standalone sampling (probabilities are phase-blind) but
    shifts every controlled eigenphase by one half, which silently corrupts
    amplitude estimation built on the controlled iterate.

    Caller-owned inputs are deep-copied at construction.  Building this block
    therefore does not build, retarget, or otherwise mutate either input.

    This convention is Eq. (1) of Brassard, Hoyer, Mosca, and Tapp,
    *Quantum Amplitude Amplification and Estimation*,
    arXiv:quant-ph/0005055.  Their zero-state reflection is
    ``S0 = I - 2|0><0|`` and their iterate is ``-A S0 A^-1 S_chi``.
    OpenQARP's ``ReflectionBlock`` is ``R0 = -S0``, yielding the exact form above.

    ``power`` repeats the iterate: the block implements ``Q^power``.  The default
    ``power=1`` is the single iterate ``Q``; ``power=0`` is the identity (empty
    circuit).  This is the only knob a fixed-schedule amplitude-amplification
    consumer needs — Grover applies ``Q^k`` after a uniform preparation, and
    maximum-likelihood amplitude estimation runs several powers ``Q^{m_k}``
    (including ``m_0 = 0``) after ``A``.

    Args:
        state_preparation: Unitary ``A`` preparing the initial state from
            ``|0...0>``.
        oracle: Unitary implementing exactly ``I - 2 Pi_good`` on the same
            register as ``state_preparation``.
        target_qubits: Optional placement of the complete iterate.
        name: Block name.
        power: Non-negative number of times to repeat the iterate ``Q`` (default
            ``1``; ``0`` is the identity). This option is keyword-only.
    """

    def __init__(
        self,
        state_preparation: AnyBlock,
        oracle: AnyBlock,
        target_qubits: Optional[List[int]] = None,
        name: str = "AmplitudeAmplification",
        *,
        power: int = 1,
    ) -> None:
        n_qubits = validate_amplification_blocks(state_preparation, oracle)
        self.power = validate_power(power)
        self.state_preparation = deepcopy(state_preparation)
        self.oracle = deepcopy(oracle)
        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        local_qubits = list(range(self.n_qubits))

        # power copies of Q = A R0 A_dagger O_good; power=0 emits nothing (identity).
        for _ in range(self.power):
            # Each child is an independent copy: materialising a pending dagger or
            # symbol operation on one child cannot alter another occurrence of A.
            oracle = deepcopy(self.oracle)
            state_dagger = deepcopy(self.state_preparation).dagger()
            reflection = ReflectionBlock(self.n_qubits)
            state_preparation = deepcopy(self.state_preparation)

            for child in (oracle, state_dagger, reflection, state_preparation):
                # Input placement belongs to the caller's surrounding circuit;
                # inside Q, A and O_good both act on the complete local register.
                child.target_qubits = local_qubits
                self.add_wired_child(child)
