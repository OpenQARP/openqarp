from copy import deepcopy
from typing import List, Optional

import qarpx as qx

from .._block import (
    AnyBlock,
    CompositeBlockBase,
    ControlledBlock,
    MeasureBlock,
    SimpleBlock,
)


class HadamardTestBlock(CompositeBlockBase):
    """Hadamard test for ``⟨ψ|U|ψ⟩`` using one ancilla qubit.

    Pattern B (composite).  Layout: qubit 0 is the ancilla; qubits
    ``1 .. state.n_qubits`` hold the state register.

    The circuit:

    1. ``H`` on ancilla.
    2. ``state`` preparation on the state register.
    3. ``C-U`` on ``(ancilla, state)`` controlled on ``ancilla = |1⟩``.
    4. (optional) ``C-U†`` on ``(ancilla, state)`` controlled on ``ancilla = |0⟩``.
    5. (optional) ``Sdg`` on ancilla — switches the post-measurement basis
       so the ancilla expectation gives the imaginary part of ``⟨ψ|U|ψ⟩``.
    6. ``H`` on ancilla.
    7. (optional) ``Measure`` ancilla into cbit 0.

    The ``unitary`` (and ``unitary_dagger``, if provided)
    must be qarpx blocks whose flattened command stream uses gates that
    ``ControlledBlock`` can lift to single-controlled form
    (X, Y, Z, H, S, Sdg, T, Tdg, Rx, Ry, Rz, P, U, CX, SWAP, GPhase,
    Barrier).  Other gates would need to be decomposed first.
    """

    def __init__(
        self,
        state: AnyBlock,
        unitary: AnyBlock,
        unitary_dagger: Optional[AnyBlock] = None,
        estimate_imaginary: bool = False,
        measure: bool = False,
        target_qubits: Optional[List[int]] = None,
        name: str = "HadamardTest",
    ):
        """Build the Hadamard test circuit.

        Args:
            state: Block preparing the ket state on the state register.
            unitary: The unitary operator U to test.  Must have the same
                ``n_qubits`` as ``state``.
            unitary_dagger: Optional U†.  When provided, the test estimates
                ``⟨ψ|U|ψ⟩ - ⟨ψ|U†|ψ⟩``-style quantities by also conjugating
                with the negative-control wrapping of U†.
            estimate_imaginary: If True, applies an Sdg before the final H so
                the ancilla expectation gives ``Im⟨ψ|U|ψ⟩``.
            measure: If True, measures the ancilla into cbit 0.
            target_qubits: standard Block kwarg.
        """
        if not isinstance(state, qx.Block):
            raise TypeError("state must be a Block instance")
        if not isinstance(unitary, qx.Block):
            raise TypeError("unitary must be a Block instance")
        if unitary_dagger is not None and not isinstance(unitary_dagger, qx.Block):
            raise TypeError("unitary_dagger must be a Block instance or None")
        if state.n_qubits != unitary.n_qubits:
            raise ValueError(
                f"state ({state.n_qubits} qubits) and unitary ({unitary.n_qubits} qubits) "
                "must act on the same number of qubits"
            )
        if unitary_dagger is not None and unitary_dagger.n_qubits != unitary.n_qubits:
            raise ValueError("unitary_dagger and unitary must have the same n_qubits")

        n_state = state.n_qubits
        super().__init__(
            n_qubits=1 + n_state,
            target_qubits=target_qubits,
            name=name,
        )

        self.state = state
        self.unitary = unitary
        self.unitary_dagger = unitary_dagger
        self.estimate_imaginary = estimate_imaginary
        self.measure_at_end = measure  # bool flag — distinct from `self.measure(q,c)`

    def build_vanilla(self) -> None:
        ancilla = 0
        state_qubits = list(range(1, self.n_qubits))

        # 1. H on ancilla
        ancilla_h_pre = SimpleBlock(1, name="AncillaH_pre")
        ancilla_h_pre.h(0)
        ancilla_h_pre.target_qubits = [ancilla]
        self.add_wired_child(ancilla_h_pre)

        # 2. State preparation on state register.  Deepcopy first so we don't
        # mutate the caller's shared ``state`` block (assigning target_qubits
        # to ``self.state.build()`` would otherwise rewrite its qubit frame
        # for any downstream caller that reused ``state`` directly).
        state_built = deepcopy(self.state.build())
        state_built.target_qubits = state_qubits
        self.add_child(state_built)

        # 3/4. Controlled branches.  For a transition/overlap test the two
        # branch unitaries are usually instances of the same ansatz (with
        # different parameters).  Controlling both complete circuits makes
        # every shared entangling gate a Toffoli-like operation.  Synthesize
        # the multiplexor command-by-command instead: gates that are exactly
        # common to both branches are applied once, unconditionally, while
        # only the differing runs are controlled.  The resulting unitary is
        # unchanged because the branch projectors are orthogonal.
        if self.unitary_dagger is None:
            unitary_built = deepcopy(self.unitary.build())
            c_u = ControlledBlock(unitary_built, num_controls=1, ctrl_state=[True], name="C-U")
            c_u.build()
            c_u.target_qubits = [ancilla] + state_qubits
            self.add_child(c_u)
        else:
            self._append_controlled_multiplexor(
                false_branch=self.unitary_dagger,
                true_branch=self.unitary,
                ancilla=ancilla,
                state_qubits=state_qubits,
            )

        # 5. Optional Sdg for imaginary-part estimation
        if self.estimate_imaginary:
            sdg = SimpleBlock(1, name="AncillaSdg")
            sdg.sdg(0)
            sdg.target_qubits = [ancilla]
            self.add_wired_child(sdg)

        # 6. Final H on ancilla
        ancilla_h_post = SimpleBlock(1, name="AncillaH_post")
        ancilla_h_post.h(0)
        ancilla_h_post.target_qubits = [ancilla]
        self.add_wired_child(ancilla_h_post)

        # 7. Optional measurement of ancilla into cbit 0
        if self.measure_at_end:
            self.add_child(MeasureBlock(ancilla, 0))

    @staticmethod
    def _flat_commands(block: AnyBlock) -> List[qx.Command]:
        """Return a private, local-frame copy of a block's commands."""
        built = deepcopy(block.build())
        built.target_qubits = list(range(block.n_qubits))
        return list(built.flatten())

    @staticmethod
    def _commands_match(left, right) -> bool:
        """Return true only for exactly identical branch commands."""
        if left is None or right is None:
            return False
        # Do not use Command.approx_equal here: sharing a nearly equal
        # rotation would change the transition amplitude rather than merely
        # change its compiled representation.
        try:
            left_cbits = list(getattr(left, "cbits", []))
            right_cbits = list(getattr(right, "cbits", []))
        except TypeError:
            return False
        if left_cbits != right_cbits:
            return False
        try:
            params_match = list(left.params) == list(right.params)
        except (TypeError, RuntimeError):
            params_match = False
        if not params_match:
            return False
        if left.gate != right.gate or list(left.qubits) != list(right.qubits):
            return False
        return True

    @staticmethod
    def _command_block(command, n_qubits: int, name: str) -> SimpleBlock:
        """Materialise one or more already-built commands in a leaf block."""
        block = SimpleBlock(n_qubits, name=name)
        block.set_commands(command if isinstance(command, list) else [command])
        block.build()
        return block

    def _append_controlled_multiplexor(
        self,
        false_branch: AnyBlock,
        true_branch: AnyBlock,
        ancilla: int,
        state_qubits: List[int],
    ) -> None:
        """Append ``|0><0|⊗false + |1><1|⊗true`` efficiently."""
        false_commands = self._flat_commands(false_branch)
        true_commands = self._flat_commands(true_branch)
        n_state = len(state_qubits)
        n_steps = max(len(false_commands), len(true_commands))

        false_run: List[qx.Command] = []
        true_run: List[qx.Command] = []
        common_run: List[qx.Command] = []

        def flush_common_run() -> None:
            if common_run:
                common = self._command_block(
                    common_run,
                    n_state,
                    name="Shared-branch-run",
                )
                common.target_qubits = state_qubits
                self.add_wired_child(common)
                common_run.clear()

        def flush_controlled_runs() -> None:
            if false_run:
                false_block = self._command_block(false_run, n_state, name="C-Udg-run")
                controlled_false = ControlledBlock(
                    false_block, num_controls=1, ctrl_state=[False], name="C-Udg"
                )
                controlled_false.build()
                controlled_false.target_qubits = [ancilla] + state_qubits
                self.add_child(controlled_false)
                false_run.clear()
            if true_run:
                true_block = self._command_block(true_run, n_state, name="C-U-run")
                controlled_true = ControlledBlock(
                    true_block, num_controls=1, ctrl_state=[True], name="C-U"
                )
                controlled_true.build()
                controlled_true.target_qubits = [ancilla] + state_qubits
                self.add_child(controlled_true)
                true_run.clear()

        for index in range(n_steps):
            false_command = false_commands[index] if index < len(false_commands) else None
            true_command = true_commands[index] if index < len(true_commands) else None

            if self._commands_match(false_command, true_command):
                flush_controlled_runs()
                common_run.append(false_command)
            else:
                flush_common_run()
                if false_command is not None:
                    false_run.append(false_command)
                if true_command is not None:
                    true_run.append(true_command)

        flush_common_run()
        flush_controlled_runs()

    def __repr__(self) -> str:
        part = "Im" if self.estimate_imaginary else "Re"
        return (
            f"HadamardTestBlock(estimate_{part.lower()}={self.estimate_imaginary}, "
            f"measure={self.measure_at_end})"
        )
