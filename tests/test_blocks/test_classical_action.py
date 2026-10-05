"""The ``classical_action`` protocol and the permutations structured execution
finds in block trees (§13, §14 *Structured execution*).

Oracles: analytic bit maps and modular arithmetic, and ``unitary_matrix()``
(the §14 oracle path, never planned)."""

import inspect

import numpy as np
import pytest

import qarp.blocks as qb
from qarp import _program
from qarp.blocks import CompositeBlock, ControlledBlock, ModularMultiplicationBlock, SimpleBlock


@pytest.fixture(autouse=True)
def _plan_small_registers(monkeypatch):
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)


def _permutation_of(unitary: np.ndarray) -> np.ndarray:
    """The map a unitary applies to basis states; asserts it is one, phase included."""
    rows = np.argmax(np.abs(unitary), axis=0)
    np.testing.assert_allclose(unitary[rows, np.arange(unitary.shape[1])], 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(unitary).sum(axis=0), 1.0, atol=1e-12)
    return rows


def _only_permutation(block) -> np.ndarray:
    """The single permutation table the planner lowers ``block`` to."""
    plan = _program.plan(block, block.n_qubits)
    assert plan is not None and plan.program().kinds() == ["permutation"]
    x = np.arange(1 << block.n_qubits)
    psi = np.zeros(1 << block.n_qubits, dtype=complex)
    image = np.empty_like(x)
    for i in x:
        psi[:] = 0.0
        psi[i] = 1.0
        out = block.statevector(psi, structured=True)
        image[i] = int(np.argmax(np.abs(out)))
        assert abs(out[image[i]] - 1.0) < 1e-12
    return image


# ── declared actions ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("m", "N"), [(2, 5), (3, 7), (5, 8), (7, 15)])
def test_modular_multiplication_action_is_multiplication_mod_n(m, N):
    block = ModularMultiplicationBlock(m, N)
    x = np.arange(1 << block.n_qubits)
    expected = np.array([m * v % N if v < N else v for v in x])
    np.testing.assert_array_equal(block.classical_action(x), expected)


# Every block class overriding classical_action, with small instances.
DECLARING = {
    ModularMultiplicationBlock: [
        lambda: ModularMultiplicationBlock(2, 5),
        lambda: ModularMultiplicationBlock(7, 15),
        lambda: ModularMultiplicationBlock(3, 32),
    ],
}


def _overrides_classical_action(cls) -> bool:
    action = getattr(cls, "classical_action", None)
    return action is not None and not getattr(action, "_qarp_default", False)


def test_every_declaring_block_class_is_registered():
    declaring = {
        cls
        for _, cls in inspect.getmembers(qb, inspect.isclass)
        if issubclass(cls, (SimpleBlock, qb.CompositeBlockBase))
        and _overrides_classical_action(cls)
    }
    assert declaring == set(DECLARING)


@pytest.mark.parametrize("factory", [f for fs in DECLARING.values() for f in fs])
def test_declared_action_matches_the_unitary(factory):
    block = factory().build()
    x = np.arange(1 << block.n_qubits)
    np.testing.assert_array_equal(
        block.classical_action(x), _permutation_of(np.asarray(block.unitary_matrix()))
    )


# ── permutations found in gate streams ──────────────────────────────────────


class _ClassicalGates(SimpleBlock):
    def build_vanilla(self):
        self.x(1)
        self.cx(1, 3)
        self.ccx(0, 3, 2)
        self.swap(0, 2)
        self.cswap(3, 1, 0)
        self.mcx(0, 1, 3, 2)


def _bit(v, q):
    return (v >> q) & 1


def _analytic_classical_gates(v: int) -> int:
    def flip(v, q):
        return v ^ (1 << q)

    def swap(v, a, b):
        return v if _bit(v, a) == _bit(v, b) else flip(flip(v, a), b)

    v = flip(v, 1)
    if _bit(v, 1):
        v = flip(v, 3)
    if _bit(v, 0) and _bit(v, 3):
        v = flip(v, 2)
    v = swap(v, 0, 2)
    if _bit(v, 3):
        v = swap(v, 1, 0)
    if _bit(v, 0) and _bit(v, 1) and _bit(v, 3):
        v = flip(v, 2)
    return v


def test_classical_gates_plan_to_their_bit_map():
    block = _ClassicalGates(4).build()
    expected = [_analytic_classical_gates(v) for v in range(16)]
    np.testing.assert_array_equal(_only_permutation(block), expected)


class _PhaseAfterX(SimpleBlock):
    def build_vanilla(self):
        self.x(0)
        self.z(0)
        self.cz(0, 1)
        self.x(1)


def test_a_relative_phase_is_never_a_permutation():
    plan = _program.plan(_PhaseAfterX(2).build(), 2)
    assert plan is None or "permutation" not in plan.program().kinds()


# ── composition through the tree ────────────────────────────────────────────


def test_controlled_declared_block_lifts_by_the_controlled_unitary_contract():
    # §6.1: the inner map applies exactly where the control matches its state.
    inner = ModularMultiplicationBlock(2, 5)  # 3 work qubits
    block = ControlledBlock(inner, num_controls=1, ctrl_state=[False]).build()
    expected = []
    for v in range(16):
        ctrl, work = v & 1, v >> 1
        image = (2 * work % 5 if work < 5 else work) if ctrl == 0 else work
        expected.append((image << 1) | ctrl)
    np.testing.assert_array_equal(_only_permutation(block), expected)


def test_dagger_plans_to_the_inverse_permutation():
    forward = [_analytic_classical_gates(v) for v in range(16)]
    inverse = np.empty(16, dtype=int)
    inverse[forward] = np.arange(16)
    np.testing.assert_array_equal(_only_permutation(_ClassicalGates(4).build().dagger()), inverse)


def test_a_power_plans_to_the_composed_permutation():
    step = [_analytic_classical_gates(v) for v in range(16)]
    cubed = [step[step[step[v]]] for v in range(16)]
    block = (_ClassicalGates(4).build() ** 3).build()
    np.testing.assert_array_equal(_only_permutation(block), cubed)


class _Spread(SimpleBlock):
    def build_vanilla(self):
        self.h(0)
        self.h(1)
        self.cx(0, 1)


def test_a_declared_child_is_placed_by_its_parent(monkeypatch):
    # The H layers keep the root from being one permutation and its ten
    # qubits from one dense kernel, so the planner recurses and the
    # multiplier's declared action lands on qubits [4, 1, 5].
    calls = []
    declared = ModularMultiplicationBlock.classical_action

    def spy(self, indices):
        calls.append(self)
        return declared(self, indices)

    monkeypatch.setattr(ModularMultiplicationBlock, "classical_action", spy)
    block = CompositeBlock(
        [
            _Spread(2, target_qubits=[1, 4]),
            _Spread(2, target_qubits=[8, 9]),
            ModularMultiplicationBlock(3, 7, target_qubits=[4, 1, 5]),
            _ClassicalGates(4, target_qubits=[0, 6, 2, 7]),
            _Spread(2, target_qubits=[3, 6]),
        ],
        n_qubits=10,
    ).build()
    plan = _program.plan(block, 10)
    assert plan is not None and "permutation" in plan.program().kinds()
    assert calls, "the declared action was not used"
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    psi /= np.linalg.norm(psi)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


def test_a_daggered_declared_block_plans_to_the_inverse():
    # 7 · 13 = 1 (mod 15): the pending dagger must win over the declaration.
    block = ModularMultiplicationBlock(7, 15).build().dagger()
    expected = [13 * v % 15 if v < 15 else v for v in range(16)]
    np.testing.assert_array_equal(_only_permutation(block), expected)


class _Rotations(SimpleBlock):
    def build_vanilla(self):
        for q in range(self.n_qubits):
            self.ry(q, 0.1 + 0.2 * q)


@pytest.mark.parametrize("under_control", [False, True])
def test_a_declared_block_edited_after_build_is_planned_from_its_gates(under_control):
    # The declaration is trusted for the gates the block was built with; a Z
    # appended afterwards is not in it, so the planner must derive the span.
    edited = ModularMultiplicationBlock(7, 15, target_qubits=[0, 1, 2, 3]).build()
    edited.z(0)
    if under_control:
        edited = ControlledBlock(edited, num_controls=1, target_qubits=[9, 0, 1, 2, 3])
    block = CompositeBlock([_Rotations(10), edited], n_qubits=10).build()
    rng = np.random.default_rng(1)
    psi = rng.standard_normal(1 << 10) + 1j * rng.standard_normal(1 << 10)
    psi /= np.linalg.norm(psi)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )
