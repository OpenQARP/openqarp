"""Conformance suite for the ``PreparesKnownState`` declaration.

Every block that declares the contract is checked against its own declared
``|0…0⟩`` column: phase-exact amplitudes on ``state_qubits``, and the ancilla
claim verified rather than trusted.

What this suite does **not** do is prove a declaration *right* — it compares a
circuit with the declaration next to it.  §18's independent oracle stays the
job of the per-block tests in this directory (Dicke against ``comb()``, the
QRAM blocks against their dataset, and so on).  A wrong ``target_statevector()``
that matches a wrong circuit passes here and fails there.

Construction is deferred into each test: the parametrize list holds plain
strings, so no qarpx-backed object is retained by a module global or by pytest's
``CallSpec2`` for the session (CLAUDE.md, "Never build qarpx-backed objects at
module scope in tests").
"""

from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import pytest

import qarp.blocks as qb
import qarp.blocks._prepares_known_state as _registry
from qarp import PostSelection
from qarp.blocks import PreparesKnownState, declaring_blocks, prepares_known_state
from qarp.operators import BravyiKitaev

# ── Construction cases ──────────────────────────────────────────────────────
# name → case id → zero-arg factory.  Factories are called inside the test.

_CASES: Dict[str, Dict[str, Callable[[], Any]]] = {
    "ComputationalBasisStateBlock": {
        "1011": lambda: qb.ComputationalBasisStateBlock(basis_state=[1, 0, 1, 1]),
        "zero": lambda: qb.ComputationalBasisStateBlock(basis_state=[0, 0, 0]),
    },
    "MappedONVStateBlock": {
        "jw_1100": lambda: qb.MappedONVStateBlock(occupation_number_vector=[1, 1, 0, 0]),
    },
    "GHZLikeStateBlock": {
        "plain": lambda: qb.GHZLikeStateBlock(basis_state=[1, 1, 0, 1]),
        "dephased": lambda: qb.GHZLikeStateBlock(basis_state=[1, 1, 0, 1], dephase=True),
        "empty_mask": lambda: qb.GHZLikeStateBlock(basis_state=[0, 0, 0]),
    },
    "DickeStateBlock": {
        "n4_k2": lambda: qb.DickeStateBlock(n_qubits=4, hamming_weight=2),
        "n3_k1": lambda: qb.DickeStateBlock(n_qubits=3, hamming_weight=1),
        "n4_k4": lambda: qb.DickeStateBlock(n_qubits=4, hamming_weight=4),
        "n3_k0": lambda: qb.DickeStateBlock(n_qubits=3, hamming_weight=0),
    },
    "HypergraphStateBlock": {
        "mixed_order": lambda: qb.HypergraphStateBlock(
            n_qubits=4, edges=[(0, 1), (1, 2), (0, 1, 2)]
        ),
        "single_vertex": lambda: qb.HypergraphStateBlock(n_qubits=3, edges=[(1,), (0, 2)]),
    },
    "SynthesizedStateBlock": {
        "complex_2q": lambda: qb.SynthesizedStateBlock(
            2, [0.5, 0.5j, -0.5, 0.5 * np.exp(1j * 0.7)]
        ),
        "sparse_dict": lambda: qb.SynthesizedStateBlock(
            3, {(0, 0, 0): np.sqrt(0.3), (1, 0, 1): np.sqrt(0.7)}
        ),
    },
    "CVQRAMStateBlock": {
        "n3": lambda: qb.CVQRAMStateBlock({(1, 0, 0): np.sqrt(0.5), (0, 1, 1): np.sqrt(0.5)}),
    },
    "CVOQRAMStateBlock": {
        "n3": lambda: qb.CVOQRAMStateBlock({(1, 0, 1): 0.6, (0, 1, 0): -0.8}),
    },
    "SparseStateBlock": {
        "n3_real": lambda: qb.SparseStateBlock(
            3, {(1, 0, 0): np.sqrt(0.5), (0, 1, 1): np.sqrt(0.5)}
        ),
        "n4_complex": lambda: qb.SparseStateBlock(
            4,
            {
                (0, 0, 0, 0): 0.5,
                (1, 1, 0, 0): 0.5j,
                (0, 1, 1, 1): -0.5,
                (1, 0, 1, 1): 0.5 * np.exp(1j * 0.7),
            },
        ),
        "single_basis_state": lambda: qb.SparseStateBlock(3, {(0, 0, 0): 1j}),
    },
    "MultiONVStateBlock": {
        "jw_two_det": lambda: qb.MultiONVStateBlock(
            {(1, 1, 0, 0): np.sqrt(0.5), (1, 0, 0, 1): np.sqrt(0.5)}
        ),
        "bk_three_det": lambda: qb.MultiONVStateBlock(
            {(1, 1, 0, 0): 0.6, (1, 0, 1, 0): 0.8j, (0, 1, 1, 0): 0.5},
            mapping=BravyiKitaev(),
        ),
    },
    "SlaterDeterminantBlock": {
        "n4_m2": lambda: qb.SlaterDeterminantBlock(
            np.linalg.qr(np.array([[1.0, 0.3], [0.2, 1.0], [0.5, -0.4], [-0.1, 0.6]]))[0]
        ),
        "n3_m0": lambda: qb.SlaterDeterminantBlock(np.zeros((3, 0))),
        "n3_m3": lambda: qb.SlaterDeterminantBlock(np.linalg.qr(np.eye(3) + 0.1)[0]),
    },
    "CSFStateBlock": {
        "singlet_2open": lambda: qb.CSFStateBlock(
            n_spatial_orbitals=3,
            core_orbitals=[0],
            open_shell_orbitals=[1, 2],
            coupling_path=[0.5, 0],
            Ms=0,
        ),
        "triplet_2open_ms0": lambda: qb.CSFStateBlock(
            n_spatial_orbitals=3,
            core_orbitals=[0],
            open_shell_orbitals=[1, 2],
            coupling_path=[0.5, 1],
            Ms=0,
        ),
        "doublet_3open_pathB": lambda: qb.CSFStateBlock(
            n_spatial_orbitals=4,
            core_orbitals=[0],
            open_shell_orbitals=[1, 2, 3],
            coupling_path=[0.5, 1, 0.5],
            Ms=0.5,
        ),
        "closed_shell": lambda: qb.CSFStateBlock(
            n_spatial_orbitals=2,
            core_orbitals=[0, 1],
            open_shell_orbitals=[],
            coupling_path=[],
            Ms=0,
        ),
    },
    "LowRankStateBlock": {
        "truncated_rank1": lambda: qb.LowRankStateBlock(
            4,
            list(
                np.random.default_rng(1).normal(size=16)
                + 1j * np.random.default_rng(2).normal(size=16)
            ),
            cut=2,
            max_schmidt_rank=1,
        ),
        "truncated_rank3": lambda: qb.LowRankStateBlock(
            4,
            list(
                np.random.default_rng(3).normal(size=16)
                + 1j * np.random.default_rng(4).normal(size=16)
            ),
            cut=2,
            max_schmidt_rank=3,
        ),
        "full_rank_exact": lambda: qb.LowRankStateBlock(
            4,
            list(
                np.random.default_rng(5).normal(size=16)
                + 1j * np.random.default_rng(6).normal(size=16)
            ),
            cut=2,
        ),
        "asymmetric_cut": lambda: qb.LowRankStateBlock(
            5,
            list(
                np.random.default_rng(7).normal(size=32)
                + 1j * np.random.default_rng(8).normal(size=32)
            ),
            cut=1,
            max_schmidt_rank=2,
        ),
    },
    "MPSStateBlock": {
        "bell_like_2site": lambda: qb.MPSStateBlock(
            [
                np.array([[[0.6, 0.0], [0.0, 0.8]]], dtype=complex),
                np.array([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=complex),
            ]
        ),
        "product_state_3site": lambda: qb.MPSStateBlock(
            [
                np.array([[[1.0], [0.0]]], dtype=complex),
                np.array([[[0.0], [1.0]]], dtype=complex),
                np.array([[[1.0], [0.0]]], dtype=complex),
            ]
        ),
    },
    "UniformSuperpositionBlock": {
        "m11": lambda: qb.UniformSuperpositionBlock(11),
        "m1": lambda: qb.UniformSuperpositionBlock(1),
        "power_of_two_m8": lambda: qb.UniformSuperpositionBlock(8),
        "padded_register": lambda: qb.UniformSuperpositionBlock(5, n_qubits=5),
    },
    "PiecewiseLinearStateBlock": {
        "single_piece": lambda: qb.PiecewiseLinearStateBlock(3, [], [0.15], [0.2]),
        "two_pieces": lambda: qb.PiecewiseLinearStateBlock(3, [3], [0.1, 0.3], [0.0, -0.2]),
        "three_pieces": lambda: qb.PiecewiseLinearStateBlock(
            4, [4, 9], [0.05, -0.1, 0.2], [0.1, 0.5, -0.3]
        ),
    },
}

_IDS: List[Tuple[str, str]] = [(name, case) for name, cases in _CASES.items() for case in cases]


# ── The registry gate ───────────────────────────────────────────────────────


def _declaring_names(candidates: Iterable[Any]) -> Set[str]:
    """Names of the classes among ``candidates`` that carry the declaration.

    ``issubclass`` goes through the metaclass, so an undecorated subclass of a
    declaring block is reported even though it never registered.
    """
    return {
        obj.__name__
        for obj in candidates
        if isinstance(obj, type)
        and obj is not PreparesKnownState
        and issubclass(obj, PreparesKnownState)
    }


def _public_declaring_names() -> Set[str]:
    return _declaring_names(getattr(qb, name) for name in qb.__all__)


@pytest.fixture
def isolated_registry(monkeypatch):
    """Classes decorated inside a test must not outlive it in the registry."""
    monkeypatch.setattr(_registry, "_DECLARING", dict(_registry._DECLARING))


def test_every_declaring_block_has_a_conformance_case():
    """A new state-prep block is covered the moment it declares the contract.

    Failing here means a block declares ``PreparesKnownState`` but registers no
    construction case above, so nothing checks its column.  Add a case; do not
    remove the declaration.  The registry alone would miss an undecorated
    subclass, so the public surface is walked as well.
    """
    declaring = set(declaring_blocks()) | _public_declaring_names()
    missing = sorted(declaring - set(_CASES))
    assert not missing, (
        f"blocks declare PreparesKnownState but have no conformance case: {missing}. "
        f"Add one to _CASES in {__file__}."
    )


def test_conformance_cases_name_live_blocks():
    """The inverse: a case naming a block that no longer declares the contract."""
    stale = sorted(set(_CASES) - set(declaring_blocks()))
    assert not stale, f"conformance cases for non-declaring blocks: {stale}"


def test_undecorated_subclass_is_caught_by_the_walk_not_the_registry():
    """Inheritance carries the declaration but not the registration (10.1)."""

    class GHZSubclass(qb.GHZLikeStateBlock):
        pass

    block = GHZSubclass(basis_state=[1, 0, 1])
    assert isinstance(block, PreparesKnownState)
    assert "GHZSubclass" not in declaring_blocks()
    assert _declaring_names([GHZSubclass, object, 42]) == {"GHZSubclass"}


def test_decorating_a_second_class_with_a_registered_name_raises(isolated_registry):
    """A same-named class in another module must not silently overwrite (0.3)."""
    impostor = type("DickeStateBlock", (), {})
    with pytest.raises(ValueError, match="already registered"):
        prepares_known_state(impostor)
    assert declaring_blocks()["DickeStateBlock"] is qb.DickeStateBlock


def test_redecorating_the_same_class_is_a_no_op(isolated_registry):
    before = declaring_blocks()
    assert prepares_known_state(qb.DickeStateBlock) is qb.DickeStateBlock
    assert declaring_blocks() == before


# ── The fixed-bit rule for ancilla_postselection ────────────────────────────


def _ghz_with_one_ancilla(condition: PostSelection):
    """A declaring block claiming qubit 3 as an ancilla under ``condition``."""

    class Claimed(qb.GHZLikeStateBlock):
        @property
        def state_qubits(self):
            return (0, 1, 2)

        @property
        def ancilla_postselection(self):
            return condition

    block = Claimed(basis_state=[1, 1, 0, 1])
    block.build()
    return block


def test_prepared_statevector_rejects_sector_postselection():
    """A sector spec keeps full width, so no state on ``state_qubits`` exists."""
    block = _ghz_with_one_ancilla(PostSelection.hamming_weight([3], 0))
    with pytest.raises(ValueError, match=r"Claimed.*fixed-bit"):
        block.prepared_statevector()


def test_prepared_statevector_rejects_fixed_bits_on_wrong_qubits():
    block = _ghz_with_one_ancilla(PostSelection({2: 0}))
    with pytest.raises(ValueError, match=r"exactly ancilla_qubits=\(3,\)"):
        block.prepared_statevector()


# ── The column contract ─────────────────────────────────────────────────────


@pytest.mark.parametrize("class_name,case", _IDS, ids=[f"{n}-{c}" for n, c in _IDS])
def test_declaration_is_well_formed(class_name: str, case: str):
    """``state_qubits`` partitions the register and the target is a unit vector."""
    block = _CASES[class_name][case]()
    block.build()

    state, ancillas = block.state_qubits, block.ancilla_qubits
    assert sorted(state + ancillas) == list(range(block.n_qubits)), (
        "state_qubits and ancilla_qubits must partition the register"
    )
    assert list(state) == sorted(state), "state_qubits must be ascending"

    target = block.target_statevector()
    assert target.shape == (2 ** len(state),)
    assert np.isclose(np.linalg.norm(target), 1.0, atol=1e-10)


@pytest.mark.parametrize("class_name,case", _IDS, ids=[f"{n}-{c}" for n, c in _IDS])
def test_circuit_matches_declared_column(class_name: str, case: str):
    """The built circuit leaves exactly the declared state, global phase included.

    Compared elementwise rather than modulo phase: a prep's global phase is a
    physical relative phase under ``ControlledBlock``, which is precisely what
    ``AmplitudeEstimationBlock`` does to it (§13, §18).
    """
    block = _CASES[class_name][case]()
    block.build()

    prepared, probability = block.prepared_statevector()
    condition = block.ancilla_postselection

    if condition is None:
        # The deterministic claim is verified, not taken on trust: any ancilla
        # must return to |0> with certainty for the block to be control-safe.
        assert probability == pytest.approx(1.0, abs=1e-10), (
            f"{class_name} declares no ancilla_postselection but leaves "
            f"{probability:.6f} of the mass on |0> ancillas"
        )
    else:
        assert condition.is_fixed, "a sector condition cannot reduce to state_qubits"
        assert condition.qubits == block.ancilla_qubits, (
            "ancilla_postselection must fix exactly the ancilla qubits"
        )
        assert probability > 0.0

    target = block.target_statevector()
    if block.is_exact:
        np.testing.assert_allclose(prepared, target, atol=1e-10)
    else:
        infidelity = 1.0 - abs(np.vdot(target, prepared)) ** 2
        assert infidelity <= block.error_bound + 1e-9, (
            f"{class_name} declares error_bound={block.error_bound:.3e} but the built "
            f"circuit's infidelity to target_statevector() is {infidelity:.3e}"
        )
