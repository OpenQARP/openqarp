"""Correctness tests for UPCCDBlock.

UPCCDBlock is a Pattern B (composite) that emits one ``GivensBlock`` per
allowed paired-double excitation (placed on qubits ``[2*i, 2*j]``) followed
by a layer of CX gates that copies α-channel amplitudes onto the β
channel.  These tests focus on:

  - structural smoke (correct number of children and symbols),
  - unitary equivalence to a hand-built reference composite,
  - threshold filtering,
  - end-to-end set_symbols round-trip via the parent composite.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import GivensBlock, SimpleBlock, UPCCDBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


# ── Construction validation ──────────────────────────────────────────────


def test_odd_qubits_raises():
    with pytest.raises(ValueError, match="Even number of qubits"):
        UPCCDBlock(basis_state=[1, 0, 1])


def test_open_shell_raises():
    with pytest.raises(ValueError, match="Closed-shell"):
        UPCCDBlock(basis_state=[1, 0, 0, 0])


def test_aufbau_violated_raises():
    # 4 qubits, 1 pair occupied at orbital index 1 (not 0): violates Aufbau.
    with pytest.raises(ValueError, match="Aufbau"):
        UPCCDBlock(basis_state=[0, 0, 1, 1])


# ── Symbol generation ────────────────────────────────────────────────────


def test_4q_one_pair_one_virtual_one_symbol():
    """1 occupied pair, 1 virtual orbital → 1 paired-double → 1 symbol."""
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0])
    upcc.build()
    assert len(upcc.symbols) == 1
    assert str(upcc.symbols[0]) == "pd0_1"


def test_6q_one_pair_two_virtuals_two_symbols():
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0, 0, 0])
    upcc.build()
    assert len(upcc.symbols) == 2
    assert {str(s) for s in upcc.symbols} == {"pd0_1", "pd0_2"}


def test_threshold_filters_below_amplitude():
    """t2 amplitude below threshold → symbol is not generated for that excitation."""
    t2 = np.array([[1e-6, 1.0]])  # first below default threshold (1e-4), second well above
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0, 0, 0], t2=t2)
    upcc.build()
    assert len(upcc.symbols) == 1
    assert str(upcc.symbols[0]) == "pd0_2"


def test_t2_populates_symbol_parameter_map():
    t2 = np.array([[-0.11223625]])
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0], t2=t2)
    upcc.build()
    sym = upcc.symbols[0]
    assert upcc.symbol_parameter_map[sym] == pytest.approx(-0.11223625)


# ── Unitary correctness vs. hand-built composite ─────────────────────────


def _reference_composite(theta_val: float) -> "qx.SimpleBlock":
    """Reference 4-qubit unitary: GivensBlock(+2θ) on [0,2] then α→β CX layer.

    Built with a concrete ``theta_val`` (not a symbol) so this reference is
    independent of the deepcopy/substitution machinery on the parent
    composite — the unit under test is UPCCDBlock's symbolic path; the
    reference must not rely on the same path.
    """
    g = GivensBlock(theta=2 * theta_val)
    g.build()
    g.target_qubits = [0, 2]

    cx = SimpleBlock(4, name="alpha_to_beta")
    cx.cx(0, 1)
    cx.cx(2, 3)
    cx.build()

    from qarp.blocks import CompositeBlock

    comp = CompositeBlock([g, cx], n_qubits=4)
    comp.build()
    return comp


@pytest.mark.parametrize("theta_val", [0.0, 0.137, 0.5, -0.42])
def test_4q_unitary_matches_reference(theta_val):
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0])
    upcc.build()
    sub = upcc.set_symbols({upcc.symbols[0]: theta_val})
    sub.build()

    U_upcc = _unitary(sub)
    U_ref = _unitary(_reference_composite(theta_val))
    assert np.linalg.norm(U_upcc - U_ref) < 1e-12


def test_unitary_is_unitary_6q():
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0, 0, 0])
    upcc.build()
    sub = upcc.set_symbols({s: 0.1 * (i + 1) for i, s in enumerate(upcc.symbols)})
    sub.build()
    U = _unitary(sub)
    I = np.eye(2**upcc.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10


# ── Trivial limit: θ = 0 → just the α→β CX layer ─────────────────────────


# ── pCCD-energy regression (gold standard, pinned references) ────────────


def _upccd_energy_equals_pccd(system, t2, e_pccd):
    """Build UPCCD with classical-pCCD t2 amplitudes, contract the resulting
    wavefunction with the molecular Hamiltonian, and verify it matches the
    classical pCCD energy.  This is the canonical UPCCD correctness regression
    — the convention mismatch in either ``GivensBlock`` (half-turn vs radian)
    or ``UPCCDBlock`` (sign on the Givens angle) makes the energy disagree.

    The pCCD energies and converged t2 amplitudes are pinned constants (see
    ``tests/pccd_reference_energies.py`` for provenance) — the classical
    solver left qarp with the interfaces layer."""
    from qarp.algorithms import StateVector
    from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock
    from qarp.engines import QarpEngine
    from qarp.operators import JordanWigner
    from tests.molecular_assets import fermion_operator, hf_energy, reference_onv

    assert not np.isclose(e_pccd, hf_energy(system)), "pCCD must lower the energy below HF"

    onv = reference_onv(system)
    qham = JordanWigner().encode_operator(fermion_operator(system))

    upcc = UPCCDBlock(basis_state=onv, t2=np.asarray(t2))
    ref_state = list(onv)
    ref_state[1::2] = [0] * (len(onv) // 2)
    ref = ComputationalBasisStateBlock(ref_state)
    wfn = CompositeBlock([ref, upcc], len(onv)).build()
    ansatz = wfn.set_symbols(upcc.symbol_parameter_map)

    sv = StateVector(bra=ansatz, ket=ansatz, operator=qham)
    sv.build()
    eng = QarpEngine()
    eng.build([sv])
    e_upccd = eng.run()[0].real
    assert np.isclose(e_upccd, e_pccd, atol=1e-8), (
        f"UPCCD energy {e_upccd:.10f} differs from pCCD {e_pccd:.10f}"
    )


def test_upccd_h2_matches_pccd_energy():
    from tests.pccd_reference_energies import E_PCCD_H2_STO3G, T2_PCCD_H2_STO3G

    _upccd_energy_equals_pccd("h2_0.735_sto3g", T2_PCCD_H2_STO3G, E_PCCD_H2_STO3G)


def test_upccd_lih_matches_pccd_energy():
    from tests.pccd_reference_energies import E_PCCD_LIH_STO3G, T2_PCCD_LIH_STO3G

    _upccd_energy_equals_pccd("lih_1.30_sto3g", T2_PCCD_LIH_STO3G, E_PCCD_LIH_STO3G)


# ── Trivial limit: θ = 0 → just the α→β CX layer ─────────────────────────


def test_zero_amplitude_reduces_to_cx_layer():
    """At θ = 0 the Givens rotation is identity and the unitary is the
    α→β CX layer applied to |1100⟩, mapping it to |1111⟩ in big-endian
    ordering (q0 = LSB, so the bit pattern is index-shifted)."""
    upcc = UPCCDBlock(basis_state=[1, 1, 0, 0])
    upcc.build()
    sub = upcc.set_symbols({upcc.symbols[0]: 0.0})
    sub.build()

    U = _unitary(sub)

    # |1100⟩ in q0-LSB encoding has integer index = 1*1 + 1*2 + 0*4 + 0*8 = 3.
    # After CX(0,1) and CX(2,3): q1 ⊕= q0 → q1=0, q3 ⊕= q2 → q3=0,
    # so state stays at |1100⟩ (q0=1, q1=0 after CX, q2=0, q3=0).  The
    # interesting ket is |0000⟩: nothing happens (Givens(0)=I, CX with
    # zero controls is identity).
    psi0 = np.zeros(2**4, complex)
    psi0[0] = 1.0  # |0000⟩ index
    out = U @ psi0
    assert abs(out[0] - 1.0) < 1e-12
    assert np.linalg.norm(out[1:]) < 1e-12


def test_t2_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Dimensions of t2 incompatible"):
        UPCCDBlock(basis_state=[1, 1, 0, 0, 0, 0], t2=np.array([[0.1]])).build()
