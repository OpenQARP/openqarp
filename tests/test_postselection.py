"""PostSelection utility tests.

Conventions under test: LSB-first tuple keys (qubit q at position q),
statevector index bit q = qubit q, fixed-bit specs remove the selected
qubits from the output, sector specs keep full register width.
"""

import numpy as np
import pytest

import qarp
from qarp import PostSelection
from qarp.algorithms import Sampler
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine

ATOL = 1e-10


# ── Fixed-bit conditions on distributions ─────────────────────────────────


def test_fixed_bits_analytic():
    dist = {(0, 0, 0): 0.5, (1, 0, 1): 0.25, (1, 1, 1): 0.25}
    out = PostSelection({1: 0}).apply(dist)
    assert out.success_rate == pytest.approx(0.75)
    assert out.distribution == {
        (0, 0): pytest.approx(0.5 / 0.75),
        (1, 1): pytest.approx(0.25 / 0.75),
    }


def test_fixed_bits_multiple_qubits():
    dist = {(0, 0, 0, 0): 0.4, (1, 0, 0, 1): 0.4, (1, 1, 1, 0): 0.2}
    out = PostSelection({1: 0, 2: 0}).apply(dist)
    assert out.success_rate == pytest.approx(0.8)
    # Surviving qubits 0 and 3, ascending.
    assert out.distribution == {(0, 0): pytest.approx(0.5), (1, 1): pytest.approx(0.5)}


def test_zero_success_mass_returns_empty():
    out = PostSelection({0: 1}).apply({(0, 0): 1.0})
    assert out.success_rate == 0.0
    assert out.distribution == {}


def test_all_qubits_selected_yields_scalar_key():
    out = PostSelection({0: 1, 1: 0}).apply({(1, 0): 0.25, (0, 0): 0.75})
    assert out.success_rate == pytest.approx(0.25)
    assert out.distribution == {(): pytest.approx(1.0)}


def test_out_of_range_qubit_raises():
    ps = PostSelection({5: 0})
    with pytest.raises(ValueError, match="only 2 qubits"):
        ps.apply({(0, 0): 1.0})
    with pytest.raises(ValueError, match="only 2 qubits"):
        ps.apply_statevector(np.array([1, 0, 0, 0], dtype=complex), 2)


def test_invalid_spec_rejected():
    with pytest.raises(ValueError):
        PostSelection({})
    with pytest.raises(ValueError):
        PostSelection({0: 2})
    with pytest.raises(ValueError):
        PostSelection({-1: 0})
    with pytest.raises(ValueError):
        PostSelection.hamming_weight([0, 0, 1], k=1)


def test_spec_is_hashable_value_type():
    a = PostSelection({0: 0, 3: 1})
    b = PostSelection({3: 1, 0: 0})
    assert a == b
    assert {a: "cached"}[b] == "cached"
    assert a != PostSelection({0: 0})


# ── Statevector path ──────────────────────────────────────────────────────


def _ghz(n):
    sv = np.zeros(2**n, dtype=complex)
    sv[0] = sv[-1] = 1 / np.sqrt(2)
    return sv


def test_ghz_collapse():
    sv_red, p = PostSelection({0: 0}).apply_statevector(_ghz(3), 3)
    assert p == pytest.approx(0.5)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(sv_red, expected, atol=ATOL)

    sv_red, p = PostSelection({0: 1}).apply_statevector(_ghz(3), 3)
    assert p == pytest.approx(0.5)
    expected = np.zeros(4, dtype=complex)
    expected[-1] = 1.0
    np.testing.assert_allclose(sv_red, expected, atol=ATOL)


def test_statevector_distribution_commutation():
    """Conditioning commutes with the Born rule: |apply_sv(ψ)|² == apply(|ψ|²)."""
    rng = np.random.default_rng(0)
    n = 4
    sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    sv /= np.linalg.norm(sv)
    born = {tuple((i >> q) & 1 for q in range(n)): abs(sv[i]) ** 2 for i in range(2**n)}
    ps = PostSelection({1: 0, 3: 1})

    sv_red, p_sv = ps.apply_statevector(sv, n)
    out = ps.apply(born)

    assert p_sv == pytest.approx(out.success_rate, abs=ATOL)
    for i, amp in enumerate(sv_red):
        key = tuple((i >> j) & 1 for j in range(n - 2))
        assert abs(amp) ** 2 == pytest.approx(out.distribution.get(key, 0.0), abs=ATOL)


def test_statevector_zero_success():
    sv = np.zeros(4, dtype=complex)
    sv[0] = 1.0  # |00⟩
    sv_red, p = PostSelection({0: 1}).apply_statevector(sv, 2)
    assert p == 0.0
    np.testing.assert_allclose(sv_red, np.zeros(2), atol=ATOL)


def test_statevector_shape_validated():
    with pytest.raises(ValueError, match="expected"):
        PostSelection({0: 0}).apply_statevector(np.zeros(3, dtype=complex), 2)


# ── Sector conditions ─────────────────────────────────────────────────────


def test_hamming_weight_sector_on_distribution_keeps_full_keys():
    dist = {(0, 1): 0.3, (1, 0): 0.2, (0, 0): 0.5}
    out = PostSelection.hamming_weight([0, 1], k=1).apply(dist)
    assert out.success_rate == pytest.approx(0.5)
    assert out.distribution == {(0, 1): pytest.approx(0.6), (1, 0): pytest.approx(0.4)}


def test_hamming_weight_sector_rates_on_singlet():
    singlet = np.zeros(4, dtype=complex)
    singlet[1] = 1 / np.sqrt(2)  # |01⟩ (LSB: qubit0=1)
    singlet[2] = -1 / np.sqrt(2)  # |10⟩

    in_sector = PostSelection.hamming_weight([0, 1], k=1)
    sv_out, p = in_sector.apply_statevector(singlet, 2)
    assert p == pytest.approx(1.0)
    # Sector projection keeps full width and, here, the full state.
    assert sv_out.shape == (4,)
    np.testing.assert_allclose(sv_out, singlet, atol=ATOL)

    _, p0 = PostSelection.hamming_weight([0, 1], k=0).apply_statevector(singlet, 2)
    assert p0 == 0.0


def test_parity_sector_on_bell():
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    _, p_even = PostSelection.parity([0, 1], even=True).apply_statevector(bell, 2)
    _, p_odd = PostSelection.parity([0, 1], even=False).apply_statevector(bell, 2)
    assert p_even == pytest.approx(1.0)
    assert p_odd == 0.0


def test_sector_projection_renormalises():
    """W-like state: project onto the k=1 sector of the first two qubits."""
    sv = np.zeros(8, dtype=complex)
    sv[0b001] = sv[0b010] = sv[0b100] = 1 / np.sqrt(3)  # |100⟩+|010⟩+|001⟩ (LSB labels)
    ps = PostSelection.hamming_weight([0, 1], k=1)
    sv_out, p = ps.apply_statevector(sv, 3)
    assert p == pytest.approx(2 / 3)
    assert np.linalg.norm(sv_out) == pytest.approx(1.0)
    assert sv_out[0b001] == pytest.approx(1 / np.sqrt(2))
    assert sv_out[0b010] == pytest.approx(1 / np.sqrt(2))


# ── Engine round-trip: sampled vs EXACT ───────────────────────────────────


def _bell_ket():
    b = SimpleBlock(2, name="bell")
    b.h(0)
    b.cx(0, 1)
    b.build()
    return b


def test_sampled_vs_exact_agreement():
    ps = PostSelection({0: 0})

    exact = Sampler(ket=_bell_ket(), n_shots=qarp.EXACT)
    eng = QarpEngine()
    eng.build([exact])
    out_exact = ps.apply(eng.run()[0])
    assert out_exact.success_rate == pytest.approx(0.5, abs=ATOL)
    assert out_exact.distribution == {(0,): pytest.approx(1.0, abs=ATOL)}

    sampled = Sampler(ket=_bell_ket())
    eng = QarpEngine(n_shots=4000, seed=0)
    eng.build([sampled])
    out_sampled = ps.apply(eng.run()[0])
    assert out_sampled.success_rate == pytest.approx(0.5, abs=0.05)
    # Bell correlations: conditioning q0=0 forces q1=0.
    assert out_sampled.distribution == {(0,): 1.0}
