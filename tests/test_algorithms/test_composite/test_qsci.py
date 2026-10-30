"""QSCI, Phase 1 (docs/contributions/qsci_plan.md).

Every energy is checked against an independent reference: closed forms for the
two-site models, pyscf for H2, and dense diagonalisation of ``sparse_matrix()``
(which shares no code with the Pauli-walk build) for interlacing.
"""

import functools
import itertools
import math
import warnings

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

import qarp
from qarp.algorithms import QSCI, Sampler, StateVector
from qarp.algorithms._composite import qsci as qsci_module
from qarp.blocks import ComputationalBasisStateBlock, MultiONVStateBlock, SimpleBlock
from qarp.endianness import bits_to_label, label_to_bits
from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.functions import jordan_wigner
from qarp.operators.models import fermi_hubbard, transverse_field_ising

H2_GEOMETRY = "H 0 0 0; H 0 0 0.74"


@functools.lru_cache(maxsize=None)
def _h2_mean_field():
    # Deliberate cache of a pyscf object; no qarpx object is held here.
    from tests.pyscf_recipes import rhf

    return rhf(H2_GEOMETRY, "sto-3g")


def _sector_configurations(n_qubits, n_alpha, n_beta):
    return [
        bits
        for bits in itertools.product((0, 1), repeat=n_qubits)
        if sum(bits[0::2]) == n_alpha and sum(bits[1::2]) == n_beta
    ]


def _uniform(configurations):
    return {bits: 1.0 / len(configurations) for bits in configurations}


def _bits(label, n_qubits):
    return tuple(label_to_bits(label, n_qubits))


# ── Energies against independent oracles ───────────────────────────────


@pytest.mark.parametrize("model", ["hubbard", "tfim"])
def test_analytic_model_energies(model):
    if model == "hubbard":
        # Fermionic path: two-site Hubbard at half filling, (1, 1) sector.
        t, u = 1.0, 4.0
        algorithm = QSCI(
            fermi_hubbard([2], t, u),
            distribution=_uniform(_sector_configurations(4, 1, 1)),
            n_electrons=(1, 1),
        )
        expected = (u - math.sqrt(u**2 + 16 * t**2)) / 2
    else:
        # Qubit path, no sector filter: two-site transverse-field Ising.
        j, h = 1.0, 0.7
        algorithm = QSCI(
            transverse_field_ising([2], j, h),
            distribution=_uniform(list(itertools.product((0, 1), repeat=2))),
        )
        expected = -math.sqrt(j**2 + 4 * h**2)
    energies, _ = algorithm.build().run()
    assert energies[0] == pytest.approx(expected, abs=1e-12)


def test_hartree_fock_determinant_energy():
    pytest.importorskip("pyscf")
    from qarp.operators.pyscf import fermion_operator_from_mf, onv_from_mf

    mf = _h2_mean_field()
    hamiltonian = fermion_operator_from_mf(mf)
    hf = tuple(onv_from_mf(mf))
    assert hf == (1, 1, 0, 0)

    energies, _ = QSCI(hamiltonian, distribution={hf: 1.0}, n_electrons=(1, 1)).build().run()
    assert energies[0] == pytest.approx(mf.e_tot, abs=1e-10)

    # The bit-reversed string is the doubly excited determinant, so an
    # MSB-first label packing would land far from the Hartree-Fock energy.
    reversed_hf = hf[::-1]
    wrong = QSCI(hamiltonian, distribution={reversed_hf: 1.0}, n_electrons=(1, 1))
    assert wrong.build().run()[0][0] - mf.e_tot > 1.0


def test_full_support_matches_fci_roots():
    pytest.importorskip("pyscf")
    from pyscf import fci

    from qarp.operators.pyscf import fermion_operator_from_mf

    mf = _h2_mean_field()
    # singlet=False keeps the Ms=0 triplet components, which QSCI's Sz sector contains.
    expected = fci.FCI(mf, singlet=False).kernel(nroots=3)[0]
    energies, _ = (
        QSCI(
            fermion_operator_from_mf(mf),
            distribution=_uniform(_sector_configurations(4, 1, 1)),
            n_electrons=(1, 1),
            n_roots=3,
        )
        .build()
        .run()
    )
    np.testing.assert_allclose(energies, expected, atol=1e-10)


def test_fci_state_recovers_fci_energy():
    """The block path end to end.  QSCI discards amplitudes, so this sees the
    FCI vector's support and endianness, not its sign conventions."""
    pytest.importorskip("pyscf")
    from pyscf import fci

    from qarp.operators.pyscf import fermion_operator_from_mf, onv_coefficients_from_civec

    mf = _h2_mean_field()
    e_fci, vectors = fci.FCI(mf, singlet=False).kernel(nroots=1)
    coefficients = onv_coefficients_from_civec(vectors, 2, (1, 1))
    algorithm = QSCI(
        fermion_operator_from_mf(mf),
        ket=MultiONVStateBlock(coefficients),
        primitive=Sampler(n_shots=qarp.EXACT),
        n_electrons=(1, 1),
    ).build()
    energies, _ = algorithm.run()
    assert energies[0] == pytest.approx(e_fci, abs=1e-10)
    assert set(algorithm.configurations) == set(coefficients)


def test_interlacing_bounds():
    """Invariant, additional to the value oracles above: a subspace spectrum
    interlaces the full in-sector spectrum from both sides."""
    hamiltonian = fermi_hubbard([4], 1.0, 3.0)
    sector = _sector_configurations(8, 2, 2)
    dense = hamiltonian.sparse_matrix(8).toarray()
    indices = [bits_to_label(bits) for bits in sector]
    reference = scipy.linalg.eigvalsh(dense[np.ix_(indices, indices)])
    n_full = len(reference)

    rng = np.random.default_rng(7)
    for size in (4, 10, 20):
        chosen = [sector[i] for i in rng.choice(n_full, size=size, replace=False)]
        energies, _ = (
            QSCI(hamiltonian, distribution=_uniform(chosen), n_electrons=(2, 2), n_roots=3)
            .build()
            .run()
        )
        for k, mu in enumerate(energies):
            assert reference[k] <= mu + 1e-10
            assert mu <= reference[k + n_full - size] + 1e-10


# ── Selection mechanics ──────────────────────────────────────────────────


def _pipeline_distribution():
    # 6 qubits, sector (n_alpha, n_beta) = (2, 1): alpha on 0/2/4, beta on 1/3/5.
    # Dyadic weights, in-sector mass exactly 1/2, so renormalisation doubles them.
    weights = {
        7: 1 / 8,  # in-sector, 1/4 after renormalisation
        13: 1 / 8,  # in-sector, 1/4
        25: 3 / 32,  # in-sector, 3/16: kept only because filtering precedes the floor
        19: 1 / 16,  # in-sector, exactly 1/8 = the floor; ties with 22
        22: 1 / 16,  # in-sector, exactly 1/8; LSB and MSB tie-breaks disagree with 19
        28: 1 / 32,  # in-sector, 1/16: below the floor
        52: 0.0,  # in-sector, zero weight: never selected
        3: 1 / 4,  # leaks the alpha sublattice only
        15: 1 / 4,  # leaks the beta sublattice only
    }
    return {_bits(label, 6): weight for label, weight in weights.items()}


def _number_operator(n_modes):
    operator = FermionOperator()
    for mode in range(n_modes):
        operator += FermionOperator(f"{mode}^ {mode}")
    return operator


def test_selection_pipeline():
    distribution = _pipeline_distribution()

    def msb_label(bits):  # the MSB-first packing §1 warns against
        return int("".join(str(bit) for bit in bits), 2)

    # The tied pair orders one way under LSB labels and the other under MSB.
    assert bits_to_label(_bits(19, 6)) < bits_to_label(_bits(22, 6))
    assert msb_label(_bits(19, 6)) > msb_label(_bits(22, 6))

    def run(dist, max_subspace_dim):
        algorithm = QSCI(
            _number_operator(6),
            distribution=dist,
            n_electrons=(2, 1),
            prob_threshold=1 / 8,
            max_subspace_dim=max_subspace_dim,
        ).build()
        with pytest.warns(UserWarning, match="0.5 of the sampled probability"):
            algorithm.run()
        return algorithm

    capped = run(distribution, 4)
    assert capped.configurations == tuple(_bits(label, 6) for label in (7, 13, 19, 25))
    assert capped.in_sector_mass == 0.5
    assert capped.rejected == {_bits(3, 6): 0.25, _bits(15, 6): 0.25}
    assert capped.distribution == distribution

    uncapped = run(distribution, None)
    assert uncapped.configurations == tuple(_bits(label, 6) for label in (7, 13, 19, 22, 25))

    shuffled = dict(reversed(list(distribution.items())))
    assert run(shuffled, 4).configurations == capped.configurations


def test_block_pooling():
    # Equal weight per block at finite shots: basis states sample deterministically.
    states = ([1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0])
    pooled = QSCI(
        QubitOperator("Z0") + QubitOperator("Z3"),
        ket=[ComputationalBasisStateBlock(state) for state in states],
        primitive=Sampler(n_shots=64),
    ).build()
    pooled.run()
    assert pooled.distribution == {tuple(state): 1 / 3 for state in states}

    # Pool before filtering: a half-leaking block pooled with a clean one gives
    # a = 1/3, b = 2/3 in-sector, so a floor of 0.4 keeps only b.  Filtering each
    # block first would give 1/2 each and keep both.
    leaky = SimpleBlock(4)
    leaky.h(0)  # half (1,0,0,0) in-sector, half (0,0,0,0) out of the (1, 0) sector
    algorithm = QSCI(
        FermionOperator("0^ 0") + FermionOperator("2^ 2", 2.0),
        ket=[leaky, ComputationalBasisStateBlock([0, 0, 1, 0])],
        primitive=Sampler(n_shots=qarp.EXACT),
        n_electrons=(1, 0),
        prob_threshold=0.4,
    ).build()
    with pytest.warns(UserWarning):
        algorithm.run()
    assert algorithm.configurations == ((0, 0, 1, 0),)
    assert algorithm.in_sector_mass == pytest.approx(0.75, abs=1e-12)


def test_default_sampler():
    # No primitive: QSCI samples with a private Sampler().  <1|Z|1> = -1.
    algorithm = QSCI(QubitOperator("Z0"), ket=ComputationalBasisStateBlock([1, 0])).build()
    energies, coefficients = algorithm.run()
    assert algorithm.configurations == ((1, 0),)
    assert energies[0] == pytest.approx(-1.0, abs=1e-12)
    assert energies is algorithm.energies and coefficients is algorithm.coefficients

    algorithm.build()  # a rebuild clears the previous run's outputs
    assert algorithm.energies is None and algorithm.distribution is None


# ── Solver dispatch ──────────────────────────────────────────────────────


def test_solver_dispatch(monkeypatch):
    hubbard = fermi_hubbard([4], 1.0, 3.0)
    full = _uniform(_sector_configurations(8, 2, 2))

    def energies(hamiltonian, distribution, n_electrons, n_roots):
        return (
            QSCI(hamiltonian, distribution=distribution, n_electrons=n_electrons, n_roots=n_roots)
            .build()
            .run()[0]
        )

    dense = energies(hubbard, full, (2, 2), 3)
    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    np.testing.assert_allclose(energies(hubbard, full, (2, 2), 3), dense, atol=1e-10)

    # n_roots >= dim - 1 goes dense even above the cutoff; ARPACK would refuse it.
    small = _uniform(_sector_configurations(4, 1, 1))
    two_site = fermi_hubbard([2], 1.0, 4.0)
    forced = energies(two_site, small, (1, 1), 3)
    monkeypatch.undo()
    np.testing.assert_allclose(forced, energies(two_site, small, (1, 1), 3), atol=1e-12)

    # A real operator stored with complex coefficients is solved as a real matrix.
    labels = np.array(sorted(bits_to_label(bits) for bits in full), dtype=np.int64)
    built = qsci_module._subspace_matrix(jordan_wigner(hubbard), labels)
    assert np.iscomplexobj(built.data)
    assert not np.iscomplexobj(qsci_module._hermitian_real(built).data)

    # A genuinely complex Hermitian operator stays complex and solves correctly.
    both = {(0,): 0.5, (1,): 0.5}
    np.testing.assert_allclose(energies(QubitOperator("Y0"), both, None, 2), [-1.0, 1.0])

    # A non-Hermitian operator is refused rather than handed to a Hermitian solver.
    with pytest.raises(ValueError, match="not Hermitian"):
        energies(QubitOperator("X0", 1j), both, None, 1)


# ── Input validation ─────────────────────────────────────────────────────

_VALIDATION_CASES = [
    "both_inputs",
    "no_inputs",
    "primitive_with_distribution",
    "non_sampling_primitive",
    "partial_measurement",
    "reordered_measurement",
    "fermionic_without_sector",
    "ragged_block_widths",
    "empty_distribution",
    "ragged_keys",
    "non_binary_key",
    "negative_weight",
    "nan_weight",
    "non_unit_mass",
    "zero_max_subspace_dim",
    "bool_max_subspace_dim",
    "threshold_above_one",
    "nan_threshold",
    "zero_n_roots",
    "bool_n_roots",
    "bool_electron_count",
    "negative_electron_count",
    "too_many_electrons",
    "odd_width_sector",
    "hamiltonian_wider_than_register",
    "empty_subspace_after_filter",
    "n_roots_above_dimension",
    "empty_block_sequence",
    "non_block_ket",
    "run_before_build",
]


def _invalid_run(case):
    z = QubitOperator("Z0")
    fermionic = FermionOperator("0^ 0")
    good = {(1, 0): 1.0}
    block2 = ComputationalBasisStateBlock([1, 0])
    arguments = {
        "both_inputs": (z, dict(ket=block2, distribution=good)),
        "no_inputs": (z, {}),
        "primitive_with_distribution": (z, dict(distribution=good, primitive=Sampler())),
        "non_sampling_primitive": (z, dict(ket=block2, primitive=StateVector())),
        "partial_measurement": (z, dict(ket=block2, primitive=Sampler(measured_qubits=[0]))),
        "reordered_measurement": (
            z,
            dict(ket=block2, primitive=Sampler(measured_qubits=[1, 0])),
        ),
        "fermionic_without_sector": (fermionic, dict(distribution=good)),
        "ragged_block_widths": (
            z,
            dict(ket=[block2, ComputationalBasisStateBlock([1, 0, 0, 0])]),
        ),
        "empty_distribution": (z, dict(distribution={})),
        "ragged_keys": (z, dict(distribution={(1, 0): 0.5, (1,): 0.5})),
        "non_binary_key": (z, dict(distribution={(2, 0): 1.0})),
        "negative_weight": (z, dict(distribution={(1, 0): 1.5, (0, 0): -0.5})),
        "nan_weight": (z, dict(distribution={(1, 0): math.nan})),
        "non_unit_mass": (z, dict(distribution={(1, 0): 0.9})),
        "zero_max_subspace_dim": (z, dict(distribution=good, max_subspace_dim=0)),
        "bool_max_subspace_dim": (z, dict(distribution=good, max_subspace_dim=True)),
        "threshold_above_one": (z, dict(distribution=good, prob_threshold=1.5)),
        "nan_threshold": (z, dict(distribution=good, prob_threshold=math.nan)),
        "zero_n_roots": (z, dict(distribution=good, n_roots=0)),
        "bool_n_roots": (z, dict(distribution=good, n_roots=True)),
        "bool_electron_count": (fermionic, dict(distribution=good, n_electrons=(True, 0))),
        "negative_electron_count": (fermionic, dict(distribution=good, n_electrons=(-1, 0))),
        "too_many_electrons": (fermionic, dict(distribution=good, n_electrons=(2, 0))),
        "odd_width_sector": (z, dict(distribution={(1, 0, 0): 1.0}, n_electrons=(1, 0))),
        "hamiltonian_wider_than_register": (
            QubitOperator("Z3"),
            dict(distribution=good),
        ),
        "empty_subspace_after_filter": (
            fermionic,
            dict(distribution={(0, 0): 1.0}, n_electrons=(1, 0)),
        ),
        "n_roots_above_dimension": (z, dict(distribution=good, n_roots=2)),
        "empty_block_sequence": (z, dict(ket=[])),
        "non_block_ket": (z, dict(ket="not a block")),
    }
    if case == "run_before_build":
        return QSCI(z, distribution=good).run()
    hamiltonian, kwargs = arguments[case]
    return QSCI(hamiltonian, **kwargs).build().run()


@pytest.mark.parametrize("case", _VALIDATION_CASES)
def test_input_validation(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the all-leaked case warns first
        with pytest.raises(ValueError):
            _invalid_run(case)


def test_rejects_non_operator_hamiltonian():
    with pytest.raises(TypeError, match="QubitOperator or FermionOperator"):
        QSCI("Z0", distribution={(1,): 1.0})  # type: ignore[arg-type]


# ── Review regressions (plan R1–R3) ──────────────────────────────────────


@pytest.mark.parametrize("n_qubits", [64, 70])
def test_wide_registers(n_qubits):
    """R1: labels and Pauli masks beyond 63 bits, against analytic energies."""
    top = n_qubits - 1
    empty = (0,) * n_qubits
    high = (0,) * top + (1,)

    def solve(operator, distribution, n_roots=1):
        algorithm = QSCI(operator, distribution=distribution, n_roots=n_roots).build()
        return algorithm.run()[0], algorithm

    # Qubit 0 empty in a configuration that occupies only the top qubit: <Z0> = +1.
    assert solve(QubitOperator("Z0"), {high: 1.0})[0][0] == pytest.approx(1.0)
    # A Pauli acting on the top qubit itself.
    assert solve(QubitOperator(f"Z{top}"), {empty: 1.0})[0][0] == pytest.approx(1.0)
    assert solve(QubitOperator(f"Z{top}"), {high: 1.0})[0][0] == pytest.approx(-1.0)

    pair = {empty: 0.5, high: 0.5}
    # X couples the two configurations: [[0, 1], [1, 0]].
    np.testing.assert_allclose(solve(QubitOperator(f"X{top}"), pair, 2)[0], [-1.0, 1.0], atol=1e-12)
    # Y's phase: in (empty, high) order the matrix is [[0, -i], [i, 0]], whose -1
    # eigenvector is (1, -i)/sqrt(2).
    y_energies, algorithm = solve(QubitOperator(f"Y{top}"), pair, 2)
    np.testing.assert_allclose(y_energies, [-1.0, 1.0], atol=1e-12)
    assert algorithm.configurations == (empty, high)
    c_empty, c_high = algorithm.coefficients[:, 0]
    assert c_high / c_empty == pytest.approx(-1j)


def test_wide_build_matches_closed_form(monkeypatch):
    """R1: the arbitrary-width build, forced onto a problem with a closed form
    (two-site Hubbard; its Jordan-Wigner hopping carries XX and YY terms)."""
    monkeypatch.setattr(qsci_module, "_VECTORISED_MAX_BITS", 0)
    t, u = 1.0, 4.0
    energies, _ = (
        QSCI(
            fermi_hubbard([2], t, u),
            distribution=_uniform(_sector_configurations(4, 1, 1)),
            n_electrons=(1, 1),
        )
        .build()
        .run()
    )
    assert energies[0] == pytest.approx((u - math.sqrt(u**2 + 16 * t**2)) / 2, abs=1e-12)


def test_exact_sampling_output_is_reusable():
    """R2: QSCI's own pooled samples are valid ``distribution=`` input.  Every
    amplitude but |0...0>'s is pruned by exact sampling (5e-13 <= 1e-12), so the
    raw pool falls short of one; <Z0> on |0...0> is +1."""
    n = 10
    state = np.full(1 << n, np.sqrt(5e-13), dtype=complex)
    state[0] = np.sqrt(1 - ((1 << n) - 1) * 5e-13)
    original = QSCI(
        QubitOperator("Z0"),
        ket=SimpleBlock(n),
        primitive=Sampler(n_shots=qarp.EXACT, initial_state=state),
    ).build()
    original_energies, _ = original.run()
    assert original_energies[0] == pytest.approx(1.0, abs=1e-12)
    assert math.fsum(original.distribution.values()) == pytest.approx(1.0, abs=1e-12)

    replay = QSCI(QubitOperator("Z0"), distribution=original.distribution).build()
    replay_energies, _ = replay.run()
    assert replay.configurations == original.configurations
    np.testing.assert_allclose(replay_energies, original_energies, atol=1e-12)


def test_zero_projected_hamiltonian_on_sparse_path():
    """R3: X0 maps every selected configuration (bit 0 clear) outside the
    subspace, so the projected Hamiltonian is identically zero."""
    distribution = {(0,) + tuple((i >> q) & 1 for q in range(11)): 1 / 2048 for i in range(2048)}
    assert len(distribution) > qsci_module._DENSE_EIGH_MAX_DIM  # the sparse dispatch
    algorithm = QSCI(QubitOperator("X0"), distribution=distribution, n_roots=3).build()
    energies, coefficients = algorithm.run()
    np.testing.assert_array_equal(energies, np.zeros(3))
    np.testing.assert_allclose(coefficients.conj().T @ coefficients, np.eye(3), atol=1e-12)
    labels = [bits_to_label(bits) for bits in algorithm.configurations]
    matrix = qsci_module._subspace_matrix(QubitOperator("X0"), labels)
    assert np.abs(matrix @ coefficients).max() == 0.0

    # An explicitly stored zero is still a zero matrix, not structure for ARPACK.
    stored_zero = scipy.sparse.csr_matrix((np.zeros(1), ([0], [0])), shape=(2100, 2100))
    assert stored_zero.nnz == 1
    zero_energies, zero_vectors = qsci_module._eigensolve(stored_zero, 2)
    np.testing.assert_array_equal(zero_energies, np.zeros(2))
    np.testing.assert_allclose(zero_vectors.T @ zero_vectors, np.eye(2))
