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
from qarp import SamplingDistribution
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


def test_blocks_share_the_seed_state():
    # Seed (|00> + |10>)/sqrt(2); the second block flips qubit 1, so the pool is
    # uniform over all four configurations and the Z0 + Z1 minimum is -2 at (1, 1).
    seed = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    flip = SimpleBlock(2)
    flip.x(1)
    algorithm = QSCI(
        QubitOperator("Z0") + QubitOperator("Z1"),
        ket=[SimpleBlock(2), flip],
        primitive=Sampler(n_shots=qarp.EXACT, initial_state=seed),
    ).build()
    energies, _ = algorithm.run()
    assert algorithm.distribution == pytest.approx(
        {bits: 0.25 for bits in itertools.product((0, 1), repeat=2)}, abs=1e-12
    )
    assert energies[0] == pytest.approx(-2.0, abs=1e-12)
    # One array for all K samplers, not a 2**n copy each.
    seeds = [sampler.initial_state for sampler in algorithm.engine._primitives]
    assert len(seeds) == 2 and seeds[0] is seeds[1] is algorithm.primitive.initial_state


def test_zero_weight_never_selected_without_cuts():
    # No floor and no cap, so only the zero-weight rule keeps (0, 0) out.
    algorithm = QSCI(QubitOperator("Z0"), distribution={(1, 0): 1.0, (0, 0): 0.0}).build()
    algorithm.run()
    assert algorithm.configurations == ((1, 0),)


def test_qubit_path_sector_filter():
    # Opt-in sector on a QubitOperator: (0, 1) leaks alpha, (1, 1) leaks beta.
    distribution = {(1, 0): 0.5, (0, 1): 0.25, (1, 1): 0.25}
    algorithm = QSCI(QubitOperator("Z1"), distribution=distribution, n_electrons=(1, 0)).build()
    with pytest.warns(UserWarning, match="0.5 of the sampled probability"):
        algorithm.run()
    assert algorithm.configurations == ((1, 0),)
    assert algorithm.in_sector_mass == 0.5


def test_clean_sector_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        QSCI(FermionOperator("0^ 0"), distribution={(1, 0): 1.0}, n_electrons=(1, 0)).build().run()


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
    sparse = QSCI(hubbard, distribution=full, n_electrons=(2, 2), n_roots=3).build()
    sparse_energies, vectors = sparse.run()
    np.testing.assert_allclose(sparse_energies, dense, atol=1e-10)
    # eigsh's eigenvectors against sparse_matrix(): eigenpair residuals, orthonormality.
    index = [bits_to_label(bits) for bits in sparse.configurations]
    block = hubbard.sparse_matrix(8).toarray()[np.ix_(index, index)]
    np.testing.assert_allclose(block @ vectors, vectors * sparse_energies, atol=1e-8)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(3), atol=1e-10)

    # ARPACK refuses n_roots >= dim - 1 for complex input and n_roots = dim for
    # real input, so both go dense above the cutoff.  Spectrum: ±1 ± 0.5.
    two_qubits = _uniform(list(itertools.product((0, 1), repeat=2)))
    complex_h = QubitOperator("Y0") + QubitOperator("Z1", 0.5)
    np.testing.assert_allclose(
        energies(complex_h, two_qubits, None, 3), [-1.5, -0.5, 0.5], atol=1e-12
    )
    real_h = QubitOperator("X0") + QubitOperator("Z1", 0.5)
    np.testing.assert_allclose(
        energies(real_h, two_qubits, None, 4), [-1.5, -0.5, 0.5, 1.5], atol=1e-12
    )
    monkeypatch.undo()

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


def test_solve_follows_input_order():
    """``_solve`` takes configurations in any order; coefficient rows follow it."""
    hamiltonian = fermi_hubbard([2], 1.0, 4.0)
    sector = _sector_configurations(4, 1, 1)
    algorithm = QSCI(hamiltonian, distribution=_uniform(sector), n_electrons=(1, 1)).build()
    # Its argsort is a 4-cycle, so placing rows by the argsort instead of its
    # inverse gives a different order.
    shuffled = [sector[i] for i in (1, 2, 0, 3)]
    energies, coefficients = algorithm._solve(shuffled)
    index = [bits_to_label(bits) for bits in shuffled]
    block = hamiltonian.sparse_matrix(4).toarray()[np.ix_(index, index)]
    np.testing.assert_allclose(block @ coefficients, coefficients * energies, atol=1e-10)
    with pytest.raises(ValueError, match="duplicates"):
        algorithm._solve([sector[0], sector[0]])


def test_solve_refuses_more_roots_than_configurations():
    # X0 projects to zero on one configuration, which reaches the zero-matrix shortcut.
    algorithm = QSCI(QubitOperator("X0"), distribution={(0, 0): 1.0}, n_roots=2).build()
    with pytest.raises(ValueError, match="exceeds the subspace dimension"):
        algorithm._solve([(0, 0)])
    with pytest.raises(ValueError, match="exceeds the subspace dimension"):
        algorithm._solve([])


@pytest.mark.parametrize(("letters", "seed"), [("XZ", 2), ("XYZ", 0)])
def test_sparse_path_degenerate_roots(monkeypatch, letters, seed):
    """Qubits 8 and 9 are idle, so every level is at least fourfold degenerate and
    the three roots share one level.  XYZ is complex: eigs leaves its copies
    non-orthonormal."""
    rng = np.random.default_rng(seed)
    hamiltonian = QubitOperator()
    for _ in range(30):
        i, j = sorted(rng.choice(8, 2, replace=False))
        a, b = rng.choice(list(letters), 2)
        hamiltonian += QubitOperator(f"{a}{i} {b}{j}", float(rng.normal()))
    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    configurations = list(itertools.product((0, 1), repeat=10))
    algorithm = QSCI(hamiltonian, distribution=_uniform(configurations), n_roots=3).build()
    energies, vectors = algorithm.run()
    # Every configuration is selected, so row i is label i of sparse_matrix().
    dense = hamiltonian.sparse_matrix(10).toarray()
    np.testing.assert_allclose(energies, scipy.linalg.eigvalsh(dense)[:3], atol=1e-10)
    np.testing.assert_allclose(dense @ vectors, vectors * energies, atol=1e-10)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(3), atol=1e-10)
    # A fixed start vector: the same input gives the same roots.
    np.testing.assert_array_equal(algorithm.run()[1], vectors)


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


# The checks after selection: without a message, a ValueError raised further
# downstream would pass for them.
_RUN_TIME_MATCHES = {
    "empty_subspace_after_filter": "selected an empty subspace",
    "n_roots_above_dimension": "exceeds the selected subspace dimension",
}


@pytest.mark.parametrize("case", _VALIDATION_CASES)
def test_input_validation(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the all-leaked case warns first
        with pytest.raises(ValueError, match=_RUN_TIME_MATCHES.get(case)):
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


def test_sparse_path_ring_degenerate_roots():
    # The periodic 8-site ring at U = 4 with (4, 3) electrons: dimension 3920 is
    # above the dense cutoff, and the lowest four roots are two twofold levels.
    hamiltonian = fermi_hubbard([8], 1.0, 4.0, periodic=True)
    configurations = _sector_configurations(16, 4, 3)
    algorithm = QSCI(
        hamiltonian, distribution=_uniform(configurations), n_electrons=(4, 3), n_roots=4
    ).build()
    energies, _ = algorithm.run()
    index = [bits_to_label(bits) for bits in configurations]
    block = hamiltonian.sparse_matrix(16)[index][:, index].toarray()
    np.testing.assert_allclose(energies, scipy.linalg.eigvalsh(block)[:4], atol=1e-10)


def test_sparse_path_eightfold_level():
    # 2-local terms on qubits 0-7 of 11: every level is eightfold degenerate, at
    # dimension 2048.  The check must start fresh: from the solve's own start it
    # misses a copy on this seed.
    rng = np.random.default_rng(10)
    hamiltonian = QubitOperator()
    for _ in range(30):
        i, j = sorted(rng.choice(8, 2, replace=False))
        a, b = rng.choice(list("XZ"), 2)
        hamiltonian += QubitOperator(f"{a}{i} {b}{j}", float(rng.normal()))
    configurations = list(itertools.product((0, 1), repeat=11))
    algorithm = QSCI(hamiltonian, distribution=_uniform(configurations), n_roots=8).build()
    energies, _ = algorithm.run()
    dense = hamiltonian.sparse_matrix(11).toarray()
    np.testing.assert_allclose(energies, scipy.linalg.eigvalsh(dense)[:8], atol=1e-10)


@pytest.mark.parametrize(
    ("verify_roots", "n_roots", "n_solves"), [(True, 2, 2), (False, 2, 1), (True, 1, 1)]
)
def test_verify_roots_sets_the_extra_solve(monkeypatch, verify_roots, n_roots, n_solves):
    # The open 4-site chain at (2, 2) is nondegenerate, so the check finds nothing
    # and never re-solves; a single root is never checked.
    calls = []
    eigsh = scipy.sparse.linalg.eigsh

    def counting(*args, **kwargs):
        calls.append(kwargs["k"])
        return eigsh(*args, **kwargs)

    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    monkeypatch.setattr(qsci_module.scipy.sparse.linalg, "eigsh", counting)
    configurations = _sector_configurations(8, 2, 2)
    algorithm = QSCI(
        fermi_hubbard([4], 1.0, 4.0),
        distribution=_uniform(configurations),
        n_electrons=(2, 2),
        n_roots=n_roots,
        verify_roots=verify_roots,
    ).build()
    algorithm.run()
    assert len(calls) == n_solves


def test_missed_root_retries_are_capped(monkeypatch):
    # A check that always fires must stop at the cap, not climb to k = dim - 2.
    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    monkeypatch.setattr(qsci_module, "_MAX_ROOT_RETRIES", 2)
    monkeypatch.setattr(qsci_module, "_missed_root", lambda *args: True)
    algorithm = QSCI(
        fermi_hubbard([4], 1.0, 4.0),
        distribution=_uniform(_sector_configurations(8, 2, 2)),
        n_electrons=(2, 2),
        n_roots=2,
    ).build()
    with pytest.raises(RuntimeError, match="after 2 retries"):
        algorithm.run()


def test_window_inside_a_level_is_not_a_miss():
    """Three copies of a fourfold level: the fourth lies outside the window at
    exactly the top root, which is not a miss; two copies and the next level are.
    Exact eigenvectors, because whether ARPACK's own solve misses depends on the
    platform's BLAS."""
    rng = np.random.default_rng(2)
    hamiltonian = QubitOperator()
    for _ in range(30):
        i, j = sorted(rng.choice(8, 2, replace=False))
        a, b = rng.choice(list("XZ"), 2)
        hamiltonian += QubitOperator(f"{a}{i} {b}{j}", float(rng.normal()))
    matrix = qsci_module._hermitian_real(
        qsci_module._subspace_matrix(hamiltonian, list(range(1 << 10)))
    )
    values, vectors = scipy.linalg.eigh(matrix.toarray())
    assert values[3] - values[0] < 1e-10 < values[4] - values[3]  # one fourfold level
    v0 = np.random.default_rng(0).standard_normal(matrix.shape[0])
    assert not qsci_module._missed_root(matrix, vectors[:, :3], values[2], v0)
    assert qsci_module._missed_root(matrix, vectors[:, [0, 1, 4]], values[4], v0)


# ── Build, pooling and run-state edge cases ──────────────────────────────


def test_wide_build_drops_images_outside_the_subspace():
    """X63 maps |0...0> outside the one-configuration subspace, so only
    <Z0> = +1 remains; an image misfiled onto row 0 would give 2."""
    hamiltonian = QubitOperator("X63") + QubitOperator("Z0")
    energies, _ = QSCI(hamiltonian, distribution={(0,) * 64: 1.0}).build().run()
    assert energies[0] == pytest.approx(1.0, abs=1e-12)


def test_parity_reaches_high_bits():
    # 60 qubits stays on the int64 build; every Z acts on an occupied qubit.
    occupied = tuple(int(q in (20, 35, 50)) for q in range(60))
    hamiltonian = QubitOperator("Z20") + QubitOperator("Z35", 0.5) + QubitOperator("Z50", 0.25)
    energies, _ = QSCI(hamiltonian, distribution={occupied: 1.0}).build().run()
    assert energies[0] == pytest.approx(-1.75, abs=1e-12)


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_tie_break_independent_of_block_order(order):
    """Labels 1 and 4 pool the same three weights, so they tie at 1/2 and the
    cap keeps label 1 in any block order."""
    low, high = _bits(1, 4), _bits(4, 4)
    blocks = [{high: 0.1, low: 0.2}, {high: 0.2, low: 0.7}, {high: 0.7, low: 0.1}]
    pooled = qsci_module._pool([blocks[i] for i in order])
    assert pooled[low] == pooled[high]
    assert qsci_module._select(pooled, 4, None, 1, None).configurations == (low,)


def test_equal_counts_tie_however_they_split():
    """Labels 1 and 4 both have 3 of 20 counts, split 3 + 0 and 1 + 2, so they
    tie and the cap keeps label 1."""
    low, high, top = _bits(1, 4), _bits(4, 4), _bits(8, 4)
    blocks = [
        SamplingDistribution.from_dict({low: 0.3, high: 0.1, top: 0.6}, n_shots=10),
        SamplingDistribution.from_dict({high: 0.2, top: 0.8}, n_shots=10),
    ]
    for order in (blocks, blocks[::-1]):
        pooled = qsci_module._pool(order)
        assert pooled[low] == pooled[high] == 3 / 20
        assert qsci_module._select(pooled, 4, None, 2, None).configurations == (low, top)


def test_failed_run_keeps_the_samples(monkeypatch):
    """A run that fails after sampling keeps the pool, ``in_sector_mass`` and
    ``rejected`` and clears the solve outputs; one that fails while sampling
    clears every output rather than keep the last run's."""

    def build(n_roots):
        # The (1, 0) sector of two qubits holds one configuration.
        return QSCI(
            FermionOperator("0^ 0"),
            ket=ComputationalBasisStateBlock([1, 0]),
            primitive=Sampler(n_shots=100),
            n_electrons=(1, 0),
            n_roots=n_roots,
        ).build()

    first = build(2)
    with pytest.raises(ValueError, match="exceeds the selected subspace dimension"):
        first.run()
    assert first.distribution == {(1, 0): 1.0}
    assert first.in_sector_mass == 1.0 and first.rejected == {}
    assert first.energies is None

    second = build(1)
    assert second.run()[0][0] == pytest.approx(1.0)  # <10| n_0 |10> = 1
    second.n_roots = 2
    with pytest.raises(ValueError, match="exceeds the selected subspace dimension"):
        second.run()
    assert second.energies is None and second.coefficients is None
    assert second.configurations is None

    # A sampler output failing the full-register check.
    third = build(1)
    third.run()
    monkeypatch.setattr(third.engine, "run", lambda *args, **kwargs: [{(1,): 1.0}])
    with pytest.raises(ValueError, match="full register"):
        third.run()
    assert third.energies is None and third.in_sector_mass is None
    assert third.distribution is None and third.rejected is None
