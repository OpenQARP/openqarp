"""qarp.operators.functions — eigenspectrum (differential against
openfermion) and the pin that the MSB matrix surface no longer lives here.
The openfermion-interop surface (compat) is tested in test_compat.py.
"""

import numpy as np
import pytest

import qarp.operators.functions as functions
from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.functions import eigenspectrum, hermitian_conjugated


@pytest.fixture
def openfermion():
    return pytest.importorskip("openfermion")


def random_fermion_pair(openfermion, rng, n_modes, n_terms):
    ours, theirs = FermionOperator(), openfermion.FermionOperator()
    for _ in range(n_terms):
        body = rng.choice([1, 2])
        modes = rng.integers(0, n_modes, size=2 * body)
        term = " ".join(f"{mode}{'^' if i < body else ''}" for i, mode in enumerate(modes))
        coeff = complex(rng.normal(), rng.normal())
        ours += FermionOperator(term, coeff)
        theirs += openfermion.FermionOperator(term, coeff)
    return ours, theirs


def test_get_sparse_operator_is_not_in_functions():
    # The MSB layout is interop-only (compat); the loud ImportError here is
    # what protects stale callers from a silently bit-reversed matrix.
    assert "get_sparse_operator" not in functions.__all__
    with pytest.raises(ImportError):
        from qarp.operators.functions import get_sparse_operator  # noqa: F401


def test_eigenspectrum_of_sparse_matrix_matches_openfermion(openfermion):
    # The dominant downstream use: eigvalsh of the dense (LSB) matrix.
    rng = np.random.default_rng(7)
    ours, theirs = random_fermion_pair(openfermion, rng, 4, n_terms=6)
    herm_ours = ours + hermitian_conjugated(ours)
    herm_theirs = theirs + openfermion.hermitian_conjugated(theirs)
    ev_ours = np.linalg.eigvalsh(herm_ours.sparse_matrix().toarray())
    ev_theirs = openfermion.eigenspectrum(herm_theirs)
    np.testing.assert_allclose(ev_ours, ev_theirs, atol=1e-10)


def test_eigenspectrum_function_matches_openfermion(openfermion):
    rng = np.random.default_rng(11)
    ours, theirs = random_fermion_pair(openfermion, rng, 3, n_terms=5)
    herm_ours = ours + hermitian_conjugated(ours)
    herm_theirs = theirs + openfermion.hermitian_conjugated(theirs)
    np.testing.assert_allclose(
        eigenspectrum(herm_ours), openfermion.eigenspectrum(herm_theirs), atol=1e-10
    )


def test_eigenspectrum_non_hermitian_and_invalid():
    # Non-hermitian: lexicographically sorted eigvals (openfermion semantics).
    op = QubitOperator("X0", 1.0) + QubitOperator("Y0", 1.0j)  # nilpotent → all zeros
    np.testing.assert_allclose(eigenspectrum(op), [0.0, 0.0], atol=1e-12)
    with pytest.raises(TypeError):
        eigenspectrum("not an operator")


def test_curated_package_exports():
    """The §15 surface: generic operator construction is flat on qarp.operators;
    the fermionic function surface is namespace-qualified (direction (b) of the
    export-surface plan) and deliberately *not* reachable flat; the raw C++
    transform aliases stay in ``functions``."""
    import importlib

    import qarp.operators as ops

    for name in (
        "fermion_operator_from_tensor",
        "rotate_tensor",
        "orbital_rotation_generator",
        "orbital_rotation_matrix",
        "orbital_rotation_parameters",
    ):
        assert callable(getattr(ops, name)), name
    qualified = {
        "integrals": (
            "spin_orbital_integrals_to_fermion_operator",
            "restricted_integrals_to_fermion_operator",
            "unrestricted_integrals_to_fermion_operator",
            "active_space_integrals",
            "unrestricted_active_space_integrals",
            "spatial_to_spin_orbital",
        ),
        "ucc": (
            "ucc_singles",
            "ucc_doubles",
            "ucc_singles_and_doubles",
            "adjacent_singles",
            "spin_adapted_singles",
            "spin_adapted_doubles",
        ),
        "models": ("fermi_hubbard", "lipkin", "transverse_field_ising", "xy_model"),
        "onv": ("onv_from_spatial_occupations",),
    }
    for namespace, names in qualified.items():
        assert namespace in ops.__all__, namespace
        mod = importlib.import_module(f"qarp.operators.{namespace}")
        for name in names:
            assert callable(getattr(mod, name)), f"{namespace}.{name}"
            assert not hasattr(ops, name), (
                f"{name} must be reached through qarp.operators.{namespace}"
            )
    for generic in ("freeze", "active_space", "is_alpha", "jordan_wigner"):
        assert not hasattr(ops, generic), f"{generic} should stay module-qualified"


def test_hermitian_conjugated_maps_over_a_list_by_comprehension():
    # The list-mapped `dagger` helper is gone; `hermitian_conjugated` is the
    # single-operator conjugate and callers map it themselves.
    import qarpx as qx
    from qarp.operators.functions import hermitian_conjugated

    ops = [qx.FermionOperator("0^ 1"), qx.FermionOperator("2^ 3")]
    assert [hermitian_conjugated(o).terms for o in ops] == [
        {((1, 1), (0, 0)): 1.0},
        {((3, 1), (2, 0)): 1.0},
    ]


def test_antihermitize_is_single_operator():
    """``A - A†`` for one operator; the list form is a comprehension."""
    import qarpx as qx
    from qarp.operators.functions import antihermitize, hermitian_conjugated

    op = qx.FermionOperator("0^ 1")
    assert antihermitize(op) == op - hermitian_conjugated(op)
    # Oracle: (a0† a1) − (a1† a0), the two terms with opposite sign.
    assert antihermitize(op).terms == {((0, 1), (1, 0)): 1.0, ((1, 1), (0, 0)): -1.0}


def test_retired_dagger_aliases_are_gone():
    from qarp.operators import functions

    for name in ("dagger", "op_dagger", "op_antihermitize"):
        assert not hasattr(functions, name), f"{name} was consolidated away"
        assert name not in functions.__all__
