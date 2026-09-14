"""UCC excitation-pool generators.

The pool *set* is pinned by the counts below (unchanged since the
pre-dissolution ``FermionicUCC`` era — see the closed forms in comments) and
by structural/physics oracles; names and generation order were re-pinned
deliberately when the pool became a direct enumeration; a bridge check
proved set-equality with the previous pool in 39/39 (ONV × flags)
configurations.
"""

import numpy as np
import pytest

from qarp.blocks import TrotterAnsatzBlock
from qarp.operators.functions import jordan_wigner
from qarp.operators.ucc import (
    adjacent_singles,
    spin_adapted_doubles,
    spin_adapted_singles,
    ucc_doubles,
    ucc_singles,
    ucc_singles_and_doubles,
)

ONV_11110000 = [1, 1, 1, 1, 0, 0, 0, 0]
ONV_11000000 = [1, 1, 0, 0, 0, 0, 0, 0]


# ── Counts (the pinned pool set; closed forms in comments) ───────────────


def test_singles_counts_11110000():
    # canonical spin-conserving: occ_α×virt_α + occ_β×virt_β = 2·2 + 2·2 = 8
    ops, syms = ucc_singles(ONV_11110000)
    assert len(ops) == len(syms) == 8
    # generalised spin-conserving: C(4,2) per spin channel = 6 + 6 = 12
    ops, syms = ucc_singles(ONV_11110000, generalised=True)
    assert len(ops) == len(syms) == 12
    # canonical unrestricted: occ×virt = 4·4 = 16
    ops, _ = ucc_singles(ONV_11110000, spin_conserving=False)
    assert len(ops) == 16
    # generalised unrestricted: C(8,2) = 28
    ops, _ = ucc_singles(ONV_11110000, spin_conserving=False, generalised=True)
    assert len(ops) == 28


def test_doubles_counts_11110000():
    # canonical spin-conserving: Sz-matched occ-pair × virt-pair combos:
    # 1·1 (αα) + 4·4 (αβ) + 1·1 (ββ) = 18
    ops, syms = ucc_doubles(ONV_11110000)
    assert len(ops) == len(syms) == 18
    # generalised spin-conserving: unordered disjoint Sz-matched index pairs:
    # 3 (αα-αα) + 72 (αβ-αβ) + 3 (ββ-ββ) = 78
    ops, syms = ucc_doubles(ONV_11110000, generalised=True)
    assert len(ops) == len(syms) == 78


def test_doubles_counts_11000000():
    # occ pair {0,1} (Sz 0) × the 9 mixed virtual pairs = 9
    ops, syms = ucc_doubles(ONV_11000000)
    assert len(ops) == len(syms) == 9
    # unrestricted: 1 occ pair × C(6,2) virtual pairs = 15
    ops, _ = ucc_doubles(ONV_11000000, spin_conserving=False)
    assert len(ops) == 15


def test_paired_doubles_counts():
    # canonical: fully-occupied spatials × fully-virtual spatials = 2·2 = 4
    ops, syms = ucc_doubles(ONV_11110000, paired=True)
    assert len(ops) == len(syms) == 4
    # generalised: unordered spatial pairs C(4,2) = 6
    ops, syms = ucc_doubles(ONV_11110000, generalised=True, paired=True)
    assert len(ops) == len(syms) == 6
    # paired conserves spin by construction — the flag is irrelevant
    ops, _ = ucc_doubles(ONV_11110000, spin_conserving=False, generalised=True, paired=True)
    assert len(ops) == 6


def test_adjacent_singles_count():
    ops, syms = adjacent_singles(8)
    assert len(ops) == len(syms) == 4


# ── Names: source→target, self-grouping pairs ────────────────────────────


def test_singles_names_are_ordered_source_to_target():
    _, syms = ucc_singles(ONV_11110000)
    assert [s.name for s in syms] == [
        "s_0to4",
        "s_0to6",
        "s_1to5",
        "s_1to7",
        "s_2to4",
        "s_2to6",
        "s_3to5",
        "s_3to7",
    ]


def test_doubles_names_membership_and_pairing():
    _, syms = ucc_doubles(ONV_11110000)
    names = {s.name for s in syms}
    assert len(names) == 18
    # mixed-spin pairs are named α→α then β→β
    assert "d_0to4_1to5" in names
    # same-spin pairs are ascending-aligned
    assert "d_0to4_2to6" in names
    assert "d_1to5_3to7" in names


def test_paired_double_names_show_the_spatial_pattern():
    _, syms = ucc_doubles(ONV_11110000, paired=True)
    assert {s.name for s in syms} == {
        "d_0to4_1to5",
        "d_0to6_1to7",
        "d_2to4_3to5",
        "d_2to6_3to7",
    }


def test_adjacent_singles_names():
    _, syms = adjacent_singles(4)
    assert [s.name for s in syms] == ["e_0to1", "e_2to3"]


# ── Orientation: the operator reads like its symbol ──────────────────────


def test_single_operator_matches_symbol_orientation():
    ops, syms = ucc_singles([1, 1, 0, 0])
    by_name = {s.name: op for s, op in zip(syms, ops, strict=True)}
    # s_0to2 ↔ a†_2 a_0 − a†_0 a_2
    assert by_name["s_0to2"].terms == {((2, 1), (0, 0)): 1.0, ((0, 1), (2, 0)): -1.0}


def test_plain_excitation_operators_are_single_term():
    ops, _ = ucc_singles_and_doubles(ONV_11110000, antihermitized=False)
    assert all(len(op.terms) == 1 for op in ops)


def test_adjacent_singles_raw_ladder_operators():
    ops, _ = adjacent_singles(4, antihermitized=False)
    assert ops[0].terms == {((1, 1), (0, 0)): 1.0}
    assert ops[1].terms == {((3, 1), (2, 0)): 1.0}


# ── Structure: antihermitized generators are 2-term (c, −c) ──────────────


@pytest.mark.parametrize("generalised", [False, True])
def test_generator_structure(generalised):
    onv = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    for op in ucc_singles(onv, generalised=generalised)[0]:
        coefficients = [complex(c) for c in op.terms.values()]
        assert len(coefficients) == 2
        assert coefficients[0] == -coefficients[1]
    for op in ucc_doubles(onv, generalised=generalised)[0]:
        coefficients = [complex(c) for c in op.terms.values()]
        assert len(coefficients) == 2
        assert coefficients[0] == -coefficients[1]


def test_singles_and_doubles_is_exact_concatenation():
    sd, sdsym = ucc_singles_and_doubles(ONV_11110000)
    singles, ssym = ucc_singles(ONV_11110000)
    doubles, dsym = ucc_doubles(ONV_11110000)
    assert len(sd) == len(sdsym) == len(singles) + len(doubles)
    # Singles first, then doubles — pinned pairing order (§17).
    assert sdsym == ssym + dsym


# ── Dimension/reference resolution ───────────────────────────────────────


def test_missing_reference_raises():
    with pytest.raises(ValueError, match="required unless generalised=True"):
        ucc_singles(n_spin_orbitals=4)
    with pytest.raises(ValueError, match="required unless generalised=True"):
        ucc_doubles(n_spin_orbitals=4)


def test_no_dimension_at_all_raises():
    with pytest.raises(ValueError, match="occupation-number vector or n_spin_orbitals"):
        ucc_singles(generalised=True)


def test_conflicting_dimensions_raise():
    with pytest.raises(ValueError, match="conflicts"):
        ucc_singles([1, 1, 0, 0], n_spin_orbitals=6)


def test_generalised_from_dimension_only():
    ops, _ = ucc_singles(n_spin_orbitals=12, generalised=True)
    assert len(ops) == 30  # C(6,2) per spin channel = 15 + 15


# ── Spin-adapted pools (own functions, spatial-orbital labels) ───────────


def test_spin_adapted_counts_and_names():
    ops, syms = spin_adapted_singles(3)
    assert len(ops) == len(syms) == 3
    assert [s.name for s in syms] == ["sas_0to1", "sas_0to2", "sas_1to2"]
    ops, syms = spin_adapted_doubles(3)
    assert len(ops) == len(syms) == 3
    assert [s.name for s in syms] == ["sad_0to1", "sad_0to2", "sad_1to2"]


def test_spin_adapted_operators_commute_with_s_squared():
    # openfermion supplies the S² reference model only (converted at the
    # boundary — openfermion is a test-only extra since the operator
    # migration).
    openfermion = pytest.importorskip("openfermion")
    from qarp.operators.compat import from_openfermion

    n_spatial = 3
    s2 = from_openfermion(openfermion.hamiltonians.s_squared_operator(n_spatial))
    s2mat = s2.sparse_matrix(2 * n_spatial).toarray()

    ops = spin_adapted_singles(n_spatial)[0] + spin_adapted_doubles(n_spatial)[0]
    for op in ops:
        op_mat = op.sparse_matrix(2 * n_spatial).toarray()
        assert np.linalg.norm(op_mat @ s2mat - s2mat @ op_mat) < 1e-8


def test_spin_adapted_symbols_pair_one_to_one_with_trotter_generators():
    """Each spatial-pair generator has one stable spatial-labelled symbol."""
    sops, ssyms = spin_adapted_singles(3)
    dops, dsyms = spin_adapted_doubles(3)
    ops, symbols = sops + dops, ssyms + dsyms

    assert [s.name for s in symbols] == [
        "sas_0to1",
        "sas_0to2",
        "sas_1to2",
        "sad_0to1",
        "sad_0to2",
        "sad_1to2",
    ]

    qubit_exponents = [jordan_wigner(op) for op in ops]
    assert len(symbols) == len(qubit_exponents) == 6

    ansatz = TrotterAnsatzBlock(
        n_qubits=6,
        qubit_exponents=qubit_exponents,
        symbols=symbols,
        imaginary=True,
    )
    ansatz.build()
    assert len(ansatz.symbol_qop_pairs) == 6


def test_spin_adapted_plain_excitations():
    # antihermitized=False: the bare singlet excitation halves — two ladder
    # terms for singles (alpha + beta), one for the paired double.
    ops, _ = spin_adapted_singles(3, antihermitized=False)
    assert all(len(op.terms) == 2 for op in ops)
    ops, _ = spin_adapted_doubles(3, antihermitized=False)
    assert all(len(op.terms) == 1 for op in ops)
