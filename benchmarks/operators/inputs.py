"""Deterministic, stack-agnostic input specs.

Plain tuples only — no SDK objects — so input generation is excluded from the
measured kernel identically on every stack.  A qubit-term spec is
(factors, coeff) with factors = ((qubit, 'X'|'Y'|'Z'), ...) sorted by qubit,
qubits unique.  A ladder-term spec is (ops, coeff) with
ops = ((mode, action), ...), action 1 = creation, 0 = annihilation.
"""

from collections.abc import Iterator

import numpy as np

SEED = 42
N_QUBITS = 20  # register for the random-Pauli workloads
_PAULIS = "XYZ"

# Fixed symbol pool + substitution values for the symbolic workload's check.
SYMBOL_NAMES = tuple(f"a{i}" for i in range(8))
SYMBOL_VALUES = {name: 0.25 + 0.125 * i for i, name in enumerate(SYMBOL_NAMES)}


def pauli_term_specs(
    n_terms: int, n_qubits: int = N_QUBITS, max_weight: int = 4, seed: int = SEED
) -> list[tuple[tuple, complex]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_terms):
        weight = int(rng.integers(1, max_weight + 1))
        qubits = np.sort(rng.choice(n_qubits, size=weight, replace=False))
        factors = tuple((int(q), _PAULIS[int(rng.integers(3))]) for q in qubits)
        out.append((factors, complex(rng.normal(), rng.normal())))
    return out


def pauli_term_spec_stream(
    n_terms: int, n_qubits: int = 24, max_weight: int = 4, seed: int = SEED
) -> Iterator[tuple[tuple, complex]]:
    """Generator variant for memory_hold: the spec list must not dominate the
    peak-RSS measurement, so terms are produced one at a time."""
    rng = np.random.default_rng(seed)
    for _ in range(n_terms):
        weight = int(rng.integers(1, max_weight + 1))
        qubits = np.sort(rng.choice(n_qubits, size=weight, replace=False))
        factors = tuple((int(q), _PAULIS[int(rng.integers(3))]) for q in qubits)
        yield (factors, complex(rng.normal(), rng.normal()))


def colliding_term_specs(
    n_terms: int, n_qubits: int = N_QUBITS, max_weight: int = 4, seed: int = SEED
) -> list[tuple[tuple, complex]]:
    """~50% key collisions: keys drawn from a pool of n_terms//2 distinct
    factors, coefficients fresh per draw."""
    pool = [f for f, _ in pauli_term_specs(max(1, n_terms // 2), n_qubits, max_weight, seed)]
    rng = np.random.default_rng(seed + 1)
    return [
        (pool[int(rng.integers(len(pool)))], complex(rng.normal(), rng.normal()))
        for _ in range(n_terms)
    ]


def term_string(factors: tuple) -> str:
    """openfermion/qarp string form: ((0,'X'),(3,'Y')) -> "X0 Y3"."""
    return " ".join(f"{letter}{qubit}" for qubit, letter in factors)


def ladder_term_specs(n_modes: int, seed: int = SEED) -> list[tuple[tuple, complex]]:
    """Random 1- and 2-body ladder terms on n_modes modes, 4·n² terms —
    the "random 2-body FermionOperator (n modes)" input of the mapping rows."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(4 * n_modes * n_modes):
        body = int(rng.integers(1, 3))
        modes = rng.integers(0, n_modes, size=2 * body)
        ops = tuple((int(mode), 1 if i < body else 0) for i, mode in enumerate(modes))
        out.append((ops, complex(rng.normal(), rng.normal())))
    return out


def ladder_string(ops: tuple) -> str:
    """openfermion/qarp string form: ((2,1),(1,0)) -> "2^ 1"."""
    return " ".join(f"{mode}^" if action else f"{mode}" for mode, action in ops)


def symbolic_term_specs(
    n_terms: int, n_qubits: int = N_QUBITS, max_weight: int = 4, seed: int = SEED
) -> list[tuple[tuple, complex, str]]:
    """(factors, numeric_coeff, symbol_name): each stack builds
    coeff · Symbol(name) with its own symbolic backend."""
    base = pauli_term_specs(n_terms, n_qubits, max_weight, seed)
    rng = np.random.default_rng(seed + 2)
    return [
        (factors, coeff, SYMBOL_NAMES[int(rng.integers(len(SYMBOL_NAMES)))])
        for factors, coeff in base
    ]
