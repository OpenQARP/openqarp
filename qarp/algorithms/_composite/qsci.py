"""Quantum-selected configuration interaction (QSCI).

Samples one or more prepared states, or takes samples gathered elsewhere, keeps
the most frequent configurations in the declared particle-number/Sz sector, and
diagonalises the Hamiltonian classically in the subspace they span (Kanno et
al., arXiv:2302.11320).  Pooling several blocks is the time-averaged form of
Mikkelsen and Nakagawa, arXiv:2412.13839.  Design contract:
``docs/contributions/qsci_plan.md``.
"""

import copy
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Self, Union, cast

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

import qarpx as qx

from ..._postselection import PostSelection
from ..._types import SamplingDictionary
from ...blocks import AnyBlock
from ...endianness import bits_to_label
from ...engines import Engine, Runnable
from ...operators import FermionOperator, QubitOperator
from ...operators.functions import count_qubits, jordan_wigner
from .._primitives import PrimitiveAlgorithm, Sampler
from .composite_algorithm import CompositeAlgorithm

# Dense ``eigh`` up to this subspace dimension, ARPACK ``eigsh`` above it.
_DENSE_EIGH_MAX_DIM = 2000

# External distributions only: exact sampling prunes probabilities <= 1e-12,
# so a primitive's own result can legitimately fall short by more than this.
_MASS_TOL = 1e-10

# i**n_Y, indexed by n_Y mod 4.
_Y_PHASE = (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)

# Labels and Pauli masks up to this many bits use the vectorised int64 build;
# wider ones (bit 63 is int64's sign bit) take the arbitrary-width path.
_VECTORISED_MAX_BITS = 63


@dataclass(frozen=True)
class _Selection:
    configurations: tuple[tuple[int, ...], ...]
    in_sector_mass: float
    leaked_fraction: float
    rejected: SamplingDictionary


class QSCI(CompositeAlgorithm):
    """Quantum-selected configuration interaction.

    Exactly one of ``ket`` and ``distribution`` supplies the samples.  They are
    pooled, filtered to the declared sector, cut by ``prob_threshold`` and then
    by ``max_subspace_dim`` (in that order), and the Hamiltonian is diagonalised
    in the subspace spanned by the surviving configurations.

    Args:
        hamiltonian: A ``QubitOperator``, or a ``FermionOperator`` which is
            Jordan-Wigner-transformed.  Restricted and unrestricted molecular
            integrals both arrive as the same abab spin-orbital operator.
        ket: A block, or a sequence of blocks of equal width, to sample.  A
            sequence is pooled with **equal weight per block**, which is the
            time average only for equally spaced time points; a nonuniform
            grid over-weights its dense stretches.
        distribution: A ``{bits: probability}`` mapping gathered elsewhere
            (LSB-first keys).  Its weights must sum to one within ``1e-10``;
            it is validated, never renormalised.
        max_subspace_dim: Keep at most this many configurations (the papers'
            R).  Ties break by ascending ``bits_to_label``.  ``None``: no cap.
        prob_threshold: Inclusive floor on a configuration's probability
            within the sector (a probability, not an amplitude).  ``None``: no
            floor.  Zero-probability configurations never enter the subspace.
        n_electrons: ``(n_alpha, n_beta)``: the sector, with alpha on even and
            beta on odd qubits (abab).  Required for a ``FermionOperator``.  For
            a ``QubitOperator`` it is optional and asserts a Jordan-Wigner
            occupation basis; ``None`` means no sector filter.
        n_roots: Number of lowest eigenpairs to return.
        primitive: A ``Sampler`` measuring every qubit in register order; a
            default ``Sampler()`` when ``ket`` is given.  Must be ``None`` with
            ``distribution``.
        engine: Execution engine for ``ket``.
        verbose: Print a one-line summary after each run.

    ``run()`` returns ``(energies, coefficients)`` and sets, like ``QSE``:

    Attributes:
        energies: The ``n_roots`` lowest eigenvalues, ascending.
        coefficients: ``(n_selected, n_roots)`` eigenvectors; row ``i`` is the
            amplitude on ``configurations[i]``.
        configurations: The diagonalised basis as LSB-first bit tuples (qubit
            ``q`` at position ``q``), in ascending ``bits_to_label`` order.
        distribution: The pooled samples before any filter or cut, normalised to
            unit total mass; with ``distribution=`` it is the input itself.  Passing it to a new
            ``QSCI`` re-selects from the same samples, e.g. for an
            energy-versus-``max_subspace_dim`` curve.
        in_sector_mass: The share of ``distribution`` inside the declared
            sector; ``1.0`` when no sector is declared.
        rejected: The out-of-sector part of ``distribution``, unrenormalised.

    Jordan-Wigner is a precondition, not something QSCI can check: the sector
    filter and the reading of bitstrings as occupations both assume the sampled
    states were prepared under it.
    """

    def __init__(
        self,
        hamiltonian: Union[QubitOperator, FermionOperator],
        ket: Optional[Union[AnyBlock, Sequence[AnyBlock]]] = None,
        distribution: Optional[SamplingDictionary] = None,
        *,
        max_subspace_dim: Optional[int] = None,
        prob_threshold: Optional[float] = None,
        n_electrons: Optional[tuple[int, int]] = None,
        n_roots: int = 1,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
        verbose: bool = False,
    ) -> None:
        if isinstance(hamiltonian, FermionOperator):
            fermionic = True
        elif isinstance(hamiltonian, QubitOperator):
            fermionic = False
        else:
            raise TypeError(
                "hamiltonian must be a QubitOperator or FermionOperator, "
                f"got {type(hamiltonian).__name__}"
            )
        if (ket is None) == (distribution is None):
            raise ValueError(
                "QSCI takes exactly one of ket (blocks to sample) or "
                "distribution (samples gathered elsewhere)"
            )
        if max_subspace_dim is not None:
            _check_positive_int("max_subspace_dim", max_subspace_dim)
        if prob_threshold is not None:
            _check_probability(prob_threshold)
        _check_positive_int("n_roots", n_roots)

        blocks: tuple[AnyBlock, ...] = ()
        if distribution is not None:
            if primitive is not None:
                raise ValueError("primitive must be None when a distribution is given")
            n_qubits = _validate_distribution(distribution, "distribution", check_mass=True)
        else:
            blocks = _as_blocks(ket)
            widths = sorted({int(block.n_qubits) for block in blocks})
            if len(widths) != 1:
                raise ValueError(f"every block must have the same register width, got {widths}")
            n_qubits = widths[0]
            if primitive is None:
                primitive = Sampler()
            elif not isinstance(primitive, Sampler):
                raise ValueError(
                    "QSCI samples its blocks, so primitive must be a Sampler "
                    f"(Target.SAMPLING), got {type(primitive).__name__}"
                )
            measured = primitive.measured_qubits
            if measured is not None and list(measured) != list(range(n_qubits)):
                raise ValueError(
                    "the Sampler must measure every qubit in register order, "
                    f"got measured_qubits={list(measured)}"
                )

        width = int(count_qubits(hamiltonian))
        if width > n_qubits:
            raise ValueError(f"hamiltonian acts on {width} qubits but the register has {n_qubits}")
        if n_electrons is None:
            if fermionic:
                raise ValueError("n_electrons=(n_alpha, n_beta) is required for a FermionOperator")
        else:
            n_electrons = _check_sector(n_electrons, n_qubits)

        # The base accepts primitive=None at runtime (deepcopy guard).
        super().__init__(primitive=primitive, engine=engine)  # type: ignore[arg-type]

        self.hamiltonian = hamiltonian
        self.blocks = blocks
        self.distribution: Optional[SamplingDictionary] = (
            dict(distribution) if distribution is not None else None
        )
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.max_subspace_dim = max_subspace_dim
        self.prob_threshold = prob_threshold
        self.n_roots = n_roots
        self.verbose = verbose
        self.energies: Optional[NDArray[np.float64]] = None
        self.coefficients: Optional[NDArray] = None
        self.configurations: Optional[tuple[tuple[int, ...], ...]] = None
        self.in_sector_mass: Optional[float] = None
        self.rejected: Optional[SamplingDictionary] = None
        self._fermionic = fermionic
        self._qubit_hamiltonian: Optional[QubitOperator] = None

    def build(self) -> Self:
        """Map the Hamiltonian to qubits and compile one sampler per block."""
        self._qubit_hamiltonian = (
            jordan_wigner(cast(FermionOperator, self.hamiltonian))
            if self._fermionic
            else cast(QubitOperator, self.hamiltonian)
        )
        if self.blocks:
            samplers: list[Runnable] = []
            for block in self.blocks:
                sampler = cast(Sampler, copy.deepcopy(self.primitive))
                sampler.ket = block
                sampler.measured_qubits = list(range(self.n_qubits))
                samplers.append(sampler)
            self.engine.build(samplers)
            self.distribution = None  # pooled by run(); stale after a rebuild
        self.energies = self.coefficients = self.configurations = None
        self.in_sector_mass = None
        self.rejected = None
        return self

    def run(self) -> tuple[NDArray[np.float64], NDArray]:
        """Sample (for blocks), select the subspace, and diagonalise once."""
        if self._qubit_hamiltonian is None:
            raise ValueError("Call build() before run().")
        if self.blocks:
            outputs = self.engine.run()
            samples: list[SamplingDictionary] = []
            for index, output in enumerate(outputs):
                what = f"sampler result for block {index}"
                if not isinstance(output, Mapping):
                    raise ValueError(f"{what} is not a distribution")
                if _validate_distribution(output, what, check_mass=False) != self.n_qubits:
                    raise ValueError(f"{what} does not cover the full register")
                samples.append(output)
            pooled = _pool(samples)
        else:
            pooled = cast(SamplingDictionary, self.distribution)

        selection = _select(
            pooled, self.n_qubits, self.n_electrons, self.max_subspace_dim, self.prob_threshold
        )
        if selection.leaked_fraction > 0:
            warnings.warn(
                f"QSCI: {selection.leaked_fraction:.3g} of the sampled probability lies "
                f"outside the (n_alpha, n_beta) = {self.n_electrons} sector and was "
                "excluded from selection; it is kept on QSCI.distribution and QSCI.rejected",
                UserWarning,
                stacklevel=2,
            )
        if not selection.configurations:
            raise ValueError(
                "QSCI selected an empty subspace: no in-sector configuration with "
                "nonzero probability survived the sector filter and cuts"
            )
        if self.n_roots > len(selection.configurations):
            raise ValueError(
                f"n_roots={self.n_roots} exceeds the selected subspace dimension "
                f"{len(selection.configurations)}"
            )

        energies, coefficients = self._solve(selection.configurations)
        self.energies, self.coefficients = energies, coefficients
        self.configurations = selection.configurations
        self.distribution = pooled
        self.in_sector_mass = selection.in_sector_mass
        self.rejected = selection.rejected
        if self.verbose:
            print(
                f"QSCI: {len(selection.configurations)} configurations, in-sector mass "
                f"{selection.in_sector_mass:.6f}, energies {energies}"
            )
        return energies, coefficients

    def _solve(
        self, configurations: Sequence[tuple[int, ...]]
    ) -> tuple[NDArray[np.float64], NDArray]:
        """Lowest ``n_roots`` eigenpairs over an arbitrary configuration set.

        Coefficient rows follow the order of ``configurations``.
        """
        qubit_hamiltonian = cast(QubitOperator, self._qubit_hamiltonian)
        labels = [bits_to_label(bits) for bits in configurations]
        order = sorted(range(len(labels)), key=labels.__getitem__)
        sorted_labels = [labels[i] for i in order]
        if len(set(sorted_labels)) != len(sorted_labels):
            raise ValueError("configurations contain duplicates")
        matrix = _hermitian_real(_subspace_matrix(qubit_hamiltonian, sorted_labels))
        energies, vectors = _eigensolve(matrix, self.n_roots)
        coefficients = np.empty_like(vectors)
        coefficients[order] = vectors
        return energies, coefficients


# ── Validation ──────────────────────────────────────────────────────────


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _check_positive_int(name: str, value: object) -> None:
    if not _is_int(value) or cast(int, value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_probability(value: object) -> None:
    if not _is_real_number(value) or not math.isfinite(cast(float, value)):
        raise ValueError(f"prob_threshold must be a finite number, got {value!r}")
    if not 0.0 <= cast(float, value) <= 1.0:
        raise ValueError(f"prob_threshold must lie in [0, 1], got {value!r}")


def _check_sector(n_electrons: object, n_qubits: int) -> tuple[int, int]:
    if (
        not isinstance(n_electrons, (tuple, list))
        or len(n_electrons) != 2
        or not all(_is_int(n) and n >= 0 for n in n_electrons)
    ):
        raise ValueError(
            f"n_electrons must be a pair of non-negative integers, got {n_electrons!r}"
        )
    if n_qubits % 2:
        raise ValueError(
            f"sector filtering needs an even-width abab register, got {n_qubits} qubits"
        )
    n_alpha, n_beta = (int(n) for n in n_electrons)
    if max(n_alpha, n_beta) > n_qubits // 2:
        raise ValueError(
            f"n_electrons={(n_alpha, n_beta)} exceeds the {n_qubits // 2} orbitals per spin"
        )
    return n_alpha, n_beta


def _validate_distribution(distribution: object, what: str, *, check_mass: bool) -> int:
    """Structural checks, plus unit total mass when ``check_mass``; returns the width."""
    if not isinstance(distribution, Mapping) or len(distribution) == 0:
        raise ValueError(f"{what} must be a non-empty mapping of bit tuples to probabilities")
    width: Optional[int] = None
    total = 0.0
    for key, weight in distribution.items():
        if (
            not isinstance(key, tuple)
            or len(key) == 0
            or not all(_is_int(bit) and bit in (0, 1) for bit in key)
        ):
            raise ValueError(f"{what} key {key!r} is not a non-empty tuple of 0/1 bits")
        if width is None:
            width = len(key)
        elif len(key) != width:
            raise ValueError(f"{what} keys have unequal widths ({width} and {len(key)})")
        if not _is_real_number(weight) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{what} weight {weight!r} for {key} is not finite and non-negative")
        total += float(weight)
    if check_mass and abs(total - 1.0) > _MASS_TOL:
        raise ValueError(
            f"{what} weights sum to {total!r}, not 1 within {_MASS_TOL}; "
            "QSCI does not renormalise external samples"
        )
    return cast(int, width)


def _as_blocks(ket: object) -> tuple[AnyBlock, ...]:
    if isinstance(ket, qx.Block):
        return (ket,)
    if isinstance(ket, Sequence) and not isinstance(ket, (str, bytes)):
        blocks = tuple(ket)
        if blocks and all(isinstance(block, qx.Block) for block in blocks):
            return blocks
    raise ValueError("ket must be a block or a non-empty sequence of blocks")


# ── Pooling and selection ───────────────────────────────────────────────


def _pool(samples: Sequence[SamplingDictionary]) -> SamplingDictionary:
    """Equal-weight mean of the per-block distributions, normalised to unit mass.

    Exact sampling prunes probabilities <= 1e-12, so a block can fall short of
    one; normalising after pooling makes the pool valid ``distribution=`` input.
    """
    totals: dict[tuple[int, ...], float] = {}
    for sample in samples:
        for key, weight in sample.items():
            totals[key] = totals.get(key, 0.0) + float(weight)
    # The mean's 1/len(samples) cancels in the normalisation: one division.
    mass = math.fsum(totals.values())
    return {key: weight / mass for key, weight in totals.items()}


def _select(
    distribution: SamplingDictionary,
    n_qubits: int,
    n_electrons: Optional[tuple[int, int]],
    max_subspace_dim: Optional[int],
    prob_threshold: Optional[float],
) -> _Selection:
    """Filter, then threshold, then cap; see the plan's order of operations."""
    total = math.fsum(distribution.values())
    if n_electrons is None:
        in_sector = dict(distribution)
        rejected: SamplingDictionary = {}
    else:
        n_alpha, n_beta = n_electrons
        alpha = PostSelection.hamming_weight(list(range(0, n_qubits, 2)), n_alpha)
        beta = PostSelection.hamming_weight(list(range(1, n_qubits, 2)), n_beta)
        members = beta.apply(alpha.apply(distribution).distribution).distribution.keys()
        in_sector = {key: w for key, w in distribution.items() if key in members}
        rejected = {key: w for key, w in distribution.items() if key not in members}

    # One division from raw weights, so dyadic inputs stay exact at the threshold.
    in_mass = math.fsum(in_sector.values())
    in_sector_mass = in_mass / total if total > 0 else 0.0
    leaked = math.fsum(rejected.values())
    leaked_fraction = leaked / total if total > 0 else 0.0

    candidates = [
        (float(weight) / in_mass, bits_to_label(key), tuple(int(bit) for bit in key))
        for key, weight in in_sector.items()
        if weight > 0
    ]
    if prob_threshold is not None:
        candidates = [c for c in candidates if c[0] >= prob_threshold]
    candidates.sort(key=lambda c: (-c[0], c[1]))
    if max_subspace_dim is not None:
        candidates = candidates[:max_subspace_dim]
    candidates.sort(key=lambda c: c[1])
    return _Selection(
        configurations=tuple(c[2] for c in candidates),
        in_sector_mass=in_sector_mass,
        leaked_fraction=leaked_fraction,
        rejected=rejected,
    )


# ── Subspace Hamiltonian and eigensolve ─────────────────────────────────


def _parity(values: NDArray[np.int64]) -> NDArray[np.int64]:
    """Bitwise parity of non-negative int64 values (popcount mod 2)."""
    folded = values.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        folded ^= folded >> shift
    return folded & 1


def _pauli_groups(qubit_hamiltonian: QubitOperator) -> dict[int, list[tuple[int, complex]]]:
    """Terms keyed by X/Y-support mask, as ``(Z/Y-support mask, coefficient * i**n_Y)``.

    A Pauli string maps ``|b>`` to ``i**n_Y (-1)**popcount(b & z) |b ^ x>``, so
    terms sharing ``x`` share their image.
    """
    groups: dict[int, list[tuple[int, complex]]] = {}
    for term, coefficient in qubit_hamiltonian.terms.items():
        x_mask = z_mask = n_y = 0
        for qubit, pauli in term:
            bit = 1 << int(qubit)
            if pauli == "X":
                x_mask |= bit
            elif pauli == "Y":
                x_mask |= bit
                z_mask |= bit
                n_y += 1
            else:
                z_mask |= bit
        groups.setdefault(x_mask, []).append((z_mask, complex(coefficient) * _Y_PHASE[n_y % 4]))
    return groups


def _subspace_matrix(
    qubit_hamiltonian: QubitOperator, labels: Sequence[int]
) -> scipy.sparse.csr_matrix:
    """``<labels[i]| H |labels[j]>`` by walking Pauli terms; ``labels`` sorted, unique.

    Vectorised over int64 when every label and mask fits in
    ``_VECTORISED_MAX_BITS`` bits, otherwise exact on Python integers.
    """
    groups = _pauli_groups(qubit_hamiltonian)
    integer_labels = [int(label) for label in labels]
    masks = [
        mask for x_mask, weighted in groups.items() for mask in (x_mask, *(z for z, _ in weighted))
    ]
    widest = max([*integer_labels, *masks], default=0).bit_length()
    if widest <= _VECTORISED_MAX_BITS:
        return _subspace_matrix_int64(groups, np.asarray(integer_labels, dtype=np.int64))
    return _subspace_matrix_wide(groups, integer_labels)


def _subspace_matrix_wide(
    groups: dict[int, list[tuple[int, complex]]], labels: list[int]
) -> scipy.sparse.csr_matrix:
    dim = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    for x_mask, weighted in groups.items():
        for col, label in enumerate(labels):
            row = index.get(label ^ x_mask)
            if row is None:
                continue
            rows.append(row)
            cols.append(col)
            values.append(
                sum(
                    weight * (-1 if (label & z_mask).bit_count() & 1 else 1)
                    for z_mask, weight in weighted
                )
            )
    return scipy.sparse.coo_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, cols)), shape=(dim, dim)
    ).tocsr()


def _subspace_matrix_int64(
    groups: dict[int, list[tuple[int, complex]]], labels: NDArray[np.int64]
) -> scipy.sparse.csr_matrix:
    dim = len(labels)
    rows: list[NDArray] = []
    cols: list[NDArray] = []
    values: list[NDArray] = []
    for x_mask, weighted in groups.items():
        images = labels ^ np.int64(x_mask)
        positions = np.minimum(np.searchsorted(labels, images), dim - 1)
        (sources,) = np.nonzero(labels[positions] == images)
        if sources.size == 0:
            continue
        occupied = labels[sources]
        amplitudes = np.zeros(sources.size, dtype=np.complex128)
        for z_mask, weight in weighted:
            amplitudes += weight * (1 - 2 * _parity(occupied & np.int64(z_mask)))
        rows.append(positions[sources])
        cols.append(sources)
        values.append(amplitudes)
    if not values:
        return scipy.sparse.csr_matrix((dim, dim), dtype=np.complex128)
    return scipy.sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    ).tocsr()


def _hermitian_real(matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """Reject a non-Hermitian build; cast to real when the imaginary part vanishes.

    scipy reroutes a complex matrix from ``eigsh`` to non-Hermitian Arnoldi and
    silently drops imaginary eigenvalue parts, so real input must stay real.
    """
    if matrix.nnz == 0:
        return matrix.real.tocsr()
    scale = max(1.0, float(abs(matrix).max()))
    if float(abs(matrix - matrix.conj().T).max()) > 1e-10 * scale:
        raise ValueError(
            "QSCI subspace Hamiltonian is not Hermitian; check that the operator is Hermitian"
        )
    if float(np.abs(matrix.data.imag).max()) <= 1e-14 * scale:
        return matrix.real.tocsr()
    return matrix


def _eigensolve(
    matrix: scipy.sparse.csr_matrix, n_roots: int
) -> tuple[NDArray[np.float64], NDArray]:
    """Lowest ``n_roots`` eigenpairs, ascending: dense ``eigh`` or ARPACK ``eigsh``.

    ARPACK refuses ``n_roots >= dim - 1`` for complex matrices, so that goes dense
    regardless of size.
    """
    dim = matrix.shape[0]
    if matrix.count_nonzero() == 0:
        # ARPACK fails on an all-zero matrix ("starting vector is zero").
        return np.zeros(n_roots), np.eye(dim, n_roots, dtype=matrix.dtype)
    if dim <= _DENSE_EIGH_MAX_DIM or n_roots >= dim - 1:
        energies, vectors = scipy.linalg.eigh(matrix.toarray(), subset_by_index=[0, n_roots - 1])
    else:
        energies, vectors = scipy.sparse.linalg.eigsh(matrix, k=n_roots, which="SA")
        order = np.argsort(energies)
        energies, vectors = energies[order], vectors[:, order]
    return np.asarray(energies, dtype=np.float64), vectors
