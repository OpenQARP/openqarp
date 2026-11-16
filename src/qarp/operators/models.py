"""Model Hamiltonian builders.

Two return types, split by physics — check before encoding:

- ``fermi_hubbard`` → :class:`FermionOperator` (map it with a
  :class:`~qarp.operators.Mapping` before use on qubits);
- ``lipkin``, ``transverse_field_ising``, ``xy_model`` →
  :class:`QubitOperator`, native to the qubit basis — **no**
  fermion-to-qubit mapping is involved or needed.

Fermionic models are built from numpy tensors via
:func:`qarp.operators.fermion_operator_from_tensor`.

Lattice models share one geometry: sites of an n-dimensional box ``dims``
are indexed with the first dimension fastest
(``site = x + dims[0]·y + dims[0]·dims[1]·z + ...``), and nearest-neighbour
bonds optionally wrap around dimensions of size > 2 (a wrap on a smaller
dimension would double an existing bond).
"""

import itertools
import math
from collections.abc import Sequence
from typing import Union

import numpy as np

from ._fermion_operator import FermionOperator
from ._qubit_operator import QubitOperator
from ._tensor_operators import fermion_operator_from_tensor


def _hypercubic_bonds(dims: Sequence[int], periodic: bool) -> list[tuple[int, int]]:
    """Nearest-neighbour site pairs of the ``dims`` box, one entry per bond."""
    if not dims or any(extent < 1 for extent in dims):
        raise ValueError(f"Lattice dimensions must be positive, got {tuple(dims)}.")
    strides = [1]
    for extent in dims[:-1]:
        strides.append(strides[-1] * extent)

    def site(coords: Sequence[int]) -> int:
        return sum(c * s for c, s in zip(coords, strides, strict=True))

    bonds = []
    for coords in itertools.product(*(range(extent) for extent in dims)):
        for axis, extent in enumerate(dims):
            neighbour = list(coords)
            if coords[axis] + 1 < extent:
                neighbour[axis] += 1
            elif periodic and extent > 2:
                neighbour[axis] = 0
            else:
                continue
            bonds.append((site(coords), site(neighbour)))
    return bonds


def _per_site(value: Union[float, Sequence[float]], n_sites: int) -> list[float]:
    """Broadcast a scalar, or validate a per-site sequence."""
    if isinstance(value, (int, float)):
        return [float(value)] * n_sites
    values = [float(v) for v in value]
    if len(values) != n_sites:
        raise ValueError(f"Expected {n_sites} per-site values, got {len(values)}.")
    return values


def fermi_hubbard(
    dims: Sequence[int], t: float, U: float, V: float = 0.0, periodic: bool = False
) -> FermionOperator:
    """(Extended) Fermi-Hubbard Hamiltonian on a hypercubic lattice.

    .. math::

        H = -t \\sum_{\\langle i,j \\rangle, \\sigma}
            (a^\\dagger_{i\\sigma} a_{j\\sigma} + \\text{h.c.})
            + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}
            + V \\sum_{\\langle i,j \\rangle} n_i n_j

    Site ``s`` owns the abab spin-orbital pair ``(2s, 2s + 1)``; for a 2D
    lattice the conventions coincide with openfermion's ``fermi_hubbard``
    (the test oracle).  A chain is ``dims=(n,)``, a 3D box ``(n_x, n_y, n_z)``.

    Args:
        dims: Lattice extents, e.g. ``(4,)``, ``(3, 2)``, ``(2, 2, 2)``.
        t: The kinetic (hopping) energy.
        U: The on-site repulsion.
        V: Nearest-neighbour density-density coupling (extended Hubbard).
        periodic: If true, wrap neighbours around each dimension of size > 2.

    Returns:
        The Hamiltonian as a FermionOperator.
    """
    bonds = _hypercubic_bonds(dims, periodic)
    n_sites = math.prod(dims)
    n_spin_orbitals = 2 * n_sites

    one_body = np.zeros((n_spin_orbitals, n_spin_orbitals))
    for s1, s2 in bonds:
        # The alpha and beta channels hop identically along each lattice bond.
        for a, b in ((2 * s1, 2 * s2), (2 * s1 + 1, 2 * s2 + 1)):
            one_body[a, b] = -t
            one_body[b, a] = -t

    # Density products n_a n_b = a†_a a†_b a_b a_a for a ≠ b (creators-first
    # order absorbs the anticommutation sign).
    two_body = np.zeros((n_spin_orbitals,) * 4)
    for s in range(n_sites):
        two_body[2 * s, 2 * s + 1, 2 * s + 1, 2 * s] = U
    for s1, s2 in bonds:
        for a in (2 * s1, 2 * s1 + 1):
            for b in (2 * s2, 2 * s2 + 1):
                two_body[a, b, b, a] = V
    return fermion_operator_from_tensor(one_body) + fermion_operator_from_tensor(two_body)


def transverse_field_ising(
    dims: Sequence[int],
    j: float,
    h_x: Union[float, Sequence[float]],
    h_z: Union[float, Sequence[float]] = 0.0,
    periodic: bool = False,
) -> QubitOperator:
    """Transverse-field Ising Hamiltonian on a hypercubic lattice (one qubit per site).

    .. math::

        H = -j \\sum_{\\langle a,b \\rangle} Z_a Z_b
            - \\sum_s h^x_s X_s - \\sum_s h^z_s Z_s

    ``h_x`` and ``h_z`` accept a scalar or a per-site sequence — per-site
    fields give the random-field Ising model, with the caller owning the
    randomness (pass a seeded draw).

    Args:
        dims: Lattice extents, e.g. ``(6,)`` or ``(3, 3)``.
        j: The ZZ coupling.
        h_x: Transverse field, scalar or one value per site.
        h_z: Longitudinal field, scalar or one value per site.
        periodic: If true, wrap neighbours around each dimension of size > 2.

    Returns:
        The Hamiltonian as a QubitOperator.
    """
    n_sites = math.prod(dims)
    transverse = _per_site(h_x, n_sites)
    longitudinal = _per_site(h_z, n_sites)
    operator = QubitOperator()
    for a, b in _hypercubic_bonds(dims, periodic):
        operator += QubitOperator(f"Z{a} Z{b}", -j)
    for s in range(n_sites):
        if transverse[s]:
            operator += QubitOperator(f"X{s}", -transverse[s])
        if longitudinal[s]:
            operator += QubitOperator(f"Z{s}", -longitudinal[s])
    return operator


def xy_model(dims: Sequence[int], j: float, periodic: bool = False) -> QubitOperator:
    """Isotropic XY Hamiltonian on a hypercubic lattice (one qubit per site).

    .. math::

        H = -\\frac{j}{2} \\sum_{\\langle a,b \\rangle} (X_a X_b + Y_a Y_b)

    On a chain this is exactly the Jordan-Wigner image of free-fermion
    hopping (the test oracle), and it is the hardcore Bose-Hubbard model.

    Args:
        dims: Lattice extents, e.g. ``(6,)`` or ``(3, 3)``.
        j: The XY coupling.
        periodic: If true, wrap neighbours around each dimension of size > 2.

    Returns:
        The Hamiltonian as a QubitOperator.
    """
    operator = QubitOperator()
    for a, b in _hypercubic_bonds(dims, periodic):
        operator += QubitOperator(f"X{a} X{b}", -j / 2)
        operator += QubitOperator(f"Y{a} Y{b}", -j / 2)
    return operator


def lipkin(n: int, t: float, V: float) -> QubitOperator:
    """Lipkin-model Hamiltonian, natively in the qubit basis (no mapping required).

    .. math::

        H = \\frac{t}{2} \\sum_p Z_p
            - \\frac{V}{2} \\sum_{p < q} (X_p X_q - Y_p Y_q)

    Args:
        n: The number of sites in the chain.
        t: The kinetic energy.
        V: The two-site coupling constant.

    Returns:
        The Hamiltonian as a QubitOperator.
    """
    operator = QubitOperator()
    for p in range(n):
        operator += t * QubitOperator(f"Z{p}", 0.5)
    for p, q in itertools.combinations(range(n), 2):
        operator += -V * (QubitOperator(f"X{p} X{q}", 0.5) - QubitOperator(f"Y{p} Y{q}", 0.5))
    return operator
