from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np

import qarpx as qx
from qarp.operators import FermionOperator

from ..blocks import AnyBlock, ComputationalBasisStateBlock
from ..blocks._block import CompositeBlockBase
from ..endianness import label_to_bits

# Re-export: the operator ↔ graph conversions live in qarp.graphs._utils.
from ..operators import JordanWigner


def nqubit_states_with_k_ones(nqubits: int, k: int) -> list[int]:
    """
    Finds all the integer values corresponding to the binary (computational) states of nqubits with k ones (fixed k Hamming weight)

    Based on Gosper's hack

    Args:
        nqubits: number of qubits of the computational states
        k: number of ones (Hamming weight) of the desired computational states

    Returns:
        result: list of integers corresponding to the computational states with fixed k ones (e.g. bin(i in list) = "1001", "1100"... for n=4, k=2)
    """
    if k > nqubits or k < 0:
        return []
    result = []
    x = (1 << k) - 1  # smallest number with k ones
    limit = 1 << nqubits
    while x < limit:
        result.append(x)
        if x == 0:
            break  # k == 0: only the empty state; Gosper's step needs x != 0
        # Gosper's hack for next combination
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r
    return result


def generate_states_new_basis(
    U: AnyBlock,
    hamming_weight: Optional[Union[int, List[int]]] = None,
    get_statevector: bool = True,
) -> Tuple[List[np.ndarray], List[AnyBlock], List[int]]:
    r"""Apply a unitary block ``U`` to every computational basis state and
    (optionally) compute the resulting statevectors.

    The new basis is :math:`\{ U |i\rangle : i \in \mathrm{indices} \}`.

    WARNING: exponential scaling — the indices set is ``2^n_qubits`` unless
    ``hamming_weight`` filters it down.

    Args:
        U: A built :class:`qarp.blocks.AnyBlock` whose unitary is the change of basis.
        hamming_weight: If int, restrict to computational basis states with
            exactly that Hamming weight.  If list[int], restrict to the union
            of those weights.  If None, use every basis state.
        get_statevector: If True, compute each new basis state's statevector
            via the qarpx simulator (still exponential).  If False, return an
            empty statevector list.

    Returns:
        ``(new_basis_states, new_basis_blocks, basis_states_indices)`` where:

        * ``new_basis_states`` — list of length-``2^n_qubits`` ndarrays (only
          populated when ``get_statevector=True``).
        * ``new_basis_blocks`` — :class:`Block` per basis state:
          ``ComputationalBasisStateBlock(bitstring) + U``.
        * ``basis_states_indices`` — the integer indices the basis was built from.
    """
    n_qubits: int = U.n_qubits

    if hamming_weight is not None:
        if isinstance(hamming_weight, int):
            basis_states_indices = nqubit_states_with_k_ones(n_qubits, hamming_weight)
        elif isinstance(hamming_weight, list):
            temp_indices: List[int] = []
            for hw_val in hamming_weight:
                if not isinstance(hw_val, int):
                    raise TypeError(
                        f"All elements in hamming_weight list must be integers, got {type(hw_val)}"
                    )
                temp_indices.extend(nqubit_states_with_k_ones(n_qubits, hw_val))
            basis_states_indices = temp_indices
        else:
            raise TypeError("hamming_weight must be int, list[int], or None")
    else:
        basis_states_indices = list(range(2**n_qubits))

    sim = qx.QarpSimulator() if get_statevector else None

    full_qubits = list(range(n_qubits))
    new_basis_states: List[np.ndarray] = []
    new_basis_blocks: List[AnyBlock] = []

    # Build U once; reuse for every basis state.  Lazy
    # ``_pending_substitutions`` / ``_pending_replacements`` on ``U`` get
    # auto-materialised by ``CompositeBlockBase.add_child`` — no manual
    # flatten/SimpleBlock dance needed.
    U.build()

    for i in basis_states_indices:
        # LSB-first bitstring matching ComputationalBasisStateBlock's convention.
        bitstring = label_to_bits(i, n_qubits)
        prep = ComputationalBasisStateBlock(bitstring)
        prep.build()
        prep.target_qubits = full_qubits

        u_built = U.build()
        u_built.target_qubits = full_qubits

        composite = CompositeBlockBase(n_qubits=n_qubits, name=f"new_basis_{i}")
        composite.add_child(prep)
        composite.add_child(u_built)
        composite.build()

        new_basis_blocks.append(composite)

        if sim is not None:
            sv = np.asarray(sim.statevector(composite.flatten(), n_qubits))
            new_basis_states.append(sv)

    return new_basis_states, new_basis_blocks, basis_states_indices


def map_binary_to_integer_keys(probs):
    """
    Maps binary keys to integer keys in a dictionary.

    Args:
        probs (dict): A dictionary with binary keys and float values.

    Returns:
        dict: A new dictionary with integer keys and float values.
    """
    strdict = {int("".join(str(i) for i in key), 2): value for key, value in probs.items()}
    return defaultdict(float, strdict)


def find_occupation_numbers(hamiltonian, n_qubits, tolerance=1e-15, verbose=False):
    """
    Finds the occupation numbers of the eigenstates of a Hamiltonian.

    Args:
        hamiltonian (FermionOperator): The Hamiltonian operator.
        n_qubits (int): The number of qubits.
        tolerance (float): The tolerance for determining occupation numbers.
        verbose (bool): If True, prints the eigenvalues and occupation numbers.

    Returns:
        np.ndarray: An array of occupation numbers for the eigenstates.
    """
    number_operator = FermionOperator()
    for i in range(n_qubits):
        number_operator += FermionOperator(f"{i}^ {i}", 1)
    numop_matrix = JordanWigner().encode_operator(number_operator).sparse_matrix(n_qubits).toarray()
    ham_matrix = hamiltonian.sparse_matrix(n_qubits).toarray()
    eigs, eigvs = np.linalg.eigh(ham_matrix)
    occ_numbers = np.zeros(len(eigs))
    for i in range(len(eigs)):
        coeffs = np.abs(eigvs[:, i])
        # LSB basis labels rendered q0-first (ket reading order).
        state = [
            "".join(str(b) for b in label_to_bits(j, n_qubits))
            for j in range(len(coeffs))
            if coeffs[j] > tolerance
        ]
        occ_numbers[i] = np.real(eigvs[:, i].T.conj() @ numop_matrix @ eigvs[:, i])
        if verbose:
            print(f"Eig: {eigs[i]:.4} Occupation #: {occ_numbers[i]:.2} States: {state}")
    return occ_numbers


def find_eigenspectrum_degeneracy(eigs, tolerance=1e-15, verbose=False):
    """
    Finds the degeneracy of the eigenvalues in the eigenspectrum.

    Args:
        eigs (np.ndarray): The eigenvalues of the Hamiltonian.
        tolerance (float): The tolerance for determining degeneracy.
        verbose (bool): If True, prints the eigenvalues and their degeneracy.

    Returns:
        dict: A dictionary with eigenvalues as keys and their degeneracy as values.
    """
    degeneracy = defaultdict(int)
    for eig in eigs:
        if any(abs(eig - key) < tolerance for key in degeneracy.keys()):
            for key in degeneracy.keys():
                if abs(eig - key) < tolerance:
                    degeneracy[key] += 1
                    break
        else:
            degeneracy[eig] += 1
    if verbose:
        for ele in degeneracy.items():
            print("Eigenvalue: ", ele[0], "Degeneracy: ", ele[1])
    return degeneracy


def find_unique_eigs_and_occupation_numbers(
    eigs, occ_numbers, select_occ=None, tolerance=1e-15, verbose=False
):
    """
    Finds unique eigenvalues and their corresponding occupation numbers.

    Args:
        eigs (np.ndarray): The eigenvalues of the Hamiltonian.
        occ_numbers (np.ndarray): The occupation numbers corresponding to the eigenvalues.
        select_occ (int, optional): If provided, filters the unique eigenvalues by this occupation number.
        tolerance (float): The tolerance for determining degeneracy.
        verbose (bool): If True, prints the unique eigenvalues and their occupation numbers.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary with unique eigenvalues as keys and their degeneracy as values.
            - list: A list of unique occupation numbers corresponding to the unique eigenvalues.
    """
    if len(eigs) != len(occ_numbers):
        raise ValueError("Eigenvalues and occupation numbers must have the same length.")
    unique_eigs_dict = find_eigenspectrum_degeneracy(eigs, tolerance=tolerance, verbose=verbose)
    unique_eigs = list(unique_eigs_dict.keys())
    unique_occ_numbers = np.zeros(len(unique_eigs))
    for idx, eig in enumerate(unique_eigs):
        index = np.where(np.isclose(eigs, eig, atol=tolerance))[0][
            0
        ]  # find the index of the eigenvalue in the eigs array
        unique_occ_numbers[idx] = occ_numbers[index]
    unique_occ_numbers = np.round(unique_occ_numbers).astype(int)  # convert to int
    if select_occ is not None:
        selected_indices = [i for i, occ in enumerate(unique_occ_numbers) if occ == select_occ]
        unique_eigs_dict = {
            k: unique_eigs_dict[k]
            for i, k in enumerate(unique_eigs_dict.keys())
            if i in selected_indices
        }
        unique_occ_numbers = [unique_occ_numbers[i] for i in selected_indices]
        unique_eigs = list(unique_eigs_dict.keys())
    if verbose:
        for eig, occ in zip(unique_eigs, unique_occ_numbers, strict=True):
            print(f"Eigenvalue: {eig:.4} Unique Occupation #: {occ}")
    return unique_eigs_dict, unique_occ_numbers


def dirichlet_kernel_squared(x, phi, N):
    """
    Dirichlet kernel squared function. Used to reconstruct the phase information in the QPE algorithm.

    Args:
        x (array-like): Input values (e.g. eigenvalues).
        phi (float): Phase to fit.
        N (int): Number of qubits.

    Returns:
        array-like: Squared Dirichlet kernel values.
    """
    x = np.asarray(x)
    delta = (phi - x) % 1
    delta = np.where(delta > 0.5, delta - 1, delta)  # map to [-0.5, 0.5]

    small = np.isclose(delta, 0.0, atol=1e-10)
    result = np.zeros_like(delta)
    result[small] = 1.0
    result[~small] = (np.sin(np.pi * N * delta[~small]) / (N * np.sin(np.pi * delta[~small]))) ** 2
    return result


_linear_terms_from_of_operator = lambda operator: {
    term: complex(coeff).real for term, coeff in operator.terms.items() if len(term) == 1
}
