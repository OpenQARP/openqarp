"""Fermion-to-qubit mappings.

``encode_operator`` delegates to the qarpx C++ transform kernels
(:mod:`qarp.operators.functions`); ``encode_state`` converts an
occupation-number vector (abab list, index ``i`` = spin orbital ``i`` =
qubit ``i``, LSB — see :mod:`qarp.operators.onv`) into the mapped
computational-basis bitstring, in the same qubit-indexed LSB order.

These stay *classes* (unlike the function-first fermionic surface around
them) because blocks dispatch polymorphically on ``mapping=`` parameters
(``MappedONVStateBlock``, ``UCCBlock``, ``PauliBlockFactory``) — the
encode_operator/encode_state pair travels as one object.  ``Parity`` and
``BravyiKitaev`` carry constructor state because their transforms of an
*operator* depend on the total register size, which the operator alone
cannot fix (JW never needs it).  ``Parity`` requires it; ``BravyiKitaev``
defaults to inferring it — per operator when given one, and once across
the whole list when given several, so every member of a list is mapped on
the same register.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from ._fermion_operator import FermionOperator
from ._qubit_operator import QubitOperator
from .functions import bravyi_kitaev, count_qubits, jordan_wigner, parity_transform
from .onv import Onv


class Mapping(ABC):
    """Abstract base class for fermion-to-qubit mappings."""

    @abstractmethod
    def encode_operator(
        self, op: Union[list[FermionOperator], FermionOperator]
    ) -> Union[list[QubitOperator], QubitOperator]:
        """Map the provided FermionOperator(s) to QubitOperator(s)."""

    @abstractmethod
    def encode_state(self, onv: Onv) -> list[int]:
        """Map an occupation-number vector to the mapped qubit basis state."""


class JordanWigner(Mapping):
    """A class for mapping FermionOperators to qubit operators using the Jordan-Wigner mapping."""

    def encode_operator(self, op: Union[list[FermionOperator], FermionOperator]):
        """Perform a Jordan-Wigner mapping of the provided operator(s).

        Args:
            op: A list of, or single, FermionOperator object.

        Returns:
             The result of the mapping as a list of, or single, QubitOperator.
        """
        qop = None
        if isinstance(op, FermionOperator):
            qop = jordan_wigner(op)
        if isinstance(op, list):
            qop = [jordan_wigner(i) for i in op]
        if qop is None or not (isinstance(op, FermionOperator) or isinstance(op, list)):
            raise RuntimeError("Unrecognised type passed to .encode_operator().")
        return qop

    def encode_state(self, onv: Onv) -> list[int]:
        """Perform a state mapping of a provided onv to an initial bitrepresentation.

        Args:
            onv: An occupation-number vector (abab list).

        Returns:
            A list of 0s and 1s representing the initial qubit state.
        """
        # A copy: callers store the result and must not alias the input.
        return list(onv)


def _beta(onv: Onv):
    """Computes the beta matrix associated to transform from occupation number basis to Bravyi-Kitaev basis.

    Args:
        onv: An occupation-number vector.

    Returns:
        beta: transformation matrix as a numpy array
    """
    nqubits = round(
        np.ceil(np.log2(len(onv)))
    )  # ceil because we need to add ancilla qubits to store all orbitals
    iden = np.eye(2)
    matrix = np.array([[1]])
    for x in range(nqubits):
        matrix = np.kron(iden, matrix)
        matrix[-1, : 2**x] = 1
    return matrix[: len(onv), : len(onv)]


class BravyiKitaev(Mapping):
    """A class for mapping FermionOperators to qubit operators using the Bravyi-Kitaev mapping.

    The BK transform of an operator depends on the register width: the update,
    parity and remainder sets of orbital ``j`` change with ``n_qubits``.  A
    list is therefore mapped on **one** width — ``n_qubits`` when given, else
    the largest ``count_qubits`` over the list — so its members compose
    (e.g. UCC generators against a Hamiltonian).  A single operator with
    ``n_qubits=None`` keeps the per-operator inference.

    Args:
        n_qubits: register width for every ``encode_operator`` call, or
            ``None`` to infer it (per call, as described above).
    """

    def __init__(self, n_qubits: Optional[int] = None):
        self.n_qubits = n_qubits

    def encode_operator(self, op: Union[list[FermionOperator], FermionOperator]):
        """Perform a Bravyi-Kitaev mapping of the provided operator(s).

        Args:
            op: A list of, or single, FermionOperator object.

        Returns:
             The result of the mapping as a list of, or single, QubitOperator.
        """
        if isinstance(op, FermionOperator):
            return bravyi_kitaev(op, self.n_qubits)
        if isinstance(op, list):
            if not op:
                return []
            width = self.n_qubits
            if width is None:
                width = max(count_qubits(f) for f in op)
            return bravyi_kitaev(op, width)
        raise RuntimeError("Unrecognised type passed to .encode_operator().")

    def encode_state(self, onv: Onv) -> list[int]:
        """Perform a state mapping of a provided onv to the Bravyi-Kitaev basis.

        Args:
            onv: An occupation-number vector (abab list).

        Returns:
            A list of 0s and 1s representing the initial qubit state in the Bravyi-Kitaev basis.
        """
        beta = _beta(onv)
        onvBK = np.mod(beta @ onv, 2)
        return [int(i) for i in onvBK]


class Parity(Mapping):
    """A class for mapping FermionOperators to qubit operators using the parity mapping.
    Args:
       n_qubits: number of qubits in the qubit register.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def encode_operator(self, op: Union[list[FermionOperator], FermionOperator]):
        """Perform a parity mapping of the provided operator(s).

        The transform runs in the qarpx C++ kernel (a verbatim port of the
        previous hand-rolled Python implementation, term order included).

        Args:
            op: A list of, or single, FermionOperator object.

        Returns:
             The result of the mapping as a list of, or single, QubitOperator.
        """
        if isinstance(op, (FermionOperator, list)):
            return parity_transform(op, self.n_qubits)
        raise RuntimeError("Unrecognised type passed to .encode_operator().")

    def encode_state(self, onv: Onv) -> list[int]:
        """Perform a state mapping of a provided onv to the parity basis.

        Args:
            onv: An occupation-number vector (abab list).

        Returns:
            A list of 0s and 1s representing the initial qubit state in the parity basis.
        """
        ones = np.ones((len(onv), len(onv)))
        pi = np.tril(ones)
        onvP = np.mod(pi @ onv, 2)
        return [int(i) for i in onvP]
