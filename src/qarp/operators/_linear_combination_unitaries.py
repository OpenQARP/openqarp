import math
from typing import List, Optional, Tuple

import numpy as np

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MATRIX_MAP = {"I": I, "X": X, "Y": Y, "Z": Z}
PAULI_SYMPLECTIC = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


class LinearCombinationUnitaries:
    """Pauli/LCU decomposition of a matrix.

    The decomposition represents a square power-of-two matrix as

        A = sum_P c_P P

    where P is a tensor product of single-qubit Pauli matrices. If the input
    matrix is not square power-of-two sized and ``pad=True``, it is decomposed
    after zero-padding to the next square power-of-two dimension.

    ``A`` is read in the qarpx LSB basis (row/column bit ``k`` ↔ qubit ``k``,
    conventions §1) and Pauli strings are qubit-ordered: character ``k`` acts
    on qubit ``k``, so ``"XZ"`` is ``Z_1 X_0 = kron(Z, X)``.  This is the same
    indexing as ``QubitOperator`` terms and ``PauliBlock`` strings, and
    ``to_QubitOperator().sparse_matrix()`` reproduces ``A``.
    """

    def __init__(self, A: np.ndarray, *, pad: bool = True, tol: float = 1e-12):
        self.A = self._validate_matrix(A)
        self.pad = pad
        self.tol = tol

    def decomposition(self, tol: Optional[float] = None) -> List[Tuple[str, complex]]:
        """Return the Pauli decomposition as ``[(pauli_string, coefficient), ...]``."""
        threshold = self.tol if tol is None else tol
        A = self.padded_matrix()

        H = 0.5 * (A + A.conj().T)
        iS = 0.5j * (A - A.conj().T)

        coeffs: dict[str, complex] = {}

        # For Hermitian inputs, Pauli coefficients are real. Taking the real
        # part also removes small numerical imaginary noise from the TPD pass.
        for pstr, coef in self.pauli_basis2ppoly(self.tpd(H).real, tol=threshold):
            coeffs[pstr] = coeffs.get(pstr, 0.0) + coef

        for pstr, coef in self.pauli_basis2ppoly(self.tpd(iS).real, tol=threshold):
            coeffs[pstr] = coeffs.get(pstr, 0.0) - 1j * coef

        result = []
        for pstr, coef in coeffs.items():
            cleaned = self._clean_scalar(coef, threshold)
            if abs(cleaned) > threshold:
                result.append((pstr, cleaned))
        return result

    def to_QubitOperator(self, tol: Optional[float] = None):
        """Convert the decomposition to a ``qarp.operators.QubitOperator``."""
        from qarp.operators._qubit_operator import QubitOperator

        qo = QubitOperator()
        for paulis, coef in self.decomposition(tol=tol):
            term = " ".join(f"{pauli}{qubit}" for qubit, pauli in enumerate(paulis) if pauli != "I")
            qo += QubitOperator(term, coef)
        return qo

    def padded_matrix(self) -> np.ndarray:
        """Return the matrix used for decomposition, padding if requested."""
        A = self.A.astype(complex, copy=True)
        if self.assert_qubitlike_shape(A):
            return A
        if not self.pad:
            raise ValueError(
                "Input matrix must be square with dimension 2**n. "
                "Use pad=True to zero-pad it automatically."
            )
        return self.adjust_padding(A)

    def pauli_str_to_matrix(self, pauli_string: str) -> np.ndarray:
        """Convert a qubit-ordered Pauli string such as ``'XXYI'`` to its
        LSB matrix ``P_{n-1} ⊗ … ⊗ P_0``."""
        mat = np.array([[1]], dtype=complex)
        for pauli_letter in reversed(pauli_string):
            try:
                pauli = PAULI_MATRIX_MAP[pauli_letter]
            except KeyError as exc:
                raise ValueError(f"Invalid Pauli letter {pauli_letter!r}.") from exc
            mat = np.kron(mat, pauli)
        return mat

    def assert_qubitlike_shape(self, matrix: Optional[np.ndarray] = None) -> bool:
        """Return True when the matrix is square with dimension ``2**n``."""
        A = self.A if matrix is None else matrix
        if A.ndim != 2:
            return False
        rows, cols = A.shape
        return rows == cols and self._is_power_of_two(rows)

    def adjust_padding(self, matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a zero-padded square power-of-two copy of ``matrix``."""
        A = self.A if matrix is None else self._validate_matrix(matrix)
        rows, cols = A.shape
        target_dim = self._next_power_of_two(max(rows, cols))

        padded = np.zeros((target_dim, target_dim), dtype=np.result_type(A, complex))
        padded[:rows, :cols] = A
        return padded

    def tpd(self, matrix: np.ndarray) -> np.ndarray:
        """Tensorized Pauli decomposition core transform.

        This returns the coefficient matrix in Pauli-basis indexing. The input
        is copied, so callers do not need to guard against mutation.
        """
        H = np.array(matrix, dtype=complex, copy=True)
        if not self.assert_qubitlike_shape(H):
            raise ValueError("TPD input must be square with dimension 2**n.")

        size = H.shape[0]
        nqubits = int(math.log2(size))
        block_size = size

        for step in range(nqubits):
            nblocks = 2**step
            block_size //= 2
            stride = 2 * block_size

            for row_block in range(nblocks):
                row = row_block * stride
                for col_block in range(nblocks):
                    col = col_block * stride

                    top_left = H[row : row + block_size, col : col + block_size].copy()
                    bottom_right = H[
                        row + block_size : row + stride,
                        col + block_size : col + stride,
                    ].copy()
                    top_right = H[
                        row : row + block_size,
                        col + block_size : col + stride,
                    ].copy()
                    bottom_left = H[
                        row + block_size : row + stride,
                        col : col + block_size,
                    ].copy()

                    H[row : row + block_size, col : col + block_size] = top_left + bottom_right
                    H[
                        row + block_size : row + stride,
                        col + block_size : col + stride,
                    ] = top_left - bottom_right
                    H[row : row + block_size, col + block_size : col + stride] = (
                        top_right + bottom_left
                    )
                    H[row + block_size : row + stride, col : col + block_size] = 1j * (
                        top_right - bottom_left
                    )

        return H / size

    def pauli_basis2ppoly(
        self, cmat: np.ndarray, tol: Optional[float] = None
    ) -> List[Tuple[str, complex]]:
        """Transform a Pauli-basis coefficient matrix into Pauli-string terms."""
        threshold = self.tol if tol is None else tol
        cmat = np.asarray(cmat)
        if not self.assert_qubitlike_shape(cmat):
            raise ValueError("Coefficient matrix must be square with dimension 2**n.")

        nqubits = int(math.log2(cmat.shape[0]))
        rows, cols = np.nonzero(np.abs(cmat) > threshold)

        ppoly = []
        for i, j in zip(rows, cols, strict=True):
            coef = self._clean_scalar(cmat[i, j], threshold)
            if abs(coef) > threshold:
                ppoly.append((self.ij_code2_pstr((int(i), int(j)), nqubits), coef))
        return ppoly

    def reconstruct(
        self,
        ppoly: List[Tuple[str, complex]],
        *,
        nqubits: Optional[int] = None,
        crop: bool = False,
    ) -> np.ndarray:
        """Reconstruct a matrix from a Pauli decomposition.

        If ``crop=True``, the reconstructed padded matrix is cropped back to the
        original input shape.
        """
        if nqubits is None:
            if ppoly:
                nqubits = len(ppoly[0][0])
            else:
                nqubits = int(math.log2(self.padded_matrix().shape[0]))

        mat = self.itpd_core(self.ppoly2pauli_basis(ppoly, nqubits=nqubits))
        if crop:
            rows, cols = self.A.shape
            return mat[:rows, :cols]
        return mat

    def ppoly2pauli_basis(
        self, ppoly: List[Tuple[str, complex]], *, nqubits: Optional[int] = None
    ) -> np.ndarray:
        """Convert weighted Pauli strings into the TPD coefficient matrix."""
        if nqubits is None:
            if not ppoly:
                raise ValueError("nqubits is required when ppoly is empty.")
            nqubits = len(ppoly[0][0])

        size = 2**nqubits
        mat = np.zeros((size, size), dtype=complex)

        for pstr, weight in ppoly:
            if len(pstr) != nqubits:
                raise ValueError(f"Expected Pauli strings of length {nqubits}, got {pstr!r}.")
            i, j = self.pstr2ij_code(pstr)
            mat[i, j] += weight
        return mat

    def ij_code2_pstr(self, ns: Tuple[int, int], length: int) -> str:
        return self.sym_code2pstr(self.ij_code2sym_code(*ns), length)

    def sym_code2pstr(self, ns: Tuple[int, int], length: int) -> str:
        if length <= 0:
            raise ValueError("length must be positive.")

        nx, nz = ns
        max_int = 2**length
        if not (0 <= nx < max_int and 0 <= nz < max_int):
            raise ValueError(f"Symplectic code {ns!r} does not fit length {length}.")

        # Character k ↔ symplectic bit k ↔ qubit k (LSB, §1).
        result = []
        for k in range(length):
            x_bit = (nx >> k) & 1
            z_bit = (nz >> k) & 1
            if x_bit == 0 and z_bit == 0:
                result.append("I")
            elif x_bit == 1 and z_bit == 0:
                result.append("X")
            elif x_bit == 1 and z_bit == 1:
                result.append("Y")
            else:
                result.append("Z")
        return "".join(result)

    def pstr2sym_code(
        self, pstr: str, sim_code: Optional[dict[str, Tuple[int, int]]] = None
    ) -> Tuple[int, int]:
        pauli_sim_dict = PAULI_SYMPLECTIC if sim_code is None else sim_code
        x_num = 0
        z_num = 0
        place_value = 1

        for pauli in pstr:  # character k ↔ bit k
            try:
                nx, nz = pauli_sim_dict[pauli]
            except KeyError as exc:
                raise ValueError(f"Invalid Pauli letter {pauli!r}.") from exc
            x_num += nx * place_value
            z_num += nz * place_value
            place_value *= 2
        return x_num, z_num

    def ij_code2sym_code(self, i: int, j: int) -> Tuple[int, int]:
        return i ^ j, i

    def pstr2ij_code(self, pstr: str) -> Tuple[int, int]:
        return self.sym_code2ij_code(*self.pstr2sym_code(pstr))

    def sym_code2ij_code(self, x: int, z: int) -> Tuple[int, int]:
        return z, x ^ z

    def itpd_core(self, matrix: np.ndarray) -> np.ndarray:
        """Inverse TPD transform: restore the standard matrix representation."""
        mat = np.array(matrix, dtype=complex, copy=True)
        if not self.assert_qubitlike_shape(mat):
            raise ValueError("Inverse TPD input must be square with dimension 2**n.")

        size = mat.shape[0]
        unit_size = 1

        while unit_size < size:
            stride = 2 * unit_size

            for row in range(0, size, stride):
                for col in range(0, size, stride):
                    top_left = mat[row : row + unit_size, col : col + unit_size].copy()
                    bottom_right = mat[
                        row + unit_size : row + stride,
                        col + unit_size : col + stride,
                    ].copy()
                    top_right = mat[
                        row : row + unit_size,
                        col + unit_size : col + stride,
                    ].copy()
                    bottom_left = mat[
                        row + unit_size : row + stride,
                        col : col + unit_size,
                    ].copy()

                    mat[row : row + unit_size, col : col + unit_size] = top_left + bottom_right
                    mat[
                        row + unit_size : row + stride,
                        col + unit_size : col + stride,
                    ] = top_left - bottom_right
                    mat[row : row + unit_size, col + unit_size : col + stride] = (
                        top_right - 1j * bottom_left
                    )
                    mat[row + unit_size : row + stride, col : col + unit_size] = (
                        top_right + 1j * bottom_left
                    )

            unit_size = stride

        return mat

    @staticmethod
    def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
        A = np.asarray(matrix)
        if A.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got array with ndim={A.ndim}.")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("Input matrix must not be empty.")
        return A

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    @classmethod
    def _next_power_of_two(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive.")
        if cls._is_power_of_two(value):
            return value
        return 1 << (value - 1).bit_length()

    @staticmethod
    def _clean_scalar(value: complex, tol: float) -> complex:
        z = complex(value)
        real = 0.0 if abs(z.real) <= tol else z.real
        imag = 0.0 if abs(z.imag) <= tol else z.imag
        cleaned = real + 1j * imag
        return cleaned.real if cleaned.imag == 0.0 else cleaned
