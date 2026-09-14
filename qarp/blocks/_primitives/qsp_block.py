from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from .._block import SimpleBlock

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


class QSPBlock(SimpleBlock):
    def __init__(
        self,
        a,
        P_angles,
        target_qubits: Optional[List[int]] = None,
        name: str = "QSP",
    ):
        r"""Construct the Quantum Signal Processing for a real parameter "a" between [-1,1]

        It builds the circuit that transforms "a" according to a chosen polynomial

        Based on Eqs (1-3) of https://arxiv.org/pdf/2105.02859

        Args:
            a: float in [-1,1]
            P_angles: optimal angles (radians) to reproduce the polynomial transformation
        """
        self.a = a
        self.P_angles = P_angles

        super().__init__(
            n_qubits=1,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # qarpx rz/rx take radians with U = exp(-i θ/2 P): each QSP phase α
        # enters as rz(-2α) = exp(+iα Z); the signal rotation is rx(-2·arccos(a)).
        rx_angle = -2 * np.arccos(self.a)
        for i in range(len(self.P_angles) - 1, 0, -1):
            self.rz(0, -2 * self.P_angles[i])
            self.rx(0, rx_angle)
        self.rz(0, -2 * self.P_angles[0])


class QSPAngleFinder:
    r"""Returns the optimal angles to build a certain polynomial transformation

    Args:
        P: polynomial transformation (has to be all even or all odd)
    """

    def __init__(self, P):
        self.P = P

    # Function to create the rotation matrix for a given Pauli matrix and angle
    def rotation_matrix(self, pauli_matrix, theta):
        return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * pauli_matrix

    def S(self, phi):
        return self.rotation_matrix(Z, -2 * phi)

    def W(self, a):
        return self.rotation_matrix(X, -2 * np.arccos(a))

    def U_phi(self, phi, a):
        Usp = self.S(phi[0])
        for i in range(1, len(phi)):
            Usp = Usp @ self.W(a) @ self.S(phi[i])
        return Usp

    # This function computes the angles for the QSP algorithm as a minimization of a loss function
    def QSP(self):

        x_vals = np.linspace(-1, 1, 100)
        target = [np.polyval(self.P[::-1], xi) for xi in x_vals]

        def loss(phi, x):
            loss = sum((target[i] - self.U_phi(phi, x[i])[0, 0].real) ** 2 for i in range(len(x)))
            return loss

        phi0 = [0.5 * np.pi] * len(self.P)
        options = {"xatol": 1e-12}

        result = minimize(loss, phi0, args=(x_vals,), method="Nelder-Mead", options=options)

        return result.x

    # This function the angles for the QSVT algorithm based on the QSP plus an update according to Eq A5 "Reflection convention for QSP" in https://arxiv.org/abs/2105.02859
    def QSVT(self):

        angles = self.QSP()
        num_angles = len(angles)
        d = num_angles - 1

        # From the paper, explicitly
        angles[0] += (2 * d - 1) * np.pi / 4
        angles[-1] += -np.pi / 4
        angles[1:-1] += -np.pi / 2

        return angles
