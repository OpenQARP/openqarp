from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray
from quimb.tensor import TensorNetwork
from scipy.linalg import expm
from scipy.optimize import minimize

from qarp.operators._vumpo_network import (
    Boundary,
    Engine,
    NetworkSpec,
    brickwork_gate_order,
    make_engine,
    polynomial_value_and_w,
)

ParamVector = NDArray[np.float64]
GateParams = NDArray[np.float64]
GateParamList = List[GateParams]
StateVector = Sequence[int]
ParamLike = Union[Sequence[float], NDArray[Any]]


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def qubit_operator_to_mpo(H, n_qubits: int, cutoff: float = 1e-12):
    """quimb MPO of a :class:`QubitOperator`, term by term, never dense.

    Each Pauli string is a bond-1 product-operator MPO; the sum is compressed
    by SVD after every term, so the bond dimension follows the operator's
    locality rather than ``n_qubits`` and the result approximates the operator
    to ``cutoff``.  quimb site ``n`` is qubit ``n``, the convention ``VUMPO``
    and ``VUMPOBrickworkBlock`` share; no bit reversal is involved, unlike the
    ``from_dense(lsb_to_msb_matrix(...))`` route.

    Args:
        H: The operator; ``H.terms`` maps ``((qubit, "X"), ...)`` to a coefficient.
        n_qubits: Number of sites of the MPO.
        cutoff: Singular-value cutoff of the compression after each term.

    Returns:
        ``quimb.tensor.MatrixProductOperator`` with ``n_qubits`` sites.
    """
    mpo = None
    for term, coeff in H.terms.items():
        arrays = [_PAULI["I"]] * n_qubits
        for q, pauli in term:
            if q >= n_qubits:
                raise ValueError(
                    f"term {term} acts on qubit {q}, outside an {n_qubits}-qubit register"
                )
            arrays[q] = _PAULI[pauli]
        arrays = [a.copy() for a in arrays]
        arrays[0] = arrays[0] * complex(coeff)
        product = qtn.MPO_product_operator(arrays)
        mpo = product if mpo is None else mpo.add_MPO(product, compress=True, cutoff=cutoff)
    if mpo is None:
        return qtn.MPO_product_operator([np.zeros((2, 2), dtype=complex)] * n_qubits)
    return mpo


@dataclass
class _Term:
    """One scalar network of the objective: ``weight * Re(value)`` or
    ``weight * |value|^2`` (deflation), with its arrays and boundaries."""

    weight: float
    squared: bool
    spec: NetworkSpec
    engine: Engine
    arrays: List[NDArray[Any]]
    left: List[Optional[Boundary]] = field(default_factory=list)
    right: List[Optional[Boundary]] = field(default_factory=list)


def _skew_gradient(
    W: NDArray[np.complex128], S: NDArray[np.complex128], d: int
) -> NDArray[np.float64]:
    """``Re Tr(W dG/dθ_i)`` for ``G = expm(S(θ))`` over ``VUMPO.skew_hermitian``'s
    parameter layout: ``d`` imaginary diagonal entries, then ``(re, im)`` per
    upper-triangular pair."""
    lam, V = np.linalg.eigh(-1j * S)  # S = V diag(i lam) V^dag
    e = np.exp(1j * lam)
    diff = 1j * (lam[:, None] - lam[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(np.abs(diff) > 1e-12, (e[:, None] - e[None, :]) / diff, e[:, None])
    Z = V.conj() @ ((V.conj().T @ W @ V).T * phi) @ V.T  # d cost = Re sum_ab S_i[a,b] Z[a,b]
    g = [-Z[j, j].imag for j in range(d)]
    for j in range(d):
        for k in range(j + 1, d):
            g.append((Z[j, k] - Z[k, j]).real)
            g.append(-(Z[j, k] + Z[k, j]).imag)
    return np.array(g)


class VUMPO:
    def __init__(
        self,
        H_mpo,
        initial_state: Optional[Sequence[int]] = None,
        n_layers: int = 1,
        n_sweeps: int = 10,
        optimizer: str = "L-BFGS-B",
        maxiter_local: int = 5,
        maxiter_global: int = 2,
        alpha: float = 0.0,
        beta: float = 0.0,
        mode: str = "diag",
        opt: str = "local",
        hwp: bool = False,
        verbose: bool = False,
        tol: float = 1e-6,
    ) -> None:
        r"""
        Implements the Variational Unitary Matrix Product Operator (VUMPO) algorithm.

        VUMPO constructs a layered, brickwork circuit of nearest-neighbour two-qubit
        unitaries parameterized via skew-Hermitian generators and expm, forming a
        unitary Matrix Product Operator (uMPO) ansatz. It supports two primary
        optimization modes: variational diagonalization of a Hamiltonian
        (minimizing off-diagonal weight of :math:`U^\dagger H U`) and variational
        ground-state search (minimizing :math:`\langle \psi | U^\dagger H U | \psi \rangle`).
        Local (DMRG-style) sweeps or global optimization can be used.

        References:
            - Pollmann, F., Khemani, V., Cirac, J. I., and Sondhi, S. L.,
              "Efficient variational diagonalization of fully many-body localized
              Hamiltonians", Phys. Rev. B 94, 041116(R).

        Args:
            H_mpo: MatrixProductOperator (quimb) representing the target Hamiltonian H.
                MPO site ``n`` is qubit ``n`` (``VUMPOBrickworkBlock`` places the
                site-``n`` gates on qubits ``n, n+1``).  quimb's ``from_dense`` is
                kron-ordered (site 0 = leftmost factor), so build it from the
                bit-reversed qarpx matrix:
                ``from_dense(lsb_to_msb_matrix(op.sparse_matrix().toarray()), dims=(2,)*L)``.
            initial_state: Optional initial computational-basis state as a list of ints;
                used in ground-state mode. If None, defaults to all zeros (or
                ``[1] + [0] * (L - 1)`` when ``hwp=True``, a HW=1 state.).
            n_layers: Number of VUMPO layers (circuit depth).
            n_sweeps: Cap on the number of local-optimisation sweeps; ``tol`` usually
                stops the sweep earlier.
            optimizer: SciPy optimizer name for local/global updates (e.g., "L-BFGS-B").
            maxiter_local: Maximum iterations per local gate optimization.
            maxiter_global: Maximum iterations for the global optimization.
            alpha: Weight for adding the energy penalty in diagonalization mode
                (``mode="diag"``). Ignored otherwise.
            beta: Weight for orthogonality penalties against previously found states.
            mode: Optimization objective. Use "diag" for variational diagonalization or
                "gs" for ground-state energy minimization.
            opt: Optimization strategy. Use "local" (DMRG-style sweeps) or "global".
            hwp: If True, uses a HW preserving parametrization (6 parameters per gate);
                otherwise use the full 4x4 parametrization (16 parameters per gate).
            tol: Relative change of the cost between sweeps below which the local
                sweep stops (paper step iv, "until convergence"); ``0`` runs all
                ``n_sweeps``.  In ``mode="diag"`` the cost differs from the paper's
                summed variance by the constant ``Tr H^2``, so the change tested is
                the same quantity.

        Attributes:
            H_mpo: The input Hamiltonian as an MPO.
            L: Number of physical sites (qubits), inferred from ``H_mpo``.
            initial_state: The working initial state used during optimization.
            n_layers: Number of VUMPO layers.
            n_gates: Total number of two-qubit gates across all layers (brickwork pattern).
            n_sweeps: Number of local optimization sweeps.
            maxiter_local: Maximum iterations per local gate update.
            maxiter_global: Maximum iterations for the global optimization.
            optimizer_options_local: Options dict passed to SciPy for local updates.
            optimizer_options_global: Options dict passed to SciPy for global updates.
            optimizer: Name of the SciPy optimizer.
            alpha: Energy penalty weight for diagonalization mode.
            beta: Orthogonality penalty weight (not currently active).
            mode: Active optimization mode ("diag" or "gs").
            opt: Active optimization strategy ("local" or "global").
            hwp: Whether Hamming-weight preserving parametrization is used.
            params_per_gate: Number of parameters per two-qubit gate (6 or 16).
            tol: Sweep convergence tolerance.
            sweeps_run: Sweeps executed by the last ``optimize_local_sweep`` call.

        Raises:
            ValueError: If ``mode`` is not one of {"diag", "gs"} when evaluating the cost.
            NotImplementedError: If ``opt`` is not one of {"local", "global"} in ``build()``.

        Notes:
            - Brickwork layout: for layer m, gates act on pairs (n, n+1) with n starting
                at ``m % 2`` (even/odd tiling). This is specifically to use in the VUMPOBrickworkBlock.
            - Gate construction: each two-qubit unitary is built from a skew-Hermitian
                generator via ``scipy.linalg.expm``. For ``hwp=True``, the generator is Hamming-weight
                preserving; otherwise, a full 4x4 parametrization of SU(4)
                is used for each gate.
            - Cost evaluation uses tensor-network contractions (quimb), scaling with the
                MPO bond dimension and circuit depth.
            - The algorithm is particularly effective in many-body localized (MBL) regimes,
                where shallow uMPOs approximately diagonalize H.

        Examples:
            Optimize a VUMPO with local sweeps in diagonalization mode:

                >>> vumpo = VUMPO(H_mpo, n_layers=3, mode="diag", opt="local", n_sweeps=5)
                >>> params = vumpo.build(valid_diag=True, plot=False)

            Compute the energy of a prepared state in ground-state mode:

                >>> vumpo = VUMPO(H_mpo, n_layers=2, mode="gs", opt="local")
                >>> params = vumpo.build(valid_diag=False, plot=False)
                >>> E = vumpo.cost_fn(params, initial_state=[0] * vumpo.L)
        """

        self.H_mpo = H_mpo
        self.L = len(H_mpo.tensors)

        self.initial_state = list(initial_state) if initial_state is not None else None
        self.n_layers = n_layers
        self.n_sweeps = n_sweeps
        self.optimizer = optimizer
        self.maxiter_local = maxiter_local
        self.maxiter_global = maxiter_global
        self.optimizer_options_local = {"maxiter": maxiter_local}
        self.optimizer_options_global = {"maxiter": maxiter_global}
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.opt = opt
        self._hwp = hwp
        self.verbose = verbose
        self.tol = tol
        self.sweeps_run = 0
        self._trace_h2: Optional[float] = None

        # Fixed at construction: params_per_gate, the network specs and every
        # stored parameter vector depend on it, so it is exposed read-only.
        self.params_per_gate = 6 if hwp else 16
        self._networks: Dict[str, Tuple[NetworkSpec, Engine]] = {}
        self.n_gates = len(brickwork_gate_order(self.L, self.n_layers))

    @property
    def hwp(self) -> bool:
        """Whether gates use the Hamming-weight-preserving parametrisation (6
        parameters) rather than the full one (16).  Read-only."""
        return self._hwp

    def _default_initial_state(self, initial_state: Optional[Sequence[int]]) -> List[int]:
        """The ket a call uses: the argument, else the instance's, else the
        parametrisation's default (``|10..0>`` for hwp, ``|0..0>`` otherwise).
        Never written back: evaluations are read-only."""
        if initial_state is not None:
            return list(initial_state)
        if self.initial_state is not None:
            return list(self.initial_state)
        return [1] + [0] * (self.L - 1) if self.hwp else [0] * self.L

    def skew_hermitian(self, params: ParamLike, d: int) -> NDArray[np.complex128]:
        """Constructs a `d × d` skew-Hermitian matrix from a real parameter vector,
        as the generator of a matrix exponential.

        Args:
            params: A flat sequence of real numbers encoding the diagonal and
                off-diagonal components: ``d`` diagonal imaginary entries followed
                by 2 parameters (real and imaginary parts) per upper-triangular
                pair, for an expected length of ``d + 2 * (d * (d - 1) / 2)``.
            d: Dimension of the resulting square matrix.

        Returns:
            numpy.ndarray: A `(d, d)` complex skew-Hermitian matrix such that
            `A† = -A`.

        Examples:
            >>> params = [0.1, 0.2, 0.3, 0.4, 0.5]
            >>> A = vumpo.skew_hermitian(params, d=2)
            >>> np.allclose(A.conj().T, -A)
            True
        """
        A = np.zeros((d, d), dtype=complex)
        idx = 0

        if len(params) < d * d:
            raise ValueError("Not enough parameters for skew-Hermitian generator.")

        for j in range(d):
            A[j, j] = 1j * params[idx]
            idx += 1

        for j in range(d):
            for k in range(j + 1, d):
                A[j, k] = params[idx] + 1j * params[idx + 1]
                A[k, j] = -params[idx] + 1j * params[idx + 1]
                idx += 2

        return A

    def build_gate(self, params: ParamLike) -> NDArray[np.complex128]:
        """Construct a two-qubit unitary from a real parameter vector.

        In **Hamming-Weight Preserving (HWP)** mode (``self.hwp=True``), the gate is
        block-diagonal in the computational basis with two independent phases on
        ``|00⟩`` and ``|11⟩`` and a general unitary acting on the Hamming-weight 1
        subspace ``span{|01⟩, |10⟩}``. Specifically:

          - ``gate[0, 0] = exp(i * params[0])`` (phase on ``|00⟩``),
          - ``gate[3, 3] = exp(i * params[1])`` (phase on ``|11⟩``),
          - the 2×2 block on indices (``|01⟩``, ``|10⟩``) is
            ``expm(skew_hermitian(params[2:6], d=2))``.

        In **full** mode (``self.hwp=False``), a general two-qubit unitary is generated
        as ``expm(skew_hermitian(params[:16], d=4))``.

        Args:
            params: Real parameter vector. In HWP mode (``self.hwp=True``), expects
                **6** parameters ``[φ00, φ11, a0, a1, a2, a3]`` where the last 4
                define the 2×2 skew-Hermitian generator for the single-excitation
                block; in full mode (``self.hwp=False``), expects **16** parameters
                forming the 4×4 skew-Hermitian generator.

        Returns:
            numpy.ndarray: A complex array of shape ``(4, 4)`` representing the
            two-qubit unitary.

        Raises:
            ValueError: If ``params`` has an incompatible length for the selected mode.

        Examples:
            Hamming-weight-preserving (HWP) gate::

                >>> # phases on |00>, |11> and a general 2x2 unitary on {|01>, |10>}
                >>> params = [0.1, 0.2,  0.3, 0.4, 0.5, 0.6]
                >>> vumpo = VUMPO(H_mpo, hwp=True)
                >>> U = vumpo.build_gate(params)
                >>> U.shape
                (4, 4)

            Full 4x4 unitary via skew-Hermitian generator:

                >>> params = np.linspace(0.0, 1.5, 16)
                >>> vumpo = VUMPO(H_mpo, hwp=False)
                >>> U = vumpo.build_gate(params)
                >>> np.allclose(U.conj().T @ U, np.eye(4))
                True
        """
        if self.hwp:
            if len(params) < 6:
                raise ValueError("HWP mode requires 6 parameters.")
            gate = np.zeros((4, 4), dtype=complex)
            gate[0, 0] = np.exp(1j * params[0])
            gate[3, 3] = np.exp(1j * params[1])
            block = expm(self.skew_hermitian(params[2:6], 2))
            gate[1:3, 1:3] = block
        else:
            if len(params) < 16:
                raise ValueError("Full SU(4) mode requires 16 parameters.")
            gate = expm(self.skew_hermitian(params[:16], 4))

        return gate

    def layerise_params(self, params: ParamLike) -> List[List[Tuple[int, GateParams]]]:
        """Group a flat parameter vector into VUMPO layers following the brickwork pattern.

        This method partitions the full parameter array into a list of layers, where
        each layer contains the parameters for the two‑qubit gates applied at that
        depth. For layer ``m``, gates act on qubit pairs ``(n, n + 1)``, starting from
        ``n = m % 2`` (even/odd staggering). Each gate consumes
        ``self.params_per_gate`` consecutive parameters.

        Args:
            params: A flat sequence of real parameters of length
                ``self.n_gates * self.params_per_gate``. The sequence is consumed in
                order as gates are assigned to layers.

        Returns:
            List[List[Tuple[int, Sequence[float]]]]: A nested list where:
                - the outer list has length ``self.n_layers``,
                - each inner list corresponds to one layer,
                - each element of the inner list is a tuple ``(n, p)`` where:
                    * ``n`` is the left qubit index of the two‑qubit gate,
                    * ``p`` is the parameter slice for that gate.

                For example, the structure looks like::

                    [
                        [(0, params_for_gate_0), (2, params_for_gate_1), ...],   # layer 0
                        [(1, params_for_gate_k), (3, params_for_gate_k+1), ...], # layer 1
                        ...
                    ]

        Raises:
            ValueError: If ``params`` does not hold exactly
                ``n_gates * params_per_gate`` entries; a longer vector is never
                truncated, since that silently builds the wrong circuit.

        """

        flat = np.asarray(params, dtype=float).ravel()

        layers: List[List[Tuple[int, GateParams]]] = []
        idx = 0

        total = self.n_gates * self.params_per_gate
        if flat.size != total:
            raise ValueError(
                f"expected {total} parameters ({self.n_gates} gates x {self.params_per_gate}), got {flat.size}"
            )

        # brickwork_gate_order is the one definition of the tiling; the flat
        # vector is consumed in that order.
        layers = [[] for _ in range(self.n_layers)]
        for m, n in brickwork_gate_order(self.L, self.n_layers):
            layers[m].append((n, cast(GateParams, flat[idx : idx + self.params_per_gate])))
            idx += self.params_per_gate

        return layers

    def _to_flat_params(self, params: Union[ParamLike, GateParamList]) -> ParamVector:
        if isinstance(params, list):
            return np.concatenate([np.asarray(p, dtype=float).ravel() for p in params])
        return np.asarray(params, dtype=float).ravel()

    def _add_circuit(
        self,
        tensors: List[qtn.Tensor],
        params: Union[ParamLike, GateParamList],
        prefix: str,
        wire_in: Dict[int, str],
        conj: bool = False,
    ) -> Dict[int, str]:
        """Insert a VUMPO circuit into a tensor list and update wire labels.

        This method expands the parameter vector into VUMPO layers (via
        ``layerise_params``), constructs each two‑qubit unitary using ``build_gate``,
        reshapes it into a rank‑4 tensor with indices
        ``(out1, out2, in1, in2)``, and appends it to the given ``tensors`` list.

        The tensor network is wired according to the brickwork pattern:

            - For layer ``m``, gates act on qubits ``(n, n+1)`` with
              ``n`` starting at ``m % 2``.
            - If ``conj=True``, layers are traversed in **reverse order** and each gate
              is replaced by its Hermitian conjugate ``U†``.
            - Wire labels are updated after each gate insertion to reflect the
              "flow" of indices through the circuit.

        Args:
            tensors: A list of ``quimb`` Tensor objects. New gate tensors will be
                appended to this list in-place.
            params: A flat sequence of real parameters defining all VUMPO gates.
                Must contain ``self.n_gates * self.params_per_gate`` entries.
            prefix: String prefix used to name output indices, e.g.,
                ``"U"`` for forward application or ``"D"`` for adjoint.
            wire_in: A dictionary mapping qubit index → input wire label for that
                position. These labels are used as the lower (input) indices of the
                first layer of gates.
            conj: If True, apply the circuit in reverse order with conjugated gates
                (i.e., build ``U†`` instead of ``U``).

        Returns:
            Dict[int, str]: A dictionary mapping qubit index → final wire label after
            all layers have been applied. These output wires may be connected to
            Hamiltonian MPO legs, bra/ket vectors, or later contractions.

        Raises:
            ValueError: If ``params`` is too short to populate all VUMPO layers.

        Examples:
            >>> tensors = []
            >>> wire_in = {q: f"in{q}" for q in range(vumpo.L)}
            >>> out_wires = vumpo._add_circuit(tensors, params, prefix="U", wire_in=wire_in)
            >>> len(tensors)  # number of gate tensors inserted
            vumpo.n_gates
            >>> out_wires[0]
            'U_0_0'   # example output index name
        """
        flat_params = self._to_flat_params(params)
        layers = self.layerise_params(flat_params)
        wire = dict(wire_in)

        layer_range = reversed(range(self.n_layers)) if conj else range(self.n_layers)

        for m in layer_range:
            for n, p in layers[m]:
                gate = self.build_gate(p)
                if conj:
                    gate = gate.conj().T

                arr = gate.reshape(2, 2, 2, 2)
                in1, in2 = wire[n], wire[n + 1]
                out1 = f"{prefix}_{m}_{n}"
                out2 = f"{prefix}_{m}_{n + 1}"

                tensors.append(qtn.Tensor(arr, inds=[out1, out2, in1, in2]))

                wire[n] = out1
                wire[n + 1] = out2

        return wire

    def _gates(self, params: Union[ParamLike, GateParamList]) -> List[NDArray[np.complex128]]:
        """Gate unitaries in ``brickwork_gate_order``, the order ``layerise_params`` uses."""
        flat = self._to_flat_params(params)
        return [self.build_gate(p) for layer in self.layerise_params(flat) for _, p in layer]

    def _network(self, kind: str) -> Tuple[NetworkSpec, Engine]:
        """Spec and engine for one scalar network, built once per instance and shape."""
        if kind not in self._networks:
            if kind == "overlap":
                spec = NetworkSpec.overlap(self.L, self.n_layers)
            else:
                spec = getattr(NetworkSpec, kind)(self.H_mpo, self.n_layers)
            self._networks[kind] = (spec, make_engine(spec))
        return self._networks[kind]

    def _cost_tn(self, params: ParamLike) -> float:
        """Build and contracts tensor network for the full-diagonalization cost.

        Constructs a TN whose scalar contraction gives
        ``sum_i (diag(U† H U))_i^2``, i.e. the sum of squared diagonal elements of the rotated Hamiltonian.
        Two independent copies of the sandwich ``U† H U`` (labelled *a* and *b*) are connected by the
        rank-4 delta tensors that enforce a trace over matching computational-basis indices, yielding the
        squared-diagonal sum in a single contraction.

        Args:
            params: Flat real parameter vector of length ``self.n_gates * self.params_per_gate``.
                Encodes all two-qubit gates in the brickwork circuit.

        Returns:
            A real valued float corresponding to the cost function ``sum_i (diag(U† H U))_i^2``

        Note:
            The cost function used in ``cost_fn`` for ``mode='diag'`` negates the output
            which minimizing diagonalizes H_mpo.
        """
        gates = self._gates(params)
        spec, engine = self._network("doubled")
        return float(np.real(engine.value(spec.arrays(gates))))

    def compute_energy_tn(
        self, params: ParamLike, initial_state: Optional[Sequence[int]] = None
    ) -> float:
        """Compute energy of MPO state using tensor network contractions.

        Constructs and contracts a TN whose scalar contraction gives the energy
        ``<s|U† H U|s>`` where ``|s>`` is a computational basis state (e.g. ``|1100>``).
        The network sandwiches the Hamiltonian MPO between the forward circuit U and
        its conjugate U†, with basis-state vectors closing the ends.

        Args:
            params: Flat real parameter vector of length ``self.n_gates * self.params_per_gate``.
                Encodes all two-qubit gates in the brickwork circuit.
            initial_state: Computational basis state as a list of 0s and 1s,
                one per qubit (e.g. ``[1, 1, 0, 0]``). For use in a
                ComputationalBasisStateBlock. If None, defaults to ``|0>^{n}``
                for non-Hamming weight preserving and ``|1>|0>^{(n-1)}`` for HW
                preserving circuits.

        Returns:
            A real valued float corresponding to the cost function ``<s|U† H U|s>``

        """
        gates = self._gates(params)
        spec, engine = self._network("energy")
        ket = self._default_initial_state(initial_state)
        return float(np.real(engine.value(spec.arrays(gates, ket=ket))))

    def _overlap_tn(
        self,
        params_a: ParamLike,
        params_b: ParamLike,
        init_a: Sequence[int],
        init_b: Sequence[int],
    ) -> complex:
        """Build the tensor network for a circuit-to-circuit overlap.

        Constructs and contracts TN which computes ``<s_a|U†_a U_b|s_b>`` where ``|s_a>, |s_b>`` are computational basis states (e.g. ``|1100>``).
        This measures the overlap between two variational states prepared by different circuits from (potentially different)
        computational basis states. Used in ``mode='gs'`` deflation to penalise overlap with previously found eigenstates.

        Args:
            params_a: Flat parameter vector for circuit *U_a*
            params_b: Flat parameter vector for circuit *U_b*
            init_a: Computational basis bra state ``<s_a|``
            init_b: Computational basis bra state ``|s_b>``


        Returns:
            The complex amplitude ``<s_a|U†_a U_b|s_b>``; ``cost_fn`` squares its modulus
        """
        spec, engine = self._network("overlap")
        arrays = spec.arrays(
            self._gates(params_a),
            gates_b=self._gates(params_b),
            ket=list(init_a),
            ket_b=list(init_b),
        )
        return complex(engine.value(arrays))

    def build_tn(
        self,
        params: Union[ParamLike, GateParamList],
    ) -> TensorNetwork:
        """Build brickwork unitary as a quimb TensorNetwork.

        Returned TN has open indices:

        - ``k0, k1, ...`` (input/comp basis side)
        - ``b0, b1, ...`` (output/physical side)

        Contracting gives a 2^L x 2^L unitary matrix (operator).

        Args:
            params: Flat real parameter vector of length ``self.n_gates * self.params_per_gate``.
                Encodes all two-qubit gates in the brickwork circuit.

        Returns:
            ``TensorNetwork`` representing the circuit unitary with 2L open indices.
        """
        tensors: list[qtn.Tensor] = []
        wire_out = self._add_circuit(tensors, params, "U", {q: f"k{q}" for q in range(self.L)})
        tn = qtn.TensorNetwork(tensors)
        for q in range(self.L):
            tn.reindex_({wire_out[q]: f"b{q}"})
        return tn

    def cost_fn(
        self,
        params: ParamLike,
        initial_state: Optional[Sequence[int]] = None,
        prev_states: Optional[List[Tuple[Union[ParamLike, NDArray[Any]], Sequence[int]]]] = None,  # type: ignore[arg-type]
    ) -> float:
        """The VUMPO cost function.

        The behaviour depends on ``self.mode``:

        * 'diag': ``-sum(diag(U† H U))^2 + self.alpha * <s|U† H U|s>``
            The first term diagonalises H, the optional alpha term biases the diagonalisation to
            favour the ground state.
        * 'gs': ``<s|U† H U|s> + self.beta * sum_k |<s|U†_a U_k|s_k>|^2``.
            Finds ground state energy, with optional deflation penalty to enforce orthogonality
            to previously converged eigenstates.

        Args:
            params: Flat real parameter vector of length ``self.n_gates * self.params_per_gate``.
                Encodes all two-qubit gates in the brickwork circuit.
            initial_state: Computational basis state as a list of 0s and 1s,
                    one per qubit (e.g. ``[1, 1, 0, 0]``). For use in a
                    ComputationalBasisStateBlock. If None, defaults to ``|0>^{otimes n}``
                    for non-Hamming weight preserving and ``|1>|0>^{otimes (n-1)}`` for HW
                    preserving circuits.
            prev_states: List of (params_k, init_k) tuples for previously
                converged states. The deflation penalty penalises the overlap with
                mode = 'gs' only.
        Returns:
            Scalar cost value.

        Raises:
            ValueError: If ``self.mode`` is not 'diag' or 'gs'.

        """
        if self.mode == "diag":
            cost = -self._cost_tn(params)
            if self.alpha != 0:
                # compute_energy_tn resolves a missing state to the instance default
                cost += self.alpha * self.compute_energy_tn(params, initial_state)
            return cost

        elif self.mode == "gs":
            cost = self.compute_energy_tn(params, initial_state)
            if prev_states and self.beta != 0:
                s_a = self._default_initial_state(initial_state)
                for p_k, s_k in prev_states:
                    ov = self._overlap_tn(params, p_k, s_a, s_k)
                    cost += self.beta * abs(ov) ** 2
            return cost
        else:
            raise ValueError("Unknown mode for cost_fn.")

    # ── objective terms, environments and gradients ──────────────────────

    def _terms(
        self,
        gates: Sequence[NDArray[np.complex128]],
        initial_state: Optional[Sequence[int]],
        prev_states: Optional[Sequence[Tuple[Any, Sequence[int]]]],
    ) -> List[_Term]:
        """The scalar networks whose weighted sum is ``cost_fn``, with arrays."""
        terms: List[_Term] = []
        if self.mode == "diag":
            spec, engine = self._network("doubled")
            terms.append(_Term(-1.0, False, spec, engine, spec.arrays(gates)))
            if self.alpha != 0:
                state = self._default_initial_state(initial_state)
                spec, engine = self._network("energy")
                terms.append(_Term(self.alpha, False, spec, engine, spec.arrays(gates, ket=state)))
        elif self.mode == "gs":
            state = self._default_initial_state(initial_state)
            spec, engine = self._network("energy")
            terms.append(_Term(1.0, False, spec, engine, spec.arrays(gates, ket=state)))
            if prev_states and self.beta != 0:
                spec, engine = self._network("overlap")
                for p_k, s_k in prev_states:
                    arrays = spec.arrays(
                        gates, gates_b=self._gates(p_k), ket=state, ket_b=list(s_k)
                    )
                    terms.append(_Term(self.beta, True, spec, engine, arrays))
        else:
            raise ValueError("Unknown mode for cost_fn.")
        return terms

    @staticmethod
    def _fill_boundaries(terms: Sequence[_Term], *, left: bool, right: bool) -> None:
        for t in terms:
            if left:
                t.left = t.engine.left_boundaries(t.arrays)
            if right:
                t.right = t.engine.right_boundaries(t.arrays)

    def _visit_order(self, sweep: int) -> List[Tuple[int, List[int]]]:
        """Column-major: ``(left_site, gates at that site across layers)``,
        left to right on even sweeps and mirrored on odd ones."""
        order = brickwork_gate_order(self.L, self.n_layers)
        cols = [
            (n, [k for k, (_, site) in enumerate(order) if site == n]) for n in range(self.L - 1)
        ]
        cols = [(n, ks) for n, ks in cols if ks]
        if sweep % 2 == 1:
            cols = [(n, ks[::-1]) for n, ks in reversed(cols)]
        return cols

    def _local_objective(
        self, k: int, terms: Sequence[_Term]
    ) -> Callable[[NDArray[np.float64]], Tuple[float, NDArray[np.float64]]]:
        """``cost_fn`` and its exact gradient as a function of gate ``k`` alone,
        every other gate frozen into the environments computed here once."""
        n = brickwork_gate_order(self.L, self.n_layers)[k][1]
        envs = [
            (t, t.engine.environment(k, t.arrays, t.left[n], t.right[n + 2]), t.spec.slot_conj(k))
            for t in terms
        ]

        def objective(p: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
            G = self.build_gate(p)
            value = 0.0
            W = np.zeros((4, 4), dtype=complex)
            for t, env, conj in envs:
                val, w = polynomial_value_and_w(env, G, conj)
                if t.squared:
                    # d|v|^2 = 2 Re(conj(v) dv); a pure-conjugate polynomial has dv = conj(Tr(W dG))
                    value += t.weight * abs(val) ** 2
                    W += t.weight * 2.0 * (val if all(conj) else val.conjugate()) * w
                else:
                    value += t.weight * val.real
                    W += t.weight * w
            return value, self._gate_gradient(W, p)

        return objective

    def _value_and_gradient(
        self,
        params: ParamLike,
        initial_state: Optional[Sequence[int]],
        prev_states: Optional[Sequence[Tuple[Any, Sequence[int]]]],
    ) -> Tuple[float, NDArray[np.float64]]:
        """``cost_fn`` and its gradient in every parameter: one environment per
        gate on boundaries built once, so linear in ``L``."""
        flat = self._to_flat_params(params)
        terms = self._terms(self._gates(flat), initial_state, prev_states)
        self._fill_boundaries(terms, left=True, right=True)
        ppg = self.params_per_gate
        value = 0.0
        grads = []
        for k in range(self.n_gates):
            value, g = self._local_objective(k, terms)(flat[k * ppg : (k + 1) * ppg])
            grads.append(g)
        return value, np.concatenate(grads)

    def _gate_gradient(self, W: NDArray[np.complex128], params: ParamLike) -> NDArray[np.float64]:
        """``Re Tr(W dG/dθ_i)`` for every parameter of ``build_gate``.

        Daleckii–Krein form of the Fréchet derivative of ``expm``: one ``eigh``
        of the generator, no ``expm_frechet`` per direction."""
        p = np.asarray(params, dtype=float)
        if self.hwp:
            g = np.empty(6)
            g[0] = (1j * np.exp(1j * p[0]) * W[0, 0]).real
            g[1] = (1j * np.exp(1j * p[1]) * W[3, 3]).real
            g[2:6] = _skew_gradient(W[1:3, 1:3], self.skew_hermitian(p[2:6], 2), 2)
            return g
        return _skew_gradient(W, self.skew_hermitian(p[:16], 4), 4)

    def optimize_local_sweep(
        self,
        initial_params: GateParamList,
        initial_state: Optional[Sequence[int]] = None,
        prev_states: Optional[List[Tuple[ParamLike, Sequence[int]]]] = None,
    ) -> NDArray[np.float64]:
        """Run alternating gate-by-gate sweeps, column-major and DMRG-like.

        Each gate is minimised while every other gate is frozen into its
        environment (the network with that gate's insertions left open), so
        one visit costs O(1) in ``L`` and one sweep O(L).  Even sweeps visit
        columns left to right, all layers at a column before the next; odd
        sweeps mirror that order.  This is not the layer-major order earlier
        releases used, so results from an ``opt="local"`` run are not
        bit-reproducible across that change.  The local minimiser receives
        the exact gradient.

        Args:
            initial_params: Mutable list of per-gate parameter arrays
            initial_state: Computational basis state as a list of 0s and 1s,
                one per qubit (e.g. ``[1, 1, 0, 0]``). For use in a
                ComputationalBasisStateBlock. If None, defaults to ``|0>^{otimes n}``
                for non-Hamming weight preserving and ``|1>|0>^{otimes (n-1)}`` for HW
                preserving circuits.
            prev_states: List of (params_k, init_k) tuples for previously
                converged states. The deflation penalty penalises the overlap with
                mode = 'gs' only.
        """

        gate_params = [np.asarray(g, dtype=float) for g in initial_params]
        gates = [self.build_gate(g) for g in gate_params]

        prev_cost: Optional[float] = None
        self.sweeps_run = 0
        for sweep in range(self.n_sweeps):
            forward = sweep % 2 == 0
            terms = self._terms(gates, initial_state, prev_states)
            # the far side is precomputed once; the near side advances with the sweep
            self._fill_boundaries(terms, left=not forward, right=forward)
            for t in terms:
                if forward:
                    t.left = [None] * (self.L + 1)
                else:
                    t.right = [None] * (self.L + 1)
            done = 0 if forward else self.L  # columns absorbed into the running boundary

            for n, ks in self._visit_order(sweep):
                if forward:
                    while done < n:
                        for t in terms:
                            t.left[done + 1] = t.engine.advance(t.left[done], t.arrays, done)
                        done += 1
                else:
                    while done > n + 2:
                        done -= 1
                        for t in terms:
                            t.right[done] = t.engine.advance(t.right[done + 1], t.arrays, done)
                for k in ks:
                    res = minimize(
                        self._local_objective(k, terms),
                        gate_params[k],
                        jac=True,
                        method=self.optimizer,
                        options=self.optimizer_options_local,
                    )
                    gate_params[k] = np.asarray(res.x, dtype=float)
                    gates[k] = self.build_gate(gate_params[k])
                    for t in terms:
                        t.spec.update_gate(t.arrays, k, gates[k])

            self.sweeps_run = sweep + 1
            cost = self.cost_fn(np.concatenate(gate_params), initial_state, prev_states)
            if prev_cost is not None and abs(prev_cost - cost) <= self.tol * max(
                abs(prev_cost), 1.0
            ):
                break
            prev_cost = cost

        return np.concatenate(gate_params)

    def optimize_global(
        self,
        initial_params: GateParamList,
        initial_state: Optional[List[int]] = None,
        prev_states: Optional[List[Tuple[GateParams, List[int]]]] = None,
    ) -> NDArray[np.float64]:
        """Optimize all parameters at once (similar to VQE).

        Args:
            initial_params: Mutable list of per-gate parameter arrays
            initial_state: Computational basis state as a list of 0s and 1s,
                    one per qubit (e.g. ``[1, 1, 0, 0]``). For use in a
                    ComputationalBasisStateBlock. If None, defaults to ``|0>^{otimes n}``
                    for non-Hamming weight preserving and ``|1>|0>^{otimes (n-1)}`` for HW
                    preserving circuits.
            prev_states: List of (params_k, init_k) tuples for previously
                converged states. The deflation penalty penalises the overlap with
                mode = 'gs' only.
        """
        flat_params = np.concatenate([np.asarray(p) for p in initial_params])

        def global_cost(p: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
            return self._value_and_gradient(p, initial_state, prev_states)  # type: ignore[arg-type]

        results = minimize(
            global_cost,
            flat_params,
            jac=True,
            method=self.optimizer,
            options=self.optimizer_options_global,
        )

        return np.asarray(results.x, dtype=float)

    def _build_circuit(
        self,
        params: Union[ParamLike, GateParamList],
    ) -> qtn.Circuit:
        """
        Build a quimb circuit from the tensors for validation. This is
        primarily for testing purposes.
        """
        circuit = qtn.Circuit(self.L)
        for m, layer in enumerate(self.layerise_params(self._to_flat_params(params))):
            for n, p in layer:
                circuit.apply_gate_raw(self.build_gate(p), (n, n + 1), gate_round=m)
        return circuit

    def _check_diag(
        self,
        params: Union[ParamLike, GateParamList],
        max_dense_qubits: int = 12,
    ) -> Tuple[NDArray, NDArray, int]:
        """Check quality exact diagonalization by comparing to exact diagonalization.

        Contracts the circuit to a dense unitary, U, forms U†HU and compares it's
        diagonal to exact eigenvalues. Computes some diagnostic information
        like off diagonal norm ratio, max eigenvalue error, and the index of the
        true ground state basis vector (for QMC applications).

        Args:
            params: Flat real parameter vector of length ``self.n_gates * self.params_per_gate``.
                Encodes all two-qubit gates in the brickwork circuit.
            max_dense_qubits: Refuse to densify above this many qubits.

        Raises:
            ValueError: ``L > max_dense_qubits``; ``energy_variance`` and
                ``off_diagonal_ratio`` give the same quality measures without
                the dense matrix.
        """
        if self.L > max_dense_qubits:
            raise ValueError(
                f"_check_diag densifies a 2^{self.L} x 2^{self.L} matrix. Use off_diagonal_ratio() "
                "or energy_variance(), which contract the MPO directly, or raise "
                "max_dense_qubits deliberately."
            )
        dim = 2**self.L

        flat_params = self._to_flat_params(params)

        H = self.H_mpo.copy()

        tn = self.build_tn(flat_params)
        out_inds = [f"b{q}" for q in range(self.L)]
        in_inds = [f"k{q}" for q in range(self.L)]
        U = tn.contract(output_inds=out_inds + in_inds).data.reshape(dim, dim)

        upper = [H.upper_ind(q) for q in range(self.L)]
        lower = [H.lower_ind(q) for q in range(self.L)]
        H_dense = H.contract(output_inds=upper + lower).data.reshape(dim, dim)

        UHU = U.conj().T @ H_dense @ U

        diag_vals_og = np.real(np.diag(H_dense))
        off_diag_vals_og = H_dense - np.diag(diag_vals_og)
        ratio_original = np.linalg.norm(off_diag_vals_og) / np.linalg.norm(H_dense)

        diag_vals = np.real(np.diag(UHU))
        off_diag = UHU - np.diag(diag_vals)

        ratio = np.linalg.norm(off_diag) / np.linalg.norm(UHU)

        exact_eigs = np.sort(np.linalg.eigvalsh(H_dense))

        approx_eigs = np.sort(diag_vals)
        max_err = np.max(np.abs(exact_eigs - approx_eigs))
        gs_idx = int(np.argmin(diag_vals))
        print(f"Off diagonal ratio, original = {ratio_original: .6f}")

        print(f"Off diagonal ratio, vumpo = {ratio:.6f}")
        print(f"Diagonalization error = {max_err: .6f}")
        return exact_eigs, approx_eigs, gs_idx

    # ── the paper's diagnostics (Appendix A) ─────────────────────────────

    def trace_h2(self) -> float:
        """``Tr H^2`` from the MPO, never dense; the constant first term of the
        paper's cost (Appendix A).  Cached on the instance."""
        if self._trace_h2 is None:
            H = self.H_mpo
            env = np.ones((1, 1), dtype=complex)
            for q in range(self.L):
                t = H[q]
                names = list(t.inds)
                left = H.bond(q - 1, q) if q > 0 else None
                right = H.bond(q, q + 1) if q + 1 < self.L else None
                order = [
                    names.index(i)
                    for i in (left, right, H.upper_ind(q), H.lower_ind(q))
                    if i is not None
                ]
                W = np.transpose(t.data, order)
                if left is None:
                    W = W[None]
                if right is None:
                    W = W[:, None]
                # Tr over the physical legs of W_q W_q: (a,c,s,t)(a',c',t,s)
                T = np.einsum("acst,bdts->abcd", W, W)
                env = np.einsum("ab,abcd->cd", env, T)
            self._trace_h2 = float(np.real(env[0, 0]))
        return self._trace_h2

    def energy_variance(self, params: ParamLike) -> float:
        """The paper's Eq. (4) cost ``sum_tau <H^2>_tau - <H>_tau^2 >= 0`` over
        all ``2^L`` approximate eigenstates: ``trace_h2() - sum_i d_i^2``."""
        return self.trace_h2() - self._cost_tn(params)

    def off_diagonal_ratio(self, params: ParamLike) -> float:
        """``||U^dag H U - diag||_F / ||H||_F = sqrt(1 - sum_i d_i^2 / Tr H^2)``,
        the paper's cost normalised, by unitary invariance of the norm."""
        return float(np.sqrt(max(1.0 - self._cost_tn(params) / self.trace_h2(), 0.0)))

    def build(self, valid_diag: bool = False, plot: bool = False) -> NDArray[np.float64]:
        """Build the VUMPO circuit parameters with ground state targeting or
        exact diagonalization.

        Args:
            valid_diag: bool indicating whether to check exact diagonalization.
                False by default to avoid expense.
        """
        # Identity start: every gate is exp(0) = 1, so the starting eigenstates
        # are the product states the layers then dress.  A random or all-ones
        # start lands in local minima an order of magnitude above this one on
        # the paper's W = 8 chain.
        gate_params = [np.zeros(self.params_per_gate, dtype=float) for _ in range(self.n_gates)]

        if self.initial_state is None:
            self.initial_state = self._default_initial_state(None)

        if self.opt == "local":
            params = self.optimize_local_sweep(
                gate_params,
                initial_state=self.initial_state,
            )
        elif self.opt == "global":
            params = self.optimize_global(
                gate_params,
                initial_state=self.initial_state,
            )
        else:
            raise NotImplementedError("opt must be 'local' or 'global'.")

        cost = self.cost_fn(params, initial_state=self.initial_state)
        print(f"Final cost = {cost:.3f}")

        if valid_diag:
            exact, approx, idx = self._check_diag(params=params)
            print("Diagonalization check complete.")
            self.exact_eigs = exact
            self.approx_eigs = approx
            self.gs_idx = idx

        return params
