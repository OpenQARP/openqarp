from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Self, Union, cast

import numpy as np
from sympy import Symbol

from qarp.operators import QubitOperator

from ...blocks import AnyBlock
from ...engines import Engine
from ...engines._gradients import gradient_method_from_flag
from ...optimizers import Optimizer, ScipyOptimizer
from .. import PrimitiveAlgorithm, StateVector
from . import CompositeAlgorithm
from ._params import resolve_initial_parameters


def _returns_probability(prim) -> bool:
    """Overlap primitives whose ``run()`` is already ``|⟨bra|ket⟩|²``, as declared
    by the ``returns_probability`` class flag."""
    return bool(getattr(prim, "returns_probability", False))


def _gradient_is_squared_overlap(prim) -> bool:
    """``True`` iff ``run_gradient`` already hands back ``∂|⟨bra|ket⟩|²``.

    Two routes: a ``StateVector`` OVERLAP (``gradient_kind == "squared_overlap"``:
    ``run()`` is the amplitude, the engine differentiates ``|o|²``), or a
    probability-returning primitive (``run()`` is already ``|o|²``).  Distinct
    from ``_gradients._value_transform``, the engine-side question of whether the
    *forward value* must be squared before differencing — true for the first only.
    """
    return getattr(prim, "gradient_kind", "none") == "squared_overlap" or _returns_probability(prim)


def squared_overlap(prim, value) -> float:
    """``|⟨bra|ket⟩|²`` from one overlap primitive's ``run()`` value: the value
    itself when ``run()`` already returns the probability, ``|o|²`` of the
    amplitude ``o`` otherwise.  ``Re(o²)`` is not ``|o|²`` once ``o`` is complex."""
    if _returns_probability(prim):
        return float(np.real(value))
    return float(abs(complex(value)) ** 2)


def squared_overlap_gradient(prim, grad, value):
    """``∂|⟨bra|ket⟩|²/∂θ`` for one built overlap primitive.  A primitive whose
    gradient is already of the squared overlap needs no correction; anything
    else (a Hadamard test) returns ``∂o`` of the amplitude ``o`` that ``run()``
    gave, so the chain rule ``2·Re(o*·∂o)`` is applied here."""
    if _gradient_is_squared_overlap(prim):
        return np.real(grad)
    o = complex(value)
    return 2.0 * np.real(np.conj(o) * np.asarray(grad))


def deflation_gradient(primitives, weights, term_gradients, forward_values):
    """``Σᵢ wᵢ·∂|⟨ψᵢ|ψ⟩|²/∂θ`` — the deflation half of a VQD-style gradient.

    ``term_gradients`` is the engine's whole gradient list, energy first, so the
    overlap for ``primitives[i]`` is ``term_gradients[i + 1]``.  ``forward_values``
    is a thunk returning the same list of ``run()`` values: it is called only when
    an amplitude-returning primitive is present, the one case whose chain rule
    needs the overlap at θ, so the extra circuit evaluation is never paid for
    otherwise.
    """
    values: list = [None] * (len(primitives) + 1)
    if any(not _gradient_is_squared_overlap(p) for p in primitives):
        values = list(forward_values())
    total = np.zeros(np.shape(term_gradients[0]), dtype=float)
    for i, (prim, weight) in enumerate(zip(primitives, weights, strict=True), start=1):
        total += weight * squared_overlap_gradient(prim, term_gradients[i], values[i])
    return total


class VQD(CompositeAlgorithm):
    def __init__(
        self,
        operator: Union[QubitOperator, AnyBlock],
        kets: List[AnyBlock],
        weights: List[float],
        initial_parameters: Optional[Iterable[Iterable[float]]] = None,
        verbose: bool = False,
        gradient: Union[bool, str] = False,
        optimizer: Optional[Optimizer] = None,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """Variational Quantum Deflation (VQD) algorithm for computing excited states.

        VQD is a variational quantum algorithm that sequentially computes multiple eigenstates
        of a Hamiltonian by deflating the energy landscape. For each excited state, the algorithm
        minimizes an objective function combining the energy expectation value with penalty terms
        that enforce orthogonality to previously computed states. This deflation mechanism prevents
        collapse to lower-energy states while maintaining computational efficiency through shared
        parameterized ansatz structures across different eigenstates.

        Args:
            operator: The Hamiltonian operator or Block whose eigenstates are computed.
            kets: List of parameterized state blocks (ansatze) for each eigenstate.
            weights: Penalty coefficients for orthogonality constraints with lower-lying states.
            initial_parameters: Starting parameters for each state's optimization — one entry per
                ket, each a ``{symbol: value}`` mapping (order-proof, preferred) or a vector
                positional against that ket's ``symbols``. If None, initialized to zeros.
            verbose: Whether to print progress information during optimization.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            optimizer: Classical optimizer for parameter updates (defaults to Scipy's conjugate gradient).
            primitive: Algorithm for computing expectation values and overlaps between states.
            engine: Execution engine for running quantum circuits.
        """

        if optimizer is None:
            optimizer = ScipyOptimizer("CG")
        if primitive is None:
            primitive = StateVector()
        super().__init__(engine=engine, primitive=primitive)

        if len(kets) != len(weights) + 1:
            raise RuntimeError(
                "The number of weights should be one fewer than the number of "
                "kets: one deflation penalty per previously found state."
            )

        self.operator = operator
        self.state_parameters: list[dict] = []
        self.weights = weights
        self.kets = kets
        self.verbose = verbose
        self.gradient_method = gradient_method_from_flag(gradient)
        self.gradient = self.gradient_method is not None
        self.optimizer = optimizer
        self.iter = 0

        for ket in kets:
            if not ket.symbols:
                raise ValueError(
                    "Cannot build VQD: at least one ket has no parameters to "
                    "optimize (symbols is None or empty). Pass parameterized "
                    "ansatz blocks, not bare state-preparation blocks."
                )
        if initial_parameters is None:
            self.initial_parameters: list = [[0.0] * len(ket.symbols) for ket in kets]
        else:
            # Per-state entries may each be a {symbol: value} mapping
            # (order-proof) or a vector positional against that ket's symbols.
            entries = list(initial_parameters)
            if len(entries) != len(kets):
                raise ValueError(
                    f"initial_parameters has {len(entries)} entries for "
                    f"{len(kets)} kets; one entry per ket is required."
                )
            self.initial_parameters = [
                resolve_initial_parameters(ket.symbols, entry)
                for ket, entry in zip(kets, entries, strict=True)
            ]
        self.state_parameters = []
        self.energies: list[Any] = [None] * len(self.kets)
        self._measurements: list = []

    def build(self) -> Self:
        if self.verbose:
            if self.gradient:
                gradstr = f"analytic ({self.gradient_method}) via {self.engine}"
            else:
                gradstr = "No analytic gradients"

            print("VQD Build:")
            print(f"\tEnergy extraction: {self.primitive}")
            print(f"\tOverlap extraction: {self.primitive}")
            print(f"\tEngine: {self.engine}")
            print("\tGradient: " + gradstr + ".")
        return self

    @property
    def optimal_parameters(self) -> List[Dict[Symbol, float]]:
        """Per-state optimized parameters keyed by symbol (alias of
        ``state_parameters`` — already the order-proof surface)."""
        if len(self.state_parameters) < len(self.kets):
            raise RuntimeError("No complete optimization result — call run() first.")
        return self.state_parameters

    def get_final_state_block(self, index: int) -> AnyBlock:
        """Ket ``index`` bound at its optimal parameters."""
        bound = self.kets[index].set_symbols(self.optimal_parameters[index])
        bound.build()
        return bound

    def objective(self, theta: Iterable[float]):
        results = self.engine.run(
            params=dict(
                zip(
                    self.kets[self.iter].symbols,  # type: ignore[arg-type]
                    np.array(theta),
                    strict=True,
                )
            )
        )
        ev = cast(complex, results[0])  # an estimator result is a scalar
        # Every overlap primitive is a copy of self.primitive, so one flag lookup
        # covers all of them; the built primitives are not consulted because the
        # engine may be a stub carrying no _measurements.
        penalty = sum(
            w * squared_overlap(self.primitive, o)
            for w, o in zip(self.weights[: self.iter], results[1:], strict=True)
        )
        self.energies[self.iter] = ev  # type: ignore[assignment]
        return float(np.real(ev)) + float(penalty)

    def objective_gradient(self, theta: Iterable[float]):
        p = dict(zip(self.kets[self.iter].symbols, np.array(theta), strict=True))  # type: ignore[arg-type]
        term_gradients = self.engine.run_gradient(p, method=self.gradient_method or "default")
        grads = np.array(np.real(term_gradients[0]), dtype=float)
        if self.iter == 0:
            return grads
        return grads + deflation_gradient(
            self._measurements[1:],
            self.weights[: self.iter],
            term_gradients,
            lambda: self.engine.run(p),
        )

    def _build_iteration(self) -> list:
        """Build this macroiteration's primitives: the energy of ket ``iter``
        and one overlap with every previously fixed state."""
        en = deepcopy(self.primitive)
        en.bra = self.kets[self.iter]
        en.operator = self.operator
        en.ket = self.kets[self.iter]
        measurements: list[Any] = [en]
        for i, bra in enumerate(self.kets[: self.iter]):
            fixed_bra = bra.set_symbols(symbol_parameter_map=self.state_parameters[i])
            fixed_bra.build()
            ovlp = deepcopy(self.primitive)
            ovlp.bra = fixed_bra
            ovlp.ket = self.kets[self.iter]
            measurements += [ovlp]

        # engine.build() builds each primitive itself — no pre-build needed.
        self.engine.build(measurements)
        self._measurements = measurements
        return measurements

    def iterate(self):
        self._build_iteration()
        state = self.iter

        def objective(x):
            val = self.objective(x)
            self._last_objective = val
            return val

        grad = None
        if self.gradient:

            def grad(x):
                gradients = self.objective_gradient(x)
                self._last_gradnorm = float(np.linalg.norm(gradients))
                return gradients

        def callback(x):
            # Reads the last-evaluation caches — no extra quantum work.
            # energies[state] is written by the same objective call, so the
            # deflation penalty is recovered without re-running the overlaps.
            energy = self._last_objective
            ovlps_norm = float(np.real(self.energies[state] - energy))
            dE = 0.0 if callback.previous_energy is None else energy - callback.previous_energy
            step_size = (
                0.0
                if callback.previous_theta is None
                else float(np.linalg.norm(np.asarray(x) - callback.previous_theta))
            )
            # Pre-formatted: the shared Cost column prints its value verbatim,
            # and the overlap norm keeps the tabular width of the other columns.
            self._log_iteration(
                callback.iter,
                energy,
                dE,
                step_size,
                cost=f"{ovlps_norm:{'.10f' if ovlps_norm < 0 else ' .10f'}}",
                label=f"Macroiteration {state}",
            )
            callback.previous_energy = energy
            callback.previous_theta = np.asarray(x, dtype=float).copy()
            callback.iter += 1

        callback.iter = 0
        callback.previous_energy = None
        callback.previous_theta = None

        p = self.initial_parameters[self.iter]
        result = self._minimize(
            objective,
            p,
            self.optimizer,
            gradient=grad,
            callback=callback if self.verbose else None,
            success_label=None,
        )
        # Re-evaluate at result.x so energies[iter] is the energy AT the
        # returned parameters, not scipy's last probe point.
        self.objective(result.x)
        self.state_parameters.append(dict(zip(self.kets[self.iter].symbols, result.x, strict=True)))
        self.iter += 1
        return result.fun, result.x

    def run(self):
        """Run the VQD algorithm to find multiple eigenstates.

        Returns:
            A tuple containing:
                - List of computed energies for each eigenstate.
                - List of optimized parameter dictionaries for each eigenstate.
        """

        if self.verbose:
            print("VQD Run:")

        for _ in range(len(self.kets)):
            self.iterate()
        return self.energies, self.state_parameters
