from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult

from ._optimizer import Optimizer


def lr_schedule(
    t: int,
    lr: float,
    T: int,
    schedule: Optional[str] = None,
    lr_min: float = 0.0,
    gamma: float = 0.99,
    power: int = 2,
):
    """
    A learning rate scheduler for Rotosolve.

    Args:
        t: current step or epoch (0-based).
        schedule: one of {"linear", "exponential", "polynomial", "cosine"}.
        lr: initial learning rate.
        T: total steps/epochs for schedules that need horizon
                                    (linear, polynomial, cosine).
        lr_min: minimum LR floor for linear/polynomial/cosine (default 0).
        gamma: exponential decay factor per step for 'exponential' (default 0.99).
        power: polynomial decay power (default 2).

    Returns:
        float: learning rate at step t.
    """
    if schedule is None:
        return lr
    else:
        schedule = schedule.lower()

    if schedule == "exponential":
        return float(lr * (gamma**t))
    x = max(0.0, min(1.0, t / T))

    if schedule == "linear":
        return float(lr_min + (lr - lr_min) * (1.0 - x))

    if schedule == "polynomial":
        return float(lr_min + (lr - lr_min) * ((1.0 - x) ** power))

    if schedule == "cosine":
        return float(lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * x)))

    raise ValueError("schedule must be one of {'linear','exponential','polynomial','cosine'}")


class RotosolveOptimizer(Optimizer):
    """
    Rotosolve optimizer.
    See Ostaszewski et al., arXiv: 1905.09692 (2021)
    This optimizer is designed for optimizing parameters in variational circuits and assumes that the
    objective function is periodic, with a period of 2*pi, with respect to each parameter.
    Args:
        maxiter: maximum number of Rotosolve macroiterations
        tol: convergence criterion
        verbose: print verbose output every macroiteration and every microiteration
    """

    supports_bounds = False

    def __init__(
        self,
        maxiter=100,
        tol=1e-7,
        lr=1,
        schedule=None,
        gamma=0.99,
        power=2,
        verbose=False,
        flat_tol=1e-10,
        max_restarts=3,
        kick_size=0.1,
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        # Coordinate descent can converge to a coordinate-wise saddle: a point
        # where every single parameter is individually optimal but a joint move
        # lowers the cost (the all-zero start of a hardware-efficient ansatz is
        # the canonical example — there ``m_φ1 == m_φ2`` exactly by symmetry, so
        # rotosolve keeps every coordinate put).  On convergence we apply up to
        # ``max_restarts`` deterministic symmetry-breaking kicks of magnitude
        # ``kick_size`` (rad) from the best point seen and re-optimise, keeping
        # the best result.  A genuine minimum is recovered unchanged (the kick is
        # rejected); a saddle is escaped.
        self.max_restarts = max_restarts
        self.kick_size = kick_size
        # A parameter is "flat" when both arguments of the arctan2 update are
        # below this threshold — i.e. the three sampled costs are equal to
        # within numerical noise, so the analytic update direction is decided
        # by floating-point rounding rather than signal.  Such updates are
        # skipped (see ``minimize``).  Chosen well above statevector roundoff
        # (~1e-16) and well below any real energy variation (~1e-2).
        self.flat_tol = flat_tol
        # qarpx parametric gates use radians (Rz(θ) rotates by θ rad).
        # The shift / constant terms in the update formula are scaled by
        # this angle so the optimizer steps over a full 2π period.
        self.PI_ANGLE = np.pi

        self.schedule = lambda t: lr_schedule(
            t, lr=lr, T=self.maxiter, schedule=schedule, lr_min=1e-5, gamma=gamma, power=power
        )

    def minimize(
        self,
        objective_function: Callable,
        initial_parameters: Union[List, np.ndarray],
        callback: Optional[Callable] = None,
        gradient: Optional[Callable] = None,
        tol: Optional[float] = None,
        bounds: Optional[Iterable] = None,
    ) -> Any:
        """
        Minimize the objective function provided, starting at the initial parameters.
        Args:
            objective_function: Function to minimize.
            initial_parameters: Initial values of parameters. It is usually best to start from a zero vector.
            callback: An optional callable to call with the parameters after each update (as in scipy optimizers).
            gradient: Must be None.
            tol: Must be None — the convergence criterion is the constructor's ``tol``.
            bounds: Must be None — the update rule assumes an unbounded 2π-periodic domain.

        Returns:
            A SciPy Result object.
        """
        if tol is not None:
            raise ValueError(
                f"{type(self).__name__} does not support the tol argument; "
                "set the convergence criterion via the constructor's tol."
            )
        if bounds is not None:
            raise ValueError(f"{type(self).__name__} does not support the bounds argument.")

        x = np.asarray(initial_parameters, dtype=float).copy()
        num_params = x.shape[0]
        f = objective_function

        results = OptimizeResult()
        results.success = False
        results.x = x
        prev_val = None
        best_val: Optional[float] = None
        best_x = np.copy(x)
        restarts = 0
        if self.verbose:
            print("Rotosolve")
            print(f"Number of parameters: {num_params}")
            print("    Iter\tMicroIter\tEnergy\t\tDifference")
        for num_it in range(self.maxiter):
            lr_t = self.schedule(num_it)
            for d in range(num_params):
                phi = x[d]
                x_temp = np.copy(x)
                x_temp[d] = phi
                m_phi = f(x_temp)
                if prev_val is None:
                    prev_val = m_phi
                if self.verbose:
                    print(f"{num_it:6}\t\t{d:6}\t\t{m_phi:.6f}\t")
                x_temp[d] = phi + self.PI_ANGLE / 2
                m_phi1 = f(x_temp)
                x_temp[d] = phi - self.PI_ANGLE / 2
                m_phi2 = f(x_temp)
                # eq. (1) from arXiv: 1905.09692
                numerator = 2 * m_phi - m_phi1 - m_phi2
                denominator = m_phi1 - m_phi2
                # Skip parameters the cost is (numerically) flat in: there both
                # arctan2 arguments are pure rounding noise, so the analytic
                # update points in a random direction.  At symmetric starts
                # (e.g. all-zero parameters, where a layer's phase/Rz gates leave
                # |0…0⟩ unchanged) this stalls the optimizer.  Leaving the
                # parameter put is exact — a flat direction has no preferred
                # angle — and it becomes informative once a neighbour moves.
                if max(abs(numerator), abs(denominator)) < self.flat_tol:
                    continue
                x[d] = phi - lr_t * (
                    self.PI_ANGLE / 2 + np.arctan2(numerator, denominator) * self.PI_ANGLE / np.pi
                )

            new_val = f(x)
            results.nit = num_it
            if best_val is None or new_val < best_val:
                best_val = new_val
                best_x = np.copy(x)

            if callback is not None:
                callback(x)
            if prev_val is not None:
                if self.verbose:
                    print(f"{num_it:6}\t      \t\t{new_val:.6f}\t\t{abs(new_val - prev_val):.6f}")
                if abs(new_val - prev_val) < self.tol:
                    if restarts < self.max_restarts:
                        # Possible coordinate-wise saddle — kick the best point
                        # along a deterministic direction and re-optimise.
                        restarts += 1
                        x = best_x + self._symmetry_break(num_params, restarts)
                        prev_val = f(x)
                        continue
                    results.success = True
                    break
            prev_val = new_val

        # Return the best point seen across all restarts (a kick that failed to
        # improve leaves the genuine minimum as the best; one that escaped a
        # saddle becomes the new best).
        results.x = best_x
        results.fun = best_val if best_val is not None else results.fun
        return results

    def _symmetry_break(self, num_params: int, restart: int) -> np.ndarray:
        """Deterministic small perturbation to break coordinate-wise symmetry.

        Varies per coordinate and per restart (no RNG, so runs are
        reproducible).  Magnitude is bounded by ``kick_size``.
        """
        k = np.arange(num_params)
        return self.kick_size * np.cos(k + 1.0 + restart * (np.pi / 3.0))
