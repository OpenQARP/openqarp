import inspect
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult

from ._optimizer import Optimizer

GRADIENT_METHODS = {"fd", "spsa"}
ParamsLike = Union[List[float], np.ndarray]


def compute_fd_gradients(
    objective: Callable, params: ParamsLike, step: int = 0, fd_eps: float = 1e-5
) -> np.ndarray:
    """Compute gradients using forward finite differences.

    Approximates the gradient of a scalar objective function using
    forward finite differences:

        g_i ≈ (f(x + ε e_i) - f(x)) / ε

    The ``step`` argument is currently unused, but is included for
    compatibility with optimizers that support iteration-dependent
    behavior (e.g., learning-rate decay).

    Args:
        objective (Callable):
            Objective function to differentiate. Must accept a NumPy
            array of parameters and return a scalar value.
        params (List, np.ndarray):
            Point at which to evaluate the gradient. Will be converted
            to a numpy array for compatibility.
        step (int, optional):
            Optimization step or iteration index. Not used in this
            implementation. Default is 0.
        fd_eps (float, optional):
            Finite-difference step size epsilon. Defaults is 1e-5.

    Returns:
        np.ndarray:
            Numpy array containing the forward
            finite-difference gradient with respect to each parameter.
    """

    params = np.asarray(params, dtype=float)
    d = params.size
    base_loss_val = objective(params)
    losses = np.empty(d, dtype=float)

    for i in range(d):
        p = params.copy()
        p[i] += fd_eps
        losses[i] = objective(p)

    return (losses - base_loss_val) / fd_eps


def _accepts_two_positionals(fn: Callable) -> bool:
    """Whether ``fn`` can be called as ``fn(params, step)``.

    Builtins and C-extension callables expose no signature; those are treated
    as single-argument, the shape every in-tree gradient callable has.
    """
    try:
        inspect.signature(fn).bind(None, 0)
    except (TypeError, ValueError):
        return False
    return True


def _ck_schedule(step: int, c0: float = 1e-2, gamma: float = 0.101) -> float:
    """SPSA decay schedule: c_k = c0 / (k+1)^gamma"""
    return c0 / ((step + 1) ** gamma)


def compute_spsa_gradients(
    objective: Callable,
    params: ParamsLike,
    ck: float = 1e-2,
    num_perturbations: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate the gradient using the SPSA estimator.

    This computes a Simultaneous Perturbation Stochastic Approximation (SPSA)
    gradient estimate by sampling random +/-1 perturbation vectors and using a
    two term finite difference.

    Args:
        objective: Function that maps a parameter vector to a scalar.
        params (List, np.ndarray):
            Point at which to evaluate the gradient. Will be converted
            to a numpy array for compatibility.
        ck: Perturbation size used in finite differences/SPSA algorithm.
        num_perturbations: Number of perturbation samples to average.
        seed: Optional random seed.

    Returns:
        A numpy array containing the SPSA gradient estimate.

    """

    params = np.asarray(params, dtype=float)
    d = params.size
    rng = np.random.default_rng(seed)
    acc = np.zeros(d, dtype=float)

    for _ in range(int(num_perturbations)):
        delta = rng.choice([-1.0, 1.0], size=d)
        x_plus = params + ck * delta
        x_minus = params - ck * delta
        f_plus = objective(x_plus)
        f_minus = objective(x_minus)
        acc += (f_plus - f_minus) / (2.0 * ck) * delta

    return acc / max(1, int(num_perturbations))


class EarlyStopper:
    def __init__(
        self,
        objective_fn: Callable,
        initial_params: ParamsLike,
        es_tol: float = 1e-6,
        patience: int = 10,
        ema_beta: float = 1.0,
    ):
        """
        Implements simple early stopping for iterative optimization.

        This tracks the best (optionally EMA-smoothed) objective value seen so far
        and triggers early stopping once there have been too many consecutive steps
        without sufficient improvement.

        Args:
            objective_fn: Function that maps a parameter vector to a scalar
                objective value.
            initial_params (List, np.ndarray):
                Initial parameter vector used to evaluate the starting objective.
                Will be converted to a numpy array.
            es_tol: Minimum required improvement to reset the patience counter.
            patience: Number of consecutive non improving steps allowed before
                stopping.
            ema_beta: Exponential moving average decay factor. If 1.0, no smoothing
                is applied.

        Returns:
            None. Use the instance as a callable: stop = stopper(params).
            The call returns **True** when optimization should stop.

        """

        self.objective_fn = objective_fn
        self.es_tol = float(es_tol)
        self.patience = int(patience)
        self.beta = float(ema_beta)

        initial = float(objective_fn(initial_params))
        self.best = initial
        self.best_params = np.array(initial_params, dtype=float)
        self.no_improve = 0

        self.ema_val = initial

    def __call__(self, params: ParamsLike) -> bool:
        """
        Evaluate new parameters and decide whether to stop.

        Args:
            params (List, np.ndarray):
                Current parameter vector being evaluated.

        Returns:
            True if early stopping criteria have been met, otherwise False.

        """

        params = np.asarray(params, dtype=float)
        current = float(self.objective_fn(params))

        if self.beta < 1.0:
            self.ema_val = self.beta * self.ema_val + (1 - self.beta) * current
            current_s = self.ema_val
        else:
            # No smoothing
            current_s = current

        if current_s < self.best - self.es_tol:
            self.best = current_s  # type: ignore
            self.best_params = params.copy()  # type: ignore
            self.no_improve = 0
        else:
            self.no_improve += 1

        return self.no_improve >= self.patience


class GradientDescentOptimizer(Optimizer):
    supports_bounds = False

    def __init__(self, options: Optional[dict] = None):
        """Initialize a gradient‑descent style optimizer.

        This sets up the optimizer configuration, wraps the objective function
        so that function evaluations can be counted, selects the gradient
        computation method (user provided, finite differences, or SPSA), and
        prepares the main optimization loop logic. Early stopping support is
        also initialized here, optionally using EMA‑smoothed objective values.

        Args:
            options: Optional dictionary of optimizer settings. May include:
                - 'objective': Callable objective function
                - 'grad_fn': Optional user‑provided gradient function
                - 'method': Gradient method ('fd', 'spsa')
                - 'early_stopper': Optional EarlyStopper instance
                - Any other algorithm‑specific configuration values

        Attributes:
            _objective_wrapped: Internal objective function with evaluation
                counting applied.
            _nfev: Number of function evaluations performed.

        Notes:
            Child classes **must** implement:
                compute_update(params, grads, step)

        """
        self.options = options or {}
        self._objective_wrapped: Callable
        self._nfev = 0

    def compute_update(
        self,
        params: ParamsLike,
        grads: ParamsLike,
        step: int,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Optimizer class must implement compute_update(params, grads, step)"
        )

    def reset_state(self) -> None:
        """Clear per-run accumulator state so one instance can be reused.

        Covers the moment/accumulator attributes of the shipped subclasses;
        a subclass carrying extra per-run state overrides this and calls
        ``super().reset_state()``.
        """
        for attr in ("m", "v", "u", "avg_sq", "accum"):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def _get_gradient_function(
        self, objective_function: Callable, gradient: Optional[Callable] = None
    ) -> Callable[[ParamsLike, int], np.ndarray]:
        """
        Construct and return a unified gradient function: grad(params, step).

        When the user provides a gradient function, this method wraps it so that
        optimizers may pass the current iteration index without requiring the user's
        function to accept it. If no gradient function is provided, gradients are
        computed using the method specified via the ``grad_method`` option.

        Note:
            For finite-difference gradients, the iteration index is ignored.
            For SPSA gradients, the iteration index controls the perturbation
            schedule, affecting the magnitude of the stochastic perturbations.

        Args:
            objective_function: Callable mapping parameters to a scalar objective.
            gradient: Optional user-specified gradient function. If None, a numerical
                method is used according to the optimizer's configuration.

        Returns:
            A callable grad(params, step) computing the gradient vector.

        """
        grad_method = str(self.options.get("grad_method", "fd")).lower()

        if gradient is not None:
            # Decide the arity by inspection, not by calling and catching
            # TypeError: a TypeError raised *inside* the user's gradient would
            # otherwise be swallowed and reported as an arity mismatch.
            takes_step = _accepts_two_positionals(gradient)

            def grad_function(params: ParamsLike, step: int) -> np.ndarray:
                return gradient(params, step) if takes_step else gradient(params)

            return grad_function

        if grad_method == "fd":
            fd_eps = float(self.options.get("fd_eps", 1e-5))

            def grad_function(params: ParamsLike, step: int) -> np.ndarray:
                return compute_fd_gradients(objective_function, params, step=step, fd_eps=fd_eps)

            return grad_function

        elif grad_method == "spsa":
            c0 = float(self.options.get("spsa_c0", 1e-2))
            gamma = float(self.options.get("spsa_gamma", 0.101))
            num_perturbations = int(self.options.get("num_spsa", 1))
            base_seed = self.options.get("spsa_seed", None)

            def grad_function(params: ParamsLike, step: int) -> np.ndarray:
                ck = _ck_schedule(step, c0=c0, gamma=gamma)
                seed = None if base_seed is None else int(base_seed) + int(step)
                return compute_spsa_gradients(
                    objective_function,
                    params,
                    ck=ck,
                    num_perturbations=num_perturbations,
                    seed=seed,
                )

            return grad_function

        else:
            raise ValueError(
                f"Unknown grad_method '{grad_method}', allowed: {', '.join(GRADIENT_METHODS)}"
            )

    def minimize(
        self,
        objective_function: Callable,
        initial_parameters: ParamsLike,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        gradient: Optional[Callable] = None,
        tol: Optional[float] = None,
        bounds: Optional[Iterable] = None,
    ) -> OptimizeResult:
        """
        Run a unified gradient-based optimization loop.

        This builds a gradient function (user-provided or estimated), iteratively
        updates the parameters via the child-implemented `compute_update`, and
        optionally performs early stopping with an EMA-smoothed objective. If early
        stopping is enabled, the final parameters can be snapped to the best seen.

        Args:
            objective_function: Function mapping a parameter vector to a scalar
                objective value.
            initial_parameters (List, np.ndarray):
                Starting parameter vector. Will be converted to a numpy array.
            callback: Optional function called after each parameter update with the
                current parameters.
            gradient: Optional user-provided gradient function. If not provided,
                a gradient estimator is selected based on options (e.g., finite
                differences or SPSA-based gradient estimator).
            tol: Must be None — convergence is controlled by the ``maxiter`` /
                ``early_stopping`` options.
            bounds: Must be None — the update rules are unconstrained.

        Returns:
            OptimizeResult: An object containing:
                - x: Final parameter vector (possibly early-stopped to the best seen).
                - fun: Objective value at the final parameters.
                - nit: Number of iterations performed.
                - nfev: Number of objective evaluations.
                - success: Boolean indicating successful completion.

        """
        if tol is not None:
            raise ValueError(
                f"{type(self).__name__} does not support the tol argument; "
                "use the maxiter / early_stopping options instead."
            )
        if bounds is not None:
            raise ValueError(f"{type(self).__name__} does not support the bounds argument.")

        self._nfev = 0
        self.reset_state()

        def wrapped_objective(x: ParamsLike) -> float:
            self._nfev += 1
            return float(objective_function(x))

        self._objective_wrapped = wrapped_objective

        params = np.asarray(initial_parameters, dtype=float)
        grad_function = self._get_gradient_function(wrapped_objective, gradient)
        maxiter = int(self.options.get("maxiter", 100))
        results = OptimizeResult()
        results.success = False
        results.x = params

        # Early Stopping
        if self.options.get("early_stopping", False):
            stopper = EarlyStopper(
                self._objective_wrapped,
                params,
                es_tol=float(self.options.get("es_tol", 1e-6)),
                patience=int(self.options.get("patience", 10)),
                ema_beta=float(self.options.get("ema_beta", 1)),
            )
        else:
            stopper = None

        for step in range(maxiter):
            grads = grad_function(params, step)
            update = self.compute_update(params, grads, step)
            params = params + update

            if callback:
                # With a user-supplied gradient the loop never needs the
                # objective, but every callback is written against the scipy
                # contract, where it has been evaluated at these parameters —
                # composite algorithms read the value their objective wrapper
                # caches.  Reporting costs one evaluation per iteration.
                wrapped_objective(params)
                callback(params)

            if stopper is not None:
                stop_now = stopper(params)
                if stop_now:
                    break
        nit = step + 1

        if stopper is not None and stopper.best < self._objective_wrapped(params):  # type: ignore
            params = stopper.best_params

        results.fun = self._objective_wrapped(params)
        results.x = params
        results.nit = nit  # number of iterations accounting for early stopping.
        results.nfev = self._nfev
        results.success = True
        return results


class SGDOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """Vanilla stochastic gradient descent update.
        No options for the optimize, but could be extended in future.

        SGD update rule:
            x_{k+1} = x_k - lr * gradient

        """
        super().__init__(options)

    def compute_update(self, params, grads, step):
        """Compute the SGD parameter update.

        SGD update rule for parameters::

            x_{k+1} = x_k - lr * gradient_estimate

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                Gradient vector evaluated at the current parameters.
            step (int):
                Current optimization step (unused for SGD).

        Returns:
            np.ndarray: The SGD update to be added to the parameters.

        """
        lr = float(self.options.get("lr", 0.01))
        return -lr * grads


class SPSAOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """SPSA optimizer with a decaying step-size schedule a_k.

        SPSA update rule for parameters:
            a_k = a0 / (k + 1 + A) ** alpha
            x_{k+1} = x_k - a_k * gradient_estimate

        """
        super().__init__(options)
        self.a0 = float(self.options.get("spsa_a0", 0.1))
        self.alpha = float(self.options.get("spsa_alpha", 0.602))
        self.A = float(self.options.get("spsa_A", 10.0))

    def compute_update(self, params, grads, step):
        """Compute the SPSA parameter update. Below parameters for optimizer
        are keys of `options` dictionary.

        SPSA update rule for parameters:
            a_k = a0 / (k + 1 + A) ** alpha
            x_{k+1} = x_k - a_k * gradient_estimate

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                SPSA gradient estimate at the current parameters.
            step (int):
                Current optimization step (0‑indexed), used in the decay rule.

        Returns:
            np.ndarray: The SPSA update to be added to the parameters.

        """
        ak = self.a0 / ((step + 1 + self.A) ** self.alpha)
        return -ak * grads


class RMSPropOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """RMSProp optimizer with exponentially decayed squared gradient averages.
        Below parameters for optimizer are keys of `options` dictionary.

        RMSProp update rule for parameters:
            avg_sq = beta * avg_sq + (1 - beta) * (gradient ** 2)
            x_{k+1} = x_k - lr * gradient / (sqrt(avg_sq) + eps)

        Args:
            lr (float, optional):
                Learning rate for the RMSProp optimizer. Defaults to 0.001.
            beta (float, optional):
                Exponential decay rate for the running average of squared
                gradients. Defaults to 0.9.
            eps (float, optional):
                Small constant added for numerical stability. Defaults to 1e‑8.

        """
        super().__init__(options)
        self.avg_sq = None
        self.lr = float(self.options.get("lr", 0.001))
        self.beta = float(self.options.get("beta", 0.9))
        self.eps = float(self.options.get("eps", 1e-8))

    def _ensure_state(self, params):
        """If optimizer object is passed to a routine which does multiple optimization routines,
        the internal state needs to be reset each time.

        If state variables are uninitialized or have a different shape than `params`,
        reinitialize them to zeros of the correct shape. Other checks are possible.
        """
        if (self.avg_sq is None) or (self.avg_sq.shape != params.shape):
            self.avg_sq = np.zeros_like(params)

    def compute_update(self, params, grads, step):
        """Compute the RMSProp parameter update.

        RMSProp update rule for parameters:
            avg_sq = beta * avg_sq + (1 - beta) * (gradient ** 2)
            x_{k+1} = x_k - lr * gradient / (sqrt(avg_sq) + eps)

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                Gradient vector evaluated at the current parameters.
            step (int):
                Current optimization step (unused for RMSProp).

        Returns:
            np.ndarray: The RMSProp update to be added to the parameters.

        """
        self._ensure_state(params)

        self.avg_sq = self.beta * self.avg_sq + (1 - self.beta) * (grads**2)
        update = -self.lr * grads / (np.sqrt(self.avg_sq) + self.eps)
        return update


class AdaGradOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """AdaGrad optimizer with accumulated squared gradients.
        Below parameters for optimizer are keys of `options` dictionary

        AdaGrad update rule for parameters:
            accum = accum + (gradient ** 2)
            x_{k+1} = x_k - lr * gradient / (sqrt(accum) + eps)

        Args:
            lr (float, optional):
                Base learning rate for AdaGrad. Defaults to 0.01.
            eps (float, optional):
                Small constant for numerical stability. Defaults to 1e‑8.
            initial_accumulator_value (float, optional):
                Initial value for the squared‑gradient accumulator. Using a
                positive value reduces very large early updates. Defaults to 0.0.

        """
        super().__init__(options)
        self.accum = None
        self.lr = float(self.options.get("lr", 0.01))
        self.eps = float(self.options.get("eps", 1e-8))
        self.init_acc = float(self.options.get("initial_accumulator_value", 0.0))

    def _ensure_state(self, params):
        """If optimizer object is passed to a routine which does multiple optimization routines,
        the internal state needs to be reset each time.

        If state variables are uninitialized or have a different shape than `params`,
        reinitialize them to zeros of the correct shape. Other checks are possible.
        """
        if (self.accum is None) or (self.accum.shape != params.shape):
            self.accum = np.full_like(params, self.init_acc, dtype=float)

    def compute_update(self, params, grads, step):
        """Compute the AdaGrad parameter update.

        AdaGrad update rule for parameters:
            accum = accum + (gradient ** 2)
            x_{k+1} = x_k - lr * gradient / (sqrt(accum) + eps)

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                Gradient vector evaluated at the current parameters.
            step (int):
                Current optimization step (unused for AdaGrad).

        Returns:
            np.ndarray: The AdaGrad update to be added to the parameters.

        """
        self._ensure_state(params)

        self.accum += grads**2
        update = -self.lr * grads / (np.sqrt(self.accum) + self.eps)
        return update


class AdamOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """
        Adam optimizer using first- and second‑order exponential moving averages.
        Below parameters for optimizer are keys of `options` dictionary

        Adam update rule for parameters:
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x_{k+1} = x_k - lr * m_hat / (sqrt(v_hat) + eps)

        Args:
            lr (float, optional):
                Base learning rate for the Adam optimizer. Defaults to 0.001.
            beta1 (float, optional):
                Exponential decay rate for the first‑moment (mean) estimates.
                Defaults to 0.9.
            beta2 (float, optional):
                Exponential decay rate for the second‑moment (uncentered variance)
                estimates. Defaults to 0.999.
            eps (float, optional):
                Small constant added for numerical stability to prevent division by
                zero. Defaults to 1e‑10 (as per Karpathy).
        """

        super().__init__(options)
        self.m = None
        self.v = None
        self.lr = float(self.options.get("lr", 0.001))
        self.beta1 = float(self.options.get("beta1", 0.9))
        self.beta2 = float(self.options.get("beta2", 0.999))
        self.eps = float(self.options.get("eps", 1e-10))

    def _ensure_state(self, params):
        """If optimizer object is passed to a routine which does multiple optimization routines,
        the internal state needs to be reset each time.

        If state variables are uninitialized or have a different shape than `params`,
        reinitialize them to zeros of the correct shape. Other checks are possible.
        """
        if (self.m is None) or (self.m.shape != params.shape):
            self.m = np.zeros_like(params)
        if (self.v is None) or (self.v.shape != params.shape):
            self.v = np.zeros_like(params)

    def compute_update(self, params, grads, step):
        """Compute the Adam parameter update.

        Adam update rule for parameters:
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x_{k+1} = x_k - lr * m_hat / (sqrt(v_hat) + eps)

        This applies the Adam optimization rule using exponential moving averages
        of the first and second moments of the gradients. Bias‑corrected moment
        estimates are used to form the adaptive learning rate update.

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                Gradient vector evaluated at the current parameters.
            step (int):
                Current optimization step (0 indexed), used for bias correction.

        Returns:
            np.ndarray: The Adam update to be added to the parameters.

        """

        self._ensure_state(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        m_hat = self.m / (1 - self.beta1 ** (step + 1))
        v_hat = self.v / (1 - self.beta2 ** (step + 1))

        update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return update


class AdamaxOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """Adamax optimizer using the infinity norm for second moment estimation.
        Below parameters for optimizer are keys of `options` dictionary

        Adamax update rule for parameters:
            m = beta1 * m + (1 - beta1) * gradient
            u = max(beta2 * u, abs(gradient))
            m_hat = m / (1 - beta1 ** t)
            x_{k+1} = x_k - lr * m_hat / (u + eps)

        This variant of Adam replaces the second moment accumulator with an
        exponentially weighted infinity norm, improving stability under very
        large gradients.

        Args:
            lr (float, optional):
                Base learning rate for the Adamax optimizer. Defaults to 0.002.
            beta1 (float, optional):
                Exponential decay rate for the first‑moment estimates.
                Defaults to 0.9.
            beta2 (float, optional):
                Exponential decay rate for the exponentially weighted infinity
                norm. Defaults to 0.999.
            eps (float, optional):
                Small constant for numerical stability. Defaults to 1e‑8.

        """
        super().__init__(options)
        self.m = None
        self.u = None
        self.lr = float(self.options.get("lr", 0.002))
        self.beta1 = float(self.options.get("beta1", 0.9))
        self.beta2 = float(self.options.get("beta2", 0.999))
        self.eps = float(self.options.get("eps", 1e-8))

    def _ensure_state(self, params):
        """If optimizer object is passed to a routine which does multiple optimization routines,
        the internal state needs to be reset each time.

        If state variables are uninitialized or have a different shape than `params`,
        reinitialize them to zeros of the correct shape. Other checks are possible.
        """
        if (self.m is None) or (self.m.shape != params.shape):
            self.m = np.zeros_like(params)
        if (self.u is None) or (self.u.shape != params.shape):
            self.u = np.zeros_like(params)

    def compute_update(self, params, grads, step):
        """Compute the Adamax parameter update.

        Adamax update rule for parameters:
            m = beta1 * m + (1 - beta1) * gradient
            u = max(beta2 * u, abs(gradient))
            m_hat = m / (1 - beta1 ** t)
            x_{k+1} = x_k - lr * m_hat / (u + eps)

        This applies the Adamax update rule using:
        - first moment accumulation for gradients,
        - an exponentially decayed infinity norm for the second moment,
        - bias corrected first moment estimates.

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                Gradient vector evaluated at the current parameters.
            step (int):
                Current optimization step (0 indexed).

        Returns:
            np.ndarray: The Adamax update to be added to the parameters.

        """
        self._ensure_state(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.u = np.maximum(self.beta2 * self.u, np.abs(grads))

        m_hat = self.m / (1 - self.beta1 ** (step + 1))
        update = -(self.lr * m_hat) / (self.u + self.eps)
        return update


class NadamOptimizer(GradientDescentOptimizer):
    def __init__(self, options=None):
        """Nadam optimizer: Nesterov accelerated variant of Adam.
        Below parameters for optimizer are keys of `options` dictionary.

        Nadam update rule for parameters::

            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            nesterov_term = (
                beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1 ** t)
            )
            x_{k+1} = x_k - lr * nesterov_term / (sqrt(v_hat) + eps)

        Nadam applies Nesterov momentum to Adam by modifying the predicted
        gradient step via a momentum‑corrected first moment estimate. This
        results in smoother and more responsive updates compared to Adam.

        Args:
            lr (float, optional):
                Learning rate for the Nadam optimizer. Defaults to 0.002.
            beta1 (float, optional):
                Exponential decay rate for the first moment estimates.
                Defaults to 0.9.
            beta2 (float, optional):
                Exponential decay rate for the second moment estimates.
                Defaults to 0.999.
            eps (float, optional):
                Small constant for numerical stability. Defaults to 1e‑8.

        """
        super().__init__(options)
        self.m = None
        self.v = None

        self.lr = float(self.options.get("lr", 0.002))
        self.beta1 = float(self.options.get("beta1", 0.9))
        self.beta2 = float(self.options.get("beta2", 0.999))
        self.eps = float(self.options.get("eps", 1e-8))

    def _ensure_state(self, params):
        """If optimizer object is passed to a routine which does multiple optimization routines,
        the internal state needs to be reset each time.

        If state variables are uninitialized or have a different shape than `params`,
        reinitialize them to zeros of the correct shape. Other checks are possible.
        """
        if (self.m is None) or (self.m.shape != params.shape):
            self.m = np.zeros_like(params)
        if (self.v is None) or (self.v.shape != params.shape):
            self.v = np.zeros_like(params)

    def compute_update(self, params, grads, step):
        """Compute the Nadam parameter update.

        Nadam update rule for parameters::

            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            nesterov_term = (
                beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1 ** t)
            )
            x_{k+1} = x_k - lr * nesterov_term / (sqrt(v_hat) + eps)

        This applies the Nadam update rule using:
        - exponential moving averages of first and second moments,
        - bias‑corrected moment estimates,
        - Nesterov momentum applied on the predicted gradient direction.

        Args:
            params (np.ndarray):
                Current parameter vector.
            grads (np.ndarray):
                Gradient vector evaluated at the current parameters.
            step (int):
                Current optimization step (0 indexed), used for bias corrections.

        Returns:
            np.ndarray: The Nadam update to be added to the parameters.

        """
        t = step + 1

        self._ensure_state(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        v_hat = self.v / (1 - self.beta2**t)

        nesterov_term = self.beta1 * self.m / (1 - self.beta1 ** (t + 1)) + (
            1 - self.beta1
        ) * grads / (1 - self.beta1**t)

        update = -self.lr * nesterov_term / (np.sqrt(v_hat) + self.eps)
        return update
