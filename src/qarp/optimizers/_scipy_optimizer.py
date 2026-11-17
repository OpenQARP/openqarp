from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
from scipy import optimize

from ._optimizer import Optimizer


class ScipyOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        options: Optional[dict] = None,
    ):
        """Optimizer class for interfacing to SciPy optimizers.

        Args:
            method:
                The optimization method to use, as defined by scipy.
            options:
                A dict with specific instructions for the optimizer as defined in
                SciPy documentation.
        """
        self.method = method
        self.options = options

    def minimize(
        self,
        objective_function: Callable,
        initial_parameters: Union[List, np.ndarray],
        callback: Optional[Callable] = None,
        gradient: Optional[Callable] = None,
        tol: Optional[float] = None,
        bounds: Optional[Iterable[float]] = None,
    ) -> Any:
        """Minimize the objective function provided, starting at the initial parameters.

        Args:
            objective_function:
                The objective function to minimize.
            initial_parameters:
                The parameters from which to begin the optimization.
            callback:
                An optional callable to call with the parameters after each update.
            gradient:
                A function which returns the gradient as an array in coincidence with
                the parameters provided.

        Returns:
            A SciPy Result object.
        """
        if gradient is None:
            return optimize.minimize(
                objective_function,
                initial_parameters,
                method=self.method,
                options=self.options,
                callback=callback,
                tol=tol,
                bounds=bounds,
            )
        else:
            return optimize.minimize(
                objective_function,
                initial_parameters,
                method=self.method,
                options=self.options,
                callback=callback,
                jac=gradient,
                tol=tol,
                bounds=bounds,
            )
