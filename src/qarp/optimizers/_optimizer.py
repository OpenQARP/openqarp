"""Abstract ``Optimizer`` contract shared by every concrete optimizer."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np


class Optimizer(ABC):
    """Base Optimizer class."""

    # Whether ``minimize(bounds=...)`` is honoured.  An optimizer whose update
    # rule has no notion of a box (gradient descent, Rotosolve) sets this
    # False and raises on a non-None ``bounds``; a caller that must bound its
    # search (MMQCELS) checks it at construction rather than after the data
    # is generated.
    supports_bounds: bool = True

    @abstractmethod
    def minimize(
        self,
        objective_function: Callable,
        initial_parameters: Union[List, np.ndarray],
        callback: Optional[Callable] = None,
        gradient: Optional[Callable] = None,
        tol: Optional[float] = None,
        bounds: Optional[Iterable] = None,
    ) -> Any:
        raise NotImplementedError("Abstract method to be defined")
