from typing import Optional


class __ConfigObject__:
    def __init__(self, seed: Optional[int] = None):
        """Configuration class for OpenQARP settings.

        Attributes:
            seed (int): Random seed for reproducibility.
        """
        self._seed = seed
        self._max_number_of_cuts = 6

    @property
    def seed(self) -> Optional[int]:
        """Get the random seed."""
        return self._seed

    @seed.setter
    def seed(self, value: Optional[int]):
        """Set the random seed."""
        self._seed = value

        # This will be handled internally to all relative dependencies
        # since random and numpy are used in multiple places, they're fixed here
        import random

        import numpy as np

        random.seed(value)
        np.random.seed(value)
        # Scipy uses the numpy seed internally
        # networkx uses numpy for random functions

    @property
    def max_number_of_cuts(self) -> int:
        """Get the maximum number of cuts allowed."""
        return self._max_number_of_cuts

    @max_number_of_cuts.setter
    def max_number_of_cuts(self, value: int):
        """Set the maximum number of cuts allowed."""
        if value < 1:
            raise ValueError("The maximum number of cuts must be at least 1.")
        if value > 6:
            raise ValueError(
                "The maximum number of cuts may be too high, as it will generate more than 6^n circuits.   Please consider reducing it."
            )
        self._max_number_of_cuts = value


config = __ConfigObject__()
