from abc import ABC, abstractmethod
from typing import List

from ..blocks import AnyBlock


class BlockFactory(ABC):
    """Abstract base class for block factories."""

    @abstractmethod
    def create_blocks(self, *args, **kwargs) -> List[AnyBlock]:
        """Create blocks from input data.

        Returns:
            List of block objects
        """
        pass
