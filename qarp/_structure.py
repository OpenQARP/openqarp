"""The ``Repeat`` record of a block's declared structure (§13 ``structure``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Repeat:
    """``block`` applied ``count`` times in sequence, placed by its own
    ``target_qubits`` in the declaring block's frame."""

    block: Any
    count: int

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("Repeat count must be a positive integer")
