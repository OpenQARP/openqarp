from enum import Enum


class Target(Enum):
    """Enumeration of measurement targets for primitive algorithms.

    Target specifies the type of quantum measurement being performed by a primitive algorithm.
    Different targets determine how measurement circuits are constructed and how results are
    post-processed to compute the desired quantum mechanical quantity.

    Attributes:
        SAMPLING: Return raw measurement distribution as a dictionary of bitstrings to probabilities.
        EXPECTATION_VALUE: Compute the expectation value <ket|operator|ket⟩.
        OVERLAP: Compute the overlap (inner product) ⟨bra|ket⟩.
        TRANSITION_AMPLITUDE: Compute the transition amplitude ⟨bra|operator|ket⟩.
    """

    SAMPLING = 0
    EXPECTATION_VALUE = 1
    OVERLAP = 2
    TRANSITION_AMPLITUDE = 3

    def __str__(self):
        return self.name
