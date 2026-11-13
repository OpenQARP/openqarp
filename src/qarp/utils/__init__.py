"""Random circuit instances and toy-model Hamiltonians.  Public depth: flat.
Submodules are private.
"""

from ._random_instances import generate_random_circuit
from ._toy_models import FH_ham_and_wf, FH_ham_and_wf_singles_and_doubles

__all__ = [
    "FH_ham_and_wf",
    "FH_ham_and_wf_singles_and_doubles",
    "generate_random_circuit",
]
