"""Operators.  Public depth: flat for the operator, mapping and grouping types,
plus seven qualified namespaces.  Function-first fermionic surface —
``models``, ``integrals``, ``ucc``, ``onv`` (bare ``ucc_singles`` /
``active_space_integrals`` do not carry their domain).  Interop and aliases —
``functions`` (raw C++ transforms: a bare ``jordan_wigner`` would sit beside
the ``JordanWigner`` Mapping class), ``compat`` (openfermion MSB interop, §1)
and ``pyscf`` (the pyscf recipes).  All other submodules are private.
"""

import importlib.util as _importlib_util

from .._lazy import lazy_exports as _lazy_exports

from ._basis_rotation_grouping import (
    basis_rotation_grouping,
    diagonal_group_to_masks,
    double_factorization,
)
from ._fermion_operator import FermionOperator
from ._linear_combination_unitaries import LinearCombinationUnitaries
from ._qdrift import qDRIFT
from ._qubit_operator import QubitOperator
from ._grouping import (
    FullyCommuting,
    GroupingStrategy,
    NoGrouping,
    QubitWiseCommuting,
    group_basis,
)
from ._mappings import BravyiKitaev, JordanWigner, Mapping, Parity
from ._tensor_operators import (
    fermion_operator_from_tensor,
    orbital_rotation_generator,
    orbital_rotation_matrix,
    orbital_rotation_parameters,
    rotate_tensor,
)

__all__ = [
    "BravyiKitaev",
    "FermionOperator",
    "FullyCommuting",
    "GroupingStrategy",
    "JordanWigner",
    "LinearCombinationUnitaries",
    "Mapping",
    "NoGrouping",
    "Parity",
    "QubitOperator",
    "QubitWiseCommuting",
    "basis_rotation_grouping",
    "compat",
    "diagonal_group_to_masks",
    "double_factorization",
    "fermion_operator_from_tensor",
    "functions",
    "group_basis",
    "integrals",
    "models",
    "onv",
    "orbital_rotation_generator",
    "orbital_rotation_matrix",
    "orbital_rotation_parameters",
    "pyscf",
    "qDRIFT",
    "rotate_tensor",
    "ucc",
]

# VUMPO needs quimb (the [mps] extras, ~0.7 s to import): resolved on first
# access; the __all__ gate keeps `import *` and the docs to what is installed.
if _importlib_util.find_spec("quimb") is not None:
    __all__ += ["VUMPO", "qubit_operator_to_mpo"]

__getattr__, __dir__ = _lazy_exports(
    __name__,
    {
        "VUMPO": ("._vumpo", "quimb", "mps"),
        "qubit_operator_to_mpo": ("._vumpo", "quimb", "mps"),
    },
)
