"""Algorithms.  Public depth: flat — every primitive and composite algorithm is
``qarp.algorithms.<Name>``, together with the eigenspectrum helpers.

Submodules are private implementation and may be reorganised without notice.
"""

import importlib.util as _importlib_util

from .._lazy import lazy_exports as _lazy_exports

from ._primitives import (
    Target,
    PrimitiveAlgorithm,
    Sampler,
    HadamardTest,
    SWAPTest,
    StateVector,
    MirrorTest,
    InterferometricTest,
    TermwiseHadamardTest,
    GroupedTransitionHadamardTest,
    TermwiseSWAPTest,
    PauliAveraging,
    BasisRotationAveraging,
    CuttingPrimitive,
    ShadowProtocol,
    PauliShadow,
    ShadowDataset,
    ShadowEstimator,
    ShadowEstimate,
    ShadowKernel,
    PauliKernel,
)

from ._composite import (
    VFF,
    VQA,
    VQE,
    ProjectedVQE,
    VQD,
    QPE,
    DOSQPE,
    QAOA,
    AdaptVQE,
    AdaptVQD,
    PCE,
    cPCE,
    iterativePCE,
    calculate_qubits,
    classical_function_max_cut,
    SSVQE,
    CompositeAlgorithm,
    MonteCarlo,
    WalkerState,
    MMQCELS,
    QMEGS,
    get_overlaps,
    QSE,
    QITE,
    AmplitudeAmplification,
    AmplitudeEstimation,
    Grover,
    Shor,
)

from ._utils import (
    map_binary_to_integer_keys,
    find_occupation_numbers,
    find_eigenspectrum_degeneracy,
    find_unique_eigs_and_occupation_numbers,
    dirichlet_kernel_squared,
    generate_states_new_basis,
)

__all__ = [
    "AdaptVQD",
    "AdaptVQE",
    "AmplitudeAmplification",
    "AmplitudeEstimation",
    "BasisRotationAveraging",
    "CompositeAlgorithm",
    "CuttingPrimitive",
    "DOSQPE",
    "GroupedTransitionHadamardTest",
    "Grover",
    "HadamardTest",
    "InterferometricTest",
    "MMQCELS",
    "MirrorTest",
    "MonteCarlo",
    "PCE",
    "PauliAveraging",
    "PauliKernel",
    "PauliShadow",
    "PrimitiveAlgorithm",
    "ProjectedVQE",
    "QAOA",
    "QITE",
    "QMEGS",
    "QPE",
    "QSE",
    "SSVQE",
    "SWAPTest",
    "Sampler",
    "ShadowDataset",
    "ShadowEstimate",
    "ShadowEstimator",
    "ShadowKernel",
    "ShadowProtocol",
    "Shor",
    "StateVector",
    "Target",
    "TermwiseHadamardTest",
    "TermwiseSWAPTest",
    "VFF",
    "VQA",
    "VQD",
    "VQE",
    "WalkerState",
    "cPCE",
    "calculate_qubits",
    "classical_function_max_cut",
    "dirichlet_kernel_squared",
    "find_eigenspectrum_degeneracy",
    "find_occupation_numbers",
    "find_unique_eigs_and_occupation_numbers",
    "generate_states_new_basis",
    "get_overlaps",
    "iterativePCE",
    "map_binary_to_integer_keys",
]

# SpectrumEstimator's heavy path needs cvxpy (the [convex-optim] extras, ~0.5 s
# to import): resolved on first access; the __all__ gate keeps a minimal
# install from seeing a half-broken symbol.
if _importlib_util.find_spec("cvxpy") is not None:
    __all__ += ["SpectrumEstimator"]

__getattr__, __dir__ = _lazy_exports(
    __name__, {"SpectrumEstimator": ("._spectral_estimation", "cvxpy", "convex-optim")}
)
