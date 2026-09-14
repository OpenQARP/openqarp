"""Circuit blocks.  Public depth: flat — every block is ``qarp.blocks.<Name>``.

Submodules are private implementation and may be reorganised without notice.
"""

import importlib.util as _importlib_util

from .._lazy import lazy_exports as _lazy_exports

from ._block import (
    AnyBlock,  # = qx.Block: the type to annotate with / isinstance against;
    # not a base class — subclass SimpleBlock (leaf) or CompositeBlockBase (tree).
    SimpleBlock,
    ControlledBlock,
    MeasureBlock,
    ResetBlock,
    ConditionalBlock,
    # `CompositeBlockBase` is the bare tree primitive that internal blocks
    # extend and that isinstance checks target; `composite_block.CompositeBlock`
    # below is the user-facing assembly wrapper (adds measurement-only-at-end
    # orchestration and qubit-count inference from children).
    CompositeBlockBase,
)
from ._composite_block import CompositeBlock
from ._prepares_known_state import (
    PreparesKnownState,
    declaring_blocks,
    prepares_known_state,
)

from ._primitives import (
    AGateBlock,
    BrickworkEntanglingBlock,
    BrickworkPCEBlock,
    HEABlock,
    HnBlock,
    XnBlock,
    QFTBlock,
    LinearEntanglingBlock,
    RSPBlock,
    TrotterBlock,
    QPEBlock,
    DOSQPEBlock,
    GivensBlock,
    OrbitalRotationBlock,
    UPCCDBlock,
    HadamardTestBlock,
    InterferometricMeasurementBlock,
    InterferometricStateBlock,
    SWAPTestBlock,
    TrotterAnsatzBlock,
    UCCBlock,
    QAOABlock,
    SPABlock,
    BlockEncodingBlock,
    PauliBlock,
    SelectBlock,
    ReadoutBlock,
    IdentityBlock,
    LayerBlock,
    MixedOperatorBlock,
    CostOperatorBlock,
    PhaseShiftBlock,
    ProjectedControlPhaseBlock,
    ModularMultiplicationBlock,
    OrderFindingBlock,
    QROMBlock,
    HaarRandomBlock,
    ReflectionBlock,
    AmplitudeAmplificationBlock,
    AmplitudeEstimationBlock,
    GroverBlock,
    QSPBlock,
    QSPAngleFinder,
    QSVTBlock,
    SynthesizedTimeEvolutionBlock,
    SynthesizedUnitaryBlock,
    QubitizationBlock,
    ParticleNumberProjectorBlock,
    SzProjectorBlock,
    SyProjectorBlock,
    SpinSquaredProjectorBlock,
)

from ._state_preparation import (
    ComputationalBasisStateBlock,
    DickeStateBlock,
    CVOQRAMStateBlock,
    CVQRAMStateBlock,
    MappedONVStateBlock,
    GHZLikeStateBlock,
    SynthesizedStateBlock,
    SparseStateBlock,
    MultiONVStateBlock,
    SlaterDeterminantBlock,
    CSFStateBlock,
    LowRankStateBlock,
    MPSStateBlock,
    UniformSuperpositionBlock,
    PiecewiseLinearStateBlock,
    HypergraphStateBlock,
)

__all__ = [
    "AGateBlock",
    "AmplitudeAmplificationBlock",
    "AmplitudeEstimationBlock",
    "AnyBlock",
    "BlockEncodingBlock",
    "BrickworkEntanglingBlock",
    "BrickworkPCEBlock",
    "CSFStateBlock",
    "CVOQRAMStateBlock",
    "CVQRAMStateBlock",
    "CompositeBlock",
    "CompositeBlockBase",
    "ComputationalBasisStateBlock",
    "ConditionalBlock",
    "ControlledBlock",
    "CostOperatorBlock",
    "DOSQPEBlock",
    "DickeStateBlock",
    "GHZLikeStateBlock",
    "GivensBlock",
    "GroverBlock",
    "HEABlock",
    "HaarRandomBlock",
    "HadamardTestBlock",
    "HnBlock",
    "HypergraphStateBlock",
    "IdentityBlock",
    "InterferometricMeasurementBlock",
    "InterferometricStateBlock",
    "LayerBlock",
    "LinearEntanglingBlock",
    "LowRankStateBlock",
    "MPSStateBlock",
    "MappedONVStateBlock",
    "MeasureBlock",
    "MixedOperatorBlock",
    "ModularMultiplicationBlock",
    "MultiONVStateBlock",
    "OrbitalRotationBlock",
    "OrderFindingBlock",
    "ParticleNumberProjectorBlock",
    "PauliBlock",
    "PhaseShiftBlock",
    "PiecewiseLinearStateBlock",
    "PreparesKnownState",
    "ProjectedControlPhaseBlock",
    "QAOABlock",
    "QFTBlock",
    "QPEBlock",
    "QROMBlock",
    "QSPBlock",
    "QSVTBlock",
    "QubitizationBlock",
    "RSPBlock",
    "ReadoutBlock",
    "ReflectionBlock",
    "ResetBlock",
    "SPABlock",
    "SWAPTestBlock",
    "SelectBlock",
    "SimpleBlock",
    "SlaterDeterminantBlock",
    "SparseStateBlock",
    "SpinSquaredProjectorBlock",
    "SyProjectorBlock",
    "SynthesizedStateBlock",
    "SynthesizedTimeEvolutionBlock",
    "SynthesizedUnitaryBlock",
    "SzProjectorBlock",
    "TrotterAnsatzBlock",
    "TrotterBlock",
    "UCCBlock",
    "UPCCDBlock",
    "UniformSuperpositionBlock",
    "XnBlock",
    "QSPAngleFinder",
    "declaring_blocks",
    "prepares_known_state",
]

# VUMPOBrickworkBlock needs quimb (via qarp.operators.VUMPO; the [mps]
# extras): resolved on first access; the __all__ gate keeps `import *` and
# the docs to what is installed.
if _importlib_util.find_spec("quimb") is not None:
    __all__ += ["VUMPOBrickworkBlock"]

__getattr__, __dir__ = _lazy_exports(
    __name__, {"VUMPOBrickworkBlock": ("._primitives.vumpo_brickwork_block", "quimb", "mps")}
)
