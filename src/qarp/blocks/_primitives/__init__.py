from .trotter_block import TrotterBlock, TrotterAnsatzBlock
from .qft_block import QFTBlock
from .rsp_block import RSPBlock
from .a_gate_block import AGateBlock
from .linear_entangling_block import LinearEntanglingBlock
from .hn_block import HnBlock
from .xn_block import XnBlock
from .brickwork_entangling_block import BrickworkEntanglingBlock
from .brickwork_pce_block import BrickworkPCEBlock
from .hea_block import HEABlock
from .qpe_block import QPEBlock
from .dos_qpe_block import DOSQPEBlock
from .cost_operator_block import CostOperatorBlock
from .mixed_operator_block import MixedOperatorBlock
from .givens_block import GivensBlock
from .orbital_rotation_block import OrbitalRotationBlock
from .upccd_block import UPCCDBlock
from .block_encoding_block import BlockEncodingBlock
from .pauli_block import PauliBlock
from .select_block import SelectBlock
from .qsp_block import QSPBlock, QSPAngleFinder
from .qsvt_block import QSVTBlock
from .phase_shift_block import PhaseShiftBlock
from .projected_control_phase_block import ProjectedControlPhaseBlock
from .modular_multiplication_block import ModularMultiplicationBlock
from .order_finding_block import OrderFindingBlock
from .qrom_block import QROMBlock

from .haar_random_block import HaarRandomBlock
from ..._lazy import lazy_exports as _lazy_exports


from .swap_test_block import SWAPTestBlock
from .hadamard_test_block import HadamardTestBlock
from .interferometric_state_block import (
    InterferometricMeasurementBlock,
    InterferometricStateBlock,
)
from .ucc_block import UCCBlock
from .qaoa_block import QAOABlock
from .spa_block import SPABlock
from .readout_block import ReadoutBlock
from .identity_block import IdentityBlock
from .layer_block import LayerBlock
from .synthesized_time_evolution_block import SynthesizedTimeEvolutionBlock
from .synthesized_unitary_block import SynthesizedUnitaryBlock
from .reflection_block import ReflectionBlock
from .amplitude_amplification_block import AmplitudeAmplificationBlock
from .amplitude_estimation_block import AmplitudeEstimationBlock
from .grover_block import GroverBlock
from .qubitization_block import QubitizationBlock
from .projector_blocks import (
    ParticleNumberProjectorBlock,
    SzProjectorBlock,
    SyProjectorBlock,
    SpinSquaredProjectorBlock,
)

# VUMPOBrickworkBlock needs quimb (the [mps] extras): resolved on first
# access, see qarp.blocks.
__getattr__, __dir__ = _lazy_exports(
    __name__, {"VUMPOBrickworkBlock": (".vumpo_brickwork_block", "quimb", "mps")}
)
