import numpy as np
import pytest

from qarp.blocks import ControlledBlock, selected_pair_unitary
from tests.selected_pair_oracles import pair_unitary


@pytest.mark.parametrize("angle", [0.0, 0.31, -0.45, np.pi / 2, np.pi])
def test_pair_rotation_absolute_phase_and_control(angle):
    rotations = ((0, 1, angle),)
    block = selected_pair_unitary(4, rotations)
    expected = pair_unitary(4, rotations)
    np.testing.assert_allclose(block.unitary_matrix(), expected, atol=1e-12)
    controlled = np.eye(32, dtype=complex)
    controlled[1::2, 1::2] = expected
    np.testing.assert_allclose(
        ControlledBlock(block, 1).build().unitary_matrix(), controlled, atol=1e-12
    )


def test_noncommuting_order_and_empty_preparation():
    rotations = ((0, 1, 0.31), (1, 2, -0.27))
    block = selected_pair_unitary(6, rotations)
    np.testing.assert_allclose(block.unitary_matrix(), pair_unitary(6, rotations), atol=1e-12)
    np.testing.assert_array_equal(selected_pair_unitary(4, ()).unitary_matrix(), np.eye(16))
    assert block.symbols == ()
