"""P2 bridge-cleanup regressions: single built flag, sanctioned mark_built,
order-independent raw-block deepcopy (now in bindings.cpp), and the
``add_wired_child`` wiring helper.
"""

import subprocess
import sys

import qarpx as qx
from qarp.blocks import CompositeBlockBase, ControlledBlock, SimpleBlock


def test_cpp_built_flag_is_the_single_source_of_truth():
    """Setting the C++ flag directly must be visible through the Python
    surface: the two must never desync (a caller may set only the C++ one)."""
    b = SimpleBlock(1)
    assert b.is_built is False
    b.set_built(True)
    assert b.is_built is True
    assert b._built is True


def test_mark_built_declares_external_population():
    src = SimpleBlock(1)
    src.h(0)
    src.build()

    b = SimpleBlock(1)
    b.set_commands(list(src.flatten()))
    b.mark_built()
    assert b.is_built is True
    b.build()  # idempotent — must not double-append
    assert len(list(b.flatten())) == 1


def test_raw_block_deepcopy_without_importing_qarp():
    """__deepcopy__ lives in bindings.cpp now: deepcopy of C++-created
    objects must work in a process that never imports qarp.blocks."""
    code = (
        "import copy, qarpx\n"
        "assert 'qarp.blocks' not in __import__('sys').modules\n"
        "b = qarpx.SimpleBlock(2, 'raw')\n"
        "b.h(0)\n"
        "b.build()\n"
        "c = copy.deepcopy(b)\n"
        "assert c is not b and c.n_qubits == 2 and c.is_built()\n"
        "p = copy.deepcopy(qarpx.Param(0.25))\n"
        "assert p.value() == 0.25\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_add_wired_child_matches_hand_wiring():
    def _remap_child():
        c = SimpleBlock(1, target_qubits=[1])
        c.ry(0, 0.4)
        return c

    def _ctrl_child():
        # Explicit controlisation (§13): the ControlledBlock is 2 qubits wide
        # (control at the lowest index) and is placed on [0, 1].
        inner = SimpleBlock(1)
        inner.rz(0, 0.7)
        inner.build()
        return ControlledBlock(inner, 1, [True], target_qubits=[0, 1])

    wired = CompositeBlockBase(2)
    wired.add_wired_child(_remap_child())
    wired.add_wired_child(_ctrl_child())
    wired.build()

    hand = CompositeBlockBase(2)
    c1 = _remap_child()
    c1.build()
    c1.target_qubits = [1]
    hand.add_child(c1)
    c2 = _ctrl_child()
    c2.build()
    c2.target_qubits = [0, 1]
    hand.add_child(c2)
    hand.build()

    assert list(map(str, wired.flatten())) == list(map(str, hand.flatten()))
    assert len(list(wired.flatten())) == 2


def test_cpp_base_bound_per_class():
    assert SimpleBlock._cpp_base is qx.SimpleBlock
    assert CompositeBlockBase._cpp_base is qx.CompositeBlock
