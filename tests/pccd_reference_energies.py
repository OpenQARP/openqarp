"""Pinned classical-pCCD references for the UPCCD oracle tests.

Provenance: computed once from a pyscfad-based classical pCCD solver that is
no longer in the tree.  Geometry and
SCF settings match the fixtures in
``tests/test_blocks/test_primitives/test_upccd_block.py``: sto-3g,
``symmetry=True``, ``conv_tol=1e-10``; ``PCCD(mf).run()`` at its defaults
(t2 solved in float32 — the values below are their exact float64 renderings).

The pCCD energy is the independent oracle for the UPCCD block: a convention
slip in ``GivensBlock`` (half-turn vs radian) or ``UPCCDBlock`` (Givens-angle
sign) moves the contracted energy away from these values.  The t2 amplitude
tables are inputs, not oracles — they parameterize the block exactly as the
live PCCD run used to.
"""

# H 0 0 0; H 0 0 0.735   (e_hf = -1.116998996754)
E_PCCD_H2_STO3G = -1.137305974960327
T2_PCCD_H2_STO3G = [
    [-0.11223620176315308],
]

# Li 0 0 0; H 0 0 1.3    (e_hf = -7.851953857957)
E_PCCD_LIH_STO3G = -7.865851879119873
T2_PCCD_LIH_STO3G = [
    [
        -0.0039140819571912289,
        -0.0016621322138234973,
        -0.0016621323302388191,
        -0.00032465212279930711,
    ],
    [-0.0073291906155645847, -0.032817739993333817, -0.032817739993333817, -0.098871909081935883],
]
