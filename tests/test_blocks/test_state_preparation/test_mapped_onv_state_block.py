def test_explicit_name_and_all_zero_onv():
    from qarp.blocks import MappedONVStateBlock

    named = MappedONVStateBlock([1, 0, 1, 0], name="ref")
    assert named.name == "ref"

    vacuum = MappedONVStateBlock([0, 0, 0, 0]).build()
    assert vacuum.name == "U_0"  # LSB integer of the all-zero bitstring
