"""`resolve_initial_parameters`: name-bound mappings (Symbol or str keys)
and the loud failures for missing, unknown, and wrong-shape inputs.
"""

import numpy as np
import pytest
from sympy import Symbol

from qarp.algorithms._composite._params import resolve_initial_parameters

_A, _B = Symbol("a"), Symbol("b")


def test_symbol_key_mapping_binds_by_name():
    out = resolve_initial_parameters((_A, _B), {_B: 2.0, _A: 1.0})
    assert np.array_equal(out, [1.0, 2.0])


def test_str_key_mapping_binds_by_name():
    out = resolve_initial_parameters((_A, _B), {"b": 2.0, "a": 1.0})
    assert np.array_equal(out, [1.0, 2.0])


def test_mixed_key_mapping_binds_by_name():
    out = resolve_initial_parameters((_A, _B), {_A: 1.0, "b": 2.0})
    assert np.array_equal(out, [1.0, 2.0])


def test_missing_symbol_raises():
    with pytest.raises(ValueError, match=r"missing symbols: \['b'\]"):
        resolve_initial_parameters((_A, _B), {_A: 1.0})


def test_unknown_symbol_raises():
    with pytest.raises(ValueError, match=r"unknown symbols: \['c'\]"):
        resolve_initial_parameters((_A, _B), {_A: 1.0, _B: 2.0, Symbol("c"): 3.0})


def test_wrong_length_vector_raises():
    with pytest.raises(ValueError, match="values for 2 symbols"):
        resolve_initial_parameters((_A, _B), [1.0])


def test_exact_length_vector_is_positional():
    out = resolve_initial_parameters((_A, _B), (1.0, 2.0))
    assert np.array_equal(out, [1.0, 2.0])
