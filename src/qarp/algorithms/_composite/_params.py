"""Shared parameter-vector coercion for variational algorithms."""

from typing import Iterable, Mapping, Sequence, Union

import numpy as np
from sympy import Symbol


def resolve_initial_parameters(
    symbols: Sequence[Symbol],
    initial_parameters: Union[Mapping, Iterable[float]],
) -> np.typing.NDArray[np.float64]:
    """Coerce user-supplied initial parameters to a vector aligned to ``symbols``.

    A mapping (Symbol or str keys) binds by name — the order-proof form.  A
    sequence is positional against ``symbols`` (the block's canonical order)
    and must match its length exactly.
    """
    if isinstance(initial_parameters, Mapping):
        values = []
        missing = []
        for s in symbols:
            if s in initial_parameters:
                values.append(initial_parameters[s])
            elif str(s) in initial_parameters:
                values.append(initial_parameters[str(s)])
            else:
                missing.append(str(s))
        if missing:
            raise ValueError(f"initial_parameters mapping is missing symbols: {missing}")
        extra = {str(k) for k in initial_parameters} - {str(s) for s in symbols}
        if extra:
            raise ValueError(f"initial_parameters mapping has unknown symbols: {sorted(extra)}")
        return np.array(values, dtype=float)

    vector = np.array(list(initial_parameters), dtype=float)
    if vector.shape != (len(symbols),):
        raise ValueError(
            f"initial_parameters has {vector.size} values for {len(symbols)} symbols; "
            "positional vectors align to the block's canonical `symbols` order — "
            "pass a {symbol: value} mapping to bind by name instead."
        )
    return vector
