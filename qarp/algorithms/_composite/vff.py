from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

import qarpx as qx
from qarp.blocks import (
    AnyBlock,
    CompositeBlock,
    LayerBlock,
    SimpleBlock,
    SynthesizedTimeEvolutionBlock,
    TrotterBlock,
)
from qarp.operators import QubitOperator
from qarp.optimizers import AdamOptimizer, Optimizer

from .. import PrimitiveAlgorithm, StateVector
from . import CompositeAlgorithm


class VFF(CompositeAlgorithm):
    def __init__(
        self,
        operator: Union[QubitOperator, AnyBlock],
        ansatz_block: AnyBlock,
        use_trotter: bool = False,
        t_time: float = 1.0,
        t_steps: int = 1,
        t_order: int = 1,
        initial_parameters: Optional[np.typing.NDArray[np.float64]] = None,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = True,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine=None,
    ):

        if optimizer is None:
            optimizer = AdamOptimizer()
        if primitive is None:
            primitive = StateVector()
        super().__init__(engine=engine, primitive=primitive)

        self.operator = operator
        self.ansatz_block = ansatz_block.build()
        self.use_trotter = use_trotter
        self.t_steps = t_steps
        self.t_time = t_time
        self.t_order = t_order
        self.optimizer = optimizer
        self.verbose = verbose
        self.primitive = primitive
        self.primitives: List[PrimitiveAlgorithm] = []

        self.n_qubits = self.ansatz_block.n_qubits
        if not self.ansatz_block.symbols:
            raise ValueError(
                "Cannot build VFF: ansatz_block has no parameters to optimize "
                "(symbols is None or empty). Pass a parameterized ansatz, "
                "not a bare state-preparation block."
            )
        if self.n_qubits is None:
            raise RuntimeError("Number of qubits not defined")
        # Positional surface is ansatz symbols then the n_qubits diagonal Rz
        # symbols (see ``_all_symbols``); D is only built in ``build()``, so
        # its count is taken from n_qubits here.
        n_parameters = len(self.ansatz_block.symbols) + self.n_qubits
        if initial_parameters is None:
            self.initial_parameters = np.random.rand(n_parameters)
        else:
            vector = np.asarray(initial_parameters, dtype=float)
            if vector.shape != (n_parameters,):
                raise ValueError(
                    f"initial_parameters has {vector.size} values for "
                    f"{n_parameters} symbols ({len(self.ansatz_block.symbols)} "
                    f"ansatz + {self.n_qubits} diagonal); pass one value per symbol."
                )
            self.initial_parameters = vector

    def build(self):
        if isinstance(self.operator, qx.Block):
            U_deltaT = self.operator.build()
        else:
            if self.use_trotter:
                trotterisation = TrotterBlock(
                    self.n_qubits, self.operator, self.t_steps, self.t_time, self.t_order
                )
                U_deltaT = trotterisation.build()
            else:
                U_deltaT = SynthesizedTimeEvolutionBlock(
                    self.operator, self.n_qubits, -self.t_time
                ).build()

        # ``qx.Param.linear(t_time, "theta_i")`` carries the linear
        # coefficient natively and substitutes correctly under
        # ``Block::set_symbols`` (the C++ ``replace_symbols`` preserves the
        # coefficient on rename); ``t_time * Symbol(...)`` would coerce to the
        # same Param through ``as_param`` but says less.
        thetas = [qx.Param.linear(self.t_time, f"theta_{i}") for i in range(self.n_qubits)]
        D = LayerBlock(qx.GateType.Rz, self.n_qubits, parameters=thetas)
        D = D.build()
        self.D = D
        V = CompositeBlock(
            [self.ansatz_block.dagger(), D, self.ansatz_block],
            target_qubits=[self.n_qubits + i for i in range(self.n_qubits)],
        )

        Pre_B = SimpleBlock(2 * self.n_qubits, name="Pre")
        for q in range(self.n_qubits):
            Pre_B.h(q)
            Pre_B.cx(q, q + self.n_qubits)

        for q in range(self.n_qubits):
            q2 = q + self.n_qubits
            Post_B = SimpleBlock(2 * self.n_qubits, name="Post")
            Post_B.cx(q, self.n_qubits + q)
            Post_B.h(q)

            ket = CompositeBlock([Pre_B, U_deltaT, V, Post_B])
            ket.build()

            newPrimitive = deepcopy(self.primitive)
            newPrimitive.bra = ket
            newPrimitive.operator = (
                QubitOperator(f"Z{q} Z{q2}") / 4
                + QubitOperator(f"Z{q}") / 4
                + QubitOperator(f"Z{q2}") / 4
                + 1 / 4
            )
            newPrimitive.ket = ket
            self.primitives.append(newPrimitive)

        self.engine.build(self.primitives)
        return self

    @property
    def _all_symbols(self):
        """Positional parameter surface: ansatz symbols first, then the
        diagonal layer's — a documented concatenation of two canonical
        orders, NOT globally sorted.  Available after build()."""
        return tuple(self.ansatz_block.symbols) + tuple(self.D.symbols)

    @property
    def optimal_parameters(self):
        """Optimized parameters keyed by symbol — the order-proof surface."""
        if getattr(self, "result", None) is None:
            raise RuntimeError("No optimization result — call run() first.")
        return dict(zip(self._all_symbols, self.result.x, strict=True))

    def objective(self, x):
        parameter_map = dict(zip(self._all_symbols, x, strict=True))
        vals = np.real(self.engine.run(parameter_map))
        val = 1 - sum(vals) / self.n_qubits
        self._last_objective = val
        return val

    def run(self):
        objective_function = lambda x: self.objective(x)

        def callback(intermediate_result):
            # Reads the last-evaluation cache — no extra quantum work.
            print(f"Iteration: {callback.iter} \t Loss value: {self._last_objective}")
            callback.iter += 1

        callback.iter = 0

        if self.verbose:
            res = self.optimizer.minimize(
                objective_function, self.initial_parameters, callback=callback
            )
        else:
            res = self.optimizer.minimize(objective_function, self.initial_parameters)

        self.result = res
        return res
