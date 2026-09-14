"""CompositeAlgorithm shared machinery: the `_minimize` wrapper, the
`_log_iteration` formatter, the run() contract and the amplitude gate.

MPI lockstep is gone (pipeline_hardening_plan.md P1.22): a multi-rank
launcher is refused at construction, pinned below through the environment
alone — no mpirun, no mpi4py.
"""

import numpy as np
import pytest

from qarp.algorithms import CompositeAlgorithm
from qarp.engines import QarpEngine
from qarp.optimizers import ScipyOptimizer


class _MinimalComposite(CompositeAlgorithm):
    """Bare concrete subclass: the machinery under test lives in the base."""

    def build(self):
        return self

    def run(self):
        return None


def test_engine_defaults_to_qarp_engine():
    alg = _MinimalComposite(primitive=None)
    assert isinstance(alg.engine, QarpEngine)


# ── MPI lockstep removed (pipeline_hardening_plan.md P1.22) ───────────────


def test_multi_rank_launcher_is_refused_at_construction(monkeypatch):
    from qarp import MPIConfig
    from qarp.errors import CapabilityError

    for var in MPIConfig._MPI_ENV_VARS + MPIConfig._SIZE_ENV_VARS + ["QARP_DISABLE_MPI"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    with pytest.raises(CapabilityError, match="MPI parallelism is not implemented"):
        _MinimalComposite(primitive=None)
    # A single rank under a launcher is just a process; QARP_DISABLE_MPI silences detection.
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    _MinimalComposite(primitive=None)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("QARP_DISABLE_MPI", "1")
    _MinimalComposite(primitive=None)


def test_minimize_reaches_the_analytic_minimum():
    alg = _MinimalComposite(primitive=None)
    result = alg._minimize(
        lambda x: float((x[0] - 1.0) ** 2),
        [0.0],
        ScipyOptimizer("CG"),
        gradient=lambda x: np.array([2.0 * (x[0] - 1.0)]),
        success_label=None,
    )
    assert abs(result.x[0] - 1.0) < 1e-6


@pytest.mark.parametrize(
    "suppress, expected",
    [(False, "VQE-ish minimization finished successfully\n"), (True, "")],
)
def test_minimize_success_line(capsys, suppress, expected):
    alg = _MinimalComposite(primitive=None)
    alg._minimize(
        lambda x: float(x[0] ** 2),
        [0.1],
        ScipyOptimizer("COBYLA"),
        success_label="VQE-ish",
        suppress_success=suppress,
    )
    assert capsys.readouterr().out == expected


def test_minimize_failure_line_prints_despite_suppress(capsys):
    # maxiter=3 (COBYLA's minimum legal budget) cannot converge: the failure
    # line must print even when success messages are suppressed.
    alg = _MinimalComposite(primitive=None)
    result = alg._minimize(
        lambda x: float((x[0] - 1.0) ** 2),
        [0.0],
        ScipyOptimizer("COBYLA", options={"maxiter": 3}),
        success_label="VQE-ish",
        suppress_success=True,
    )
    assert not result.success
    assert capsys.readouterr().out == "VQE-ish minimization did NOT finish successfully\n"


def test_log_iteration_header_default_label(capsys):
    alg = _MinimalComposite(primitive=None)
    alg._log_iteration(0, 0.0, 0.0, 0.0)
    assert capsys.readouterr().out == (
        "_MinimalComposite Run:\n\t\tIteration\t\tEnergy\t\t\t\t  dE\t\t\t\t|step|\n"
    )


def test_log_iteration_header_optional_columns_and_label(capsys):
    alg = _MinimalComposite(primitive=None)
    alg._log_iteration(0, 0.0, 0.0, 0.0, gradnorm=1.0, cost="c", label="Macroiteration 0")
    assert capsys.readouterr().out == (
        "Macroiteration 0 Run:\n"
        "\t\tIteration\t\tEnergy\t\t\t\t  dE\t\t\t\t|step|\t\t\t\t|grad|\t\t\t\tCost\n"
    )


def test_log_iteration_row_negative_alignment(capsys):
    # Negative values print sign-flush; non-negatives get a leading space.
    alg = _MinimalComposite(primitive=None)
    alg._log_iteration(3, -1.5, -0.25, 0.5)
    assert capsys.readouterr().out == (
        "\t\t\t3\t\t-1.5000000000\t\t-0.2500000000\t\t 0.5000000000\n"
    )


def test_log_iteration_row_gradnorm_and_cost(capsys):
    alg = _MinimalComposite(primitive=None)
    alg._log_iteration(1, 2.0, 0.0, 0.0, gradnorm=-0.5, cost="raw-cost")
    assert capsys.readouterr().out == (
        "\t\t\t1\t\t 2.0000000000\t\t 0.0000000000\t\t 0.0000000000\t\t-0.5000000000\t\traw-cost\n"
    )


# ── run() contract (pipeline_hardening_plan.md P1.4) ──────────────────────


def test_run_contract_takes_no_arguments():
    """The abstract ``run`` is ``run(self)``; every exported composite honours
    it with no required positional argument, so callers can be generic."""
    import inspect

    from qarp import algorithms

    assert list(inspect.signature(CompositeAlgorithm.run).parameters) == ["self"]
    for name in algorithms.__all__:
        cls = getattr(algorithms, name)
        if not (isinstance(cls, type) and issubclass(cls, CompositeAlgorithm)):
            continue
        if cls is CompositeAlgorithm:
            continue
        params = list(inspect.signature(cls.run).parameters.values())[1:]
        required = [p for p in params if p.default is inspect.Parameter.empty]
        assert not required, f"{name}.run requires positional arguments {required}"
        positional = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        assert not positional, f"{name}.run takes positional arguments {positional}"


# ── statevector fast-path gate (pipeline_hardening_plan.md P1.3) ──────────
#
# Helper-level coverage with stub engines; the algorithm-level rows (a noisy
# or routed engine raising through AdaptVQE / QSE / QITE / MonteCarlo) land
# with the call sites.


def _stub_composite(primitive, engine):
    return _MinimalComposite(primitive=primitive, engine=engine)


def test_amplitude_gate_default_engine_hands_back_its_simulator():
    from qarp.algorithms import StateVector

    engine = QarpEngine()
    alg = _stub_composite(StateVector(), engine)
    assert alg._amplitudes_available()
    assert alg._amplitude_simulator(1) is engine._sim


def test_amplitude_gate_rejects_counts_primitive():
    from qarp.algorithms import Sampler
    from qarp.errors import CapabilityError

    alg = _stub_composite(Sampler(), QarpEngine())
    assert not alg._amplitudes_available()
    with pytest.raises(CapabilityError, match="consumes counts"):
        alg._amplitude_simulator(1)


def test_amplitude_gate_noise_toggle_is_revalidated():
    """§14: an enabled noise model closes the gate; disabling it reopens it
    on the same engine without a rebuild."""
    import qarpx as qx
    from qarp.algorithms import StateVector
    from qarp.devices import NoiseModel
    from qarp.errors import CapabilityError

    engine = QarpEngine(n_qubits=2, noise_model=NoiseModel.depolarizing(0.01, [qx.GateType.CX]))
    alg = _stub_composite(StateVector(), engine)
    with pytest.raises(CapabilityError, match="does not provide amplitudes"):
        alg._amplitude_simulator(1)
    engine.noise_model.enabled = False
    assert alg._amplitude_simulator(1) is engine._sim


def test_amplitude_gate_rejects_routed_engine():
    import qarpx as qx
    from qarp.algorithms import StateVector
    from qarp.errors import CapabilityError

    engine = QarpEngine(n_qubits=4, architecture=qx.nearest_neighbour_architecture(2, 2))
    alg = _stub_composite(StateVector(), engine)
    with pytest.raises(CapabilityError, match="routes"):
        alg._amplitude_simulator(1)


def test_amplitude_gate_rejects_engine_without_host_statevector():
    """provides_amplitudes is a promise about primitives; the fast path also
    needs a simulator exposing ``statevector`` — an engine with none (or a
    GPU simulator lacking the host API) must raise, not AttributeError."""
    from qarp.algorithms import StateVector
    from qarp.engines import Engine
    from qarp.errors import CapabilityError

    class _NoSimEngine(Engine):
        def _sweep(self, *args, **kwargs):
            raise NotImplementedError

    alg = _stub_composite(StateVector(), _NoSimEngine())
    assert alg._amplitudes_available()
    with pytest.raises(CapabilityError, match="host statevector"):
        alg._amplitude_simulator(1)

    class _OpaqueSim:
        def run(self, *a, **k):
            raise NotImplementedError

    class _OpaqueSimEngine(_NoSimEngine):
        def __init__(self):
            super().__init__()
            self._sim = _OpaqueSim()

    with pytest.raises(CapabilityError, match="host statevector"):
        _stub_composite(StateVector(), _OpaqueSimEngine())._amplitude_simulator(1)


def test_amplitude_gate_goes_through_the_engine_hook():
    """The gate reads ``Engine._host_statevector_simulator(n_qubits)``, never
    ``engine._sim``: an engine that caps its host transfer refuses through
    the hook with the width the algorithm asked for."""
    from qarp.algorithms import StateVector
    from qarp.engines import Engine
    from qarp.errors import CapabilityError

    class _Sim:
        def statevector(self, cmds, n):
            raise NotImplementedError

    class _CappedEngine(Engine):
        cap = 3

        def __init__(self):
            super().__init__()
            self._sim = _Sim()
            self.asked: list[int] = []

        def _sweep(self, *args, **kwargs):
            raise NotImplementedError

        def _host_statevector_simulator(self, n_qubits):
            self.asked.append(n_qubits)
            if n_qubits > self.cap:
                raise CapabilityError(f"capped at {self.cap}")
            return self._sim

    engine = _CappedEngine()
    alg = _stub_composite(StateVector(), engine)
    assert alg._amplitude_simulator(3) is engine._sim
    with pytest.raises(CapabilityError, match="capped at 3"):
        alg._amplitude_simulator(4)
    assert engine.asked == [3, 4]


def test_cudaq_engine_hook_applies_the_host_cap():
    """``CudaqEngine._host_statevector_simulator`` is the fast path's way to
    the GPU simulator and applies ``_STATEVECTOR_HOST_QUBIT_CAP`` (backlog
    F4).  Constructed without ``__init__`` so no CUDA-Q build is needed."""
    from qarp.engines._cudaq_engine import _STATEVECTOR_HOST_QUBIT_CAP, CudaqEngine
    from qarp.errors import CapabilityError

    engine = CudaqEngine.__new__(CudaqEngine)
    engine._sim = object()
    assert engine._host_statevector_simulator(_STATEVECTOR_HOST_QUBIT_CAP) is engine._sim
    with pytest.raises(CapabilityError, match="statevector fast path"):
        engine._host_statevector_simulator(_STATEVECTOR_HOST_QUBIT_CAP + 1)
