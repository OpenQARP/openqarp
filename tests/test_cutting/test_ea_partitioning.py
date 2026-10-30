"""Tests for EAPartitioning."""

import pytest

import qarpx as qx
from qarp.cutting import EAPartitioning
from qarp.cutting._internals import _cmd, _CutMarker


def _toy_circuit_cmds(n_qubits: int, cx_pairs: list) -> tuple[list, int]:
    cmds = [_cmd(qx.GateType.H, i) for i in range(n_qubits)]
    for c, t in cx_pairs:
        cmds.append(qx.Command(qx.GateType.CX, c, t))
    return cmds, n_qubits


def test_empty_circuit_raises():
    """EAPartitioning requires at least one entangling gate."""
    cmds = [_cmd(qx.GateType.H, 0), _cmd(qx.GateType.H, 1)]
    with pytest.raises(AssertionError):
        EAPartitioning(cmds, 2, [2, 2])


def test_wrong_max_sizes_raises():
    cmds, n = _toy_circuit_cmds(4, [(0, 1), (1, 2), (2, 3)])
    with pytest.raises(AssertionError):
        EAPartitioning(cmds, n, [2, 1])  # sum < n_qubits


def test_manual_cut_n_cuts():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2), (0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0, 1], [2]])
    # The (1,2) connection is the cut — 2 CX gates on (1,2)
    assert result.n_cuts == 2


def test_manual_cut_subcircuit_count():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    assert result.n_subcircuits == 2


def test_manual_cut_subcircuit_qubit_indices():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    assert result.subcircuit_qubits[0] == [0]
    assert sorted(result.subcircuit_qubits[1]) == [1, 2]


def test_cutter_result_str():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    s = str(result)
    assert "subcircuits" in s.lower() or "cuts" in s.lower()


def test_cut_markers_present_in_custom_commands():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    markers = [c for c in result.custom_commands if isinstance(c, _CutMarker)]
    assert len(markers) == result.n_cuts


def test_manual_setting_invalid_qubits():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    with pytest.raises(AssertionError):
        cutter.cut(manual_setting=[[0, 1], [1, 3]])  # qubit 3 out of range


def test_subcircuits_are_filtered_correctly():
    """After cutting, each subcircuit's commands only reference local qubit indices."""
    cmds, n = _toy_circuit_cmds(4, [(0, 1), (2, 3), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0, 1], [2, 3]])
    for sub_cmds, local_n in result.subcircuits:
        for cmd in sub_cmds:
            if isinstance(cmd, qx.Command):
                assert all(q < local_n for q in cmd.qubits), (
                    f"qubit out of local range in subcircuit (local_n={local_n})"
                )


def _objective_cutter():
    commands, n = _toy_circuit_cmds(4, [(0, 1), (0, 1), (1, 2), (2, 3), (2, 3)])
    return EAPartitioning(commands, n, [2, 2], verbose=False)


def _solution(cutter, removed):
    import numpy as np

    return np.array([0.0 if tuple(sorted(pair)) in removed else 1.0 for pair in cutter.variables])


@pytest.mark.parametrize(
    ("removed", "expected"),
    [
        ((), 40.0),
        (((1, 2),), 1.0),
        (((0, 1),), 40.0),
        (((0, 1), (2, 3)), 4.0),
        (((0, 1), (1, 2), (2, 3)), 5.0),
    ],
)
def test_objective_explicit_graph_oracle(removed, expected):
    """Path weights 2/1/2 give hand-counted cuts and component-size penalties."""
    cutter = _objective_cutter()
    cutter.variables = [(2, 3), (0, 1), (1, 2)]
    assert cutter._make_objective()(_solution(cutter, removed)) == expected


@pytest.mark.parametrize(
    ("values", "expected"), [([1, 1], 20), ([0, 1], 3), ([1, 0], 3), ([0, 0], 3)]
)
def test_objective_both_orientations_count_once(values, expected):
    import numpy as np

    commands, n = _toy_circuit_cmds(2, [(0, 1), (1, 0), (0, 1)])
    cutter = EAPartitioning(commands, n, [1, 1], verbose=False)
    cutter.variables = [(1, 0), (0, 1)]
    assert cutter._make_objective()(np.array(values)) == expected


@pytest.mark.parametrize(
    ("bridge", "expected"), [(0.49, 1), (0.50, 1), (0.51, 40), (1.6, 40), (-0.6, 40)]
)
def test_objective_rounding(bridge, expected):
    cutter = _objective_cutter()
    solution = _solution(cutter, ())
    solution[cutter.variables.index((1, 2))] = bridge
    assert cutter._make_objective()(solution) == expected


def test_objective_fresh_graph_for_reused_array():
    cutter = _objective_cutter()
    objective = cutter._make_objective()
    solution = _solution(cutter, ())
    assert objective(solution) == 40
    solution[:] = _solution(cutter, ((1, 2),))
    assert objective(solution) == 1
    solution[:] = 1
    assert objective(solution) == 40


def test_objective_isolates_unequal_capacity_and_size_penalty():
    commands, n = _toy_circuit_cmds(6, [(0, 1), (1, 2), (2, 3)])
    cutter = EAPartitioning(commands, n, [3, 2, 1], verbose=False)
    objective = cutter._make_objective()
    # Sizes 4/1/1 fail 3/2/1: equal component count, excess size one, penalty 60.
    assert objective(_solution(cutter, ())) == 60
    # Sizes 2/2/1/1 fit 3/2/1 with one removed bridge.
    assert objective(_solution(cutter, ((1, 2),))) == 1


def test_objective_normal_measurements_and_zero_qubit_commands():
    from qarp.cutting._internals import _measure_cmd

    cutter = _objective_cutter()
    cutter.reconstructer.commands.extend([_measure_cmd(0, 0), qx.Command()])
    assert cutter._make_objective()(_solution(cutter, ((1, 2),))) == 1


def test_objective_absent_variables_and_unregistered_retained_edge():
    cutter = _objective_cutter()
    cutter.variables = [(0, 1), (1, 2), (10, 11)]
    # (2,3) is absent from variables but still contributes its edge; absent (10,11) costs zero.
    assert cutter._make_objective()(_solution(cutter, ((1, 2), (10, 11)))) == 1


@pytest.mark.parametrize(
    "kind",
    [
        "wide",
        "marker",
        "measure",
        "variables_tuple",
        "commands_tuple",
        "length",
        "pair_shape",
        "pair_type",
        "pair_subclass",
        "int_subclass",
        "n_vars_type",
    ],
)
def test_objective_fallback_delegates_without_early_errors(monkeypatch, kind):
    import numpy as np

    from qarp.cutting._ea_partitioning import _prepare_objective_connectivity

    cutter = _objective_cutter()
    if kind == "wide":
        cutter.reconstructer.commands.append(_wide_command())
    elif kind == "marker":
        cutter.reconstructer.commands.append(_CutMarker("cut_1", [0, 1], "CX", []))
    elif kind == "measure":
        command = qx.Command(qx.GateType.CX, 0, 1)
        command.gate = qx.GateType.Measure
        cutter.reconstructer.commands.append(command)
    elif kind == "variables_tuple":
        cutter.variables = tuple(cutter.variables)
    elif kind == "commands_tuple":
        cutter.reconstructer.commands = tuple(cutter.reconstructer.commands)
    elif kind == "length":
        cutter.variables.append((8, 9))
    elif kind == "pair_shape":
        cutter.variables[0] = (0,)
    elif kind == "pair_type":
        cutter.variables[0] = (0.0, 1)
    elif kind == "pair_subclass":

        class HostilePair(list):
            def __iter__(self):
                raise RuntimeError("pair iteration")

        cutter.variables[0] = HostilePair([0, 1])
    elif kind == "int_subclass":

        class HostileInt(int):
            def __hash__(self):
                raise RuntimeError("hash")

        cutter.variables[0] = (HostileInt(0), 1)
    else:
        cutter.n_vars = np.int64(cutter.n_vars)
    assert (
        _prepare_objective_connectivity(
            cutter.reconstructer.commands, cutter.variables, cutter.n_vars
        )
        is None
    )
    objective = cutter._make_objective()
    calls = []
    original = cutter.reconstructer.reconstruct_delete_gates

    def reconstruct(deletions):
        calls.append(deletions)
        return original(deletions)

    monkeypatch.setattr(cutter.reconstructer, "reconstruct_delete_gates", reconstruct)
    solution = np.ones(cutter.n_vars)
    if kind == "wide":
        with pytest.raises(RuntimeError, match="More than 2q"):
            objective(solution)
    else:
        assert objective(solution) == 40
    assert calls == [[]]


def test_objective_custom_pair_error_is_deferred():
    import numpy as np

    class HostilePair(list):
        def __iter__(self):
            raise RuntimeError("pair iteration")

    cutter = _objective_cutter()
    cutter.variables[0] = HostilePair([0, 1])
    objective = cutter._make_objective()
    with pytest.raises(RuntimeError, match="pair iteration"):
        objective(np.zeros(cutter.n_vars))


def test_automatic_cut_prepares_once_and_rebuilds_between_cuts(monkeypatch):
    from scipy.optimize import OptimizeResult

    from qarp.cutting import _ea_partitioning as module

    cutter = _objective_cutter()
    factory_calls = []
    reconstruction_calls = []
    original_factory = cutter._make_objective
    original_reconstruct = cutter.reconstructer.reconstruct_delete_gates

    def factory():
        factory_calls.append(True)
        return original_factory()

    def reconstruct(deletions):
        reconstruction_calls.append(deletions)
        return original_reconstruct(deletions)

    expected_cut_count = 1

    def optimizer(objective, bounds, **kwargs):
        assert bounds == [(0, 1)] * 3
        assert kwargs == dict(
            strategy="best1bin",
            maxiter=15,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
        )
        assert objective(_solution(cutter, ())) == 40
        solution = _solution(cutter, ((1, 2),))
        assert objective(solution) == expected_cut_count
        assert len(reconstruction_calls) == len(factory_calls) - 1
        return OptimizeResult(x=solution)

    monkeypatch.setattr(cutter, "_make_objective", factory)
    monkeypatch.setattr(cutter.reconstructer, "reconstruct_delete_gates", reconstruct)
    monkeypatch.setattr(module, "differential_evolution", optimizer)
    result = cutter.cut()
    assert result.n_cuts == 1
    assert result.subcircuit_qubits == [[0, 1], [2, 3]]
    cutter.reconstructer.commands.append(qx.Command(qx.GateType.CX, 1, 2))
    expected_cut_count = 2
    result = cutter.cut()
    assert result.n_cuts == 2
    assert result.subcircuit_qubits == [[0, 1], [2, 3]]
    assert len(factory_calls) == len(reconstruction_calls) == 2


def test_manual_cut_does_not_prepare_objective(monkeypatch):
    cutter = _objective_cutter()

    def forbidden():
        pytest.fail("manual cut prepared automatic objective")

    monkeypatch.setattr(cutter, "_make_objective", forbidden)
    assert cutter.cut(manual_setting=[[0, 1], [2, 3]]).n_cuts == 1


def test_automatic_wide_error_occurs_inside_optimizer(monkeypatch):
    import numpy as np

    from qarp.cutting import _ea_partitioning as module

    cutter = _objective_cutter()
    cutter.reconstructer.commands.append(_wide_command())
    events = []

    def optimizer(objective, *args, **kwargs):
        events.append("optimizer")
        return objective(np.ones(cutter.n_vars))

    monkeypatch.setattr(module, "differential_evolution", optimizer)
    with pytest.raises(RuntimeError, match="More than 2q"):
        cutter.cut()
    assert events == ["optimizer"]


def test_real_scipy_automatic_cut(monkeypatch):
    from qarp.cutting import _ea_partitioning as module

    original = module.differential_evolution

    def seeded(*args, **kwargs):
        return original(*args, **kwargs, seed=817)

    monkeypatch.setattr(module, "differential_evolution", seeded)
    commands, n = _toy_circuit_cmds(2, [(0, 1)])
    result = EAPartitioning(commands, n, [1, 1], verbose=False).cut()
    assert result.n_cuts == 1
    assert result.subcircuit_qubits == [[0], [1]]


def _wide_command():
    block = qx.SimpleBlock(3, "wide")
    block.ccx(0, 1, 2)
    block.set_built(True)
    return block.flatten()[0]
