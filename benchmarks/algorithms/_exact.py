"""Oracle stack: the numpy statevector reference under the shared optimizer.

Every energy evaluation rebuilds and simulates the bound circuit with the
same independent reference the statevector track is checked against — slow,
but it defines the trajectory every SDK column must reproduce.
"""

from benchmarks.algorithms import _shell, inputs
from benchmarks.statevector import _np

LABEL = "exact"


def prepare(problem: dict):
    n = problem["n_qubits"]
    template, terms = problem["template"], problem["terms"]

    def energy_fn(params) -> float:
        prepared = _np.build_energy(n, inputs.bind(template, params), terms)
        return float(_np.run_energy(n, prepared).real)

    return energy_fn


def run(state, problem: dict):
    return _shell.minimize(state, problem["x0"], problem["budget"], problem["rhobeg"])
