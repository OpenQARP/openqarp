"""Build-time capability validation (``Engine._validate_primitive`` /
``_validate_flat_commands``).

Pins the D3 behavior change of the sampler/statevec refactor: engines that
cannot provide exact amplitudes (noise active) REJECT
amplitude-consuming primitives at ``build()`` with ``qarp.errors.CapabilityError``
instead of silently returning noiseless numbers; ditto amplitude consumption
over true mid-circuit measurement.  Terminal measure-all layers and sampling
primitives are unaffected.
"""

import pytest

import qarpx as qx
from qarp.algorithms import Sampler, StateVector, Target
from qarp.blocks import SimpleBlock
from qarp.devices import Device, NoiseModel, get_nearest_neighbour_architecture
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator


def _h():
    """Factory, not a module global: a module-level QubitOperator stays alive
    to interpreter shutdown and nanobind reports it as leaked (see AGENTS.md)."""
    return QubitOperator("Z0")


def _plus_ket():
    b = SimpleBlock(1, name="plus")
    b.h(0)
    b.build()
    return b


def _mcm_ket():
    """H, measure, H — the measure is followed by work on the same qubit."""
    b = SimpleBlock(1, name="mcm")
    b.h(0)
    b.measure(0, 0)
    b.h(0)
    b.build()
    return b


# ── noise × amplitude consumption ────────────────────────────────────────


def test_noisy_qarp_engine_rejects_amplitude_primitive():
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05))
    sv = StateVector(ket=_plus_ket(), operator=_h())
    with pytest.raises(CapabilityError, match="disable the noise model"):
        eng.build([sv])


def test_disabled_noise_model_restores_amplitudes():
    """qx.Device copies the model at construction: toggle via engine.noise_model."""
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05))
    eng.noise_model.enabled = False
    sv = StateVector(ket=_plus_ket(), operator=_h())
    eng.build([sv])
    assert abs(eng.run()[0]) < 1e-10  # ⟨+|Z|+⟩ = 0, exact


def test_noise_toggle_after_build_caught_on_rebuild():
    """The provides_amplitudes property is dynamic — re-validated on reuse."""
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05))
    eng.noise_model.enabled = False
    sv = StateVector(ket=_plus_ket(), operator=_h())
    eng.build([sv])
    eng.noise_model.enabled = True
    with pytest.raises(CapabilityError, match="exact amplitudes"):
        eng.build([sv], rebuild=False)


def test_noisy_engine_accepts_sampling_primitive():
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05), n_shots=100, seed=0)
    samp = Sampler(ket=_plus_ket(), n_shots=100)
    eng.build([samp])
    assert isinstance(eng.run()[0], dict)


# ── mid-circuit measurement × amplitude consumption ──────────────────────


def test_mcm_ket_rejects_amplitude_primitive():
    sv = StateVector(ket=_mcm_ket(), operator=_h())
    with pytest.raises(CapabilityError, match="mid-circuit"):
        QarpEngine().build([sv])


def test_mcm_ket_fine_for_sampling():
    """QarpSimulator handles true MCM via per-shot trajectories."""
    eng = QarpEngine(n_shots=100, seed=0)
    samp = Sampler(ket=_mcm_ket(), n_shots=100)
    eng.build([samp])
    assert isinstance(eng.run()[0], dict)


def test_terminal_measure_fine_for_amplitude_primitive():
    """A measure-all layer at the END is sampling, not MCM — must pass."""
    b = SimpleBlock(1, name="terminal")
    b.h(0)
    b.measure(0, 0)
    b.build()
    sv = StateVector(ket=b, operator=_h())
    QarpEngine().build([sv])  # must not raise


# ── declared-target membership ───────────────────────────────────────────


def test_target_outside_supported_targets_rejected():
    class _WrongTarget(Sampler):
        supported_targets = frozenset({Target.EXPECTATION_VALUE})

    with pytest.raises(CapabilityError, match="supported_targets"):
        QarpEngine().build([_WrongTarget(ket=_plus_ket(), n_shots=10)])


def test_noise_composition_mutation_reaches_simulator():
    """Adding channels while ``enabled`` stays True must reach the simulator
    on the next run — the old sync rebuilt only on ``enabled`` flips and ran
    stale physics silently."""
    b = SimpleBlock(1)
    b.x(0)
    s = Sampler(ket=b, n_shots=2000)
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(1e-12), seed=3)
    eng.build([s])
    baseline = dict(eng.run()[0])
    assert baseline.get((1,), 0.0) > 0.99

    nm = eng.noise_model  # live C++ model reference
    nm += NoiseModel.bit_flip(0.5).inner
    mutated = dict(eng.run()[0])
    assert mutated.get((1,), 0.0) < 0.9


def test_run_revalidates_after_noise_toggle():
    """build → enable noise → run() must raise the friendly CapabilityError,
    not fall through to the opaque C++ statevector backstop."""
    sv = StateVector(ket=_plus_ket(), operator=QubitOperator("Z0"))
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05))
    eng.noise_model.enabled = False
    eng.build([sv])
    eng.noise_model.enabled = True
    with pytest.raises(CapabilityError, match="amplitudes"):
        eng.run()


def test_routed_device_rejects_amplitude_primitive():
    """Routing permutes the register; amplitude contraction is in logical
    order — was a silent wrong number, must reject."""
    b = SimpleBlock(4, name="routed")
    b.h(0)
    b.cx(0, 3)  # not adjacent on the 2x2 grid → forces routing
    b.build()
    sv = StateVector(ket=b, operator=QubitOperator("Z0"))
    arch = get_nearest_neighbour_architecture(2, 2)
    eng = QarpEngine(device=Device(4, architecture=arch))
    with pytest.raises(CapabilityError, match="routing permuted"):
        eng.build([sv])


# ── Uninitialised classical conditions ────────────────────────────────────
#
# Reading a cbit no Measure wrote is *defined* (the register is
# zero-initialised, so the condition reads false) — the simulator accepts it,
# because randomized property tests and deliberately-dead branches are
# legitimate.  At engine level the program is user-authored and complete, so
# it is warned about rather than silently executed.


def _feedforward(alias: bool):
    """H;Measure→c0 then `if c0: X(1)`, composed as siblings.

    ``alias=True`` points the condition at the measured cbit.  ``alias=False``
    pins it to cbit 1, which nothing writes — the offset a sibling composition
    used to apply silently.  The block layer now refuses an *implicit* offset
    at build() (P1.10), so the dead branch is spelled explicitly here to reach
    the engine-level warning it exists to test."""
    from qarp.blocks import CompositeBlockBase, ConditionalBlock

    prelude = SimpleBlock(2)
    prelude.h(0)
    prelude.measure(0, 0)
    prelude.build()

    body = SimpleBlock(2)
    body.x(1)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    cond.target_cbits = [0] if alias else [1]

    composite = CompositeBlockBase(2)
    composite.add_child(prelude)
    composite.add_child(cond)
    composite.build()
    return composite


def test_engine_warns_when_a_condition_can_never_fire():
    """Sibling composition offsets the condition past the write — warn."""
    with pytest.warns(UserWarning, match="can never run"):
        QarpEngine(n_shots=100).build([Sampler(ket=_feedforward(alias=False), n_shots=100).build()])


def test_engine_is_silent_when_cbits_are_aliased():
    """An explicit ``target_cbits`` alias is the correct spelling — no warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        QarpEngine(n_shots=100).build([Sampler(ket=_feedforward(alias=True), n_shots=100).build()])


# ── Compile-stage rejections retyped (pipeline-sweep findings F1–F3) ──────
# The C++ device pipeline (check_fits / rebase / route) rejects with
# RuntimeError; the engines retype to CapabilityError so one except clause
# catches every capability failure.


def _wide_ket(n=3):
    b = SimpleBlock(n, name="wide")
    b.h(0)
    b.cx(0, 1)
    b.cx(1, 2)
    b.build()
    return b


def test_undersized_device_rejects_with_capability_error():
    """F1: circuit wider than the device — was a C++ RuntimeError escaping
    ``engine.build()``."""
    with pytest.raises(CapabilityError, match="exposes only 2"):
        QarpEngine(n_qubits=2, n_shots=50).build([Sampler(ket=_wide_ket(), n_shots=50)])


def test_routed_undersized_device_rejects_with_capability_error():
    """F1, routed path (``check_fits`` before routing) — the unrouted twin
    above never reaches the router."""
    dev = Device(2, architecture=get_nearest_neighbour_architecture(2, 1))
    with pytest.raises(CapabilityError, match="exposes only 2"):
        QarpEngine(n_shots=50, device=dev).build([Sampler(ket=_wide_ket(), n_shots=50)])


def test_router_rejects_3q_gate_with_capability_error():
    """F2: the router supports ≤2-qubit gates; a connectivity-only device
    has no gate set to rebase CSWAP through — must reject typed."""
    b = SimpleBlock(3, name="cswap")
    b.cswap(0, 1, 2)
    b.build()
    arch = get_nearest_neighbour_architecture(3, 1)
    with pytest.raises(CapabilityError, match="0/1/2-qubit"):
        QarpEngine(n_qubits=3, architecture=arch, n_shots=50).build([Sampler(ket=b, n_shots=50)])


def test_undecomposable_gate_set_rejects_with_capability_error():
    """F2: rebase failure ("no decomposition") is a capability statement
    about the configured device gate set."""
    b = SimpleBlock(1)
    b.h(0)
    b.build()
    eng = QarpEngine(n_qubits=1, gate_set=qx.full_gateset_2q(), n_shots=50)
    with pytest.raises(CapabilityError, match="no decomposition"):
        eng.build([Sampler(ket=b, n_shots=50)])
