import numpy as np
import pytest

from qarp.algorithms import VQE, StateVector
from qarp.optimizers import (
    AdaGradOptimizer,
    AdamaxOptimizer,
    AdamOptimizer,
    EarlyStopper,
    GradientDescentOptimizer,
    NadamOptimizer,
    RMSPropOptimizer,
    SGDOptimizer,
    SPSAOptimizer,
    compute_fd_gradients,
    compute_spsa_gradients,
)
from qarp.optimizers._gradient_descent_optimizer import _ck_schedule
from qarp.utils import FH_ham_and_wf_singles_and_doubles


def quadratic_fn(x):
    return np.sum(x**2)


def grad_quadratic_fn(x):
    return 2 * x


def test_fd_gradients_matches_true_gradient():
    x = np.array([1.0, -2.0, 3.0])
    fd = compute_fd_gradients(quadratic_fn, x, fd_eps=1e-6)
    true = grad_quadratic_fn(x)
    assert np.allclose(fd, true, atol=1e-4)


def test_fd_gradients_zero_at_origin():
    x = np.zeros(3)
    fd = compute_fd_gradients(quadratic_fn, x)
    assert np.allclose(fd, np.zeros(3), atol=1e-4)


def test__ck_schedule_step0():
    assert _ck_schedule(0, c0=1.0, gamma=0.5) == 1.0


def test__ck_schedule_monotonic_decrease():
    c1 = _ck_schedule(0)
    c2 = _ck_schedule(1)
    c3 = _ck_schedule(2)
    assert c1 > c2 > c3


def test_spsa_gradients_unbiased_on_quadratic():
    x = np.array([0.5, -1.0])
    rng_seed = 123

    est = compute_spsa_gradients(quadratic_fn, x, ck=1e-3, num_perturbations=5000, seed=rng_seed)

    true = grad_quadratic_fn(x)
    assert np.allclose(est, true, atol=1e-2)


def test_spsa_gradients_zero_at_origin():
    x = np.zeros(5)
    est = compute_spsa_gradients(quadratic_fn, x, ck=1e-2, num_perturbations=2000, seed=42)
    assert np.allclose(est, np.zeros(5), atol=1e-2)


def test_user_gradient_with_without_step():
    opt = GradientDescentOptimizer()
    grad_fn = opt._get_gradient_function(quadratic_fn, gradient=grad_quadratic_fn)

    x = np.array([1.0, 2.0])
    g = grad_fn(x, 5)
    assert np.allclose(g, grad_quadratic_fn(x))

    def grad_with_step(x, step):
        return 2 * x + step

    opt = GradientDescentOptimizer()
    grad_fn = opt._get_gradient_function(quadratic_fn, gradient=grad_with_step)

    x = np.array([1.0, 2.0])
    g = grad_fn(x, 3)
    assert np.allclose(g, np.array([5.0, 7.0]))


def test_fd_gradient_matches_true_gradient():
    opt = GradientDescentOptimizer(options={"grad_method": "fd", "fd_eps": 1e-6})
    grad_fn = opt._get_gradient_function(quadratic_fn, gradient=None)

    x = np.array([1.0, -2.0])
    g = grad_fn(x, 0)
    true = grad_quadratic_fn(x)
    assert np.allclose(g, true, atol=1e-4)


def sgd_default():
    return SGDOptimizer(options={"lr": 0.01, "maxiter": 20})


def spsa_default():
    return SPSAOptimizer(
        options={
            "spsa_a0": 0.01,
            "spsa_alpha": 0.602,
            "spsa_A": 10.0,
            "spsa_c0": 1e-2,
            "spsa_gamma": 0.101,
            "spsa_seed": 123,
            "maxiter": 20,
        }
    )


def rmsprop_default():
    return RMSPropOptimizer(options={"lr": 0.01, "maxiter": 20})


def adagrad_default():
    return AdaGradOptimizer(options={"lr": 0.01, "maxiter": 20})


def adam_default():
    return AdamOptimizer(options={"lr": 0.01, "maxiter": 20})


def nadam_default():
    return NadamOptimizer(options={"lr": 0.01, "maxiter": 20})


def adamax_default():
    return AdamaxOptimizer(options={"lr": 0.01, "maxiter": 20, "eps": 1e-8})


optimizers_to_test = [
    (sgd_default, "SGD"),
    (spsa_default, "SPSA"),  # SPSA should not use analytic gradient
    (rmsprop_default, "RMSProp"),
    (adagrad_default, "AdaGrad"),
    (adam_default, "Adam"),
    (nadam_default, "Nadam"),
    (adamax_default, "Adamax"),
]


@pytest.mark.parametrize(
    "opt, name", optimizers_to_test, ids=[name for _, name in optimizers_to_test]
)
def test_optimizer_decreases_objective(opt, name):
    opt = opt()
    x0 = np.array([2.0, -1.0])
    print(x0)
    if name.lower() == "spsa":
        result = opt.minimize(quadratic_fn, x0)  # SPSA uses internal gradient
    else:
        result = opt.minimize(quadratic_fn, x0, gradient=grad_quadratic_fn)

    assert np.isfinite(quadratic_fn(result.x))
    assert np.all(np.isfinite(result.x))

    assert quadratic_fn(result.x) < quadratic_fn(np.array([2.0, -1.0]))

    assert result.nit > 0
    assert result.fun == quadratic_fn(result.x)
    assert result.nfev == opt._nfev


def test_spsa_gradients_zero_at_minimum():
    opt = GradientDescentOptimizer(
        options={
            "grad_method": "spsa",
            "spsa_c0": 1e-2,
            "spsa_gamma": 0.101,
            "num_spsa": 3000,
            "spsa_seed": 123,
        }
    )

    grad_fn = opt._get_gradient_function(quadratic_fn, gradient=None)

    x = np.zeros(3)
    g = grad_fn(x, step=0)
    assert np.allclose(g, np.zeros(3), atol=1e-2)


def test_spsa_gradients_unbiased_for_quadratic():
    opt = GradientDescentOptimizer(
        options={
            "grad_method": "spsa",
            "spsa_c0": 1e-3,
            "spsa_gamma": 0.101,
            "num_spsa": 5000,
            "spsa_seed": 42,
        }
    )

    grad_fn = opt._get_gradient_function(quadratic_fn, gradient=None)

    x = np.array([0.5, -1.0])
    g = grad_fn(x, step=10)
    true = grad_quadratic_fn(x)
    assert np.allclose(g, true, atol=1e-2)


def test_unknown_grad_method_raises():
    opt = GradientDescentOptimizer(options={"grad_method": "bad_method"})
    with pytest.raises(ValueError):
        opt._get_gradient_function(quadratic_fn)


def sgd_custom():
    return SGDOptimizer(options={"lr": 0.002, "maxiter": 100})


def spsa_custom():
    return SPSAOptimizer(
        options={
            "spsa_a0": 0.02,
            "spsa_alpha": 0.602,
            "spsa_A": 10.0,
            "spsa_c0": 1e-2,
            "spsa_gamma": 0.101,
            "spsa_seed": 123,
            "maxiter": 100,
        }
    )


def rmsprop_custom():
    return RMSPropOptimizer(options={"lr": 0.004, "maxiter": 100})


def adagrad_custom():
    return AdaGradOptimizer(options={"lr": 0.02, "maxiter": 100})


def adam_custom():
    return AdamOptimizer(options={"lr": 0.01, "maxiter": 100})


def nadam_custom():
    return NadamOptimizer(options={"lr": 0.01, "maxiter": 100})


def adamax_custom():
    return AdamaxOptimizer(options={"lr": 0.005, "maxiter": 100, "eps": 1e-8})


optimizers_to_test_cust = [
    (sgd_custom, "SGD"),
    (spsa_custom, "SPSA"),  # SPSA should not use analytic gradient
    (rmsprop_custom, "RMSProp"),
    (adagrad_custom, "AdaGrad"),
    (adam_custom, "Adam"),
    (nadam_custom, "Nadam"),
    (adamax_custom, "Adamax"),
]


@pytest.fixture
def fh_model():
    return FH_ham_and_wf_singles_and_doubles(2, generalised=False)


@pytest.mark.parametrize(
    "opt, name",
    optimizers_to_test_cust,
    ids=[name for _, name in optimizers_to_test_cust],
)
def test_vqe(opt, name, fh_model):
    ham, wfn = fh_model
    wfn.build()
    # Hand-tuned starting point for the fixed-step optimizers; signs follow
    # the generator orientation (excitation a†_virtual a_occupied — flipped
    # by the UCC pool rewrite).
    initial_parameters = np.asarray([0.0, -2.2, 0.5])
    optimizer = opt()

    vqe = VQE(
        operator=ham,
        ket=wfn,
        primitive=StateVector(),
        initial_parameters=initial_parameters,
        optimizer=optimizer,
        verbose=False,
    )
    vqe.build()

    e_vqe, x_vqe = vqe.run()
    assert np.isclose(e_vqe, -1.6450, atol=5e-1)


def constant_objective(x):
    return 1.0


def decreasing_objective(x):
    return float(x[0])


def increasing_objective(x):
    return -float(x[0])


def test_no_improvement_triggers_stop():
    """
    Early stopper should stop after patience=3 consecutive non-improvements.
    """
    stopper = EarlyStopper(
        constant_objective,
        initial_params=np.array([0.0]),
        es_tol=1e-6,
        patience=3,
    )

    params = np.array([0.0])

    # First call -> best stays 1.0, no_improve = 1
    assert stopper(params) is False
    # Second call -> no_improve = 2
    assert stopper(params) is False
    # Third call -> no_improve = 3 -> stop
    assert stopper(params) is True


def test_improvement_resets_no_improve():
    """
    Improvement resets the no_improve counter to zero.
    """
    stopper = EarlyStopper(
        decreasing_objective, initial_params=np.array([5.0]), es_tol=0.0, patience=3
    )

    # Step 1: x=5.0, best=5.0
    assert stopper(np.array([5.0])) is False
    assert stopper.no_improve == 1

    # Step 2: x=5.0 -> no improvement
    assert stopper(np.array([5.0])) is False
    assert stopper.no_improve == 2

    # Step 3: x=4.0 -> improvement = reset
    assert stopper(np.array([4.0])) is False
    assert stopper.no_improve == 0
    assert stopper.best == 4.0


def test_stop_only_after_consecutive_non_improvements():
    """
    Should not stop if improvement happens within patience window=3.
    """
    stopper = EarlyStopper(
        decreasing_objective, initial_params=np.array([10.0]), es_tol=0.0, patience=3
    )

    # no improvement twice
    stopper(np.array([10.0]))
    stopper(np.array([10.0]))
    assert stopper.no_improve == 2

    stopper(np.array([9.0]))
    assert stopper.no_improve == 0  # improvement, stopper resets
    assert stopper.best == 9.0

    # patience=3, now stop
    stopper(np.array([9.0]))  # no_improve = 1
    stopper(np.array([9.0]))  # no_improve = 2
    assert stopper(np.array([9.0])) is True  # no_improve = 3 -> stop


def test_ema_smoothed_improvement():
    """
    EMA should smooth the objective and use the smoothed value for stopping.

    stop_val_next = ema_beta * stop_val + (1-ema_beta) * stop_val_prev
    """
    stopper = EarlyStopper(
        objective_fn=lambda x: float(x[0]),
        initial_params=np.array([10.0]),
        es_tol=0.0,
        patience=3,
        ema_beta=0.5,
    )

    # 1: loss = 10 -> initial best = 10, smoothed stays 10
    assert stopper(np.array([10.0])) is False

    # 2: slight decrease -> EMA is still > new loss, improvement
    # previous ema_val = 10
    # new = 9, so ema = 0.5*10 + 0.5*9 = 9.5
    assert stopper(np.array([9.0])) is False
    assert pytest.approx(stopper.ema_val) == 9.5
    assert pytest.approx(stopper.best) == 9.5


def test_ema_does_not_improve_when_smoothed_not_lower():
    """
    Even if raw objective improves, if EMA does not drop below best - tol,
    it should count as no improvement.
    """
    stopper = EarlyStopper(
        objective_fn=lambda x: float(x[0]),
        initial_params=np.array([10.0]),
        es_tol=0.0,
        patience=2,
        ema_beta=0.9,
    )

    # 1: x=10
    stopper(np.array([10.0]))  # ema = 10

    # 2: x=9,  > ema = 0.9*10 + 0.1*9 = 9.9
    # best = 10, improvement (9.9 < 10)
    stopper(np.array([9.0]))

    # 3: x=8, >  ema = 0.9*9.9 + 0.1*8 = 9.71
    # best = 9.9 -> improvement (9.71 < 9.9)
    stopper(np.array([8.0]))

    # 4: x=9 again > ema = 0.9*9.71 + 0.1*9 = 9.639
    # best = 9.71, curr ema = 9.639 < 9.71
    # Loss INCREASED, but ema DECREASED - still an improvement
    # so optimisation continues & EarlyStopping doesn't happen
    assert stopper(np.array([9.0])) is False


def test_stopper_attributes():
    stopper = EarlyStopper(
        quadratic_fn,
        np.array([8.0]),
        es_tol=1e-3,
        patience=7,
        ema_beta=0.9,
    )

    assert isinstance(stopper, EarlyStopper)
    assert stopper.es_tol == 1e-3
    assert stopper.patience == 7
    assert stopper.beta == 0.9


# ── User-supplied gradient arity ────────────────────────────────────────


def test_user_gradient_taking_only_params_is_called_with_one_argument():
    seen = []

    def gradient(params):
        seen.append(np.asarray(params, dtype=float).copy())
        return 2.0 * np.asarray(params, dtype=float)

    opt = AdamOptimizer(options={"maxiter": 3, "lr": 0.1})
    opt.minimize(lambda p: float(np.sum(np.square(p))), [1.0, -1.0], gradient=gradient)
    assert len(seen) >= 1


def test_user_gradient_taking_step_receives_the_iteration_index():
    steps = []

    def gradient(params, step):
        steps.append(step)
        return 2.0 * np.asarray(params, dtype=float)

    opt = AdamOptimizer(options={"maxiter": 3, "lr": 0.1})
    opt.minimize(lambda p: float(np.sum(np.square(p))), [1.0, -1.0], gradient=gradient)
    assert steps == sorted(steps)
    assert len(set(steps)) > 1, "step index never advanced"


def test_type_error_inside_user_gradient_is_not_swallowed():
    """A TypeError raised *by* the gradient must propagate untouched.

    Probing arity by calling and catching TypeError would retry with one
    argument and surface an unrelated failure — masking the real cause (a
    stale binding, a bad argument deeper down) behind an arity mismatch.
    """

    def gradient(params):
        raise TypeError("incompatible function arguments: deliberate")

    opt = AdamOptimizer(options={"maxiter": 2, "lr": 0.1})
    with pytest.raises(TypeError, match="deliberate"):
        opt.minimize(lambda p: float(np.sum(np.square(p))), [1.0], gradient=gradient)


def test_type_error_inside_two_argument_gradient_is_not_swallowed():
    calls = []

    def gradient(params, step):
        calls.append(step)
        raise TypeError("incompatible function arguments: deliberate")

    opt = AdamOptimizer(options={"maxiter": 2, "lr": 0.1})
    with pytest.raises(TypeError, match="deliberate"):
        opt.minimize(lambda p: float(np.sum(np.square(p))), [1.0], gradient=gradient)
    assert len(calls) == 1, "gradient retried after its own TypeError"


# ── Callback contract: the objective is evaluated for each callback ─────


def test_objective_is_evaluated_before_each_callback_with_user_gradient():
    """Every optimizer must leave the objective evaluated at the parameters it
    hands the callback.

    Composite algorithms read the value their objective wrapper caches
    (``_last_objective``) rather than recomputing it, so a loop that only ever
    calls the gradient hands the callback a stale ``None``.  Scipy's optimizers
    evaluate the objective each iteration; this loop must match, or callbacks
    are not portable between the two.
    """
    seen = []

    def objective(p):
        seen.append(("obj", len(seen)))
        return float(np.sum(np.square(p)))

    def gradient(p):
        return 2.0 * np.asarray(p, dtype=float)

    calls = {"cb": 0}

    def callback(p):
        calls["cb"] += 1
        # At least one objective evaluation must precede this callback.
        assert seen, "callback fired before any objective evaluation"

    opt = AdamOptimizer(options={"maxiter": 4, "lr": 0.1})
    opt.minimize(objective, [1.0, -1.0], gradient=gradient, callback=callback)
    assert calls["cb"] == 4


def test_no_callback_means_no_extra_objective_evaluations():
    """Without a callback the loop stays gradient-only — the per-iteration
    objective evaluation is the cost of reporting, not of optimizing."""
    n_obj = {"n": 0}

    def objective(p):
        n_obj["n"] += 1
        return float(np.sum(np.square(p)))

    opt = AdamOptimizer(options={"maxiter": 10, "lr": 0.1})
    opt.minimize(objective, [1.0], gradient=lambda p: 2.0 * np.asarray(p, dtype=float))
    # Only the final results.fun evaluation.
    assert n_obj["n"] <= 2, f"{n_obj['n']} objective calls without a callback"


def test_vqe_verbose_with_adam_and_analytic_gradient(capsys):
    """The reported failure: verbose logging with a gradient-descent optimizer
    and analytic gradients must print real energies, not crash formatting a
    ``None``."""
    from qarp.blocks import HEABlock
    from qarp.operators import QubitOperator

    ansatz = HEABlock(2, 1, real=True, linear=True, circular=False, use_cz=True).build()
    operator = QubitOperator("Z0") + QubitOperator("Z1")
    vqe = VQE(
        operator=operator,
        ket=ansatz,
        gradient=True,
        verbose=True,
        initial_parameters=np.full(len(ansatz.symbols), 0.3),
        optimizer=AdamOptimizer(options={"maxiter": 3, "lr": 0.05}),
    )
    vqe.build()
    vqe.run()
    out = capsys.readouterr().out
    assert "None" not in out
    assert "VQE Run:" in out


def test_energy_history_with_adam_has_no_none_entries():
    """``save_energy_history`` must record real energies under a gradient-only
    loop — silently collecting ``None`` is worse than the crash."""
    from qarp.blocks import HEABlock
    from qarp.operators import QubitOperator

    ansatz = HEABlock(2, 1, real=True, linear=True, circular=False, use_cz=True).build()
    operator = QubitOperator("Z0") + QubitOperator("Z1")
    vqe = VQE(
        operator=operator,
        ket=ansatz,
        gradient=True,
        save_energy_history=True,
        initial_parameters=np.full(len(ansatz.symbols), 0.3),
        optimizer=AdamOptimizer(options={"maxiter": 3, "lr": 0.05}),
    )
    vqe.suppress_success_message = True
    vqe.build()
    vqe.run()
    assert len(vqe.energy_history) == 3
    assert all(isinstance(e, float) for e in vqe.energy_history)
