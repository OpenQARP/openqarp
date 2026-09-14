Optimizers
===========

The ``optimizers`` submodule provides objects compatible with OpenQARP features that rely on the optimization of objective functions. 
As the optimal strategy for circuit parameter optimization remains an open research question, this section outlines the usage of 
available black-box optimizers and the process for implementing custom ones.

Unless otherwise specified, the Rosenbrock function will serve as the illustrative example throughout this section.

.. math::

    f(x, y) = (a - x)^2 + b (y - x^2)^2

The corresponding Python implementation of the function and its gradient is as follows:

.. code-block:: python

    import numpy as np

    def rosenbrock_function(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def rosenbrock_gradient(x):
        jac = np.zeros_like(x)
        jac[:-1] += 200 * (x[1:] - x[:-1]**2) * (-2 * x[:-1]) - 2 * (1 - x[:-1])
        jac[1:] += 200 * (x[1:] - x[:-1]**2)
        return jac

Using the ScipyOptimizer
------------------------

By leveraging SciPy for the minimization process, a wide range of solvers becomes accessible. Wrapping these solvers within our 
own class ensures compatibility with other OpenQARP components. For a comprehensive list of available optimizers, refer to the 
documentation for ``scipy.optimize.minimize``.

As an example, we demonstrate the optimization of the Rosenbrock function using the ``BFGS`` quasi-Newton method.

To instantiate a ``ScipyOptimizer``, specify the desired method and any options to be passed to the underlying SciPy optimizer 
as a dictionary. The ``minimize`` method can then be invoked with the appropriate arguments:

.. code-block:: python

    from qarp.optimizers import ScipyOptimizer
    optimizer = ScipyOptimizer(method="BFGS", options={"disp": True})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([0., 0.]),
        callback=None,
        gradient=rosenbrock_gradient
    )

That’s all that is required. Users are encouraged to consult the SciPy documentation for details on the ``options`` dictionary 
and gradient support for each method.

.. note::

    Spectrum estimation from DOS-QPE distributions (formerly the
    ``SpectralOptimizer``) is not an optimizer, since it implements no
    ``minimize`` contract, and now lives in ``qarp.algorithms`` as
    :code:`SpectrumEstimator`.  See the *Spectrum estimation* section of the
    algorithms documentation.


RotosolveOptimizer
------------------

The :code:`RotosolveOptimizer` implements the **Rotosolve algorithm** (Ostaszewski et. al. - Quantum 5, 391 (2021)), which analytically minimizes
each circuit parameter by exploiting the periodic structure of variational
quantum circuits.

Rotosolve assumes the objective function is **periodic with period** :math:`2\pi`
with respect to each parameter and updates parameters sequentially using
three function evaluations per parameter.

The implementation supports optional **learning‑rate scheduling** and
damped updates for improved convergence stability if needed.

**Algorithm: Rotosolve**

**Input:**
Hermitian measurement operator :math:`M` encoding the objective function,
a parameterized quantum circuit :math:`U(\boldsymbol{\theta})` with fixed
structure, and a stopping criterion.

1. Initialize parameters
   :math:`\theta_d \in (-\pi, \pi]` for :math:`d = 1, \ldots, D`
   heuristically or at random.

2. Repeat for `t`:

   a. For :math:`d = 1, \ldots, D` do:

      i.   Select a value :math:`\phi \in \mathbb{R}` heuristically or at random.

      ii.  Fix all circuit parameters except the :math:`d`‑th parameter.

      iii. Estimate the expectation values
           :math:`\langle M \rangle_{\phi}`,
           :math:`\langle M \rangle_{\phi + \frac{\pi}{2}}`,
           and
           :math:`\langle M \rangle_{\phi - \frac{\pi}{2}}`
           from measurement samples.

      iv.  Update the parameter:

           .. math::

              \theta_d \leftarrow
              \phi
              - \frac{\pi}{2}
              - \operatorname{arctan2}\!\left(
              2\langle M \rangle_{\phi}
              - \langle M \rangle_{\phi + \frac{\pi}{2}}
              - \langle M \rangle_{\phi - \frac{\pi}{2}},
              \langle M \rangle_{\phi + \frac{\pi}{2}}
              - \langle M \rangle_{\phi - \frac{\pi}{2}}
              \right)

3. Until :math:`t=T_{max}` or the stopping criterion is met.

**Key Parameters:**

- :code:`maxiter`: Maximum number of Rotosolve macro-iterations
- :code:`tol`: Convergence tolerance on change in objective value
- :code:`lr`: Base learning rate applied to the analytic Rotosolve update (optional)
- :code:`schedule`: Learning-rate schedule (optional)
- :code:`gamma`: Exponential decay factor for learning-rate scheduling (optional)
- :code:`power`: Power for polynomial learning-rate schedules (optional)
- :code:`verbose`: Enable detailed macro- and micro-iteration logging

.. note::

    The default Rotosolve does not have a learning rate scheduler, we include one as an 
    heuristic option for smoother convergence according to a particular `schedule` with the 
    update rule:

           .. math::

              \theta_d \leftarrow
              \phi - \eta(t) \left(
              + \frac{\pi}{2}
              +\operatorname{arctan2}\!\left(
              2\langle M \rangle_{\phi}
              - \langle M \rangle_{\phi + \frac{\pi}{2}}
              - \langle M \rangle_{\phi - \frac{\pi}{2}},
              \langle M \rangle_{\phi + \frac{\pi}{2}}
              - \langle M \rangle_{\phi - \frac{\pi}{2}}
              \right)\right)

    :math:`\eta(t)` is the learning rate computed via the schedule at iteration :math:`t`, and
     if :math:`\eta(t) = 1 \forall t` (which is the default), the algorithm defaults to vanilla Rotosolve.



Custom Optimizers
-----------------

The structure of the optimizer classes allows for straightforward implementation of custom optimizers. To ensure compatibility 
with OpenQARP, custom optimizers must adhere to the following requirements:

1. The custom optimizer must implement a ``.minimize`` method.
2. The ``.minimize`` method must accept the following arguments: ``objective_function`` and ``initial_parameters``.
3. The return value must be an object with an ``.x`` attribute or property, consistent with SciPy’s behavior. No additional attributes are required for compatibility with other OpenQARP components.
4. An optimizer that cannot honour ``minimize(bounds=...)`` should set the class attribute ``supports_bounds = False`` (as ``GradientDescentOptimizer`` and ``RotosolveOptimizer`` do) and raise on a non-``None`` ``bounds``; an algorithm whose search is bounded (``MMQCELS``) checks the attribute at construction and refuses such an optimizer with a ``TypeError`` instead of failing mid-run. An object without the attribute is assumed to honour bounds.


Gradient-Based Optimizers
-------------------------

OpenQARP provides several built‑in gradient‑based optimizers implemented directly in
Python, using the ``ScipyOptimizer`` syntax including:

- :class:`SGDOptimizer`: Vanilla gradient descent/stochastic gradient descent optimizer.
- :class:`SPSAOptimizer`: Optimiser for the SPSA algorithm, random vector perturbations with learning rate decay.
- :class:`RMSPropOptimizer`: Gradient descent with adaptive squared‑gradient normalization.  
- :class:`AdaGradOptimizer`: Gradient descent with cumulative squared‑gradient scaling.  
- :class:`AdamOptimizer`: Gradient descent with Momentum and adaptive second-moment accumulation. 
- :class:`AdamaxOptimizer`: Infinity‑norm variant of Adam.  
- :class:`NadamOptimizer`: Nesterov‑accelerated Adam updates.  

These optimizers follow a shared interface defined by
:class:`GradientDescentOptimizer`, making them fully compatible with OpenQARP’s
variational algorithms and gradient approximation tools (finite differences ``fd``,
SPSA-based (``spsa``) (random Rademacher perturbations) gradients, or analytic gradients if provided by the user).

Each optimizer implements only its update rule; all parameter validation,
gradient construction, and input handling are centralized in the base class.

The ``fd`` and ``spsa`` estimators here call the objective once per point.
The engines expose the same estimators as batched gradient methods
(``engine.run_gradient(params, method="finite-diff")`` /
``method="spsa"``) that evaluate every point in one sweep, alongside the
analytic adjoint and parameter-shift methods — see :doc:`gradients`.


Early Stopping
^^^^^^^^^^^^^^

OpenQARP gradient-based optimizers provide an optional early‑stopping mechanism that can be enabled for any
gradient‑based optimizer. Early stopping monitors the objective value during
optimization and halts the optimization when the loss fails to improve by a
minimum threshold for a specified number of consecutive steps. This helps avoid
wasted function evaluations and prevents over‑optimization on noisy objectives.

The early‑stopping behaviour is controlled through the ``options`` dictionary
passed to the optimizer, alongside the usual keys such as ``maxiter``:

.. code-block:: python

    from qarp.optimizers import SGDOptimizer

    optimizer = SGDOptimizer(
        options={
            "maxiter": 200,
            "early_stopping": True,
            "es_tol": 1e-3,
            "patience": 5,
            "ema_beta": 0.9,
        }
    )

Parameters
~~~~~~~~~~

``early_stopping`` : bool, optional  
    Enables early stopping when set to ``True``. Default is ``False``.

``es_tol`` : float, optional  
    Minimum required improvement in the loss to be considered progress.
    If the smoothed loss (or raw loss, if EMA is disabled) does not improve by at
    least ``es_tol`` relative to the best observed value, the iteration is counted as a
    non‑improvement. Default is ``1e‑6``.

``patience`` : int, optional  
    Number of consecutive non‑improvement iterations tolerated before stopping.
    Once the number of consecutive failures to improve exceeds ``patience``,
    optimization terminates. Default is ``10``.

``ema_beta`` : float, optional  
    Exponential moving‑average smoothing factor. If set below ``1.0``, early
    stopping monitors the smoothed loss instead of the raw loss. A value such as
    ``0.9`` or ``0.99`` is typical for noisy objectives. Default is ``1.0``,
    which disables smoothing.

Usage Notes
~~~~~~~~~~~

- Early stopping works with all gradient estimation methods: analytic gradients,
  finite differences, and SPSA-computed (random directional derivative) gradients.
- When ``ema_beta`` is set, smoothing can significantly stabilize stopping
  behaviour on stochastic or noisy objectives.
- Optimizers automatically return the *best* parameters encountered when early
  stopping is triggered, rather than those from the final iteration.
- Since :class:`ScipyOptimizer` has built in convergence halting routines for the optimisation,
  the EarlyStopping functionality mimics this for :class:`GradientDescentOptimizer` subclasses,
  though it is not enabled by default.


SGDOptimizer
^^^^^^^^^^^^

The :code:`SGDOptimizer` implements the vanilla form of stochastic gradient descent
using a fixed (non-adaptive) learning rate. This optimizer represents the simplest gradient-based
method and serves as a strong baseline for smooth objective functions in both
classical and variational quantum optimization tasks.


**How it works:**
At iteration :math:`k`, the update rule is:

.. math::

    \theta_{k+1} = \theta_k - \eta \, \nabla f(\theta_k)

where:

- :math:`\eta` is a constant learning rate,
- :math:`\nabla f(\theta_k)` is obtained analytically or approximately estimated.

SGD does **not** adjust :math:`\eta` across iterations and does not maintain any
per-parameter state such as momentum or variance estimates.

**Key Options:**

- ``lr``: Constant learning rate :math:`\eta` (default: 0.01)
- ``grad_method``: ``"fd"`` or ``"spsa"`` for gradient approximation.
- ``fd_eps``: Finite-difference step used when ``grad_method="fd"`` (default: 1e‑5)

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import SGDOptimizer

    # Rosenbrock's curved valley is ill-conditioned for a fixed step size:
    # lr much above 1e-3 here diverges. Vanilla SGD also needs more
    # iterations than the adaptive methods below to make useful progress.
    optimizer = SGDOptimizer(options={"lr": 0.001, "maxiter": 3000})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([0., 0.])
    )

    print("Optimized parameters:", result.x)

.. note::

    The "stochastic" in stochastic gradient descent typically refers to a gradient computed over a batched 
    subset of input data in machine learning. When using this optimiser for variational quantum algorithms 
    outside of quantum machine learning, there is typically no data. However, the SGD optimizer can still be 
    stochastic due to measurement noise and gradient approximations.


SPSAOptimizer
^^^^^^^^^^^^^


The :code:`SPSAOptimizer` implements the Simultaneous Perturbation Stochastic
Approximation (SPSA) algorithm, a highly efficient optimization
method that requires only **two function evaluations per iteration**, regardless
of the dimensionality of the parameter vector. Technically, SPSA computes directional derivatives in random 
perturbed directions. This makes SPSA especially useful
for variational quantum algorithms (VQAs), noisy objective functions, and
high-dimensional optimization tasks.

Unlike standard finite-difference or quantum parameter-shift methods that scale linearly with the number of
parameters, SPSA achieves dimensionality-independent cost by using random
Rademacher perturbations to estimate gradients.

**How it works:**
At iteration :math:`k`, SPSA samples a random perturbation vector
:math:`\Delta_k` whose entries take values :math:`+1` or :math:`-1` with equal
probability. Two perturbed parameter vectors are evaluated:

.. math::

    \theta_k^{+} = \theta_k + c_k \Delta_k, \qquad
    \theta_k^{-} = \theta_k - c_k \Delta_k

The stochastic gradient estimate is:

.. math::

    \hat{g}_k
    = \frac{f(\theta_k^{+}) - f(\theta_k^{-})}{2 c_k \Delta_k}

The parameter update follows:

.. math::

    \theta_{k+1} = \theta_k - a_k \, \hat{g}_k

where :math:`a_k` and :math:`c_k` follow power-law decay schedules:

.. math::

    a_k = \frac{a_0}{(k + A + 1)^{\alpha}}, \qquad
    c_k = \frac{c_0}{(k+1)^{\gamma}}

Typical recommended values (Spall, 1992) are:

- :math:`\alpha \approx 0.602`
- :math:`\gamma \approx 0.101`

**Key Options:**

- ``spsa_a0``: Initial learning-rate coefficient :math:`a_0` (default: 0.1)
- ``spsa_alpha``: Decay exponent :math:`\alpha` (default: 0.602)
- ``spsa_A``: Stabilizer parameter :math:`A` in the :math:`a_k` schedule (default: 10.0)
- ``spsa_c0``: Initial perturbation scale :math:`c_0` (default: 0.1)
- ``spsa_gamma``: Perturbation decay exponent :math:`\gamma` (default: 0.101)
- ``num_spsa``: Number of SPSA perturbation pairs per iteration (default: 1)
- ``spsa_seed``: Optional base random seed for reproducibility

.. note::

    SPSA does *not* use analytical gradients. The ``grad_method`` argument is
    automatically set to ``"spsa"`` when using :code:`SPSAOptimizer`. This computes
    the "gradients" as random directional derivatives multiplied by the perturbation.

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import SPSAOptimizer

    # As with SGD, Rosenbrock's ill-conditioning means spsa_a0 much above
    # 0.01 here diverges; more iterations make up for the smaller step.
    optimizer = SPSAOptimizer(options={
        "spsa_a0": 0.01,
        "spsa_alpha": 0.602,
        "spsa_A": 10.0,
        "spsa_c0": 0.1,
        "spsa_gamma": 0.101,
        "spsa_seed": 123,
        "maxiter": 3000,
    })

    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([0., 0.])
    )

    print("Optimized parameters:", result.x)


RMSPropOptimizer
^^^^^^^^^^^^^^^^

The :code:`RMSPropOptimizer` is an adaptive gradient method that normalizes parameter
updates by an exponential moving average of squared gradients. This stabilizes
optimization under noisy or highly anisotropic landscapes.

**How it works:**
RMSProp maintains a running average:

.. math::

    v_k = \beta v_{k-1} + (1 - \beta) \, (\nabla f(\theta_k))^2

and performs the update:

.. math::

    \theta_{k+1} = \theta_k - \frac{\eta \, \nabla f(\theta_k)}{\sqrt{v_k} + \varepsilon}

where:

- :math:`\beta` controls the decay of past gradients,
- :math:`\varepsilon` avoids division by zero.

**Key Options:**

- ``lr``: Learning rate :math:`\eta` (default: 0.001)
- ``beta``: Smoothing factor (default: 0.9)
- ``eps``: Numerical stability term (default: 1e‑8)

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import RMSPropOptimizer

    optimizer = RMSPropOptimizer(options={"lr": 0.001, "beta": 0.95})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([-1.0, 1.0])
    )

    print("Optimized parameters:", result.x)


AdaGradOptimizer
^^^^^^^^^^^^^^^^

The :code:`AdaGradOptimizer` adapts the learning rate of each parameter individually
based on the accumulated squared gradients. This method is especially effective in
sparse or high‑dimensional optimization problems.

**How it works:**
AdaGrad accumulates squared gradients:

.. math::

    r_k = r_{k-1} + (\nabla f(\theta_k))^2

and updates parameters according to:

.. math::

    \theta_{k+1} = \theta_k - \frac{\eta \, \nabla f(\theta_k)}{\sqrt{r_k} + \varepsilon}

As :math:`r_k` grows, the effective learning rate decreases.

**Key Options:**

- ``lr``: Base learning rate :math:`\eta` (default: 0.01)
- ``eps``: Stability constant :math:`\varepsilon` (default: 1e‑8)
- ``initial_accumulator_value``: Starting value of the squared-gradient accumulator (default: 0.0)

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import AdaGradOptimizer

    optimizer = AdaGradOptimizer(options={"lr": 0.02})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([0., 0.])
    )

    print("Optimized parameters:", result.x)


AdamOptimizer
^^^^^^^^^^^^^

The :code:`AdamOptimizer` combines momentum with adaptive per‑parameter learning
rates, making it one of the most widely used optimization algorithms in deep
learning and variational quantum eigensolvers.

**How it works:**
Adam maintains exponential moving averages of gradients and squared gradients:

.. math::

    m_k = \beta_1 m_{k-1} + (1 - \beta_1) \nabla f(\theta_k)

.. math::

    v_k = \beta_2 v_{k-1} + (1 - \beta_2) (\nabla f(\theta_k))^2

Bias-corrected estimates:

.. math::

    \hat{m}_k = \frac{m_k}{1 - \beta_1^k}
    \qquad
    \hat{v}_k = \frac{v_k}{1 - \beta_2^k}

Update rule:

.. math::

    \theta_{k+1} = \theta_k - \frac{\eta \, \hat{m}_k}{\sqrt{\hat{v}_k} + \varepsilon}

**Key Options:**

- ``lr``: Learning rate :math:`\eta` (default: 0.001)
- ``beta1``: Momentum decay (default: 0.9)
- ``beta2``: Variance decay (default: 0.999)
- ``eps``: Numerical stability constant (default: 1e‑10)

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import AdamOptimizer

    optimizer = AdamOptimizer(options={"lr": 0.01})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([-0.5, 0.5])
    )

    print("Optimized parameters:", result.x)


AdamaxOptimizer
^^^^^^^^^^^^^^^

The :code:`AdamaxOptimizer` is an infinity-norm variant of Adam. By replacing the
second-moment estimate with a maximum-based accumulator, it improves robustness to
rare but very large gradient spikes.

**How it works:**
Adamax keeps a momentum term:

.. math::

    m_k = \beta_1 m_{k-1} + (1 - \beta_1) \nabla f(\theta_k)

and an exponentially weighted infinity norm:

.. math::

    u_k = \max(\beta_2 u_{k-1}, |\nabla f(\theta_k)|)

Update rule:

.. math::

    \theta_{k+1}
    = \theta_k - \frac{\eta}{1 - \beta_1^k} \frac{m_k}{u_k}

**Key Options:**

- ``lr``: Learning rate :math:`\eta` (default: 0.002)
- ``beta1``: Momentum weight (default: 0.9)
- ``beta2``: Infinity‑norm decay (default: 0.999)
- ``eps``: Numerical stability constant (default: 1e‑8)

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import AdamaxOptimizer

    optimizer = AdamaxOptimizer(options={"lr": 0.002})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([1.0, 1.0])
    )

    print("Optimized parameters:", result.x)



NadamOptimizer
^^^^^^^^^^^^^^

The :code:`NadamOptimizer` introduces Nesterov momentum into Adam, producing a
look‑ahead‑corrected gradient estimate that often yields smoother convergence.

**How it works:**
Momentum:

.. math::

    m_k = \beta_1 m_{k-1} + (1 - \beta_1) \nabla f(\theta_k)

Second moment:

.. math::

    v_k = \beta_2 v_{k-1} + (1 - \beta_2) (\nabla f(\theta_k))^2

Nesterov‐corrected first moment:

.. math::

    \hat{m}_k^{\text{Nadam}}
    =
    \frac{\beta_1 m_k}{1 - \beta_1^k}
    +
    \frac{(1 - \beta_1) \nabla f(\theta_k)}{1 - \beta_1^k}

Update rule:

.. math::

    \theta_{k+1} = \theta_k - \frac{\eta \, \hat{m}_k^{\text{Nadam}}}{\sqrt{v_k / (1 - \beta_2^k)} + \varepsilon}

**Key Options:**

- ``lr``: Learning rate :math:`\eta` (default: 0.002)
- ``beta1``: Momentum decay (default: 0.9)
- ``beta2``: Variance decay (default: 0.999)
- ``eps``: Numerical stability constant (default: 1e‑8)

**Basic Usage:**

.. code-block:: python

    from qarp.optimizers import NadamOptimizer

    optimizer = NadamOptimizer(options={"lr": 0.002})
    result = optimizer.minimize(
        objective_function=rosenbrock_function,
        initial_parameters=np.array([-1.0, 1.0])
    )

    print("Optimized parameters:", result.x)

