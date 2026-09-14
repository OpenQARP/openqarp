"""OpenQARP exception types."""


class CapabilityError(ValueError):
    """A declared capability is incompatible with the engine, circuit, or call.

    The single exception type for every engine-capability rejection: raised by
    the ``Engine`` template's validation hooks at ``build()``, ``run()`` and
    ``batch_run()`` time (targets, exactness, noise×amplitudes, mid-circuit
    operations, ``initial_state`` support/routing), by engine constructors for
    self-contradictory configuration, by ``StructuredQPEPlan.sample()``, and by
    primitives whose own construction detects an unsupported combination.

    Also the rejection type of the emit/absorb SDK boundary: an emitter whose
    declared gate set or ``EmitterCapabilities`` cannot represent a command
    raises it from ``emit()`` (via ``validate()``, before any SDK import),
    with the offending ``qarpx.Command`` attached as the exception's
    ``command`` attribute; an absorber handed the wrong SDK's object, or a
    parameter expression outside the single-symbol linear form, raises it
    too.  A *missing* SDK is ``ImportError`` instead — environment, not
    circuit.  Subclasses ``ValueError`` so broad ``except ValueError``
    handlers keep working.
    """
