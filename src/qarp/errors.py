"""OpenQARP exception types."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import qarpx as qx


class CapabilityError(ValueError):
    """A declared capability is incompatible with the engine, circuit, or call.

    The single exception type for every engine-capability rejection: raised by
    the ``Engine`` template's validation hooks at ``build()``, ``run()`` and
    ``batch_run()`` time (targets, exactness, noise×amplitudes, mid-circuit
    operations, ``initial_state`` support/routing), by engine constructors for
    self-contradictory configuration, and by primitives whose own construction
    detects an unsupported combination.

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

    command: Optional["qx.Command"] = None


class RemoteJobError(RuntimeError):
    """A job a remote service accepted did not end ``succeeded``.

    Raised by remote engines (``OqtopusEngine``) when a submitted job comes
    back ``failed`` or ``cancelled``, or succeeded without a result payload.
    Not a capability failure: the request was legitimate and the service
    declined to complete it.  ``job_id``, ``status`` and the service
    ``message`` are attributes for programmatic handling.
    """

    def __init__(self, job_id: str, status: str, message: "str | None" = None):
        self.job_id = job_id
        self.status = status
        self.message = message
        text = f"remote job {job_id!r} ended with status {status!r}"
        if message:
            text += f": {message}"
        super().__init__(text)
