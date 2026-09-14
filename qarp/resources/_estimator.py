"""The producer: run the real compilation pipeline, snapshot each stage.

``ResourceEstimator`` mirrors the staging engines actually perform
(an engine's ``_compile_one``: rebase → route → rebase), inserting a counted
:class:`ResourceVector` snapshot at every stage boundary.  The numbers are
therefore ground truth of the pipeline that would execute, not a parallel
cost model — the ROUTED snapshot is taken *before* the final rebase, the
only point where router-inserted SWAPs exist as SWAP gates.

Config is validated and fixed at construction (``Device`` convention);
modeled fields are delegated to an optional
:class:`~qarp.resources.ResourceModeler` (see
``Engine.resource_modeler()``).
"""

from collections.abc import Iterator, Mapping
from typing import Any, Sequence

import qarpx as qx

from ._counting import count_resources
from ._modelers import ResourceModeler
from ._synthesis import synthesize_clifford_t
from ._vector import SCHEMA_VERSION, Provenance, ResourceVector, Stage


class ResourceReport(Mapping[Stage, ResourceVector]):
    """Per-stage vectors in pipeline order; ``final`` is the last stage."""

    def __init__(self, stages: "dict[Stage, ResourceVector]") -> None:
        if not stages:
            raise ValueError("ResourceReport needs at least one stage")
        self._stages = dict(stages)

    def __getitem__(self, stage: Stage) -> ResourceVector:
        return self._stages[stage]

    def __iter__(self) -> Iterator[Stage]:
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    @property
    def final(self) -> ResourceVector:
        return self._stages[self.stages[-1]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stages": {s.value: v.to_dict() for s, v in self._stages.items()},
        }

    def __repr__(self) -> str:
        return f"ResourceReport(stages={[s.value for s in self._stages]})"


class ResourceEstimator:
    """Estimate resources of a block or command stream, stage by stage.

    Stages produced: LOGICAL always; OPTIMIZED when a ``gateset`` is given;
    ROUTED and TARGET when a ``device`` with an architecture is also given
    (routing requires the rebase, so ``device`` implies ``gateset``);
    SYNTHESIZED when ``synthesis_epsilon`` is given (requires the Clifford+T+Rz gate
    set — the Clifford+T+Rz intermediate gridsynth consumes).  The
    ``modeler`` runs on the final stage only, where the target-gate-set
    circuit is known.
    """

    def __init__(
        self,
        *,
        gateset: "qx.GateSet | None" = None,
        opt_level: "qx.OptLevel" = qx.OptLevel.O1,
        device: "qx.Device | None" = None,
        router: "qx.RouterKind" = qx.RouterKind.Sabre,
        modeler: ResourceModeler | None = None,
        device_label: str | None = None,
        synthesis_epsilon: float | None = None,
    ) -> None:
        if device is not None and gateset is None:
            raise ValueError("a device requires a gateset: routing runs on rebased circuits")
        if synthesis_epsilon is not None:
            if gateset is None or gateset.name != "clifford_t_rz":
                raise ValueError(
                    "synthesis_epsilon needs gateset=qx.clifford_t_rz_gateset(): "
                    "gridsynth consumes the Clifford+T+Rz intermediate"
                )
            if not 0.0 < synthesis_epsilon < 1.0:
                raise ValueError(f"synthesis_epsilon must be in (0, 1), got {synthesis_epsilon}")
        self._gateset = gateset
        self._opt_level = opt_level
        self._device = device
        self._router = router
        self._modeler = modeler
        self._device_label = device_label
        self._synthesis_epsilon = synthesis_epsilon
        self._transpiler: "qx.Transpiler | None" = None
        if gateset is not None:
            # A GateSet carries its own overrides (GateSet.rules); no install.
            self._transpiler = qx.Transpiler(gateset)

    def _routed(self) -> bool:
        return self._device is not None and self._device.architecture is not None

    def _pre_route_transpiler(self) -> "qx.Transpiler":
        """Sibling of ``self._transpiler`` targeting the routable subset.

        Built per call rather than cached: the object is cheap and a stale
        copy would silently diverge from a re-configured gateset.
        """
        assert self._gateset is not None
        return qx.Transpiler(qx.routable_subset(self._gateset))

    def estimate(self, source: "qx.Block | Sequence[qx.Command]") -> ResourceReport:
        if isinstance(source, qx.Block):
            if not source.is_built:
                source.build()
            cmds = source.flatten()
            n_qubits: int | None = source.n_qubits
        else:
            cmds = list(source)
            n_qubits = None

        stages: dict[Stage, ResourceVector] = {}
        final_cmds = cmds
        stages[Stage.LOGICAL] = count_resources(
            cmds, provenance=Provenance(stage=Stage.LOGICAL), n_qubits=n_qubits
        )

        if self._transpiler is not None:
            assert self._gateset is not None
            gs, opt = self._gateset.name, self._opt_level.name
            final_cmds = self._transpiler.transpile_and_optimize(cmds, self._opt_level)
            stages[Stage.OPTIMIZED] = count_resources(
                final_cmds,
                provenance=Provenance(stage=Stage.OPTIMIZED, gateset=gs, opt_level=opt),
                n_qubits=n_qubits,
            )

            if self._device is not None:
                self._device.check_fits(
                    stages[Stage.OPTIMIZED].n_qubits if n_qubits is None else n_qubits
                )
            if self._routed():
                assert self._device is not None
                opts = qx.RoutingOptions()
                opts.arch = self._device.architecture
                opts.directedness = self._device.directedness
                opts.router = self._router
                # Mirror the engine staging: the router takes 0/1/2-qubit
                # gates only, so wide gates the gateset keeps native (MCZ,
                # CCX) are lowered first.  ROUTED counts them lowered; the
                # OPTIMIZED snapshot above still shows them native.
                routed = qx.route(self._pre_route_transpiler().transpile(final_cmds), opts)
                stages[Stage.ROUTED] = count_resources(
                    routed.commands,
                    provenance=Provenance(
                        stage=Stage.ROUTED,
                        gateset=gs,
                        opt_level=opt,
                        router=self._router.name,
                        device=self._device_label,
                    ),
                    n_qubits=self._device.n_qubits,
                )
                final_cmds = self._transpiler.transpile(routed.commands)
                stages[Stage.TARGET] = count_resources(
                    final_cmds,
                    provenance=Provenance(
                        stage=Stage.TARGET,
                        gateset=gs,
                        opt_level=opt,
                        router=self._router.name,
                        device=self._device_label,
                    ),
                    n_qubits=self._device.n_qubits,
                )

            if self._synthesis_epsilon is not None:
                eps = self._synthesis_epsilon
                # O1/O2 end with fusion (Custom output, kept at OPTIMIZED);
                # gridsynth needs Clifford+T+Rz format, so re-open via the
                # exact rebase — identity on streams already in it.
                final_cmds = self._transpiler.transpile(final_cmds)
                final_cmds = synthesize_clifford_t(final_cmds, epsilon=eps)
                stages[Stage.SYNTHESIZED] = count_resources(
                    final_cmds,
                    provenance=Provenance(
                        stage=Stage.SYNTHESIZED,
                        gateset=gs,
                        opt_level=opt,
                        router=self._router.name if self._routed() else None,
                        device=self._device_label if self._device is not None else None,
                        synthesis=f"gridsynth:eps={eps:g}",
                    ),
                    n_qubits=self._device.n_qubits if self._device is not None else n_qubits,
                )

        if self._modeler is not None:
            last = next(reversed(stages))
            stages[last] = self._modeler.model(final_cmds, stages[last])

        return ResourceReport(stages)


def estimate(
    source: "qx.Block | Sequence[qx.Command]",
    *,
    gateset: "qx.GateSet | None" = None,
    opt_level: "qx.OptLevel" = qx.OptLevel.O1,
    device: "qx.Device | None" = None,
    router: "qx.RouterKind" = qx.RouterKind.Sabre,
    modeler: ResourceModeler | None = None,
    device_label: str | None = None,
    synthesis_epsilon: float | None = None,
) -> ResourceReport:
    """One-liner facade over :class:`ResourceEstimator`."""
    return ResourceEstimator(
        gateset=gateset,
        opt_level=opt_level,
        device=device,
        router=router,
        modeler=modeler,
        device_label=device_label,
        synthesis_epsilon=synthesis_epsilon,
    ).estimate(source)
