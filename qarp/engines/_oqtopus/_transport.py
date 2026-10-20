"""The OQTOPUS transport boundary (§15).

``OqtopusTransport`` is the four-method protocol the engine depends on.
``RemoteOqtopusTransport`` is the only module in qarp that imports
``oqtopus_client``, inside its constructor; ``LocalOqtopusTransport`` runs
the emitted programs on ``QarpSimulator`` and answers with the documented
payload shapes, so users, the example notebook and the tests need no
credentials.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Any, Mapping, Optional, Protocol, Sequence, TypedDict

import qarpx as qx

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

_MISSING_SDK = (
    "RemoteOqtopusTransport needs the OQTOPUS client SDK, which is not "
    'installed.  Install the extra:\n    pip install "openqarp[oqtopus]"\n'
    "or use LocalOqtopusTransport for a credential-free dry run."
)


class JobOutcome(TypedDict):
    """One finished job as ``wait_all`` reports it."""

    job_id: str
    status: str  # "succeeded" | "failed" | "cancelled"
    counts: Optional[dict[str, int]]  # bitstrings, c[0] rightmost; None unless succeeded
    message: Optional[str]


class OqtopusTransport(Protocol):
    """What ``OqtopusEngine`` needs from a service: four calls, no SDK types."""

    def get_device(self, device_id: str) -> dict:
        """The ``device_info`` JSON of ``device_id`` (see ``device_from_oqtopus``)."""
        ...

    def submit(
        self,
        *,
        program: str,
        shots: int,
        device_id: str,
        name: str,
        mitigation_info: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Submit one OpenQASM 3 sampling program; returns the job id."""
        ...

    def wait_all(self, job_ids: Sequence[str], *, timeout: float) -> list[JobOutcome]:
        """Block until every job is terminal; outcomes in ``job_ids`` order.
        Raises ``TimeoutError`` after ``timeout`` seconds."""
        ...

    def cancel(self, job_ids: Sequence[str]) -> None:
        """Best-effort cancellation; unknown or finished ids are ignored."""
        ...


def _all_to_all_device_info(n_qubits: int, device_id: str) -> dict:
    return {
        "device_id": device_id,
        "name": device_id,
        "qubits": [{"id": i, "physical_id": i} for i in range(n_qubits)],
        "couplings": [
            {"control": i, "target": j} for i in range(n_qubits) for j in range(n_qubits) if i != j
        ],
    }


class LocalOqtopusTransport:
    """Credential-free dry run: executes each QASM 3 program on ``QarpSimulator``.

    ``device_info`` is served verbatim by ``get_device``; without one,
    ``n_qubits`` builds an all-to-all device of that width.  ``seed`` makes
    the sampled counts reproducible (job ``k`` draws from ``seed + k``).
    Every call is appended to ``log`` for structural assertions.
    """

    def __init__(
        self,
        device_info: Optional[Mapping[str, Any]] = None,
        *,
        n_qubits: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if device_info is None:
            if n_qubits is None:
                raise ValueError("LocalOqtopusTransport: pass device_info or n_qubits")
            device_info = _all_to_all_device_info(n_qubits, "local")
        self._device_info = copy.deepcopy(dict(device_info))
        self._seed = seed
        self._outcomes: dict[str, JobOutcome] = {}
        self._log: list[dict] = []

    @property
    def log(self) -> list[dict]:
        return self._log

    def get_device(self, device_id: str) -> dict:
        self._log.append({"call": "get_device", "device_id": device_id})
        return copy.deepcopy(self._device_info)

    def submit(
        self,
        *,
        program: str,
        shots: int,
        device_id: str,
        name: str,
        mitigation_info: Optional[Mapping[str, Any]] = None,
    ) -> str:
        k = len(self._outcomes)
        job_id = f"local-{k}"
        self._log.append(
            {
                "call": "submit",
                "job_id": job_id,
                "program": program,
                "shots": shots,
                "device_id": device_id,
                "name": name,
                "mitigation_info": None if mitigation_info is None else dict(mitigation_info),
            }
        )
        cmds, n_qubits, n_cbits, symbols = qx.QASM3Absorber().absorb(program)
        if symbols:
            raise ValueError(
                f"LocalOqtopusTransport: program {name!r} still carries free symbols {symbols}"
            )
        seed = None if self._seed is None else (self._seed + k) % 2**32
        sr = qx.QarpSimulator().run(cmds, n_qubits, shots, seed)
        self._outcomes[job_id] = {
            "job_id": job_id,
            "status": "succeeded",
            "counts": _bitstring_counts(sr, n_cbits),
            "message": None,
        }
        return job_id

    def wait_all(self, job_ids: Sequence[str], *, timeout: float) -> list[JobOutcome]:
        self._log.append({"call": "wait_all", "job_ids": list(job_ids), "timeout": timeout})
        missing = [j for j in job_ids if j not in self._outcomes]
        if missing:
            raise ValueError(f"LocalOqtopusTransport: unknown job ids {missing}")
        return [copy.deepcopy(self._outcomes[j]) for j in job_ids]

    def cancel(self, job_ids: Sequence[str]) -> None:
        self._log.append({"call": "cancel", "job_ids": list(job_ids)})


def _bitstring_counts(sr: qx.SamplingResult, n_cbits: int) -> dict[str, int]:
    """Service-shaped counts: the classical register as a string, c[0] rightmost."""
    if n_cbits > 0:
        rows = Counter(
            "".join("1" if row[i] else "0" for i in reversed(range(n_cbits)))
            for row in sr.cbit_history
        )
        return dict(rows)
    return {format(k, f"0{sr.n_qubits}b"): n for k, n in sr.counts.items()}


class RemoteOqtopusTransport:
    """The default transport: ``oqtopus_client`` over the OQTOPUS user API.

    ``config`` is an ``oqtopus_client.OqtopusConfig``; ``None`` resolves
    ``OQTOPUS_URL`` / ``OQTOPUS_API_TOKEN`` first and the SDK's config file
    (``~/.config/oqtopus/config.ini``) otherwise.  qarp reads no credential
    itself.  Jobs are submitted with ``transpiler_lib:
    None`` because qarp has already compiled for the device (§14).
    """

    def __init__(
        self,
        config: Any = None,
        *,
        poll_interval: float = 1.0,
        poll_backoff: float = 1.2,
        max_poll_interval: float = 10.0,
    ):
        try:
            import oqtopus_client as oc
        except ImportError as exc:
            raise ImportError(_MISSING_SDK) from exc
        if config is None:
            try:
                config = oc.OqtopusConfig.from_env()
            except ValueError:
                config = None  # no OQTOPUS_URL: the client falls back to its config file
        self._client = oc.OqtopusClient(config)
        self._spec_cls = oc.OqtopusJobSpec
        self._api_error = oc.UserApiError
        self._poll_interval = poll_interval
        self._poll_backoff = poll_backoff
        self._max_poll_interval = max_poll_interval

    def get_device(self, device_id: str) -> dict:
        dev = self._client.get_device(device_id)
        info = dev.device_info
        if info and "qubits" in info:
            info = dict(info)
            info.setdefault("device_id", device_id)
            return info
        n = dev.n_qubits
        if n is None:
            raise ValueError(
                f"RemoteOqtopusTransport: device {device_id!r} publishes neither a "
                "device_info topology nor n_qubits"
            )
        return _all_to_all_device_info(int(n), device_id)

    def submit(
        self,
        *,
        program: str,
        shots: int,
        device_id: str,
        name: str,
        mitigation_info: Optional[Mapping[str, Any]] = None,
    ) -> str:
        spec = self._spec_cls.sampling(
            device_id=device_id,
            program=program,
            shots=shots,
            name=name,
            transpiler_info={"transpiler_lib": None},
            mitigation_info=None if mitigation_info is None else dict(mitigation_info),
        )
        return str(self._client.submit_job(spec).job_id)

    def wait_all(self, job_ids: Sequence[str], *, timeout: float) -> list[JobOutcome]:
        deadline = time.monotonic() + timeout
        pending = list(job_ids)
        outcomes: dict[str, JobOutcome] = {}
        interval = self._poll_interval
        while True:
            for jid in list(pending):
                if _status_name(self._client.status(jid)) in TERMINAL_STATUSES:
                    # Fetched only once terminal: presigned result URLs expire.
                    outcomes[jid] = self._fetch(jid)
                    pending.remove(jid)
            if not pending:
                return [outcomes[j] for j in job_ids]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"RemoteOqtopusTransport: jobs still pending after {timeout}s: {pending}"
                )
            time.sleep(min(interval, remaining))
            interval = min(interval * self._poll_backoff, self._max_poll_interval)

    def _fetch(self, job_id: str) -> JobOutcome:
        job = self._client.get_job(job_id)
        status = _status_name(job.status)
        counts: Optional[dict[str, int]] = None
        if status == "succeeded":
            payload: Any = job.sampling
            raw = None
            if payload is not None:
                raw = payload.counts if hasattr(payload, "counts") else payload.get("counts")
            # A succeeded job without a payload is reported with counts=None;
            # the engine turns that into RemoteJobError.
            if raw is not None:
                counts = {str(k): int(v) for k, v in dict(raw).items()}
        return {
            "job_id": job_id,
            "status": status,
            "counts": counts,
            "message": None if job.message is None else str(job.message),
        }

    def cancel(self, job_ids: Sequence[str]) -> None:
        for jid in job_ids:
            try:
                self._client.cancel_job(jid)
            except Exception:
                continue  # finished, unknown to the service, or unreachable: best effort


def _status_name(status: Any) -> str:
    return str(getattr(status, "value", status)).lower()
