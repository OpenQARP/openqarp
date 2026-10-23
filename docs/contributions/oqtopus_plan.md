# OQTOPUS integration — `OqtopusEngine` over `oqtopus-client`, `sx_rz` rebase target

**Status:** In progress — Phases 0–3a done (3a on the oqtopus-cli local stack); Phase 3b (QPU) open
**Author:** Stefano Scali (+ Claude Code)
**Date:** 2026-09-22
**Tier:** Structural (new engine; deliberate edits to §11, §14, §15)
**Branch:** `feature/oqtopus-integration`
**Green-lit:** <pending the Draft PR's `Green-lit at <sha>` comment>
**Scope:**

Phase 0 — compiled backend (ABI bump):
- `cpp/libqarpx/include/qarpx/transpiler/gateset.h`, `cpp/libqarpx/src/transpiler/gateset.cpp` (`sx_rz_gateset()`)
- `cpp/libqarpx/include/qarpx/transpiler/decompositions.h`, `cpp/libqarpx/src/transpiler/decompositions.cpp` (`sx_rz_decompositions()`, rule tag `"sx_rz"`)
- `cpp/libqarpx/src/block/controlled_block.cpp` (`decompose_h_phase_correct` moves to the shared table; the file keeps calling it)
- `cpp/libqarpx/python/bindings.cpp` (`sx_rz_gateset` binding, `SamplingResult` constructor, `QARPX_ABI_VERSION` 15 → 16)
- `qarp/_abi.py` (`EXPECTED_QARPX_ABI` 15 → 16)
- `cpp/libqarpx/python/qarpx.pyi` (regenerated stub for the two bindings)
- `cpp/libqarpx/tests/cpp/test_decomposition_table.cpp` (rule-set closure and phase-exactness)
- `tests/test_transpiler/test_target_rules.py` (Python-side `sx_rz` oracle rows)
- `tests/test_engines/test_sampling_result.py` (new — the constructor)

Phase 1 — pure-Python plumbing, offline:
- `qarp/engines/_oqtopus/__init__.py`, `_transport.py`, `_result.py` (new)
- `qarp/devices/_oqtopus.py` (new, `device_from_oqtopus`), `qarp/devices/__init__.py` (export)
- `qarp/errors.py` (`RemoteJobError`)
- `pyproject.toml` (`oqtopus` extra)
- `tests/test_devices/test_oqtopus_device.py`, `tests/oqtopus_device_topology.py` (new — the
  published example topology as a Python literal; `.json` is a forbidden file type in pre-commit)
- `tests/test_engines/test_oqtopus_transport.py` (new)

Phase 2 — the engine:
- `qarp/engines/_oqtopus/_engine.py` (new), `qarp/engines/__init__.py` (export)
- `tests/test_engines/test_oqtopus_engine.py` (new)
- `examples/engines/mwe_oqtopus.ipynb` (new)
- `docs/source/engines.rst`, `docs/source/installation.rst`, `docs/source/errors.rst` (`RemoteJobError` paragraph)
- `tests/test_engines/test_layering.py` (walks the new subpackage)
- `docs/contracts/qarp_conventions.md` (§11, §14, §15 — see Design)
- `.github/workflows/nightly.yml` (install the `oqtopus` extra in the notebooks job; run the transport suite un-skipped there)
- `.github/workflows/ci.yml` (install the `oqtopus` extra in the pytest/coverage job, so the SDK-oracle transport rows count toward the patch gate)

Phase 3 — live tier:
- `tests/test_engines/test_oqtopus_live.py` (new; gated, `slow`)

Plan bookkeeping: `docs/contributions/oqtopus_plan.md`, `docs/contributions/README.md` (index row).

---

## Why

OpenQARP runs every algorithm on two engines, both local simulators
(`QarpEngine`, `CudaqEngine`).  There is no path from a built block to real
hardware: the emit adapters hand a circuit to another SDK and stop there, so a
user who wants a VQE energy from a QPU leaves qarp at the first shot.

[OQTOPUS](https://github.com/oqtopus-team/oqtopus-cloud) (Osaka / Fujitsu,
Apache-2.0) is an open quantum-cloud stack with a documented user API: one
OpenQASM 3 program per job, a `device_id`, a shot count, counts back as
bitstrings.  Its official SDK, `oqtopus-client` (PyPI, 1.2.0), has no quantum
framework dependency.  It is the smallest real target for a first remote
engine, and everything it needs qarp already has: `to_qasm3()` emission (§12.1),
`compile_for_device` routing (§14), and the `Engine` template whose narrow hooks
(`_compile_one` / `_dispatch_one` / `_sweep`) are the single seam a job
submitter replaces.

Two gaps surfaced while probing the tree, and both are fixed here because the
integration cannot land without them:

1. **No rebase rules reach the hardware basis.**  The OQTOPUS gateway accepts
   `x, sx, rz, cx, measure, barrier` after transpilation; with server-side
   transpilation switched off (the design below routes in qarp) the program must
   already be in that basis.  `Transpiler(GateSet{X, SX, Rz, CX})` today reports
   `H, Rx, Ry, S, T, Y, Z` unreachable: the built-in table lowers *toward*
   `{H, S, T, Rx, Ry, Rz, CX, …}`, never toward `SX`.
2. **`qx.SamplingResult` has no Python constructor.**  Counts arriving from a
   service cannot be wrapped in the one result type §14 mandates.

## Design

### Decisions taken in the sit-down

- **Shape: a new `Engine` subclass**, `OqtopusEngine`, so `Sampler`,
  `PauliAveraging`, VQE and the parameter-shift / finite-diff / SPSA gradients
  run on an OQTOPUS device unchanged.  Not an export-only adapter.
- **qarp always compiles.**  The engine builds a `qx.Device` from the service's
  `device_info` (directed coupling map + the `sx_rz` gate set), runs
  `qx.compile_for_device` (check_fits → rebase → route → rebase, §14), and
  submits with `transpiler_info = {"transpiler_lib": None}` so the server
  transpiler is skipped.  Counts come back keyed by logical qubit, because
  the engine lays the measurement layer on the final physical wires itself
  (*How it fits*); nothing is reindexed.  Consequence: `multi_manual` (server-side
  packing of several programs into one wide circuit) is not used — it conflicts
  with physical-qubit routing — and `_sweep` submits one job per circuit.
- **Transport: `oqtopus-client`, lazily.**  A new `[oqtopus]` extra; the SDK is
  imported inside the constructor and a missing install is `ImportError` naming
  the extra (§15).  It is *not* added to `[full]` / `[full-dev]` (coverage-floor
  re-baseline, §15).
- **Offline mode is public API.**  `LocalOqtopusTransport` executes the emitted
  QASM 3 on `QarpSimulator` through `SimpleBlock.from_qasm3` and answers with the
  documented payload shapes.  It is the credential-free dry run for users, what
  the example notebook runs on in the nightly, and the test double.
- **Rebase target name: `sx_rz_gateset()`**, after the basis (like
  `clifford_t_rz_gateset`), not after the consumer.

### How it fits (§14)

`OqtopusEngine(Engine)` lives in `qarp/engines/_oqtopus/_engine.py` and mirrors
`CudaqEngine`'s shape:

- `provides_amplitudes = False` (static), `supports_initial_state = False`,
  `gradient_methods = {"default", "parameter-shift", "finite-diff", "spsa"}`,
  `_host_statevector_simulator()` returns `None`, `_plan_unbuilt` /
  `_plan_built` keep their defaults (no structured programs).  These declarations alone make the base template refuse
  `StateVector`, `n_shots=qarp.EXACT`, `initial_state` and the adjoint gradient
  with typed `CapabilityError`s — no per-call guards in the engine.
- `_routed()` is `True`; `_compile_one` runs `compile_for_device` and then
  **owns the measurement layer**: terminal `Measure`s are lifted out of the
  logical stream before routing and re-laid at the end on the final physical
  wire of their logical qubit (`c[c] = measure q[final(l)]`); a
  measurement-free circuit (`Sampler` kets) measures every logical qubit into
  its own bit.  The per-circuit record the base threads through is a
  `_Wiring(cbit_to_logical)`, not a `_Layout`: the service returns
  classical-register keys, whose bit order is the *cbit* order, so an
  explicitly measured circuit (`PauliAveraging` measures qubit `i` into
  `c[i]`) comes back in logical order already and a final-map reindex would
  be wrong.  `CompiledCircuit.n_qubits` is the block width — the width the
  primitives read, since the counts are already in logical order.
- `_post_compile_check` calls the shared `_reject_true_mcm` — OQTOPUS exposes no
  feed-forward — and rejects `initial_state` through the base guards.
- `_dispatch_one` and `_sweep` are the only remote code: emit each substituted
  circuit with `qx.QASM3Emitter` (the `GPhase` a rebase leaves behind has been
  dropped by the target already, so the program uses `stdgates.inc` names only),
  submit every job, wait once, convert, hand `prim.run(results)`.
  `_sweep` submits `len(param_sets) × Σ circuits` jobs up front and assembles
  `results[set_idx][prim_idx]`, which is what makes `evaluate_stencils`
  (parameter-shift) one round trip per gradient.
- **No seed.**  Hardware randomness is not seedable; the constructor takes no
  `seed`, `_seed` is `None`, `_circuit_seed` therefore returns `None`.  The
  local transport takes its own `seed` for reproducible dry runs.
- **No noise model, no `device=`.**  The device is fetched, never passed, so
  the §14 rule "an engine that does not consume `device.noise_model` must raise"
  is satisfied by construction (there is nothing to drop).
- **`mitigation_info` pass-through.**  The constructor takes
  `mitigation_info: Mapping[str, Any] | None` and every `submit` forwards it
  verbatim; qarp neither reads nor validates its keys, they belong to the
  service.  Readout mitigation returns counts in the submitted register
  layout, so the wiring is unaffected.  `transpiler_info` is *not*
  exposed: a server that routes would invalidate the wiring.
- **Result contract.**  Counts → `qx.SamplingResult(n_qubits, counts, n_shots)`
  via the new constructor, keyed by logical qubit through the circuit's
  `_Wiring`; nothing is reindexed afterwards.  Bit order: OQTOPUS keys are
  Qiskit-style strings, `c[0]` rightmost, so under the identity wiring
  `int(key, 2)` has bit *i* = `c[i]` = qubit *i* — qarp's LSB integer key (§1)
  with no reversal.  The conversion goes through one function
  (`_result.counts_to_sampling_result(counts, n_qubits=, n_shots=, wiring=)`)
  with its own oracle tests, not through an inline `int(k, 2)`.
- **Errors.**  Capability failures stay `CapabilityError`.  A job that ends
  `failed` or `cancelled` raises `qarp.errors.RemoteJobError(RuntimeError)`
  carrying `job_id`, `status` and the service `message`.  A transport
  `TimeoutError` or `KeyboardInterrupt` during `wait` — any exception leaving
  submit or wait — cancels every outstanding job of that call before
  re-raising.

### The transport boundary (§15)

```
qarp/engines/_oqtopus/
    _transport.py   OqtopusTransport (Protocol), RemoteOqtopusTransport, LocalOqtopusTransport
    _result.py      counts_to_sampling_result(...)
    _engine.py      OqtopusEngine
qarp/devices/
    _oqtopus.py     device_from_oqtopus(info) -> qx.Device
```

`OqtopusTransport` is a four-method PEP 544 protocol (below).  The engine
depends on the protocol only.  `LocalOqtopusTransport` parses programs with
`qx.QASM3Absorber` (the parser `SimpleBlock.from_qasm3` wraps) and runs them
on `qx.QarpSimulator`, so it imports no block class at all, and
`qarp/engines/` never imports `qarp.algorithms` (`test_layering.py`, extended
to walk the new subpackage).  Service-shaped keys are built from the
simulator's `cbit_history` (the classical register, `c[0]` rightmost).  `RemoteOqtopusTransport` is the default and the
sole importer of `oqtopus_client`; it builds `OqtopusJobSpec.sampling(...)`,
calls `submit_job`, polls `status()` and fetches `get_job()` once terminal
(presigned result URLs expire after one hour — never fetch early).  Config
resolution is the SDK's: an explicit `OqtopusConfig`, else `OQTOPUS_URL` +
`OQTOPUS_API_TOKEN`, else `~/.config/oqtopus/config.ini`.  qarp reads no
credential itself and stores none.

`device_from_oqtopus(info)` is a pure function over the `device_info` JSON the
service serves (fields `qubits[].id`, `couplings[].control/target`): it returns
`qx.Device(n, Architecture(n, edges, name=device_id), gate_set=sx_rz_gateset(),
directedness=True)`.  It lives under `qarp.devices` (`_oqtopus.py`) because it
is a device builder, not an engine concern, and useful for `ResourceEstimator`
runs against a real topology without any network call; `qarp.devices` cannot
re-export it from the engines package without importing the engines (which
import `qarp.devices`) during its own initialisation.  A device that publishes
no topology but an `n_qubits` is served as all-to-all by the remote transport.

### The `sx_rz` target (§11, §16)

`sx_rz_gateset()` = `{X, SX, Rz, CX, Measure, Barrier}`, `rules = "sx_rz"`.
`sx_rz_decompositions()` starts from the built-in table and overrides every
single-qubit gate that is not in the set with a phase-exact `Rz·SX·Rz` form
carrying an explicit `GPhase`, so the table stays composable (§11: rules are
phase-exact; only the *target* drops `GPhase`, via the existing
`decompose_gphase → []` boundary, as cudaq does).  Overrides: `H`
(`decompose_h_phase_correct`, moved out of `controlled_block.cpp` into the
shared table and re-used there), `Rx`, `Ry`, `S`, `Sdg`, `T`, `Tdg`, `Y`, `Z`,
`P`, `U`.  Two-qubit gates are untouched — the built-in table already reaches
`CX`, and `CZ → H·CX·H` cascades through the `H` override.  `decompositions_for`
learns the tag; the "unknown rule set" error lists it.

### Convention edits (deliberate, all three)

- **§11** — the per-target table gains the `sx_rz` row next to `clifford_t_rz`,
  and the `H` paragraph is corrected: its phase-exact lowering now lives in the
  shared table (`decompositions.cpp`), with `controlled_block.cpp` a consumer.
- **§14** — a new bullet, *Remote engines*: `OqtopusEngine` is the first;
  a remote engine takes no `seed` and no `device=`, declares
  `provides_amplitudes = False`, spells out end-of-circuit measurement, wraps
  counts through the `SamplingResult` constructor, raises `RemoteJobError` for
  a job the service failed, and cancels outstanding jobs on the way out of a
  failed `wait`.  The engine-name list adds `OqtopusEngine`.
- **§15** — the first bullet ("SDK integrations are optional and lazy … core
  code paths never route through an SDK") is amended: a *cloud client* SDK may
  be routed through by a remote engine, behind its own extra, imported inside
  the constructor, never at module scope, and never by blocks / transpiler /
  local engines.  `oqtopus-client` is named as the one such dependency.

No edit to §12 (the emitter is used as-is) or §13.

### Deliberately not in this plan

- The OQTOPUS `estimation` job type.  It re-spends the full `shots` per
  commuting group, serially, and returns one scalar with no per-term
  statistics; `PauliAveraging` over `sampling` jobs is strictly more
  informative.  Cost accepted: an energy evaluation is one sampling job per
  commuting group instead of one estimation job, and on a queued service the
  per-job latency, not the shot count, is the price; `_sweep` submitting every
  job up front is the mitigation.  A later plan may add estimation as a fast
  path in the `CudaqEngine.batch_expectation` style; the client expresses an
  estimation operator as `{pauli, coeff}` items, so that path converts qarp's
  own operator directly, with no openfermion or quri-parts step.
- `multi_manual` packing and server-side transpilation as an option
  (`transpiler_info` stays fixed, see above).
- Running qarp *inside* OQTOPUS's server-side execution (SSE) container, and
  SSE jobs submitted from the engine.  The former is a deployment of this
  engine, not a second integration: the operator pins `openqarp` in the SSE
  runtime's `pyproject.toml` and the script inside the container constructs
  `OqtopusEngine` with the in-cluster `OqtopusConfig`.  Two constraints it
  places on qarp releases: OQTOPUS has no wheel upload, so PyPI is the only
  route into the container, and a manylinux wheel for the container's Python
  avoids a from-sdist C++ build at every image rebuild; a GPU build needs C++
  build steps in the operator's Dockerfile, which is their work.
- Any change to `Device` (stays passive data, §14).

## API sketch

```python
# qarp/engines/_oqtopus/_transport.py
class JobOutcome(TypedDict):          # what wait_all returns per job
    job_id: str
    status: str                       # "succeeded" | "failed" | "cancelled"
    counts: dict[str, int] | None     # MSB-first bitstrings, c[0] rightmost
    message: str | None

class OqtopusTransport(Protocol):
    def get_device(self, device_id: str) -> dict: ...            # the device_info JSON
    def submit(self, *, program: str, shots: int, device_id: str, name: str,
               mitigation_info: Mapping[str, Any] | None = None) -> str: ...  # job_id
    def wait_all(self, job_ids: Sequence[str], *, timeout: float) -> list[JobOutcome]: ...
    def cancel(self, job_ids: Sequence[str]) -> None: ...

class RemoteOqtopusTransport:         # the only module that imports oqtopus_client
    def __init__(self, config: "oqtopus_client.OqtopusConfig | None" = None,
                 *, poll_interval: float = 1.0, poll_backoff: float = 1.2,
                 max_poll_interval: float = 10.0): ...
    # submit(): OqtopusJobSpec.sampling(device_id, shots, program, name,
    #           transpiler_info={"transpiler_lib": None},
    #           mitigation_info=mitigation_info) -> submit_job().job_id
    # wait_all(): status() polling; get_job() only once terminal

class LocalOqtopusTransport:
    """Credential-free dry run: executes each QASM 3 program on QarpSimulator."""
    def __init__(self, device_info: dict | None = None, *, n_qubits: int | None = None,
                 seed: int | None = None): ...
    # device_info=None and n_qubits=n -> an all-to-all n-qubit device_info
    # get_device / submit / wait_all / cancel per the protocol; job ids "local-<k>"
    @property
    def log(self) -> list[dict]: ...  # every call, for structural assertions

# qarp/devices/_oqtopus.py   (exported as qarp.devices.device_from_oqtopus)
def device_from_oqtopus(info: dict) -> qx.Device: ...
    # n = len(info["qubits"]); edges = [(c["control"], c["target"]) for c in info["couplings"]]
    # gate_set = qx.sx_rz_gateset(); directedness=True; Architecture name = info["device_id"]
    # ValueError on a qubit id outside range(n) or a coupling referencing one

# qarp/engines/_oqtopus/_result.py
def counts_to_sampling_result(counts: dict[str, int], *, n_qubits: int, n_shots: int,
                              wiring: Sequence[int | None] | None = None) -> qx.SamplingResult: ...
    # wiring[c] = logical qubit measured into c[c] (None = unused bit); default identity,
    # under which key -> int(key, 2).  len(key) must equal len(wiring); sum(counts) == n_shots

# qarp/engines/_oqtopus/_engine.py
class OqtopusEngine(Engine):
    supports_initial_state = False
    gradient_methods = frozenset({"default", "parameter-shift", "finite-diff", "spsa"})

    def __init__(
        self,
        device_id: str,
        *,
        transport: OqtopusTransport | None = None,   # None -> RemoteOqtopusTransport()
        n_shots: int = 1_000,
        timeout: float = 600.0,                       # per dispatch / sweep call
        job_name: str = "qarp",                       # prefix; jobs are f"{job_name}:{prim}:{k}"
        mitigation_info: Mapping[str, Any] | None = None,  # forwarded verbatim on every submit
    ): ...
    # n_shots is int only: Shots.EXACT raises CapabilityError at construction
    @property
    def device(self) -> qx.Device: ...                # built once from get_device()
    @property
    def provides_amplitudes(self) -> bool: return False
    def _routed(self) -> bool: return True
    def _compile_one(self, flat, block_n_qubits): ...       # compile_for_device + measurement layer (_Wiring)
    def _post_compile_check(self, prim, compiled, layout): ...  # _reject_true_mcm
    def _compile_stamp(self): ...                     # (type, device): foreign circuits recompile
    def _dispatch_one(self, prim, compiled, ordinal, params): ...   # CompiledCircuit.layout is the _Wiring
    def _sweep(self, primitives, compiled_per_prim, param_sets, shots_override): ...

# qarp/errors.py
class RemoteJobError(RuntimeError):
    job_id: str; status: str; message: str | None

# qarpx (bindings.cpp)
qx.sx_rz_gateset() -> qx.GateSet
qx.SamplingResult(n_qubits: int, counts: dict[int, int], n_shots: int, n_cbits: int = 0)
    # ValueError if any key >= 2**n_qubits or sum(counts.values()) != n_shots;
    # cbit_history stays empty (no per-shot register from a remote service)

# pyproject.toml
oqtopus = ["oqtopus-client>=1.3"]
```

Public surface (§15, `__all__`): `qarp.engines.OqtopusEngine`,
`qarp.engines.OqtopusTransport`, `qarp.engines.LocalOqtopusTransport`,
`qarp.engines.RemoteOqtopusTransport`, `qarp.devices.device_from_oqtopus`,
`qarp.errors.RemoteJobError`, `qarpx.sx_rz_gateset`.  Everything under
`qarp/engines/_oqtopus/` is private.

### Open items (explicit; resolved before or during the relevant phase)

1. **Program qubit index vs. `device_info.qubits[].id` / `physical_id`.**  The
   device-gateway example carries both.  Assumed: the QASM register index *is*
   `id`, and `couplings` reference `id`.  Verified only against the Qulacs
   gateway source; a QPU gateway is UNVERIFIED.  Resolution: the QPU run of
   Phase 3 runs a one-CX-per-edge probe; until then the assumption is
   documented on `device_from_oqtopus`.
2. **`transpiler_lib: None` on a QPU device.**  Settled for the Qulacs
   gateway on the running local stack: the `sx_rz` Bell program runs
   verbatim, and the same program with a raw `h` ends `failed`
   (`Unsupported instruction: h` in the gateway log), which is the evidence
   that nothing is transpiled server-side.  `test_oqtopus_live.py` asserts
   both; a QPU gateway behaving differently fails those rows and is a
   QPU-run finding, not a design change.
3. **Hardware credentials and device ids.**  None in hand.  The local stack
   from [oqtopus-cli](https://github.com/oqtopus-team/oqtopus-cli)
   (`cloud-local` + `backend` templates, default `qulacs` device, API on
   `http://127.0.0.1:8080`, token accepted empty by the service) is the
   sandbox for everything but open item 1; it is set up outside the repo and
   never in CI.  Only the QPU run waits for credentials.  Settled on first
   contact: `OqtopusConfig(url=..., api_token="")` is accepted by the stack,
   so `OQTOPUS_API_TOKEN` may be empty.  One trap: the database seeds
   `qulacs` with a 16-qubit `device_info` carrying no couplings, and the
   gateway replaces it with its own topology (16 qubits, 22 directed
   couplings) shortly after it starts; a device built from the seed is
   edgeless and the router refuses every CX.  Query the device only once
   the gateway reports `active`.
4. **Mitigated counts.**  With `mitigation_info` set, are the returned counts
   still integers summing to `shots`?  Readout mitigation can yield real,
   even negative, values; `counts_to_sampling_result` would reject those with
   a shot-sum `ValueError` after a paid job.  On the local stack
   `mitigation_info={"readout": "pseudo_inverse"}` is echoed on the job and
   the counts come back as integers summing to `shots` (the simulator has
   no readout calibration to invert); `test_oqtopus_live.py` asserts it.
   Hardware remains unverified, so the docs still say integer counts are
   expected back; a fractional payload is refused at fetch, naming
   mitigation, never truncated.
5. **`oqtopus-client` floor.**  Every SDK surface the transport uses was
   verified against 1.3.0 only, hence the `>=1.3` pin; relax if an older
   release is shown to carry them.

## Test plan

Shot-based rows assert within `k·σ` of the analytic value with `σ` from the
binomial at the stated shots (the form `tests/test_pipeline/test_engine_agreement.py`
uses).  Round-trips (engine on the local transport vs. `QarpEngine` on the same
block) appear only as *additional* rows.

| Test | Oracle | Location |
|---|---|---|
| `sx_rz_decompositions()` output for each of `H, Rx, Ry, S, Sdg, T, Tdg, Y, Z, P, U` at `θ, φ, λ ∈ {0.7, −0.4, 2.9}` equals the gate's matrix **exactly** (GPhase retained) | analytic 2×2 matrices assembled with numpy (§2 tables) | `test_target_rules.py::test_sx_rz_phase_exact` |
| Same, in C++: rule outputs contain only `{X, SX, Rz, GPhase}` and the product equals the source unitary to 1e-12 | analytic, `qarpx::unitary_matrix` of hand-built `Rz`/`SX` products | `test_decomposition_table.cpp` |
| `Transpiler(sx_rz_gateset()).transpile()` of a native-gateset sweep circuit yields only allowed gates and a unitary equal **up to global phase** to the input; `GPhase` absent from output | analytic (input built from the §2 matrices) | `test_target_rules.py::test_sx_rz_target_closure` |
| `Device(gate_set=sx_rz_gateset(), architecture=line)` through `compile_for_device`: `P_final⁻¹·U_routed·P_initial` equals the unrouted unitary up to phase | the §14 routing oracle | `test_target_rules.py::test_sx_rz_routed_device` |
| `decompositions_for` with an unknown tag still throws and the message lists `"sx_rz"` | contract | `test_decomposition_table.cpp` |
| `qx.SamplingResult(2, {0: 5, 3: 5}, 10)` exposes the fields; `sum ≠ n_shots`, `key ≥ 2ⁿ`, a negative count and `n_qubits` outside `[0, 63]` raise `ValueError`; `cbit_history` empty | contract | `test_sampling_result.py` |
| `reindex_sampling_result(SamplingResult(2, {1: 10}, 10), [1, 0]) == {2: 10}` | analytic bit permutation | `test_sampling_result.py::test_reindex_constructed` |
| `device_from_oqtopus` on the device-gateway example topology: `n_qubits`, edge list, `directedness`, gate-set name | the published device-gateway `device_topology.json` example (transcribed verbatim, with its source URL, into `tests/oqtopus_device_topology.py`) | `test_oqtopus_device.py` |
| `device_from_oqtopus` rejects a coupling naming an unknown qubit | contract | `test_oqtopus_device.py` |
| `counts_to_sampling_result({"00": 506, "11": 494}, n_qubits=2, n_shots=1000).counts == {0: 506, 3: 494}`; `{"01": 7}` → `{1: 7}` | the OQTOPUS user-API spec example payload and the device-gateway `_remap_counts` docstring ("bit index 0 rightmost") | `test_oqtopus_transport.py::test_counts_bit_order` |
| `LocalOqtopusTransport` on the spec's Bell program returns keys ⊆ `{"00", "11"}` at ≈ ½ each; on `x q[0];` returns `{"01": shots}` | OQTOPUS spec example; analytic | `test_oqtopus_transport.py::test_local_payload_shape` |
| `LocalOqtopusTransport` status values ∈ the spec enum; `cancel` on an unknown id is a no-op; `log` records every call | the `jobs.JobStatus` enum from the spec | `test_oqtopus_transport.py` |
| `RemoteOqtopusTransport.submit` builds a spec with `job_type == "sampling"` and `transpiler_info == {"transpiler_lib": None}` (client monkeypatched at `submit_job`) | `oqtopus_client.OqtopusJobSpec.to_model()` — the SDK as oracle | `test_oqtopus_transport.py` (`importorskip("oqtopus_client")`) |
| Constructing `RemoteOqtopusTransport` without the SDK raises `ImportError` naming `pip install "openqarp[oqtopus]"` | contract (module blocked via `sys.modules`) | `test_oqtopus_transport.py` |
| `OqtopusEngine(..., mitigation_info=m)` → every `submit` in `transport.log` carries `m` unchanged; `RemoteOqtopusTransport.submit` puts it in the spec's `mitigation_info` verbatim | structural; the SDK's `to_model()` as oracle | `test_oqtopus_engine.py::test_mitigation_info_forwarded`, `test_oqtopus_transport.py` |
| `Sampler(GHZ₃)` on `OqtopusEngine(local)`: `{000: ½, 111: ½}` within 5σ at 4000 shots, no other key | analytic | `test_oqtopus_engine.py::test_sampler_ghz` |
| `PauliAveraging(Z₀)` on `Ry(θ)|0⟩`: `cos θ` within 5σ for θ ∈ {0.3, 1.1, 2.5} | analytic | `…::test_expectation_cos` |
| Routed: line topology `0–1–2`, ket `X(0)·CX(0,2)` → single key `101` (logical order) | analytic + §14 reindex rule | `…::test_routed_counts_logical_order` |
| SWAP-forcing route (`X(1)·CX(0,1)·CX(1,2)·CX(0,2)` on the line, `initial ≠ final` asserted) → single key `011`; reading the initial map would give `101` | analytic; mutation-checked (`final → initial` fails this row alone) | `…::test_swap_forcing_route_lays_measures_on_the_final_wires` |
| Explicit `measure(2, 0)` on the routed `X(0)·CX(0,2)` → single key `101`; an identity cbit→logical wiring would read `110` | analytic; mutation-checked | `…::test_explicit_measure_into_a_permuted_cbit_is_wired_back` |
| A symbolic circuit run with no parameters raises `ValueError` naming the symbol before any `submit` (`transport.log` empty) | contract | `…::test_unbound_symbols_are_refused_before_any_submit` |
| Engine lays the measurement layer: a measurement-free block submits `bit[n] c;` and `c[i] = measure q[i]` for every *logical* qubit; a block measuring one qubit explicitly still measures the rest (uniform H⊗3 within 5σ) | §12.1 emission table; analytic | `…::test_measure_all_appended`, `…::test_partially_measured_block_still_reports_every_qubit` |
| Transport whose `cancel` raises → the original `TimeoutError` still propagates; circuits compiled by another engine reused with `rebuild=False` are recompiled here (stamp mismatch), measurement layer included | contract; analytic `sin²(θ/2)` | `…::test_failed_cancel_does_not_mask_the_timeout`, `…::test_foreign_compiled_circuits_are_recompiled` |
| `RemoteOqtopusTransport()` with `OQTOPUS_URL` set and no config file resolves the env; `cancel` swallows transport errors | contract | `test_oqtopus_transport.py` |
| `RemoteOqtopusTransport` refuses a fractional count at fetch with `ValueError` naming mitigation; an integral float is accepted as `int` | contract (open item 4) | `…::test_remote_transport_rejects_fractional_counts`, `…::test_remote_transport_accepts_integral_float_counts` |
| `batch_run` over 3 parameter sets: `len(transport.log["submit"]) == 3 × n_circuits`, one `wait_all`; per-set values equal `run()` at the same params within 5σ | structural + analytic (`cos θ` rows) | `…::test_sweep_submits_all_then_waits` |
| Parameter-shift VQE, 1 qubit, `H = Z`, ansatz `Ry(θ)`: converges to `θ ≈ π`, energy `−1 ± 5σ` | analytic minimum | `…::test_vqe_one_qubit_parameter_shift` |
| `slow`: VQE H₂ (STO-3G, JW) on the local transport, started at the Hartree–Fock point (0.02 Ha above), reaches `−1.137 ± 0.01` Ha at 8000 shots | published FCI energy (Szabo–Ostlund; the value already pinned in `tests/pccd_reference_energies.py`) | `…::test_vqe_h2_slow` |
| `StateVector`, `n_shots=qarp.EXACT`, `initial_state=`, a mid-circuit `Measure` + conditional, and `gradient="adjoint"` each raise `CapabilityError` | §14 contract | `…::test_capability_refusals` |
| `OqtopusEngine(..., n_shots=qarp.EXACT)` raises `CapabilityError` at construction | §14 contract | `…::test_exact_rejected_at_construction` |
| Transport returning `status="failed"` → `RemoteJobError` with `job_id` and `message`; returning `"cancelled"` likewise | contract | `…::test_remote_job_error` |
| Transport raising `TimeoutError` from `wait_all` → every submitted id passed to `cancel`, then the error propagates | contract (`transport.log`) | `…::test_timeout_cancels_outstanding` |
| `import qarp.engines` does not import `oqtopus_client` (checked in `sys.modules`) | §15 laziness | `test_oqtopus_transport.py::test_importing_qarp_engines_does_not_import_the_sdk` |
| Engine layering: `qarp/engines/_oqtopus/*` imports nothing from `qarp.algorithms` | existing `test_layering.py` lint (extended to the subpackage) | `test_layering.py` |
| Additional (round-trip): `Sampler` on a random 4-qubit Clifford circuit, local transport vs. `QarpEngine`, total-variation distance < 0.05 at 4000 shots | not an oracle — agreement only | `…::test_agrees_with_qarp_engine` |
| Live, local stack (Phase 3a, gated on `OQTOPUS_URL`, `OQTOPUS_API_TOKEN`, `QARP_OQTOPUS_DEVICE=qulacs`; `slow`): `device_from_oqtopus(get_device("qulacs"))` builds; the `sx_rz` Bell program submitted with `transpiler_lib: None` ends `succeeded` and the same program with a raw `h` ends `failed`; `x q[0];` returns the single key `"01"`; the engine's Bell counts within 5σ of ½ each, with and without readout mitigation | analytic; the bit-order and pass-through claims checked on the running gateway rather than its source | `test_oqtopus_live.py` |
| Live, QPU (Phase 3b, same gating with a hardware device id; `slow`): Bell state on the named device, `P(00)+P(11) > 0.8`; one-CX-per-edge probe confirms open item 1 | analytic (loose bound — real hardware) | `test_oqtopus_live.py` |
| Notebook `mwe_oqtopus.ipynb` runs on `LocalOqtopusTransport` in the nightly (`--nbmake`) | executed example surface (§15) | `examples/engines/` |

## Phases

Each phase is landable and reviewable on its own; Phase 0 is useful without
the rest (an IBM-style rebase target and a constructible result).

### Phase 0 — compiled backend: `sx_rz` target, `SamplingResult` constructor

- [x] `sx_rz_gateset()` + `sx_rz_decompositions()`; `decompose_h_phase_correct`
      moved into the shared table, `controlled_block.cpp` re-uses it (2026-10-16)
- [x] `decompositions_for` knows `"sx_rz"`; error message lists it (2026-10-16)
- [x] `SamplingResult` constructor binding with validation (2026-10-16)
- [x] `QARPX_ABI_VERSION` / `EXPECTED_QARPX_ABI` bumped, same commit (2026-10-16; 16 after the develop merge, 2026-10-23)
- [x] C++ + Python oracle rows above (2026-10-16)
- [x] §11 edit (2026-10-16)

### Phase 1 — plumbing, offline

- [x] `OqtopusTransport` protocol, `LocalOqtopusTransport`, `RemoteOqtopusTransport` (2026-10-16; renamed 2026-10-20)
- [x] `device_from_oqtopus` + `tests/oqtopus_device_topology.py` (2026-10-16)
- [x] `counts_to_sampling_result` (2026-10-16)
- [x] `RemoteJobError`; `[oqtopus]` extra (2026-10-16)
- [x] Transport / device / result oracle rows above (2026-10-16)

### Phase 2 — the engine

- [x] `OqtopusEngine` per the API sketch, `mitigation_info` included; exports (2026-10-16; ported to the `CompiledCircuit` template 2026-10-23)
- [x] Engine oracle rows above (incl. the `slow` H₂ row and the layering lint) (2026-10-16; discriminating rows added 2026-10-23)
- [x] `docs/source/engines.rst` section "OqtopusEngine — remote execution";
      `installation.rst` paragraph for the extra (2026-10-16)
- [x] `examples/engines/mwe_oqtopus.ipynb` on `LocalOqtopusTransport` (2026-10-16)
      (an inline copy of the example topology — `examples/` never imports
      `tests/`, §15 — GHZ sampling, one VQE step, and the
      `RemoteOqtopusTransport` construction shown but not executed)
- [x] `nightly.yml`: install `[…,oqtopus]` in the notebooks job; run
      `tests/test_engines/test_oqtopus_transport.py` un-skipped there (2026-10-16)
- [x] §14 and §15 edits (2026-10-16)
- [x] Index row added (2026-10-16; the index carries no status column, this header does)

### Phase 3 — live tier (gated + `slow`, never in CI)

3a, local stack (open items 2–4):

- [x] `test_oqtopus_live.py` against the oqtopus-cli stack, device `qulacs`
      (cloud v1.8.0, engine v2.2.1, gateway v1.1.16): 6 passed, QPU probe
      skipped (2026-10-19; re-run green after the merge, 2026-10-23)
- [x] Open item 2 closed for the Qulacs gateway; open item 3's empty token
      accepted; open item 4 settled for the simulator only (2026-10-19)

3b, QPU (blocked on credentials; open item 1):

- [ ] The same file against a hardware device id; the one-CX-per-edge probe
- [ ] Fold the findings into `device_from_oqtopus`'s docstring and this plan

## Deviations log

Declared during implementation and folded into the sections above:

- **Measurement layer and no reindex** (Design, "How it fits", "Result
  contract").  The plan reindexed physical counts by the final map and left
  explicitly measured circuits alone.  The service returns classical-register
  keys, so an explicitly measured circuit comes back in cbit order, which a
  final-map reindex would scramble.  The engine now lifts terminal `Measure`s
  out before routing and re-lays them on the final wires with an explicit
  `_Wiring`; `counts_to_sampling_result` gained a `wiring=` argument.
- **`device_from_oqtopus` lives in `qarp/devices/_oqtopus.py`**, not under the
  engines package (import cycle: `qarp.devices` ← engines ← `qarp.devices`).
- **The sx_rz `H` rule is its own function** (`decompose_h_sx_rz`, one `SX`);
  `decompose_h_phase_correct` still moved into the shared table for the
  multi-control basis.  `Sdg`/`Tdg` get direct `Rz` rules; `P` keeps the
  built-in one.
- **`LocalOqtopusTransport` parses with `qx.QASM3Absorber`** rather than
  `SimpleBlock.from_qasm3` (same parser, no block import).
- **Cancellation on any exception** leaving submit/wait, not only timeout and
  Ctrl-C.
- **Remote transport fallbacks**: a device without a published topology is
  served all-to-all from `n_qubits`; a succeeded job without a payload is
  reported with `counts=None` and raised as `RemoteJobError`; with no
  `config`, `OqtopusConfig.from_env()` is tried before the SDK's config file
  (the SDK alone reads only the file).
- **Review fix pass**: every logical qubit a block leaves unmeasured is
  measured too (the Sampler contract is the full register, as on
  `QarpEngine`); a failing `cancel` never masks the error that triggered it.
- **Phase 3a rows** (Test plan, open items 2–4).  The planned
  "untranspiled Bell" row used `h`, which the gateway refuses under
  `transpiler_lib: None`; the row now submits the `sx_rz` form and a second
  row asserts the raw-`h` program ends `failed`.  A mitigation row covers
  open item 4 on the simulator.
- **`RemoteOqtopusTransport`** is the name of the SDK-backed transport (file
  layout, API, test plan).  It pairs with `LocalOqtopusTransport` on the
  distinction a user cares about, where the circuits run, rather than on the
  package it wraps.
- **Ported to the lazy-circuit / structured-execution Engine template**
  (API).  `develop` now hands hooks `CompiledCircuit` records: the routing
  record travels as `CompiledCircuit.layout` (the `_Wiring`) instead of an
  engine-held list, `_dispatch_one` binds parameters itself, and
  `_compile_stamp` is `(type, device)`, so circuits compiled by another
  engine or against another topology are recompiled under `rebuild=False`
  rather than refused.  No structured programs are planned: `_plan_unbuilt`
  and `_plan_built` keep their defaults, every circuit runs as a gate stream.
- **Review round before the Draft PR** (three independent conformance
  reviews).  Unbound symbols are refused in the engine before any submit
  (`ValueError` naming them), not left to the service; the remote transport
  refuses fractional counts at fetch instead of truncating; three test rows
  were made discriminating (a SWAP-forcing route, an explicit measure into
  a permuted cbit, H₂ started at Hartree–Fock) after mutation checks showed
  the original rows passed with the final map replaced by the initial one;
  `ci.yml`'s pytest job installs the `oqtopus` extra so the transport rows
  count toward the 80 % patch gate.
- **Bookkeeping after the review round.**  Three files outside the
  green-lit Scope are now listed there: `docs/source/errors.rst` (the
  `RemoteJobError` paragraph), `cpp/libqarpx/python/qarpx.pyi` (the stub
  `develop` now ships, regenerated for the two bindings) and
  `tests/test_engines/test_layering.py`.  The ABI bump is 15 → 16 after the
  `develop` merge.  The `oqtopus-client` floor is `>=1.3` (open item 5).
  The `SamplingResult` constructor also rejects a negative count and
  `n_qubits` outside `[0, 63]`, now in its test row.  `CompiledCircuit.n_qubits`
  is the block width, not the device width the base comment describes,
  because the counts are already in logical order.  The notebook's device is a
  hand-written three-qubit line, not the device-gateway example; its "VQE
  step" is one energy and one gradient.  Phase 3b's rows switch on
  `QARP_OQTOPUS_QPU=1` rather than on the device id alone.  The capability row
  exercises measure-then-reuse; the conditional case is the base
  `_reject_true_mcm`, shared with `QarpEngine`.  `_compile_stamp` compares the
  device record by identity, so every engine instance recompiles under
  `rebuild=False`, two on the same device id included.
