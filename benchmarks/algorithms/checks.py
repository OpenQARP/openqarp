"""Final-result checks for the algorithms track.

Two fields with two different rigours:

- ``energy_x0`` — the energy at the shared starting point, before any
  optimization.  Trajectory-independent and tight (CHECK_RTOL): a wrong
  Hamiltonian, circuit, or convention fails here deterministically.
- ``energy`` — the converged value, compared within an absolute convergence
  band.  COBYLA amplifies last-bit summation differences between stacks into
  genuinely different trajectories (measured: two clusters 2e-3 apart on
  QAOA), so bit-level agreement of finals is not a property the physics
  guarantees; landing in the same optimum neighbourhood is.

ffsim (different ansatz by design) carries only its variational-band flag.
"""

from benchmarks.common import agree as _agree_fields

FINAL_ABS_TOL = 5e-3


def result_check(energy: float, energy_x0: float | None, band_ok: bool | None = None) -> dict:
    check: dict = {"energy": float(energy)}
    if energy_x0 is not None:
        check["energy_x0"] = float(energy_x0)
    if band_ok is not None:
        check["band_ok"] = int(band_ok)
    return check


def agree(check: dict | None, oracle: dict | None, rtol: float) -> bool:
    if check is None or oracle is None:
        return False
    if "band_ok" in check:  # different-ansatz stack: its own validity flag
        return bool(check["band_ok"])
    if "energy_x0" not in check or "energy_x0" not in oracle:
        return False
    if not _agree_fields(
        {"energy_x0": check["energy_x0"]}, {"energy_x0": oracle["energy_x0"]}, rtol
    ):
        return False
    return abs(check["energy"] - oracle["energy"]) <= FINAL_ABS_TOL
