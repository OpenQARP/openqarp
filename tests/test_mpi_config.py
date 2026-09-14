"""MPIConfig environment detection — the only MPI surface OpenQARP keeps
(pipeline_hardening_plan.md P1.22).  No MPI runtime is initialised anywhere
in the suite.
"""

import pytest

from qarp import MPIConfig


@pytest.fixture(autouse=True)
def _clean_mpi_env(monkeypatch):
    for var in MPIConfig._MPI_ENV_VARS + MPIConfig._SIZE_ENV_VARS + ["QARP_DISABLE_MPI"]:
        monkeypatch.delenv(var, raising=False)


def test_disabled_flag_wins_over_mpi_env(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("QARP_DISABLE_MPI", "1")
    assert MPIConfig.is_disabled()
    assert not MPIConfig.is_mpi_env()


def test_world_size_reads_first_valid_env(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert MPIConfig.world_size() == 4


def test_world_size_skips_unparseable_values(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "not-a-number")
    monkeypatch.setenv("PMI_SIZE", "3")
    assert MPIConfig.world_size() == 3


def test_world_size_defaults_to_one():
    assert MPIConfig.world_size() == 1


def test_no_mpi4py_import_anywhere_in_qarp():
    """P1.22: detection is environment-only; no module under qarp/ may import
    mpi4py (its import runs MPI_Init_thread)."""
    import pathlib
    import re

    import qarp

    root = pathlib.Path(qarp.__file__).parent
    pattern = re.compile(r"^\s*(import mpi4py|from mpi4py)", re.MULTILINE)
    offenders = [str(f) for f in root.rglob("*.py") if pattern.search(f.read_text())]
    assert offenders == []
