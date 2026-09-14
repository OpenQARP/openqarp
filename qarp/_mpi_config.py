import os


class MPIConfig:
    """MPI launcher *detection* — nothing more.

    OpenQARP does not implement MPI parallelism: a composite algorithm constructed
    under a multi-rank launcher raises ``CapabilityError`` rather than running
    the same serial work on every rank.  Detection is environment-only so no
    ``mpi4py`` import (and no ``MPI_Init_thread``) ever happens inside OpenQARP.

    Set QARP_DISABLE_MPI=1 to silence the detection (e.g. one rank of a job
    that drives OpenQARP serially on purpose).
    """

    # Environment variables indicating MPI is active
    _MPI_ENV_VARS = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",  # PMI (MPICH, SLURM)
        "PMI_SIZE",
        "MPI_LOCALNRANKS",  # MPICH
        "MPIRUN_RANK",
        "I_MPI_INFO_NUMA_NODE_NUM",  # Intel MPI
        "MV2_COMM_WORLD_SIZE",  # MVAPICH2
        "MV2_COMM_WORLD_RANK",
    ]

    # Environment variables that contain world size
    _SIZE_ENV_VARS = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
    ]

    @staticmethod
    def is_disabled() -> bool:
        """Check if MPI is explicitly disabled via QARP_DISABLE_MPI."""
        return os.environ.get("QARP_DISABLE_MPI", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @classmethod
    def is_mpi_env(cls) -> bool:
        """Check if running under an MPI launcher (env-only, no mpi4py import)."""
        if cls.is_disabled():
            return False
        return any(var in os.environ for var in cls._MPI_ENV_VARS)

    @classmethod
    def world_size(cls) -> int:
        """Get MPI world size from environment (no mpi4py import)."""
        for var in cls._SIZE_ENV_VARS:
            val = os.environ.get(var)
            if val:
                try:
                    return int(val)
                except ValueError:
                    continue
        return 1
