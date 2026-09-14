# Eigen (vendored)

Header library vendored from [Eigen](https://gitlab.com/libeigen/eigen) at
**3.4.0** (commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c`), extracted with
`git archive` from that tag — every file is byte-identical to upstream.

Vendored rather than fetched at build time because the wheels ship object code
compiled from these headers: MPL-2.0 §3.2(a) makes us, not upstream, the party
who must supply the corresponding source, so it travels with the repo and the
sdist (everything git tracks ships in the sdist); the wheel itself contains no
source, and the root `LICENSES_bundled.txt` tells wheel recipients where to
find it.

## What is here, and what is not

`Eigen/` — for a header-only library that is the complete work, not an excerpt
— plus `unsupported/Eigen/MatrixFunctions` and its `src/MatrixFunctions/`,
which the C++ test suite uses for a reference matrix exponential
(`test_synthesis_unitary.cpp`). Nothing from `unsupported/` reaches the shipped
binary, so it carries no §3.2(a) obligation; it is vendored so the test suite
builds from a clean checkout.

Upstream's `bench/`, `test/`, `doc/`, `demos/`, `blas/`, `lapack/` and the rest
of `unsupported/` are omitted. That is also where upstream's **19
GPL-licensed** files (`bench/btl`) and **2 LGPL-licensed** files
(`unsupported/Eigen/src/IterativeSolvers/`) live: no GPL- or LGPL-covered file
is vendored here.

The 344 vendored source files are not uniformly MPL-2.0: 309 carry the MPL-2.0
header, 20 are BSD-3-Clause (Intel's MKL/BLAS/LAPACKE bridge headers, gated
behind `EIGEN_USE_BLAS`/`EIGEN_USE_LAPACKE` and so contributing no object code
to our default build), one is Apache-2.0
(`Eigen/src/Core/arch/Default/BFloat16.h`, from TensorFlow), and 14 are
umbrella or plugin headers with no license block. All upstream `COPYING.*`
files are kept verbatim so that `COPYING.README`'s cross-references resolve;
`COPYING.GPL`, `COPYING.LGPL` and `COPYING.MINPACK` are retained for that
reason alone and cover no file present here.

Two further files are MPL-2.0 overall but carry a third-party grant over part
of their contents, and both are in the include closure of the shipped build:
`Eigen/src/Geometry/AlignedBox.h` (`AlignedBox::transform` is BSD-3-Clause,
Willow Garage / Open Source Robotics Foundation) and
`Eigen/src/Core/arch/Default/Half.h` (float16 conversion routines, Fabian
Giesen, permissive). The repository-root `LICENSES_bundled.txt` reproduces both.

The build defines `EIGEN_MPL2_ONLY` (PUBLIC on `csim_static`, so every Eigen
user in the build inherits it). Upstream guards only its GPL/LGPL files with
that macro, so it turns including one of those into a compile error — it does
not exclude the BSD-3-Clause and Apache-2.0 files counted above, which is why
`LICENSES_bundled.txt` discloses them individually.

Upgrades are a straight re-extract of the same paths at the new tag; never
patch these files in place.
