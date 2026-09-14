# Contributing

Welcome and thank you for the interest in contributing to OpenQARP.

## How Contributions Flow

Contributions are tiered; the tier is the reviewer's call, and "this needs a
plan" is a legitimate PR response:

| Tier | Examples | Requirements |
|---|---|---|
| Trivial | typo, doc fix, bugfix + regression test | PR template only |
| Standard | new function/method, extension of existing abstractions | one-page plan, async review |
| Structural | new block type, new engine, anything touching [`qarp_conventions.md`](docs/contracts/qarp_conventions.md) territory | sit-down, full plan, green-lit in Draft before implementation |

The loop for standard/structural work (details and the plan scaffold live in
[`docs/contributions/`](docs/contributions/README.md)):

One PR per contribution, with two review gates inside it.

1. **Plan** — contributor + reviewer refine a plan together (AI-assisted is
   the norm).  Branch per the conventions below, commit the plan to
   `docs/contributions/` as the branch's first commit, and open a **Draft**
   PR on it.
2. **Green-light** — the reviewer approves the design in that Draft PR.
   *That* is the design approval; implementation does not start before it.
3. **Implement** — same branch, same PR.  The PR declares deviations from the
   green-lit plan (an empty section claims "implemented as green-lit", and
   reviewers check that claim).
4. **Review** — mark the PR ready; the reviewer may run an AI conformance
   diff as input. Human approval stays human.
5. **Land** — declared drift is folded back into the plan file before merge.

## General Guidelines

* Never commit to `develop` or `main` directly — every change lands through a
  pull request.
* Open an issue first, using the **Feature** or **Bug** template in the GitHub
  issue form, so the what and why are on record.
* Always branch off the latest `develop`.
* Every change ships tests in `tests/`, and every numerical assertion needs an
  **independent oracle** — an analytic value, a published number, an
  openfermion/scipy reference
  ([`qarp_conventions.md`](docs/contracts/qarp_conventions.md) §18).
  Round-trip tests are additional, never the oracle.
* Standard- and structural-tier features also ship an example in `examples/`;
  trivial fixes are exempt.
* OpenQARP is structured to ease future integration: prefer extending the
  existing abstract classes over proposing new classes or modules.
* CI must be green before review: ruff (lint + format), mypy, pytest, and the
  C++ test suite.

## Git Branch Conventions

We use Gitflow for organising our contributions to OpenQARP. Branch names should
be all lower-case, not be excessively verbose or long, and use the following
templates:

* `feature/name-of-feature` for new features (e.g. `feature/vqe`).
* `bugfix/name-of-bugfix` for bug fixes.
* `docs/name-of-file` for documentation, where the placeholder `name-of-file`
  corresponds to the name of the file that is being documented.
* `improvement/name-of-improvement` for code refactoring or any kind of
  change that modifies existing code only.
* `chore/name-of-chore` for process, tooling, CI, and other non-library work.


## Code Formatting and Linting

We use [ruff][ruff] as both linter and formatter (configuration lives in
`pyproject.toml` under `[tool.ruff]`; line length 100). CI blocks on both:

```
ruff check .
ruff format --check .
```

We use [mypy][mypy] to enforce consistency of data types:
```
mypy qarp/ tests/ --disable-error-code=import-untyped --disable-error-code=method-assign
```

Git hooks are managed by the [pre-commit][pre-commit] framework
(`.pre-commit-config.yaml`): ruff lint + format, notebook-output stripping, and
the forbidden/oversized-file policy. Do **not** write a manual
`.git/hooks/pre-commit` script. One-time setup in your dev venv:

```
pip install pre-commit
pre-commit install
```

## Tests

Python: `pytest` (defaults exclude the `slow`/`bench` markers; the full run
takes ≈ 8 minutes). C++: `python scripts/run_cpp_tests.py` — the same suite
runs in CI as the `ctest` job.

## Licensing

OpenQARP is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
Unless you state otherwise, any contribution you intentionally submit for
inclusion in OpenQARP is licensed under the same terms, per section 5 of that
license.

Happy coding!


[pre-commit]: https://pre-commit.com/
[mypy]: https://www.mypy-lang.org/
[ruff]: https://docs.astral.sh/ruff/
