<!-- One PR per contribution, two gates inside it: green-light the plan while
     this PR is a Draft, then mark it ready for implementation review.
     The loop: docs/contributions/README.md -->

## Summary

<!-- What and why, 2–4 sentences. -->

## Plan

<!-- One of the two lines below; delete the other.
     Tier definitions: docs/contributions/README.md -->
Plan: docs/contributions/<topic>_plan.md
Tier: trivial — <!-- one-line justification -->

Reviewer: @
Sit-down held: <!-- yes / no — structural tier requires yes -->

### Design summary

<!-- 3–5 sentences: the problem, the chosen approach, what it deliberately
     does NOT do. Details live in the plan file — don't duplicate. -->

### Gate 1 — green-light checklist (while Draft)

<!-- Trivial tier: delete this whole section. -->

- [ ] House-format header complete (Status / Author / Reviewer / Date /
      Tier / Scope / Branch)
- [ ] Scope is an explicit file list
- [ ] Test plan table has the oracle column filled (independent references
      named BEFORE implementation exists)
- [ ] Conventions checked (`qarp_conventions.md` §13–§17); any convention
      change named explicitly in the Design section
- [ ] Index row added in `docs/contributions/README.md` (📋 Planned)
- [ ] **Green-lit by the reviewer** — implementation starts only after this;
      flip the index row to 🚧 In progress

## Deviations from plan

<!-- Measured against the plan AS GREEN-LIT, not as later edited.
     Leave EMPTY to claim "implemented exactly as green-lit" (reviewers check
     this claim). Otherwise list each deviation; fold them back into the
     plan file before merge — declared drift is fine, silent drift is not. -->

## Gate 2 — implementation checklist (before marking ready)

- [ ] Tests added/updated; every numerical assertion has an independent
      oracle (analytic / published / openfermion / scipy — round-trips are
      additional, never the oracle)
- [ ] Example in `examples/` (standard/structural features)
- [ ] Plan file updated: deviations folded back, phase boxes dated,
      status + index row in `docs/contributions/README.md` (standard/structural)
- [ ] `ruff check .`, `ruff format --check .`, mypy, `pytest` green locally

<!-- Default expectation is untouched: a convention change is a deliberate act,
     green-lit in the plan — never a side effect of implementation. -->
- [ ] `qarp_conventions.md` **has not been edited** — or, if it has, the change
      was named in the plan and green-lit
