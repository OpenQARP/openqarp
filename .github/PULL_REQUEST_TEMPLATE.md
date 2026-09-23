<!-- Two paths.  Trivial (typo, doc fix, bugfix + regression test): keep
     Summary and "Trivial PR", delete "Planned PR" whole.  Standard or
     structural: keep Summary and "Planned PR", delete "Trivial PR" whole.
     The loop and tier table: docs/contributions/README.md

     Checkboxes.  The boxes in this description are the AUTHOR's claims; tick
     them before asking for a gate.  The REVIEWER's checks live in the
     reviewer's own text — the green-light comment and the Approve review —
     pasted from the blocks in the HTML comments below, so a reviewer's tick
     can only come from the reviewer.  A non-draft PR with an unticked box fails
     the `pr-checklist` job.  A box that does not apply is ticked with its
     text struck through and a word of reason (- [x] ~~Example in
     `examples/`~~ doc-only) — never deleted — so a skipped item and a
     non-applicable one look different. -->

## Summary

<!-- What and why, 2–4 sentences. -->

Closes #<!-- the issue, if there is one; delete this line otherwise -->

Tier: <!-- trivial / standard / structural — the author proposes, the
           reviewer confirms; "this needs a plan" converts the PR to Draft
           and adds one (see "Promotion" in the README) -->
Reviewer: @<!-- the code owner CODEOWNERS requests, or the reviewer named at
                the sit-down.  A sit-down reviewer owns the reviewer checks;
                the ruleset still requires the code owner's Approve, so an
                owner who is not the sit-down reviewer is a second Approve,
                not a replacement. -->

## Trivial PR

<!-- Delete this section for a planned PR. -->

- [ ] Tests added/updated; every numerical assertion has an independent
      oracle (analytic / published / openfermion / scipy — round-trips are
      additional, never the oracle)
- [ ] `ruff check .`, `ruff format --check .`, mypy, `pytest` green locally
- [ ] `qarp_conventions.md` not edited (a convention change is never trivial)

<!-- Reviewer — paste into the Approve review body:

- [ ] Tier confirmed trivial: no new public surface, no convention territory
- [ ] Every numerical assertion has an independent oracle; no test asserts
      the implementation against itself
- [ ] CI green
-->

## Planned PR

<!-- Delete this section for a trivial PR. -->

Plan: docs/contributions/<topic>_plan.md
Sit-down held: <!-- yes / no — structural tier requires yes -->

### Design summary

<!-- 3–5 sentences: the problem, the chosen approach, what it deliberately
     does NOT do.  Details live in the plan file — don't duplicate. -->

### Gate 1 — green-light (while Draft)

<!-- Author: tick these, then request the green-light. -->

- [ ] House-format header complete (Status / Author / Reviewer / Date /
      Tier / Scope / Branch; Green-lit stays empty until Gate 1 clears)
- [ ] Scope is an explicit file list
- [ ] Test plan table has the oracle column filled (independent references
      named BEFORE implementation exists)
- [ ] Conventions checked (`qarp_conventions.md` §13–§17); any convention
      change named explicitly in the Design section
- [ ] Index row added in `docs/contributions/README.md` (📋 Planned)

<!-- Reviewer — paste into a PR COMMENT, not an Approve review (an Approve
     on the Draft would count as the merge approval at Gate 2):

Green-lit at <commit sha>

- [ ] Tier confirmed (standard or structural; structural has a sit-down)
- [ ] Design fits the conventions, or the plan names the convention it edits
- [ ] Scope list is complete and bounded — nothing the design needs is
      missing, nothing unrelated is in
- [ ] Every oracle in the test plan is independent of the implementation

     After that comment the author fills the plan's Green-lit header (sha,
     date, plan blob id) and flips the index row to 🚧 In progress, in one
     commit.  Implementation starts after that commit. -->

## Deviations from plan

<!-- Measured against the plan AS GREEN-LIT (the blob in the plan header),
     not as later edited.  Leave EMPTY to claim "implemented exactly as
     green-lit" (the reviewer checks this claim at Gate 2).  Otherwise list
     each deviation; fold them back into the plan file before merge —
     declared drift is fine, silent drift is not. -->

### Gate 2 — implementation (before marking ready)

- [ ] Tests added/updated; every numerical assertion has an independent
      oracle (analytic / published / openfermion / scipy — round-trips are
      additional, never the oracle)
- [ ] Example in `examples/`
- [ ] Plan file updated: deviations folded back, phase boxes dated, status
      and index row flipped to ✅ Landed (a push after the Approve dismisses
      it, so the plan file is finished before marking ready)
- [ ] `ruff check .`, `ruff format --check .`, mypy, `pytest` green locally
- [ ] `qarp_conventions.md` **has not been edited** — or, if it has, the change
      was named in the plan and green-lit

<!-- Reviewer — paste into the Approve review body.  Approve reviews are
     given here, never at Gate 1.  Post the /plan-review report as a PR
     comment first, so the author sees the same findings:

- [ ] Deviations claim verified against the green-lit plan: every touched
      file is in Scope or declared above
- [ ] Every oracle named in the plan's test table appears in the assertions;
      no test asserts the implementation against itself
- [ ] Plan blob unchanged since the green-light, or every change is a
      folded-back deviation, a phase date or a status flip
- [ ] `qarp_conventions.md` untouched, or the edit was green-lit
- [ ] CI green
-->
