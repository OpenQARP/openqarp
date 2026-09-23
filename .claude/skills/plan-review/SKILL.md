---
name: plan-review
description: Conformance-review a OpenQARP implementation against the green-lit plan in docs/contributions/ (same PR — the plan is normally its first commit). Reviewer tool — produces a report for the human reviewer, never a merge decision.
---

# /plan-review — diff an implementation against its plan

Input: a plan file in `docs/contributions/` and the implementation branch
(or PR diff). If either is not given, ask. Your output is INPUT to the
human reviewer working through the Gate 2 reviewer block of the PR
template — you never approve or block; write findings, not verdicts.

## Procedure

1. Read the plan fully: Scope, Design, API sketch, Test plan (oracle
   column), Phases, Deviations log.  Take the green-lit sha and blob from
   the plan's `**Green-lit:**` header and verify the sha against the
   reviewer's own `Green-lit at <sha>` comment on the PR
   (`gh pr view <n> --comments`) — the header is author-written, the
   comment is not.  Then verify the blob: `git rev-parse <sha>:<plan file>`
   must print the header's blob (if the sha is not in the local history the
   branch was rebased; `git fetch origin <sha>` retrieves it, GitHub keeps a
   PR's rewritten commits).  A mismatch on either is the top finding: the
   author may have pointed the review at a later revision of the plan.  If
   there is no green-light comment the PR never cleared Gate 1 — report that
   and stop.
2. Get the change set: `git diff <target>...<branch> --stat` and then the
   full diff of the files that matter. Target is `develop` unless told
   otherwise. The plan file is part of this diff by design — never report
   it as out-of-scope.  It is normally the branch's first commit; a PR
   promoted from trivial adds it later, which is fine.
3. **Check the plan was not rewritten to match the code.** Plan and
   implementation share a branch, so the plan file is editable after the
   green-light — an easy way to make deviations disappear.  Fast path:
   `git rev-parse <branch>:<plan file>` equal to the green-lit blob means
   the plan is byte-identical to what was approved — skip to step 4.
   Otherwise diff from the green-lit revision
   (`git diff <blob> <branch>:<plan file>`) and read the plan as approved
   (`git cat-file -p <blob>`); that revision, not the tip, is what Scope,
   Design and the oracles are judged against.  Post-green-light edits to
   Scope, Design, API sketch or the Test plan's oracle column are a finding
   unless the PR's "Deviations from plan" section declares them.  Later
   phase-box dates, folded-back deviations, the `**Green-lit:**` fill and
   status flips are expected and fine.
4. Check, in order:
   - **Scope**: every touched file is in the plan's Scope list, or the
     deviation is declared in the PR's "Deviations from plan" section.
     Undeclared out-of-scope changes are the highest-priority finding.
   - **Design & API**: implementation matches the sketch — names,
     signatures, ownership/layering. Note drift even when it is an
     improvement; declared-and-folded-back is the rule, not silence.
   - **Test plan**: each row of the plan's test table exists in the diff,
     and each named oracle actually appears in the assertions. Flag any
     test that asserts the implementation against itself (round-trip
     posing as an oracle) — per conventions, round-trips are additional,
     never the oracle. Ask of each test: would it fail if the feature were
     wrong?
   - **Conventions**: `qarp_conventions.md` §13–§17 compliance for the
     touched areas — LSB endianness, radians, SimpleBlock/CompositeBlock
     inheritance (never `Block` internally), sorted-tuple `.symbols` and
     `parameter_map` usage, SDK imports only in adapters, `zip(strict=)` justified.
   - **Plan hygiene**: deviations folded back into the plan file; phase
     checkboxes dated; status/index row updated if this PR completes the
     plan.
5. Do not relitigate the design itself — the Draft-PR green-light approved
   it. If the design looks wrong in hindsight, flag it as exactly that: a
   question for the humans, separate from conformance findings.

## Report format

- **Verdict-free summary** (2–3 sentences: what the PR does, overall
  conformance impression).
- **Deviations claim** — one line: the green-lit sha and blob used, and
  whether the PR's "Deviations from plan" section accounts for every
  difference found (an empty section is the claim "implemented exactly as
  green-lit").
- **Findings**, most severe first, each with file:line, what the plan says,
  what the code does.
- **Unverifiable items** — anything you could not check (e.g. behaviour
  needing a run) listed explicitly, never silently skipped.
- **Suggested reviewer focus** — the 2–3 places human judgment matters most.

The reviewer posts this report as a PR comment before their Approve, so the
author sees the same findings the reviewer acted on.
