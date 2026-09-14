---
name: plan-review
description: Conformance-review a OpenQARP implementation against the green-lit plan in docs/contributions/ (same PR — the plan is its first commit). Reviewer tool — produces a report for the human reviewer, never a merge decision.
---

# /plan-review — diff an implementation against its plan

Input: a plan file in `docs/contributions/` and the implementation branch
(or PR diff). If either is not given, ask. Your output is INPUT to the
human reviewer — you never approve or block; write findings, not verdicts.

## Procedure

1. Read the plan fully: Scope, Design, API sketch, Test plan (oracle
   column), Phases, Deviations log.
2. Get the change set: `git diff <target>...<branch> --stat` and then the
   full diff of the files that matter. Target is `develop` unless told
   otherwise. The plan file is part of this diff by design (it is the
   branch's first commit) — never report it as out-of-scope.
3. **Check the plan was not rewritten to match the code.** Plan and
   implementation share a branch, so the plan file is editable after the
   green-light — an easy way to make deviations disappear. Diff the plan
   file across its own history on the branch
   (`git log -p <target>..<branch> -- <plan file>`). Post-green-light edits
   to Scope, Design, API sketch or the Test plan's oracle column are a
   finding unless the PR's "Deviations from plan" section declares them.
   Later phase-box dates, folded-back deviations and status flips are
   expected and fine.
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
- **Findings**, most severe first, each with file:line, what the plan says,
  what the code does.
- **Unverifiable items** — anything you could not check (e.g. behaviour
  needing a run) listed explicitly, never silently skipped.
- **Suggested reviewer focus** — the 2–3 places human judgment matters most.
