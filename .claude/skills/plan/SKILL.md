---
name: plan
description: Scaffold a OpenQARP contribution plan in the house format under docs/contributions/. Use at the start of any standard- or structural-tier contribution, ideally during the contributor+reviewer sit-down.
---

# /plan — scaffold a contribution plan

You are helping draft a plan that a reviewer will green-light in a Draft PR
before implementation starts. Plan and implementation share one PR, so the
green-lit plan is the contract the rest of that same PR gets diffed against
— precision here saves review time later.

## Before writing anything

1. Read `docs/contributions/README.md` (tiers, loop, oracle rule).
2. Read `docs/contracts/qarp_conventions.md` — at minimum
   §13 (blocks), §14 (engines/devices), §15 (repo), §17 (symbols), plus any
   section the topic touches. The plan must FIT these conventions; if it
   needs to change one, that is a deliberate edit to the conventions doc,
   named explicitly in the plan's Design section.
3. Skim the open rows (🚧, 📋) of the index in `docs/contributions/README.md`
   for adjacent work the plan should not collide with.

## Then

4. Copy `docs/contributions/_template.md` to
   `docs/contributions/<topic>_plan.md` (short snake_case topic).
5. Fill every section with what was agreed in the discussion — do not
   invent scope. Rules that matter most:
   - **Scope** is an explicit file list. The conformance review checks the
     implementation commits touch nothing outside it without a declared
     deviation.
   - **API sketch** concrete enough to diff an implementation against.
   - **Test plan**: one row per behaviour, and the oracle column is
     mandatory — an independent reference (analytic value, published
     number, openfermion/scipy), never the implementation's own output.
     Naming oracles now, before code exists, is the point.
   - **Phases** individually landable; date items as they complete.
6. Add the plan's row to the index table in `docs/contributions/README.md`
   (status 📋 Planned).
7. Remind the author: branch, then open a **Draft** PR whose first commit is
   the plan, using the default PR template (delete the *Trivial PR*
   section, fill *Planned PR* and tick the Gate 1 boxes; the reviewer
   blocks in the HTML comments are for the reviewer to paste, not for the
   author to tick).  The reviewer's green-light in that Draft PR — a comment
   `Green-lit at <sha>`, not an Approve review — is the design approval.
   Do not start implementing before it, and do not open a second PR.  Once
   it is posted, fill the plan's `**Green-lit:**` header with the sha, the date
   and the blob id from `git rev-parse <sha>:docs/contributions/<topic>_plan.md`,
   and flip the index row to 🚧, in one commit.  The blob identifies the
   green-lit plan by content, so rebasing the branch later is fine.
   A PR that started trivial and was asked for a plan follows the same
   steps from a later commit — the plan need not be the first commit.

Ask the user for any decision the discussion has not settled rather than
choosing silently — unresolved decisions belong in the plan as explicit
open items, not as your guesses.
