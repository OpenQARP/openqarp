# Contribution plans

Plan-first home for OpenQARP contributions.  Every standard- or structural-tier
contribution starts as a plan file here, green-lit in a **Draft PR** before
implementation begins — plan and implementation then land together in that
same PR.  The [index below](#index) tracks every plan.  Conventions are
governed by [`qarp_conventions.md`](../contracts/qarp_conventions.md); when a
plan and that doc disagree, that doc wins.

The plans behind the code as first published are not carried into this
repository; the code, its tests and `qarp_conventions.md` are the record.
This directory therefore starts with the scaffold and a worked example only —
the first plan filed after the release is the first row in the index.

## The loop

One PR per contribution, with two review gates inside it.

1. **Plan** — contributor + reviewer sit down, generate and refine the plan
   with AI.  Copy [`_template.md`](_template.md) to `<topic>_plan.md`, fill
   it, add the index row below (📋 Planned).  Branch per CONTRIBUTING
   conventions and open a **Draft PR** whose first commit is the plan.
2. **Green-light** — the reviewer posts a PR **comment** headed
   `Green-lit at <sha>` (the plan commit) with the Gate 1 reviewer checks
   pasted from the PR template and ticked.  **That comment is the design
   approval** — implementation does not start before it.  It is a comment,
   never a GitHub *Approve* review: Approve reviews are given at Gate 2
   only.  The author then fills the plan's `**Green-lit:**` header — the
   sha, the date, and the plan file's blob id from
   `git rev-parse <sha>:docs/contributions/<topic>_plan.md` — and flips the
   index row to 🚧 In progress, in one commit.
3. **Implement** — same branch, same PR.  Deviations from the green-lit plan
   go in the PR's "Deviations from plan" section as they happen.
4. **Review** — fold deviations back into the plan file, date the phase
   boxes, flip status and index row to ✅ Landed, tick the Gate 2 boxes,
   then mark the PR ready.  The reviewer posts the `/plan-review` report
   (an AI conformance diff against the green-lit blob) as a PR comment, so
   the author sees the same findings, then gives the GitHub *Approve* with
   the Gate 2 reviewer checks pasted into its body and ticked.
5. **Land** — merge.  A push after the Approve dismisses it, which is why
   the plan file is finished before review is requested; review fixes cost
   one more Approve.

Deviating from the plan is fine — *silent* drift is the violation.  The
green-light is a point in time: measure drift against the plan as it stood
when it was approved, not against the version you just edited.  That
revision is pinned by the plan file's **blob id** in the header, not by the
commit: the blob survives rebases and the squash on landing, so the branch
may be rebased freely.  The reviewer trusts the sha in their own comment,
and `git rev-parse <sha>:<plan file>` must reproduce the header's blob; a
rewritten commit can still be fetched from GitHub by sha while the PR is
open.

### Who ticks what

The checkboxes in the PR description are the **author's** claims, ticked
before asking for a gate.  The **reviewer's** checks are never ticked in the
author's description: they are pasted from the template's reviewer blocks
into the reviewer's own text — the green-light comment at Gate 1, the Approve
review body at Gate 2 — so a reviewer's tick can only come from the reviewer.
The `pr-checklist` job fails a non-draft PR whose description still has an
unticked box.  A box that does not apply is ticked with its text struck
through and a word of reason, never deleted, so a skipped item and a
non-applicable one look different.

The reviewer is the code owner GitHub requests from `CODEOWNERS`, or the
reviewer named at the sit-down.  A sit-down reviewer owns the reviewer
checks.  The branch ruleset requires the code owner's Approve either way, so
an owner who is not the sit-down reviewer gives a second Approve, not a
replacement.

### Promotion

"This needs a plan" on a PR opened as trivial converts it back to Draft; the
author adds the plan file and index row in the next commit and fills the
"Planned PR" section of the description.  The plan being the branch's first
commit is the norm for a PR that starts planned, not a requirement — the
conformance review compares against the green-lit blob, wherever that commit
sits.

## Tiers

The tier is the reviewer's call; "this needs a plan" is a legitimate PR
rejection.

| Tier | Examples | Requirements |
|---|---|---|
| Trivial | typo, doc fix, bugfix + regression test | PR template only — no plan |
| Standard | new function/method, extension of existing abstractions | one-page plan, async review |
| Structural | new block type, new engine, anything touching `qarp_conventions.md` territory | sit-down, full plan, green-lit in Draft before implementation |

## Test plans name their oracles

Every numerical feature is tested against an **independent oracle** — an
analytic value, a published number, an openfermion/scipy reference — never
solely against the implementation's own output.  Round-trip tests are
allowed *in addition to*, never *instead of*, an oracle test.  The plan's
test-plan table has an oracle column so this is decided at planning time,
before there is an implementation to rationalize.

## Roadmap

Three named windows, so a phase can say where it belongs.  Nothing structural
lands during a release freeze: the tree should not churn while a tag is being
cut.

**First post-launch window** — cleanup with no public surface change, cleared
before any layout move.  Type stubs for the compiled backend, then the
configuration and agent-guide tidy-up.

**The 0.2 cycle** — layout, so paths break once rather than twice.  The `src/`
source root and the `cpp/` flattening.

**The 1.0 item** — the only change on this roadmap that breaks user code.  One
public import name, with `qarpx` kept as a deprecating shim for a full release.

A plan not tied to one of these windows lands when its own two gates clear.

## Index

One row per plan file, open work first (🚧, 📋, then ✅, 🗄️; alphabetical
within each group).  Every plan also carries a matching `**Status:**` header.
Not a row: [`_template.md`](_template.md), the plan scaffold.

| Badge | Meaning |
|-------|---------|
| ✅ **Landed** | Implemented, tested, in production.  The doc is now a reference/contract. |
| 🚧 **In progress** | Partially landed; named work still pending. |
| 📋 **Planned** | Backlog.  Little or nothing implemented yet. |
| 🗄️ **Superseded** | Historical snapshot; a newer doc or the code is now authoritative. |

| Plan | Theme | Author | Status | Notes |
|---|---|---|:---:|---|
| [`sampler_distribution_plan.md`](sampler_distribution_plan.md) | `Sampler` returns an array-backed read-only `SamplingDistribution` | Stefano Scali | 📋 | Structural: §14 result-contract edit; breaks `isinstance(dict)` and mutation of results |
| [`sampling_distribution_utilities_plan.md`](sampling_distribution_utilities_plan.md) | Analysis utilities and shot metadata on `SamplingDistribution` | Stefano Scali | 📋 | Same PR as `sampler_distribution_plan.md`; no convention edit |
| [`example_plan.md`](example_plan.md) | Worked example of a plan | OpenQARP maintainers | 🗄️ | Illustrates the format only: a filled-in `_template.md` for a small standard-tier feature.  Not implemented; delete this row when the first real plan lands |
