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
2. **Green-light** — the reviewer approves the design *in that Draft PR*
   (comment or approval on the plan commits).  **That green-light is the
   design approval** — implementation does not start before it.  Flip the
   index row to 🚧 In progress.
3. **Implement** — same branch, same PR.  Deviations from the green-lit plan
   go in the PR's "Deviations from plan" section as they happen.
4. **Review** — mark the PR ready.  The reviewer may run an AI conformance
   diff against the green-lit plan revision as input.  Approval stays human.
5. **Land** — deviations folded back into the plan file; status flips to
   ✅ Landed.

Deviating from the plan is fine — *silent* drift is the violation.  The
green-light is a point in time: measure drift against the plan as it stood
when it was approved, not against the version you just edited.

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
| [`example_plan.md`](example_plan.md) | Worked example of a plan | OpenQARP maintainers | 🗄️ | Illustrates the format only: a filled-in `_template.md` for a small standard-tier feature.  Not implemented; delete this row when the first real plan lands |
