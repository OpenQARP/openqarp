<!-- Copy to <topic>_plan.md.  Replace every <angle-bracket> placeholder and
     delete the guidance comments.  Keep the section order — reviews and the
     conformance diff key off it. -->

# <Topic — one-line imperative summary>

**Status:** Draft
**Author:** <contributor> (+ <AI assistant, if used>)
**Reviewer:** <named reviewer — their green-light on this plan in the Draft PR is the design approval>
**Date:** <YYYY-MM-DD>
**Tier:** <Standard | Structural>
**Branch:** <type/name-of-branch>
**Scope:** <explicit list of files/dirs to be added or edited — the
conformance review checks the PR touches nothing outside it without a
declared deviation>

---

## Why

<The problem, incident, or gap.  Cite issues, failing workflows, or numbers.
If this replaces or extends existing behaviour, name it.>

## Design

<How it fits the existing architecture: which abstract classes are extended
(prefer extending over new hierarchies), which modules own what, data flow.
Cite the relevant `qarp_conventions.md` sections (§13 blocks, §14
engines/devices, §15 repo, §17 symbols) — and flag explicitly if any
convention needs a deliberate edit.>

## API sketch

<Signatures, class shapes, user-visible surface — concrete enough that the
implementation can be diffed against it.>

```python
# def new_method(self, x: ..., *, flag: bool = False) -> ...
```

## Test plan

<One row per behaviour.  The oracle is an independent reference — analytic
value, published number, openfermion/scipy — never the implementation's own
output.  Round-trip tests are additional, never the oracle.>

| Test | Oracle | Location |
|---|---|---|
| <what is asserted> | <independent reference> | `tests/...` |

## Phases

<Phased checklists; each phase should be landable/reviewable on its own.
Date items as they complete: `- [x] done thing *(2026-01-31)*`.>

### Phase 1 — <name>

- [ ] <step>

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
