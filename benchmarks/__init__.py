"""Competitor benchmark harness.

Bench-only surface: imports qarp one-way; nothing under qarp/ or cpp/ imports
this package or any competitor SDK (conventions §15).  Deps ride the [bench]
and [integrations] extras.  Timings are produced by `python -m benchmarks.run`
on the published host; `python -m benchmarks.smoke` is the CI correctness gate.
"""
