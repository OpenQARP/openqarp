# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.  Report it through
GitHub's private vulnerability reporting on this repository (Security →
Report a vulnerability) or by email to openqarp@fujitsu.com.  You will get an
acknowledgement within five working days.

## Scope

OpenQARP is a simulation and compilation library: it executes no untrusted
input by design, and its optional integrations import third-party SDKs only
when you install and call them.  Reports about those SDKs belong upstream.

## Supported versions

Fixes land on `develop` and ship in the next release; there are no
maintenance branches for earlier versions.
