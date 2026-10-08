# Releasing

How a version of OpenQARP goes from `develop` to PyPI, GitHub and Zenodo.
Cutting a release needs org-admin or `qar-fre` team rights (they move `main`
and create `v*` tags); publishing to PyPI needs a required reviewer of the
`pypi` environment.

## The model

* `develop` is where everything lands, through pull requests.  It is the
  default branch and the source of the docs site.
* `main` is the latest release.  The README, and so the PyPI page, links
  into `tree/main/...`, so `main` must show the released code.
* A release is an annotated tag `vX.Y.Z` on a `develop` commit, and `main`
  fast-forwards to that same commit.  The tag push builds the wheels and
  publishes to PyPI; the GitHub Release makes Zenodo archive it and create
  the version DOI.

`main` only ever **fast-forwards**, so it keeps the exact SHAs of `develop`
and stays its ancestor.  A pull request cannot do that: GitHub's merge
methods always write a new commit (merge, squash) or new SHAs (rebase),
leaving `main` with commits `develop` never gets.  The rulesets encode this:

| Ruleset | Rules | Bypass |
|---|---|---|
| `main-integrity` | no deletion, no force-push, the pushed commit must already pass `lint`, `wheel`, `mypy`, `pytest`, `coverage`, `docs`, `ctest` | none |
| `main-release` | restrict updates | org admins, `qar-fre` |
| `protect-tags` | `v*` tags: no creation, update or deletion | org admins, `qar-fre` |

So `main` moves only by a release manager's push, only forward, and only to
a commit that CI has already passed on `develop`.

## Versioning

[Semantic Versioning](https://semver.org/).  While on `0.x`, a patch release
(`0.1.1`) carries fixes only; anything that adds or changes public API is a
minor release (`0.2.0`).

## Steps

`X.Y.Z` is the new version.  Run the git commands from an up-to-date clone
(`git fetch origin --tags`).

### 1. Pre-flight

* The last `nightly` run on `develop` is green (slow and property tests,
  notebooks, the LAPACK C++ build, the container CPU check).  If anything
  merged since, dispatch one:
  `gh workflow run nightly --ref develop`.
* If C++ or packaging changed since the last `wheels` run, build the full
  wheel matrix without publishing — a `workflow_dispatch` run skips the
  PyPI job: `gh workflow run wheels --ref <release branch>` after step 2's
  branch is pushed.  The per-change `ci` job builds on Linux only; this is
  the first time macOS, Windows and aarch64 see the change.

### 2. Release pull request

Branch `chore/release-X.Y.Z` off `develop` and change:

* `qarp/__init__.py` — `__version__ = "X.Y.Z"` (the single source;
  `pyproject.toml` reads it).
* `CHANGELOG.md` — move the `[Unreleased]` entries under
  `## [X.Y.Z] - <date>`, fill in anything the merged PRs did not record,
  and update the compare links at the bottom.
* `CITATION.cff` — `version`, `date-released`, and remove the previous
  release's *version* DOI.  Keep the concept DOI; the new version DOI does
  not exist until step 7.

This is a trivial-tier PR into `develop` (no plan).  Merge it as usual.

### 3. Freeze `develop`

Nothing merges into `develop` from here until the tag is pushed.  `ci`
cancels an in-progress run when a newer push arrives, and a release commit
whose run was cancelled has no checks, so `main-integrity` rejects it.

### 4. Wait for green CI on the release commit

The release commit is the merge of step 2 on `develop`:

```
git fetch origin
SHA=$(git rev-parse origin/develop)
gh run list --commit "$SHA" --workflow ci
```

Wait until that `push` run has completed successfully.

### 5. Fast-forward `main`

```
git push origin "$SHA":main
```

A rejection means the commit does not carry all the required checks — the
run was cancelled, failed or has not finished.  Stop and fix that (re-run
the `ci` run for that commit) before going on; nothing has been published
yet.  Moving `main` first is deliberate: it is the last point where a
problem costs nothing.

### 6. Tag and publish to PyPI

```
git tag -a vX.Y.Z "$SHA" -m "OpenQARP X.Y.Z"
git push origin vX.Y.Z
```

The tag push starts `wheels`: twenty wheels and the sdist, then the
`publish` job, which waits for approval on the `pypi` environment.  When all
builds are green, approve it from the run page (*Review deployments*).
Check the result in a clean venv:
`pip install openqarp==X.Y.Z && python -c "import qarp; print(qarp.__version__)"`.

### 7. GitHub Release and Zenodo

```
awk '/^## \[X.Y.Z\]/{f=1; next} /^## \[/{f=0} f' CHANGELOG.md > notes.md
gh release create vX.Y.Z --verify-tag --title "OpenQARP X.Y.Z" --notes-file notes.md
```

Publishing the release makes Zenodo archive the tag and create the version
DOI (listed on the concept DOI's record,
<https://doi.org/10.5281/zenodo.22755228>).  The freeze ends here.

### 8. Record the version DOI

A trivial PR into `develop` adds the new version DOI to `CITATION.cff`:

```yaml
  - type: doi
    value: "10.5281/zenodo.<id>"
    description: "Version DOI — this release (vX.Y.Z)"
```

## When something goes wrong

* **`main` push rejected** — see step 5.  Never loosen `main-integrity` to
  get a release through.
* **A wheel build fails after the tag** — nothing is published, since
  `publish` needs every build.  Fix on `develop` and release the next patch
  version.  Do not move or recreate a pushed tag: `main` already points at
  it, and PyPI never accepts the same version twice.
* **The PyPI upload fails part-way** — PyPI rejects a file it already has,
  so re-running `publish` fails on the files that made it.  Download the
  run's `cibw-*` artifacts (kept seven days) and upload only the missing
  files with `twine upload`.
