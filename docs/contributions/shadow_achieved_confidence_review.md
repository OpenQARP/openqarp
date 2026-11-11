# Review of #24 — Report the shadow confidence the batch count achieves

Reviewed at commit `3b746b5`, against `develop` at `6c632b9`.

## Verdict

The code is correct and I found no bugs. All six items below are text edits; none touches code
behaviour.

- Please fix item 1 before merging.
- Item 2 should go in the same PR.
- Items 3–6 are small wording fixes.

## Background: the estimator and its bound

A shadow dataset has $N_s$ random settings, each giving an estimate $y_i$ of $\mu = \langle O \rangle$.
The settings are split into $K$ batches of at least $N = \lfloor N_s / K \rfloor$ settings each.
The estimate is the median of the batch means:

$$
\hat\mu = \operatorname{median}\left(\bar y_1, \dots, \bar y_K\right),
\qquad
\varepsilon = \sqrt{\frac{34\,\hat V}{N}},
$$

Here $\hat V$ is the measured variance of the $y_i$, and $\varepsilon$ is the reported `error`.

Theorem 1 of Huang, Kueng and Preskill (HKP, arXiv:2002.08953) says that if the true variance is
$\sigma^2$ and $N = 34\sigma^2/\varepsilon^2$, then

$$
\Pr\left[\,|\hat\mu - \mu| \ge \varepsilon\,\right] \le 2e^{-K/2}.
$$

The failure probability therefore depends only on the batch count $K$. The field `delta` should
report it:

$$
\delta(K) = \min\left(1,\ 2e^{-K/2}\right).
$$

The $\min$ is there because $2e^{-1/2} \approx 1.21$, so at $K = 1$ the bound says nothing.

When the user asks for a failure probability $\delta$ over $M$ observables, the code picks the
smallest $K$ with $2e^{-K/2} \le \delta/M$:

$$
K = \left\lceil 2 \ln \frac{2M}{\delta} \right\rceil .
$$

## What the PR changes

**The field now reports $\delta(K)$.** Before this PR, `delta` repeated the value the user asked
for, even when the user set $K$ directly. For example, `expval(..., delta=0.05, n_batches=1)`
reported $0.05$ although $\delta(1) = 1$, which is no guarantee at all. The values it now reports:

| $K$ | 1 | 2 | 4 | 8 | 9 | 20 |
|---|---|---|---|---|---|---|
| $\delta(K)$ | $1$ | $0.736$ | $0.271$ | $0.0366$ | $0.0222$ | $9.1\times10^{-5}$ |

**Families use the union bound.** For a family of $M$ estimates with fields $\delta_j$:

$$
\Pr\left[\text{any of the } M \text{ estimates misses}\right] \le \min\left(1,\ \sum_{j=1}^{M} \delta_j\right).
$$

The default joint mode picks $K$ so that every $\delta_j \le \delta/M$, so the sum is at most
$\delta$. For example, with $M = 2$ and $\delta = 0.05$, $K = \lceil 2\ln 80 \rceil = \lceil 8.76
\rceil = 9$ and each $\delta_j = 2e^{-4.5} = 0.0222 \le 0.025$. Before this PR each field said
$0.05$.

**Exact constants report $\varepsilon = 0$ and $\delta = 0$.** That pair is only true with a strict
inequality: $\Pr[|\hat\mu - \mu| > 0] = 0$, but $\Pr[|\hat\mu - \mu| \ge 0] = 1$. The PR therefore
states the guarantee as

$$
\Pr\left[\,|\hat\mu - \mu| > \varepsilon\,\right] \le \delta .
$$

**A rounding edge case is fixed.** When $\delta = 2M e^{-K/2}$ exactly, the formula for $K$ lands
exactly on an integer. In floating point the computed bound $2e^{-K/2}$ can then come out one ulp
(the smallest possible float step) above $\delta/M$. The PR adds a batch whenever that happens.

**Invalid input is rejected.** `delta` is checked even when `n_batches` is given, and
`n_batches` $< 1$ is rejected for a constant too.

## Items to fix

Each item gives the reason, the current text, and a suggested replacement.

### 1. The new to-do entry is in the wrong section (must fix)

`docs/contributions/to_do.md` sorts future work into two sections:

- **Breaking changes waiting for a minor release** (line 33): work that changes what users get
  back.
- **Additive, no window needed** (line 70): work that changes nothing for users.

The PR adds its new entry to the additive section, but the work it describes changes what users
get back in two ways.

**The error bars get wider.** HKP's guarantee needs the true $\sigma^2$, but the code uses
$\hat V$. Take an observable that is $100$ with probability $p = 10^{-4}$ and $0$ otherwise, so
$\mu = 0.01$. With 200 settings, no rare outcome is seen with probability
$(1 - 10^{-4})^{200} \approx 0.980$. In those runs $\hat V = 0$, so $\varepsilon = 0$ and
$\hat\mu = 0 \ne \mu$: the bar misses 98 % of the time, against a reported $\delta(8) = 0.037$. The
entry proposes replacing $\hat V$ with a certified upper bound $\sigma^2_{\max} \ge \sigma^2$, and
that can only widen the bar:

$$
\varepsilon_{\text{cert}} = \sqrt{\frac{34\,\sigma^2_{\max}}{N}} \ \ge\ \varepsilon .
$$

**`delta` changes meaning again.** The entry also proposes a tighter failure bound. By Chebyshev,
each batch mean misses with probability at most $\frac{\sigma^2/N}{\varepsilon^2} = \frac{1}{34}$.
The median misses only if at least $\lceil K/2 \rceil$ of the independent batch means miss. At even
$K$, `np.median` averages the two middle values, and the same count still holds. So

$$
\delta_{\text{Cheb}}(K) = \sum_{j=\lceil K/2 \rceil}^{K} \binom{K}{j} \left(\frac{1}{34}\right)^{j} \left(\frac{33}{34}\right)^{K-j},
$$

which gives $0.029$ at $K = 1$ and $4.8\times10^{-5}$ at $K = 8$, against HKP's $1$ and $0.037$.
The numbers in the entry are right. But this PR documents `delta` as exactly $\min(1, 2e^{-K/2})$,
so switching to $\delta_{\text{Cheb}}$ would change a public field's meaning a second time.

**Current text** (`to_do.md`, lines 70–81, under the additive heading):

```markdown
## Additive, no window needed

- **Shadow error bar from a certified variance.**  `ShadowEstimate.error`
  puts the empirical `V_hat` where HKP Theorem 1 needs the true variance.
  On an observable driven by rare outcomes `V_hat` is usually 0, and so is
  the bar: `X = 100` with probability `1e-4` over 200 settings misses at
  `K = 8` 98 % of the time against a reported `delta` of 0.037.  Bound the
  variance from above instead, by the shadow norm (state-independent) or a
  confidence bound on `V_hat`.  The same change can report the tighter
  failure bound the half-width already earns: Chebyshev gives each batch
  mean `1/34`, so a median of `K` fails at most `P[Bin(K, 1/34) >=
  ceil(K/2)]`, `2.9e-2` at `K = 1` and `4.8e-5` at `K = 8` against HKP's
  1 and 0.037.  Both change the reported values; §18 oracles for each.
```

**Suggested replacement:** delete the bullet from the additive section. Add it as the last bullet
of the breaking section, directly above the `## Additive, no window needed` heading, with its last
sentence extended:

```markdown
- **Shadow error bar from a certified variance.**  `ShadowEstimate.error`
  puts the empirical `V_hat` where HKP Theorem 1 needs the true variance.
  On an observable driven by rare outcomes `V_hat` is usually 0, and so is
  the bar: `X = 100` with probability `1e-4` over 200 settings misses at
  `K = 8` 98 % of the time against a reported `delta` of 0.037.  Bound the
  variance from above instead, by the shadow norm (state-independent) or a
  confidence bound on `V_hat`.  The same change can report the tighter
  failure bound the half-width already earns: Chebyshev gives each batch
  mean `1/34`, so a median of `K` fails at most `P[Bin(K, 1/34) >=
  ceil(K/2)]`, `2.9e-2` at `K = 1` and `4.8e-5` at `K = 8` against HKP's
  1 and 0.037.  Both change the reported values: `error` widens, and
  `delta` stops being HKP's bound, so a public field changes meaning.
  §18 oracles for each.

## Additive, no window needed
```

**Optional:** define `delta` now as an upper bound rather than as HKP's formula. A tighter bound
later then changes only the numbers, not the meaning. The wider `error` is breaking either way, so
the entry still moves.

Current text (`qarp/algorithms/_primitives/shadows/estimator.py`, lines 27–29 and 38):

```python
    the half-width for which ``Pr[|value - true| > error] <= delta``, where
    ``delta`` is HKP's bound ``min(1, 2 e^{-K/2})`` at ``K = n_batches``, whether
    ``K`` was passed or derived from a requested ``delta``.
...
    delta: float  # HKP failure bound at n_batches, min(1, 2 e^{-K/2}); 0 when exact
```

Suggested replacement:

```python
    the half-width for which ``Pr[|value - true| > error] <= delta``, where
    ``delta`` is an upper bound on that failure probability at ``K = n_batches``,
    whether ``K`` was passed or derived from a requested ``delta``; today it is
    HKP's ``min(1, 2 e^{-K/2})``.
...
    delta: float  # upper bound on failure probability at n_batches (today HKP's); 0 when exact
```

The first sentence of the `CHANGELOG.md` entry and the `docs/source/algorithms.rst` paragraph would
take the same "an upper bound, today HKP's" wording.

### 2. The example notebook still calls the error bar rigorous (should fix)

The PR changes `docs/source/algorithms.rst` to say two things:

- $\varepsilon$ is a worst-case bar.
- The guarantee holds for the true $\sigma^2$, with $\hat V$ standing in for it, so a bar driven
  by rare outcomes can be far too narrow (item 1).

The notebook still promises a rigorous bound. CI won't catch the mismatch, because the text is in
a markdown cell.

**Current text** (`examples/algorithms/primitives/mwe_classical_shadows.ipynb`, cell 4, first
paragraph):

```markdown
Each `ShadowEstimate.error` is the half-width $\varepsilon=\sqrt{34\,\hat V/N}$ that
**Theorem 1** of Huang–Kueng–Preskill controls — a rigorous $1-\delta$ bound
($\hat V$ the empirical single-setting variance, $N$ the per-batch size). Its
constant $34$ is a deliberately loose worst case (the paper's own remark), so the
*true* error is typically far inside the bar: below you'll see accurate values
carrying wide guarantees.
```

**Suggested replacement:**

```markdown
Each `ShadowEstimate.error` is the half-width $\varepsilon=\sqrt{34\,\hat V/N}$ that
**Theorem 1** of Huang–Kueng–Preskill controls at failure probability $\delta$
($\hat V$ the empirical single-setting variance, $N$ the per-batch size). It is a
worst-case bar, not a standard error: the theorem's guarantee holds at the true
variance and $\hat V$ stands in for it, so an observable driven by rare outcomes
can carry a bar far too narrow. Its constant $34$ is a deliberately loose worst
case (the paper's own remark), so the *true* error is typically far inside the
bar: below you'll see accurate values carrying wide bars.
```

### 3. One docstring uses $\ge$ where the others now use $>$

The `ShadowEstimate` docstring and the docs page now state the guarantee with $>$, which the
constant case needs (see *What the PR changes*). The module docstring's restatement still uses
$\ge$. Line 21 of the same docstring quotes HKP's theorem itself and should keep $\ge$.

**Current text** (`qarp/algorithms/_primitives/shadows/_estimators.py`, line 29):

```python
for which ``Pr[|o_hat - tr(O rho)| >= eps] <= delta``.  It is the paper's own
```

**Suggested replacement:**

```python
for which ``Pr[|o_hat - tr(O rho)| > eps] <= delta``.  It is the paper's own
```

### 4. The changelog names a private function

`n_batches_for` lives in the private `_estimators` module, and users can't import it. The
changelog should describe the effect users see.

**Current text** (`CHANGELOG.md`, lines 141–142):

```markdown
  `n_batches_for` adds a batch where float rounding of the log left the
  bound a few ulp above `delta / M`.
```

**Suggested replacement:**

```markdown
  The `n_batches` that `expval` and `expval_many` derive from `delta` can
  be one higher, where float rounding of the log left the bound a few ulp
  above `delta / M`.
```

### 5. `expval` doesn't document its arguments

After this PR, `delta` behaves in a way users can't guess: it is checked even when `n_batches` is
given, but then has no other effect. The one-line docstring doesn't say so. This gap is older than
the PR. The suggestion follows the Args/Raises style that `median_of_means` already uses.

**Current text** (`qarp/algorithms/_primitives/shadows/estimator.py`, lines 112–113):

```python
        """Median-of-means estimate of ``<observable>`` (sum-inside for a
        multi-term operator)."""
```

**Suggested replacement:**

```python
        """Median-of-means estimate of ``<observable>`` (sum-inside for a
        multi-term operator).

        Args:
            observable: a Pauli string or a Hermitian ``QubitOperator``.
            delta: target failure probability in (0, 1).  Sets the batch count
                unless ``n_batches`` is given, and is checked either way.
            n_batches: batch count overriding the one derived from ``delta``;
                at least 1.

        Raises:
            ValueError: if ``delta`` is outside (0, 1), ``n_batches`` is below 1,
                or the observable is not Hermitian or acts on a qubit the
                dataset lacks.
        """
```

### 6. The PR description says `delta` where it means `error`

The docs describe `error`, the half-width $\varepsilon$, as a worst-case bar. `delta` is a
probability.

**Current text** (PR #24 description, end of the Summary):

```markdown
The docs read `delta` as a worst-case bar, not a standard error.
```

**Suggested replacement:**

```markdown
The docs read `error` as a worst-case bar, not a standard error.
```

## Tier

The PR is filed as trivial (a bugfix with tests). That fits, since the old value was wrong. But the
to-do entry it closes was listed as a breaking change to a public field, so the tier is worth
confirming explicitly.

## What I checked

- **The rounding fix is real.** In the case pinned in the tests, $M = 3$ and
  $\delta = 6e^{-7/2}$, so $\delta/M = 2e^{-7/2}$ exactly. The old code chose $K = 7$, whose
  computed bound $0.060394766844637$ is one ulp above $\delta/M = 0.060394766844636995$. The fix
  chooses $K = 8$. I also tried thousands of cases within two ulp of $\delta = 2Me^{-K/2}$:
  wherever the fix changes $K$, it adds exactly one batch.
- **Nothing else reads `delta`.** `ShadowProtocol.run` returns only $\hat\mu$, and its minimum
  number of settings is unchanged at $\lceil 2\ln 40 \rceil = 8$.
- **The tests check against independent values.** They compare against values worked out by hand
  ($2e^{-4}$, $2e^{-4.5}$, $2e^{-10}$, and $K = 8$ and $K = 9$ from the formula above), not against
  the code's own output.
- **Everything passes.** Tests, lint and type checks pass locally, and every CI check passes. The
  branch merges cleanly with the latest `develop`.
