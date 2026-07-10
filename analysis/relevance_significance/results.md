# Statistical Tests for Relevance Patterns

All numbers below are computed from the labelled paired prompt–response dataset
(N = 1,250) by [compute_relevance_tests.py](compute_relevance_tests.py) and are
saved in [relevance_test_results.json](relevance_test_results.json). The script
first **reproduces the paper's relevance tables exactly** (Table 1, 5, 6, 7),
then runs the tests below.

Relevance $r \in \{1,2,3\}$ is the human relevance annotation; response severity
$s^c_r \in \{0,1,2,3\}$ is the highest active level per category $c$, and
$\max s_r = \max_c s^c_r$. Omnibus $r\times c$ tests use a Monte Carlo
label-permutation $\chi^2$ (50,000 permutations, both margins fixed) that is
valid under sparse/zero cells, corroborated by the asymptotic Pearson $\chi^2$;
targeted $2\times2$ contrasts use Fisher's exact test. Effect sizes are Cramér's
$V$ (omnibus) and $\phi$ ($2\times2$).

## Table 1: Tests

| Analysis | Test | N | Statistic | p-value | Effect Size | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Response severity ($\max s_r$, 0–3) × relevance (1–3) | Permutation $\chi^2$ (asymptotic) | 1250 | $\chi^2=50.17$, df=6 | $3.0\times10^{-4}$ ($4.3\times10^{-9}$) | Cramér's $V=0.142$ (bias-corr. 0.133) | Relevance is **not** independent of response severity; small-to-moderate association. |
| Severity 2 vs all other severities: Rel 3 rate | Fisher exact ($2\times2$) | 1250 | OR $=0.17$ | $5.7\times10^{-6}$ | $\phi=-0.159$ | The medium-severity relevance dip (Table 1) is **robust**, not noise. |
| Harm category × relevance among **harmful** responses (4×3, overlapping rows) | Permutation $\chi^2$ (asymptotic) | 146 rows | $\chi^2=14.41$ | $0.014$ ($0.025$) | Cramér's $V=0.222$ | Relevance among harmful responses differs by category; **exploratory** (rows overlap; small denominators). |
| Violence vs Hate harmful responses: Rel 3 rate | Fisher exact ($2\times2$) | 70 | OR $=0$ (0/37 non-Rel3 for Hate) | $2.3\times10^{-4}$ | $\phi=-0.432$ | Harmful Violence responses far less relevant than harmful Hate; **small n**. |
| Harmful ($\max s_r\ge1$) vs safe response: Rel 3 rate | Fisher exact ($2\times2$) | 1250 | OR $=0.53$ | $0.015$ | $\phi=-0.071$ | Harmful responses slightly less relevant overall; effect is **tiny** when pooled. |
| Violence: response-harmful (R+) vs response-safe (R−): Rel 3 | Fisher exact ($2\times2$) | 1250 | OR $=0.23$ | $7.0\times10^{-4}$ | $\phi=-0.115$ | Violence harmful responses reliably less relevant. |
| Sexual: response-harmful (R+) vs response-safe (R−): Rel 3 | Fisher exact ($2\times2$) | 1250 | OR $=0.49$ | $0.064$ | $\phi=-0.056$ | Same direction, **not** significant (suggestive only). |
| Self-harm: R+ vs R−: Rel 3 | Fisher exact ($2\times2$) | 1250 | OR $=0.53$ | $0.407$ | $\phi=-0.029$ | No detectable difference; small n (R+ = 18). |
| Hate: R+ vs R−: Rel 3 | Fisher exact ($2\times2$) | 1250 | OR $=\infty$ (37/37 Rel 3) | $0.043$ | $\phi=+0.057$ | Harmful Hate responses are **more** often on-task (counter-speech). |

Reported p-values are two-sided. For omnibus rows the permutation p-value is
primary; the asymptotic p-value is shown in parentheses for reference.

## Table 2: Key observed relevance pattern

| Comparison | Rel 3 Rate A | Rel 3 Rate B | Difference | Interpretation |
|---|---:|---:|---:|---|
| Severity 2 responses vs all other severities | 64.1% (25/39) | 91.2% (1104/1211) | −27.1 pp | Robust ($p=5.7\times10^{-6}$): medium-severity responses are markedly less on-task. |
| Harmful Violence responses vs harmful Hate responses | 69.7% (23/33) | 100.0% (37/37) | −30.3 pp | Robust in direction ($p=2.3\times10^{-4}$) but small denominators. |
| Violence R+ (harmful resp.) vs R− (safe resp.) | 69.7% (23/33) | 90.9% | −21.2 pp | Robust ($p=7.0\times10^{-4}$): harmful Violence output is less relevant. |
| Harmful Sexual responses vs overall baseline | 82.8% (48/58) | 90.3% (1129/1250) | −7.5 pp | Suggestive only (Sexual R+ vs R−: $p=0.064$). |
| Any-harmful prompt vs all-safe prompt (all responses) | 91.8% (784/854) | 87.1% (345/396) | +4.7 pp | Harmful prompts draw *more* on-task answers, but with a heavier Rel 1 refusal tail. |

## Interpretation

- **Response severity is significantly associated with relevance** (permutation
  $\chi^2$, $p=3.0\times10^{-4}$; asymptotic $p=4.3\times10^{-9}$), with a
  small-to-moderate effect (Cramér's $V=0.142$). The signal is concentrated in
  the severity-2 band, whose Rel 3 rate (64.1%) is 27 pp below the rest of the
  data; this contrast is highly significant on its own ($p=5.7\times10^{-6}$).
- **Among harmful responses, relevance differs by category** (permutation
  $\chi^2$, $p=0.014$; $V=0.222$), driven by Violence (69.7% Rel 3) and Sexual
  (82.8%) versus Hate (100%). The Violence-vs-Hate contrast is significant
  ($p=2.3\times10^{-4}$), but denominators are small (Hate 37, Violence 33,
  Self-harm 18), so we mark category-level relevance comparisons as
  **exploratory**.
- **The "harmful response" quadrants are less relevant for Violence** ($p=7.0
  \times10^{-4}$) and directionally for Sexual ($p=0.064$, n.s.), consistent
  with the paper's descriptive claim that Violence/Sexual persistence and drift
  cells carry lower Rel 3 rates. Self-harm and Hate show no such reduction
  (Hate harmful responses are, if anything, *more* on-task).
- These tests establish **statistical association**, not causation: they do not
  show that higher severity or a given category *causes* lower relevance. They
  support the paper's descriptive relevance analysis where denominators are
  adequate (severity association; Violence) and appropriately **qualify** the
  sparse-cell, category-level comparisons as exploratory.

## Boundary-case interpretation

The tests are not just "significant vs. not" — read together they show that the
relevance shortfall is a **localized boundary phenomenon**, invisible in the
aggregate and only surfaced by severity- and category-conditioned analysis.

**The signal is local, not global.** At the aggregate level, harmful vs. safe
responses barely differ in relevance (84.4% vs. 91.1% Rel 3, $\phi=-0.071$, a
negligible effect). Relevance and harm are almost decoupled *on average*; the
structure appears only after conditioning on the severity band ($V=0.142$) and
category ($V=0.222$). This is the core evaluation-depth message: aggregate
relevance metrics hide the boundary behaviour.

**The severity effect is non-monotonic — a medium-severity boundary band.** Rel 3
rates by $\max s_r$ are 91.1% (Sev 0), 92.6% (Sev 1), **64.1% (Sev 2)**, 85.7%
(Sev 3). Relevance does not decay smoothly with severity; it is flat-high through
Sev 0–1, **collapses at Sev 2**, then partly recovers at Sev 3. The least on-task
outputs are therefore not the most severe ones but the **medium-severity,
partially-committed** ones, where the Rel 2 "partial" tail balloons to 35.9%.
Severity 2 is precisely the boundary between hedged refusal (Rel 1) and full
compliance (Rel 3), and the significant Sev-2 contrast ($p=5.7\times10^{-6}$)
confirms this is a real band effect rather than a 39-sample artefact.

**The dip is carried by Violence, not shared evenly.** Among harmful responses,
Rel 3 rates are Hate 100% (37), Self-harm 83.3% (18), Sexual 82.8% (58), and
**Violence 69.7% (33)**. Violence is the driver: harmful Violence responses are
far less relevant than harmful Hate ($p=2.3\times10^{-4}$) and than *safe*
Violence responses ($p=7.0\times10^{-4}$). Hate is the opposite boundary case —
harmful Hate responses are uniformly Rel 3 because they are on-task counter-speech
that engages the hateful framing directly, so *harm does not imply irrelevance*
here. Sexual points the same way as Violence but is only suggestive ($p=0.064$).

**Safety reading of the direction.** Lower Rel 3 for harmful Violence/Sexual means
those harmful outputs are disproportionately **partial/hedged** rather than fully
compliant — a qualitatively milder failure mode than clean, fully-relevant harmful
content. The medium-severity Rel 2 cluster is thus best read as *partial refusals
that leak some harmful content*, i.e. the boundary between refusal and compliance,
not as high-quality harmful assistance.

**Robust vs. limited (boundary cases).** The severity association, the Sev-2 dip,
and the Violence effect are robust (adequate $n$, small $p$, non-trivial effect
size). The Sexual reduction is directional only ($p=0.064$), and the Hate (37) and
Self-harm (18) harmful cells are too small for confirmatory category claims; we
mark these as exploratory and report effect sizes alongside every p-value.

## Summary

Significance tests quantify the association between response severity and
relevance and the category-level relevance differences. Relevance is
significantly associated with response severity (Monte-Carlo permutation
$\chi^2$, $p=3.0\times10^{-4}$; asymptotic $p=4.3\times10^{-9}$; Cramér's
$V=0.142$), with the effect concentrated in the medium-severity band
(severity-2 Rel 3 rate 64.1% vs. 91.2% elsewhere, Fisher
$p=5.7\times10^{-6}$). Among harmful responses, relevance differs significantly
by category (permutation $\chi^2$, $p=0.014$; $V=0.222$), and harmful Violence
responses are markedly less relevant than harmful Hate responses
($p=2.3\times10^{-4}$) and safe Violence responses
($p=7.0\times10^{-4}$); the Sexual reduction is directionally consistent but
does not reach significance ($p=0.064$). Category comparisons with small
denominators (Hate 37, Violence 33, Self-harm 18 harmful responses) are
exploratory and include effect sizes. These results establish statistical
association only, not causation by severity or category.

## Reproduction

Run:

```bash
cd PairedSafety/analysis/relevance_significance
python3 compute_relevance_tests.py
```

The script reads the labelled dataset, prints the summary above, and writes
`relevance_test_results.json`. It first reproduces Tables 1, 5, 6 and 7 of the
paper (all match exactly), then computes the tests. Randomised components use a
fixed seed (`RNG_SEED = 20260709`); permutation p-values are stable to the third
decimal at 50,000 permutations.
