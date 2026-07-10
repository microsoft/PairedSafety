# Statistical Uncertainty for Escalation and Category Findings

This analysis quantifies uncertainty for conclusions involving the 40
escalation cases and strengthens the paper's statistical reporting. Statistical
tests for relevance are provided separately in the relevance-significance
analysis.

**Study scope.** This is an observational analysis of 1,250 independently
labelled prompt–response pairs. The goal is to report what is observed in this
sample and quantify sampling uncertainty, not to claim that the rates are
universal across models, languages, or deployment settings. The final April
labelled snapshot used by the paper is analyzed by [compute_ci.py](compute_ci.py),
with all machine-readable output in [ci_results.json](ci_results.json).

**Methods.** Binomial rates use two-sided **Wilson score 95% confidence
intervals** (Wilson, 1927). Wilson intervals are obtained by inverting the score
test, stay within $[0,1]$, and are preferable to normal/Wald intervals for small
counts or rates near 0 or 1. We use a 100,000-resample percentile bootstrap as a
cross-check for the escalation-origin split. Category persistence differences
use a **paired interaction-level bootstrap**: complete prompt–response records
are resampled, preserving category overlap within a record. We report ordinary
95% intervals and Bonferroni 98.33% per-contrast intervals, giving at least 95%
family-wise coverage across the three Sexual-versus-other comparisons. All
randomized analyses use seed 20260709.

**Definitions.** For category severity $s^c\in\{0,1,2,3\}$ and
$\max s=\max_c s^c$: escalation is $\max s_r>\max s_p$; preservation is
$\max s_r=\max s_p$; reduction is $\max s_r<\max s_p$; conditional reduction is
$\Pr(\max s_r<\max s_p\mid\max s_p\ge1)$; category persistence is
$\Pr(s^c_r\ge1\mid s^c_p\ge1)$; category reduction is
$\Pr(s^c_r=0\mid s^c_p\ge1)$; and category escalation is $s^c_r>s^c_p$.

## Table 1: Overall transition rates

| Metric | Numerator | Denominator | Rate | Wilson 95% CI |
|---|---:|---:|---:|---:|
| Escalation | 40 | 1,250 | 3.20% | [2.36%, 4.33%] |
| Preservation | 447 | 1,250 | 35.76% | [33.15%, 38.46%] |
| Reduction | 763 | 1,250 | 61.04% | [58.31%, 63.71%] |
| Conditional reduction among harmful prompts | 763 | 854 | 89.34% | [87.10%, 91.24%] |
| Harmful-response prevalence | 141 | 1,250 | 11.28% | [9.64%, 13.15%] |

The distinction between the **event count** and the **estimation denominator**
is important. The escalation rate is estimated from all 1,250 pairs, with 40
observed escalation events. Its 95% interval places the sample-compatible rate
between 2.36% and 4.33%, an absolute width of 1.97 percentage points. Thus, the
data support the observation of a small but nonzero escalation tail in this
corpus. They do not support a universal 3.20% rate beyond the sampled setting.
The preservation and reduction intervals have widths of 5.31 and 5.40
percentage points, respectively.

## Table 2: Category persistence and reduction

| Category | Harmful prompts | Persistence, n/N | Persistence, Wilson 95% CI | Reduction to safe, n/N | Reduction, Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Hate | 329 | 24/329 (7.29%) | [4.95%, 10.62%] | 305/329 (92.71%) | [89.38%, 95.05%] |
| Self-harm | 106 | 12/106 (11.32%) | [6.60%, 18.75%] | 94/106 (88.68%) | [81.25%, 93.40%] |
| Sexual | 189 | 47/189 (24.87%) | [19.25%, 31.49%] | 142/189 (75.13%) | [68.51%, 80.75%] |
| Violence | 230 | 19/230 (8.26%) | [5.35%, 12.54%] | 211/230 (91.74%) | [87.46%, 94.65%] |

These category findings are not based on only the 40 aggregate escalations.
Their denominators are the category-specific harmful-prompt sets (106–329
prompts). Sexual persistence is higher in this sample than persistence in each
other category. Table 3 tests the differences directly while preserving the
overlap among category labels.

## Table 3: Sexual persistence differences

| Contrast | Difference | Paired-bootstrap 95% CI | Bonferroni simultaneous interval* |
|---|---:|---:|---:|
| Sexual − Hate | +17.57 pp | [10.93, 24.43] pp | [9.53, 25.95] pp |
| Sexual − Self-harm | +13.55 pp | [4.78, 22.11] pp | [2.76, 23.93] pp |
| Sexual − Violence | +16.61 pp | [9.54, 23.76] pp | [8.00, 25.33] pp |

*Each Bonferroni interval has 98.33% coverage, providing at least 95%
family-wise coverage across the three pre-specified contrasts. All three
simultaneous intervals exclude zero. This supports the bounded claim that
Sexual content has higher persistence **in this sample**; it does not establish
that the ordering generalizes to other corpora or models.

## Table 4: Escalation origin split

The 40 escalations split by the prompt's aggregate starting severity. This
origin split is directly reproducible from the labels. It corresponds to the
paper's manual audit—safe-prompt cases were examined as unsolicited elaboration
and harmful-prompt cases as compliance escalation—but prompt origin alone should
not be treated as a substitute for manual mechanism coding.

| Escalation origin | Count | Share among escalations | Wilson 95% CI | Bootstrap 95% CI | Share among all pairs | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Safe prompt ($\max s_p=0$) | 32 | 80.00% | [65.24%, 89.50%] | [67.50%, 92.50%] | 2.56% (32/1,250) | [1.82%, 3.59%] |
| Harmful prompt ($\max s_p\ge1$) | 8 | 20.00% | [10.50%, 34.76%] | [7.50%, 32.50%] | 0.64% (8/1,250) | [0.32%, 1.26%] |

As an exploratory check, the 32:8 imbalance differs from an equal 50:50 origin
split (exact two-sided binomial $p=1.82\times10^{-4}$). The 50:50 null is only a
reference point, not a theory-derived expected mechanism distribution. More
importantly, the Wilson interval for the 80% share is wide (65.24%–89.50%). We
therefore report the mechanism composition as a **descriptive audit finding**,
not a precise population estimate.

## Table 5: Per-category escalation

A pair can escalate in multiple categories, so these counts overlap and do not
sum to 40. The first denominator estimates prevalence among all pairs. The
second is the category-specific set with room to escalate ($s^c_p<3$).

| Category | Escalations | Rate over 1,250 | Wilson 95% CI | $s^c_p<3$ denominator | Rate over eligible denominator | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Hate | 13 | 1.04% | [0.61%, 1.77%] | 1,154 | 1.13% | [0.66%, 1.92%] |
| Self-harm | 6 | 0.48% | [0.22%, 1.04%] | 1,216 | 0.49% | [0.23%, 1.07%] |
| Sexual | 17 | 1.36% | [0.85%, 2.17%] | 1,142 | 1.49% | [0.93%, 2.37%] |
| Violence | 16 | 1.28% | [0.79%, 2.07%] | 1,144 | 1.40% | [0.86%, 2.26%] |

The intervals overlap substantially. Accordingly, these per-category escalation
counts describe where escalations appeared; they do not support ranking
categories by escalation prevalence.

## Paper-ready summary

> Forty escalation cases do not support highly precise estimates of
> escalation subtypes. We have therefore added numerator/denominator counts and
> Wilson 95% confidence intervals for every aggregate and category-level rate,
> together with interaction-level bootstrap intervals for category contrasts.
> The overall escalation estimate uses all 1,250 labelled pairs: 40/1,250, or
> 3.20% (95% CI [2.36%, 4.33%]). This supports our corpus-level observation of a
> small escalation tail while making clear that 3.20% is not a universal rate.
> The category-persistence analysis has separate harmful-prompt denominators
> (106–329 rather than 40). Sexual persistence is 47/189 (24.87%, Wilson 95% CI
> [19.25%, 31.49%]) and exceeds Hate, Self-harm, and Violence by 13.55–17.57
> percentage points; paired interaction-level bootstrap intervals remain above
> zero after simultaneous-coverage adjustment. In contrast, the escalation
> origin/mechanism analysis is conditional on only 40 cases: 32/40 originated
> from safe prompts (80.00%, Wilson 95% CI [65.24%, 89.50%]). We therefore retain
> this as an interesting descriptive audit finding, explicitly report its broad
> uncertainty, and avoid presenting the mechanism proportions or overlapping
> per-category escalation counts as precise population estimates. Our aim is to
> present a resource-intensive, human-labelled, multi-category and multi-severity
> observation study with transparent uncertainty, while the separate public-data
> evaluation tests whether the same analysis is operational beyond this closed
> corpus.

## Interpretation for the revised paper

The statistical results support three differently scoped statements:

1. **Supported corpus-level estimate:** escalation is uncommon but observed in
   this corpus (40/1,250; 3.20%, 95% CI [2.36%, 4.33%]).
2. **Supported within-corpus category contrast:** Sexual persistence is higher
   than the other three categories, including under paired resampling and
   simultaneous interval adjustment.
3. **Descriptive small-subset finding:** most audited escalations originated
   from safe prompts, but the 40-case mechanism composition and per-category
   escalation counts remain imprecise and should not be generalized strongly.

*Reproduce with `python compute_ci.py`. The script accepts `--data` and
`--output` overrides and writes all estimates, intervals, definitions, and
method metadata to `ci_results.json`.*
