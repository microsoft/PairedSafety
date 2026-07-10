# LLM-Assisted Error Analysis of Response-Side Grader Failures

*Grader:* GPT-5 few-shot grader, **response target**, using the public Azure AI
Content Safety rubric (Hate `H`, Sexual `S`, Violence `V`, Self-harm `SH`; merged
severity 0–3). *Data:* the 1,250 human-labelled paired prompt–response records
used in the paper (5,000 record-category decisions). *Predictions:* the
restricted response-target predictions used in the paper; internal storage
details are intentionally omitted.

## Scope and method

This is an error analysis of the paper's grader experiment, not an evaluation of
where a production moderation filter should be placed. The paper's main
contribution is paired analysis of human-labelled prompt and response severity.
The grader study asks a narrower question: why does the same rubric-based grader
transfer less effectively to responses than to prompts?

Error direction is computed mechanically against the human labels. One error is
one `(record, category)` mismatch. The response grader makes **329 errors**:
233 false positives (FP), 76 false negatives (FN), 13 nonzero under-severity
errors, and 7 nonzero over-severity errors.

Interpretive patterns are assigned through **LLM-assisted qualitative coding**:

1. GPT-5.2 and GPT-4.1 independently code all 329 errors from the full prompt,
   response, human severity, grader severity, and grader rationale.
2. The coders use a fixed seven-pattern codebook and select one primary observed
   pattern. Pattern labels are descriptive, not causal ground truth.
3. GPT-5.4 adjudicates primary-label disagreements.
4. The two initial coders agree on 267/329 cases (**81.2% exact agreement;
   Cohen's κ = 0.686**); 62 disagreements are adjudicated.

All three models use no-content-filter research deployments. The coding prompt
and pipeline are released in `llm_code_errors.py`; aggregate output is in
`llm_error_analysis.json`. Raw texts and per-case coding rationales remain
private because they contain explicit material. LLM coding can inherit model
biases, so the taxonomy is reported as a structured descriptive analysis rather
than human annotation.

## Metric context: rare positive response labels

Only **146/5,000 (2.9%)** response category-labels are truly harmful, compared
with **854/5,000 (17.1%)** prompt category-labels. The grader therefore emits
more false positives (233) than there are truly harmful response labels (146).
The binary counts imply 70 true positives, giving harmful-class precision
0.231 and recall 0.479, consistent with the paper's binary metrics. This class
imbalance explains why high accuracy can coexist with response macro-F1 of
0.356, but it does not itself explain the textual failure patterns; those are
examined below.

## Table 1: Primary observed error patterns

Each error receives exactly one final primary pattern after adjudication.
Percentages are over all 329 response-side errors.

| Primary observed pattern | Description | Count | % of errors | Typical direction |
|---|---|---:|---:|---|
| Educational / explanatory framing | Educational, medical, historical, fictional, analytical, or condemnatory framing is central to the mismatch. | 155 | 47.1% | 89 FP; 63 FN; 3 boundary |
| Over-triggering lexical cues | A safe response is flagged because it quotes or names harm-related, crisis-support, refusal, romantic, or violence-adjacent language without human-labelled harm. | 143 | 43.5% | 143 FP |
| Boundary ambiguity | Human and grader both mark the category harmful but disagree on nonzero severity. | 17 | 5.2% | 11 under; 6 over |
| Safe-refusal prior | Refusal or safety framing plausibly masks residual content retained by the response. | 5 | 1.5% | 5 FN |
| Other / unclear | No codebook pattern is well supported. | 5 | 1.5% | 1 FP; 4 FN |
| Category overlap / possible confusion | Co-occurring or adjacent categories make the specified category assignment uncertain. | 4 | 1.2% | 4 FN |
| Long-response dilution | A brief harmful span is missed inside a much longer, mostly safe response. | 0 | 0.0% | No case selected as the primary pattern |
| **Total** | | **329** | **100%** | |

The zero primary count for long-response dilution is informative. Although
responses are longer than prompts and some misses occur in long responses, the
full-context coders usually identify educational/explanatory framing—not length
alone—as the more specific primary pattern. The revised analysis therefore does
not retain the earlier claim that dilution is a dominant independent mechanism.

## Table 2: Error directions by category

| Category | Total errors | False negatives | False positives | Nonzero boundary errors | Dominant primary pattern |
|---|---:|---:|---:|---:|---|
| Hate | 68 | 19 | 49 | 0 | Over-triggering lexical cues (34) |
| Sexual | 68 | 34 | 23 | 11 | Educational / explanatory framing (45) |
| Violence | 137 | 12 | 118 | 7 | Over-triggering lexical cues (66; educational framing 65) |
| Self-harm | 56 | 11 | 43 | 2 | Over-triggering lexical cues (36) |
| **Total** | **329** | **76** | **233** | **20** | |

Violence is the largest false-positive source (118). Its two leading patterns
are nearly tied: lexical over-triggering (66) and educational/explanatory
framing (65), with some boundary cases accounting for the remainder. Sexual is
the largest false-negative source (34), and educational/explanatory framing is
its dominant primary pattern.

## Direction-specific findings

- **False positives:** 143/233 (61.4%) are lexical over-triggers and 89/233
  (38.2%) involve educational/explanatory framing. Thus, response-side precision
  is primarily damaged by safe text that contains harm-adjacent language.
- **False negatives:** 63/76 (82.9%) are assigned educational/explanatory
  framing, followed by five safe-refusal-prior cases, four category-overlap
  cases, and four unclear cases. The central FN problem is contextual
  interpretation, not merely response length.
- **Severity calibration:** 17/20 nonzero severity mismatches are coded as
  boundary ambiguity; the remaining three are better explained by framing.
  Sexual accounts for 11 of the 20 mechanical boundary errors.
- **Length:** FP responses have median length 1,410 characters (IQR 918–2,301),
  FN responses 1,137.5 (673–2,191), under-severity responses 2,164
  (1,363–4,289), and over-severity responses 1,650 (1,083.5–5,922.5), compared
  with 1,004 characters for all responses. Length is associated with errors but
  does not uniquely determine their coded pattern.

## Sanitized examples

The examples below paraphrase high-confidence coded cases; explicit material is
not reproduced.

1. **Educational-framing FP — Self-harm, human 0 → grader 1.** A long response
   gives supportive clinical, relapse-prevention, and urgent-care guidance. The
   coder identifies medical/harm-reduction framing as central to the grader
   mistaking support language for self-harm content.

2. **Educational-framing FP — Hate, human 0 → grader 1.** A response condemns an
   antisemitic stereotype and provides historical and allyship context. The
   grader flags the protected-group discussion despite its explicitly anti-hate
   framing.

3. **Lexical over-trigger FP — Sexual, human 0 → grader 1.** A benign fictional
   rewrite contains mild romantic wording. The grader appears to react to the
   wording even though the human label contains no sexual harm.

4. **Boundary ambiguity — Sexual, human 2 → grader 1.** Human and grader agree
   that the response is sexual, but differ by one level over whether suggestive
   physical detail crosses the Low/Medium boundary.

5. **Safe-refusal FN — Self-harm, human 1 → grader 0.** A response is dominated
   by supportive refusal and crisis-help language but retains a small amount of
   adjacent harm-reduction detail. The safety framing plausibly leads the grader
   to assign severity 0.

6. **Category overlap FN — Self-harm, human 2 → grader 0.** The response contains
   graphic accidental injury rather than intentional self-harm. The grader
   recognizes violence-like content but misses the human Self-harm category,
   exposing ambiguity at the category boundary.

## Why response grading is harder in this sample

1. **Positive labels are rare.** Harmful response category-labels are about six
   times rarer than harmful prompt labels (2.9% versus 17.1%). Consequently,
   false alarms strongly erode harmful-class precision while misses erode recall.

2. **Response errors are mainly contextual.** Educational/explanatory framing is
   the primary pattern for 47.1% of all errors and 82.9% of false negatives.
   Responses often discuss, condemn, refuse, or safely redirect harm-related
   content; these pragmatic functions are harder to infer from category terms
   than direct prompt intent.

3. **False positives dominate.** The grader produces 233 FPs versus 76 FNs.
   Lexical over-triggering and safe educational framing together account for 232
   of the 233 false positives.

4. **Category behavior differs.** Violence produces 118 FPs, Sexual produces the
   most FNs and boundary errors, and Self-harm often triggers on crisis-support
   vocabulary. A single global response threshold would obscure these distinct
   error profiles.

5. **Few-shot evidence is limited.** The paper's additions contain only two
   examples per sampled set. Their failure to close the response gap shows that
   these small interventions are insufficient; it does not establish that
   response-grader calibration is impossible.

These results support the narrow claim that response grading presents a
different contextual calibration problem from prompt grading. They do not by
themselves establish which moderation architecture should be deployed.

## Paper-ready summary

> We analyzed all
> 329 response-side record-category mismatches against the human labels,
> separating 233 false positives, 76 false negatives, and 20 nonzero severity
> errors. To avoid relying on keyword heuristics, two independent LLMs coded the
> full context of every error under a fixed taxonomy (81.2% exact agreement;
> Cohen's κ = 0.686), and a third LLM adjudicated 62 disagreements; we explicitly
> describe this as LLM-assisted qualitative coding rather than human annotation.
> The dominant primary patterns are educational/explanatory framing (155/329,
> 47.1%) and lexical over-triggering (143/329, 43.5%). In particular, 63/76
> false negatives involve framing, whereas 143/233 false positives are lexical
> over-triggers. Violence contributes the most false positives (118), while
> Sexual contributes the most false negatives (34) and nonzero boundary errors
> (11). We also clarify the metric effect of class imbalance: only 146/5,000
> response category-labels are harmful, so 233 false alarms exceed the number of
> true harmful labels, yielding harmful-class precision 0.231 and recall 0.479
> despite high accuracy. This strengthens our narrower conclusion that response
> grading requires response-specific contextual calibration. We have revised the
> moderation discussion to present this as a diagnostic implication of the
> paired analysis, not as validation of a particular deployed filter design.

## Scope and interpretation

- The exhaustive, category-specific analysis covers false positives, false
   negatives, nonzero severity-boundary errors, and sanitized examples under a
   disclosed coding protocol.
- It strengthens the empirical basis for the statement that response-side
   detection has distinct failure modes, but does not validate where a deployed
   filter should be placed; architecture optimization remains follow-up work.
- The analysis documents class imbalance, category variation, boundary
   ambiguity, framing failures, few-shot limitations, and the limitations of LLM
   coding as a qualitative instrument.

## Reproduce

```bash
cd PairedSafety/analysis/grader_error_analysis
python3 analyze_grader_errors.py
python3 llm_code_errors.py --workers 6
```

The second command requires authenticated access to the configured no-filter
Azure research deployments. The public aggregate artifact can be inspected
without endpoint access.
