---

# RESULTS

## Public Reproducibility Mini-Eval

**Reproduction code and artifacts:** [`public_mini_eval/`](.)
(prompt-set builder, generation, grader, and analysis scripts; all resumable).

### Dataset source and sample size

We draw *prompts only* from two publicly redistributable benchmarks that the
paper already cites, and generate all responses ourselves:

- **XSTest** (Röttger et al., 2024; CC-BY-4.0): benign and borderline prompts,
  each labeled `safe`/`unsafe` with a prompt type.
- **Do-Not-Answer** (Wang et al., 2024; Apache-2.0): clearly-harmful prompts,
  labeled by risk area / harm type.

To make the evaluation substantial rather than a toy, we use the **entire
XSTest benchmark (all 450 prompts, 18 types)** plus a **stratified 150-prompt
Do-Not-Answer sample** (fixed seed `20260709`, weighted toward hate, adult
content, mental-health, and illegal-activity harm types to cover the paper's
four categories), for **600 unique prompts**. The set is deliberately mixed —
250 public-`safe` and 350 public-`unsafe` prompts — so that reduction,
preservation, and escalation (including drift from benign prompts) are all
observable. Under our rubric grader, 285/600 prompts are harmful (`max sₚ ≥ 1`);
the grader's harmful/safe call agrees with the XSTest prompt label on 82%
(370/450) of XSTest items, a basic public-label sanity check.

Each of 6 models answered all 600 prompts, yielding exactly **3,600 analyzed
prompt→response pairs**. These pairs reuse **600 unique prompts** across models
and therefore are not 3,600 independent observations. The released public
prompts, generations, grader labels, and analysis code make this supporting
evaluation auditable and reproducible.

### Model and generation settings

All models were queried through content-filter-disabled research deployments so
that each model's *own* alignment behavior is measured rather than an external
filter. **No system prompt** was used, matching the paper's generation setup.
Decoding: `temperature=0.7, top_p=1.0, max_tokens=1024` for standard models; for
reasoning models (gpt-5, grok-4, gpt-oss-120b) we used API defaults with a 4,000-
token budget (reasoning tokens are consumed internally).

| Family | Models | Relative safety alignment (public knowledge) |
|---|---|---|
| Open-weight | Llama-3.3-70B-Instruct, gpt-oss-120b, DeepSeek-V3.1 | strong / moderate–strong / lighter |
| Closed | gpt-4.1, gpt-5, grok-4 | strong / strongest / more permissive |

Both prompts and responses were then labeled **independently** on the paper's
0–3 severity scale across the four categories (Hate, Sexual, Violence,
Self-harm) using the paper's public rubric grader (system prompt from
[`docs/prompts/system_prompt_compact.md`](../../docs/prompts/system_prompt_compact.md);
grader model gpt-5). Independent labeling is what makes the paired transition
analysis well-defined. Labels are LLM-as-judge, not human (see limitations).

### Aggregate transition results

Transition types use maximum severity across categories, exactly as in the paper:
escalation (`max sᵣ > max sₚ`), preservation (`=`), reduction (`<`); conditional
reduction is `Pr(max sᵣ < max sₚ | max sₚ ≥ 1)`.

| Setting | N | Reduction | Preservation | Escalation | Conditional Reduction |
|---|---:|---:|---:|---:|---:|
| llama-3.3-70b (open) | 600 | 40.2% | 54.5% | 5.3% | 84.6% |
| gpt-oss-120b (open) | 600 | 44.3% | 50.8% | 4.8% | 93.3% |
| deepseek-v3.1 (open) | 600 | 41.0% | 51.2% | 7.8% | 86.3% |
| gpt-4.1 (closed) | 600 | 41.2% | 55.3% | 3.5% | 86.7% |
| gpt-5 (closed) | 600 | 42.5% | 53.2% | 4.3% | 89.5% |
| grok-4 (closed) | 600 | 37.7% | 52.7% | 9.7% | 79.3% |
| **Pooled** | **3,600** | **41.1%** | **52.9%** | **5.9%** | **86.6%** |

The prompt-cluster bootstrap 95% interval for pooled escalation is
[4.53%, 7.42%], resampling each unique prompt together with all six model
outputs. Escalations cluster at the bottom of the severity scale (115 land at
severity 1, 78 at severity 2, only 20 at severity 3).

### Model safety comparison

Because prompts are held fixed across models, the framework doubles as a
**cross-model safety probe**. We report, per model, the response-harm rate
(`Pr(max sᵣ ≥ 1)`), harmful-prompt compliance (`Pr(max sᵣ ≥ 1 | max sₚ ≥ 1)` —
how often a model answers a harmful prompt with harmful content), benign-prompt
drift (`Pr(max sᵣ ≥ 1 | max sₚ = 0)`), and escalation.

| Model | Family | Resp-harm rate | Harmful-prompt compliance | Benign drift | Escalation | Compliance escalations |
|---|---|---:|---:|---:|---:|---:|
| **grok-4** | closed | **20.2%** | **27.0%** | **14.0%** | **9.7%** | **14** |
| deepseek-v3.1 | open | 15.5% | 17.2% | 14.0% | 7.8% | 3 |
| llama-3.3-70b | open | 14.3% | 20.0% | 9.2% | 5.3% | 3 |
| gpt-4.1 | closed | 11.5% | 17.2% | 6.3% | 3.5% | 1 |
| gpt-5 | closed | 11.0% | 14.0% | 8.3% | 4.3% | 0 |
| gpt-oss-120b | open | 8.0% | 7.4% | 8.6% | 4.8% | 2 |

Grok-4 has the highest observed response-harm rate, harmful-prompt compliance,
and escalation rate and accounts for **14 of the 23 pooled compliance
escalations**. Because models answer the same prompts, inference uses pairwise
exact McNemar tests rather than an independent-samples proportion test, with
Holm correction across the five comparisons for each metric. Grok-4's
response-harm and harmful-prompt-compliance rates exceed each comparator after
correction (all adjusted $p\le0.0046$). Its escalation rate exceeds four of the
five comparators (all adjusted $p\le1.2\times10^{-4}$), but the contrast with
DeepSeek-V3.1 is not statistically resolved (9.7% vs. 7.8%, adjusted $p=0.099$).
Full paired contingency counts and adjusted results are in
[`results/summary.md`](results/summary.md) and
[`results/metrics.json`](results/metrics.json).

Response-harm rate alone also separates several models. The added value of the
paired framework is to identify *how* that harm arises—persistence on harmful
prompts, drift from benign prompts, or an increase in severity—rather than to
claim that response-only rates contain no model signal.

### Category results

| Category | Harmful Prompts | Harmful Responses From Harmful Prompts | Persistence | Reduction |
|---|---:|---:|---:|---:|
| Hate | 546 | 40 | 7.3% | 92.7% |
| Sexual | 96 | 26 | 27.1% | 72.9% |
| Violence | 1,026 | 211 | 20.6% | 79.4% |
| Self-harm | 132 | 3 | 2.3% | 97.7% |

(Counts are over prompt×model pairs; persistence is a property of the pair.)
**Sexual again shows the highest category persistence (27.1%)** and lowest
reduction, mirroring the paper's category asymmetry, while Self-harm is the most
reliably reduced.

**Escalation origin split (pooled, 213 escalations).** The automatic rule places
23 cases in **compliance escalation** (an
already-harmful prompt answered on-task at higher severity — 14 from grok-4) and
190 **unsolicited elaborations** (a benign/low-severity prompt yielding a
higher-severity response). Of the unsolicited elaborations, 147 originate from
public-`safe` prompts, many of which are faithful answers to benign historical
questions (e.g., XSTest items on the Rwandan genocide or Auschwitz) that the
Azure rubric places at medium Violence — a public-data artifact rather than model
misbehavior, which we flag rather than hide.

### Does the framework work on public data?

**Operationally, yes.** Every metric in the framework — aggregate reduction,
preservation, escalation, conditional reduction, per-category persistence, and
the escalation-origin split — is computable on 600 publicly shareable prompts
and six models (3,600 model–prompt pairs). Two directional signals from the
human-labeled study also appear under the rubric grader: conditional reduction
on harmful prompts is 86.6%, and Sexual has the highest observed category
persistence (27.1%). These are directional observations, not rigorous
replication of the human-labeled estimates.

We do **not** claim rate-level comparability with the internal corpus: the
aggregate reduction/preservation split differs (the public prompt mix contains
many benign and borderline items that preserve at severity 0), the absolute
escalation rate is affected by content-filter-disabled endpoints and the
inclusion of deliberately permissive models, and part of the escalation mass
reflects the rubric's treatment of educational/historical content. Labels are
from an LLM rubric grader rather than human annotators, so absolute numbers are
noisier than the paper's human-labeled corpus. What the mini-eval establishes is
*operationality, conditional cross-model characterization, and directional
evidence*, not replication of the human-labeled findings.

### Summary

To assess reproducibility, data availability, and directional generalization,
we tested whether the proposed paired-transition framework can operate beyond
our internal corpus through a public-data supporting evaluation (code and
artifacts released) over the **full XSTest benchmark plus a stratified
Do-Not-Answer sample (600 prompts)** and six models spanning three open-weight
(Llama-3.3-70B, gpt-oss-120b, DeepSeek-V3.1) and three closed (gpt-4.1, gpt-5,
grok-4) systems, with prompt and response severity both labeled by our public
rubric grader on the same 0–3, four-category scale — **3,600 model–prompt pairs
over 600 unique prompts**. While not a benchmark replacement or a human-labeled
replication, it shows that every transition
metric — reduction (41.1%), preservation (52.9%), escalation (5.9%), conditional
reduction on harmful prompts (86.6%, vs. 89.3% internally), Sexual as the highest-
persistence category (27.1%), and the unsolicited-elaboration vs.
compliance-escalation split — is computable on publicly shareable data.
The matched comparison further illustrates the framework's conditional model
characterization: exact McNemar tests with Holm correction find higher
response-harm and harmful-prompt-compliance rates for grok-4 than for each other
model; its escalation rate exceeds four of five comparators, while the
DeepSeek-V3.1 contrast is not resolved. We interpret all grader-based public
results as directional rather than equivalent to the paper's human labels.
