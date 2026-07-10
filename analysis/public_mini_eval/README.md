# Public Supporting Mini-Eval

A fully public supporting evaluation of the paired prompt→response safety-transition
framework from the main paper. It shows that the framework's transition metrics
(reduction / preservation / escalation, conditional reduction, category
persistence, and the escalation-mechanism split) can be computed on publicly
shareable prompts and open / closed models — not just the internal corpus.

This is a **supporting mini-eval**, not a benchmark replacement.

## What this does

1. Builds a stratified, publicly shareable **prompt** set from two benchmarks
   that the paper already cites: **XSTest** (Röttger et al., 2024, full 450) and
   **Do-Not-Answer** (Wang et al., 2024, stratified 150) = 600 prompts. Both
   benign and harmful prompts are included so that drift/escalation from safe
   prompts is observable.
2. Generates **responses** from six models (three open-weight, three closed),
   yielding ~3,600 paired prompt→response records.
3. Labels **both** prompt and response severity with the paper's public rubric
   grader (0–3 severity across Hate, Sexual, Violence, Self-harm).
4. Computes **paired** transition metrics from the (prompt-label, response-label)
  pairs, plus **cross-model safety comparisons** (response-harm rate,
  harmful-prompt compliance, benign drift) using prompt-matched inference.

## Data sources

| Source | License / availability | Role | Prompt-level public label |
|---|---|---|---|
| XSTest | Public (CC-BY-4.0, GitHub `paul-rottger/xstest`) | benign + borderline prompts | `safe` / `unsafe` + type |
| Do-Not-Answer | Public (Apache-2.0, HF `LibrAI/do-not-answer`) | clearly-harmful prompts | risk area / harm type + `GPT4_harmful` |

Only prompts are taken from these sets; all responses are freshly generated.
Sampling is **stratified by the datasets' own type/harm labels** with a fixed
seed (`20260709`) — we do not cherry-pick only extreme harmful items. See
`01_build_prompt_set.py` for exact strata and counts.
All harmful prompts originate from these already-public benchmarks; this
evaluation introduces no newly authored unsafe prompts.

## Models and generation settings

All models are called through **content-filter-disabled** research deployments
so that each model's *own* alignment behavior is measured, not an external
filter. No system prompt is used (matching the paper's generation setup).

| Model | Family | Relative safety alignment (public knowledge) |
|---|---|---|
| Llama-3.3-70B-Instruct | open | Strong (Meta safety fine-tuning / refusal training) |
| gpt-oss-120b | open | Moderate–strong (OpenAI open-weight, safety-trained reasoning) |
| DeepSeek-V3.1 | open | Comparatively lighter / more permissive |
| gpt-4.1 | closed | Strong (production alignment) |
| gpt-5 | closed | Strongest (frontier alignment) |
| grok-4 | closed | Comparatively more permissive |

Decoding:
- Standard models: `temperature=0.7`, `top_p=1.0`, `max_tokens=1024`.
- Reasoning models (gpt-5, grok-4, gpt-oss-120b): API defaults, token budget
  4000 (reasoning tokens are consumed internally); temperature not set.
- System prompt: **none**.

## Grader

The grader is the paper's public few-shot rubric grader. Its system prompt is
loaded directly from `../../docs/prompts/system_prompt_compact.md` (shipped in
this repo). It predicts raw 0–7 severity per category and maps to the merged 0–3
scale. Grader model: **gpt-5** (strongest available rubric follower). Prompts and
responses are graded independently, which is what makes the paired transition
analysis well-defined. This is an LLM-as-judge label, not a human label (see
Limitations).

## Pipeline

```bash
python 01_build_prompt_set.py                 # -> data/prompts.jsonl
python 02_generate_responses.py               # -> data/responses/<model>.jsonl
python 03_grade.py                            # -> data/graded/{prompts,response_<model>}.jsonl
python 04_analyze.py                          # -> results/{metrics.json,summary.md,escalation_cases.json}
```

Every stage is resumable (re-running skips already-completed rows).
Authentication uses `AzureCliCredential` (`az login`).

## Metric definitions (identical to the paper)

- max severity: max over the four categories.
- aggregate transition on max severity: escalation (`max s_r > max s_p`),
  preservation (`=`), reduction (`<`).
- conditional reduction: `Pr(max s_r < max s_p | max s_p >= 1)`.
- per category `c`: persistence `Pr(s_r^c>=1 | s_p^c>=1)`,
  reduction `Pr(s_r^c=0 | s_p^c>=1)`, drift `Pr(s_p^c=0 | s_r^c>=1)`.
- escalation mechanism: unsolicited elaboration (`max s_p=0, max s_r>=1`) vs.
  compliance escalation (`max s_p>=1, max s_r>max s_p`).

## Limitations

- Labels are from an LLM rubric grader, not human annotators; absolute rates are
  noisier than the paper's human-labeled corpus. Consequently, agreement with
  the human-labelled study is interpreted as directional rather than as a
  matched replication, and absolute rates are not directly comparable.
- Public taxonomies do not map perfectly onto the four categories; category
  coverage (especially Sexual / Self-harm) is thinner than the internal set.
- Sample size (3,579 pairs) exceeds the internal corpus in pairs but reuses 600
  shared prompts across models, so pairs are not independent across settings.
- Cross-model inference therefore uses prompt-matched exact McNemar tests with
  Holm correction; pooled uncertainty resamples unique prompts as clusters.
