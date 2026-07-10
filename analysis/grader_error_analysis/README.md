# Response-Side Grader Error Analysis (Rebuttal R2)

Targeted error analysis explaining **why response-side moderation is harder than
prompt-side moderation**, addressing R2's request for an LLM-grader error
analysis (and R1/R3 concerns on empirical support and grader clarity).

## Contents

| File | Description |
|---|---|
| `analyze_grader_errors.py` | Joins the paper's response-target grader predictions to the labelled paired dataset and mechanically enumerates every FP, FN, and nonzero severity error. Its original regex taxonomy is retained only as a baseline and is not used for the revised claims. |
| `llm_code_errors.py` | Runs two independent no-filter LLM coders over all errors, reports reliability, and uses a third LLM to adjudicate primary-pattern disagreements. |
| `error_taxonomy.md` | **Deliverable.** The rebuttal-ready "Error Analysis of Response-Side Grader Failures" section: taxonomy table, per-category table, sanitized examples, and the R2 rebuttal paragraph. |
| `error_analysis.json` | Machine-readable counts (generated). |
| `llm_error_analysis.json` | Public aggregate multi-LLM coding results, agreement, cross-tabs, and length summaries. |
| `example_candidates.json` | Sanitizable example pool with grader rationales (generated). |

## Data sources (internal, not redistributed)

- Predictions: `omni/notebooks/dsats/emnlp2026/grader/preds_public_azure_compact/response_gpt-5.jsonl`
  (1,250 records; per-category human `true_label` + grader `prediction`).
- Labelled paired dataset: `omni/.../2026April_response/combined_response_label_orange.jsonl`
  (joined on `metadata.hit_id`; the file that reproduces the paper's numbers).

The grader is the GPT-5 few-shot grader scored on **response** text only, using
the public Azure AI Content Safety rubric (Hate/Sexual/Violence/Self-harm,
severity 0–3).

## Headline findings

- 329 grader errors: **233 FP, 76 FN, 20 boundary**.
- Harmful response labels are rare (**2.9%** of category-labels vs **17.1%** on
  prompts) → FPs (233) exceed true-harmful labels (146) → macro-F1 collapses
  (**0.356** response vs **0.745** prompt) despite high accuracy.
- Two independent coders agree on **267/329 (81.2%)** primary patterns
  (**Cohen's κ = 0.686**); GPT-5.4 adjudicates 62 disagreements.
- Dominant primary observed patterns: **educational/explanatory framing
  (155/329, 47.1%)** and **lexical over-triggering (143/329, 43.5%)**.
- Interpretive labels are LLM-assisted descriptive coding, not human annotation
  or causal ground truth.

## Reproduce

```bash
python3 analyze_grader_errors.py
python3 llm_code_errors.py --workers 6
```

The first script uses only the Python standard library. The second requires the
configured Azure/OpenAI dependencies and authenticated access to the no-filter
research deployments. Raw case-level codings contain explicit text and are
gitignored; only aggregate output is public.
