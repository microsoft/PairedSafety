# PairedSafety

Reproducibility artifacts for the paper *From Prompt Risk to Response Risk:
Paired Analysis of Safety Behavior of Large Language Models*. The paper studies
LLM safety as a **transition** from prompt severity to response severity across
four harm categories (Hate, Sexual, Violence, Self-harm) and four ordinal
severity levels (0 Safe, 1 Low, 2 Medium, 3 High).

The human-labeled internal corpus (1,250 pairs) cannot be released because its
hand-authored prompts contain explicit harm-category exemplars. This repository
instead provides everything needed to (a) reproduce the grader setup and
(b) run a **public-data supporting evaluation** of the framework on open benchmarks
and open/closed models.

## Repository layout

```
docs/prompts/                     Grader prompt artifacts (rubric + templates)
analysis/
  public_mini_eval/               Public-data evaluation (public prompts + open/closed models)
    01_build_prompt_set.py        Build the stratified public prompt set
    02_generate_responses.py      Generate responses (your own endpoints)
    03_grade.py                   Grade prompts + responses with the rubric grader
    04_analyze.py                 Compute all transition / category / model metrics
    common.py                     Endpoints, model registry, grader (env-configurable)
    requirements.txt
    data/                         Public prompts, raw responses, grader labels (see "Data")
    results/                      Aggregate metrics (metrics.json, escalation_cases.json)
  statistical_uncertainty/        Wilson CIs + bootstrap for transition/category rates
  relevance_significance/         Permutation / Fisher tests with effect sizes
  grader_error_analysis/          Response-side grader error taxonomy
```

## Grader prompts (`docs/prompts/`)

The grader is rendered from public Azure AI Content Safety text definitions and
examples for Hate and Fairness, Sexual, Violence, and Self-harm. It predicts raw
severity 0–7 per category and maps those to the paper's merged 0–3 labels.

- `system_prompt_compact.md` — compact system prompt for the full-corpus runs.
- `single_item_user_prompt_template.md` — user prompt for one record.
- `batch_user_prompt_template.md` — user prompt for batched records.
- `dataset_ablation_shots/` — sampled two-example sets for the ablation study.

To grade, send the system prompt plus the appropriate user prompt, with **only**
the target text (prompt for prompt-grading, response for response-grading). Any
LLM client works; no API client is bundled.

## Public supporting evaluation (`analysis/public_mini_eval/`)

A public, at-scale supporting evaluation: the full XSTest benchmark (450 prompts) plus a
stratified Do-Not-Answer sample (150) = **600 public prompts** answered by six
models (Llama-3.3-70B, gpt-oss-120b, DeepSeek-V3.1, gpt-4.1, gpt-5, grok-4),
yielding ~3,600 analyzed prompt→response pairs. Both prompts and responses are
labeled by the rubric grader on the same 0–3, four-category scale. Because this
uses 600 unique prompts repeatedly across models and LLM-grader rather than
human labels, its findings are directional and are not presented as a rigorous
replication of the internal human-labeled rates.

### Quick reproduction (no model calls)

All reported numbers are reproducible from the released grader labels alone:

```bash
cd analysis/public_mini_eval
pip install -r requirements.txt
python 04_analyze.py           # reads data/graded/ + data/prompts.jsonl
```

### Full pipeline (regenerate everything)

Point the scripts at your own content-filter-disabled deployments via
environment variables (no private endpoints are committed):

```bash
export PAIREDSAFETY_V1_ENDPOINT="https://<your-openai-compatible-resource>/openai/v1/"
export PAIREDSAFETY_AOAI_ENDPOINT="https://<your-azure-openai-resource>.openai.azure.com/"
# optional: PAIREDSAFETY_AOAI_ENDPOINT_2

# Download the public benchmark inputs into data/ first:
#   data/xstest_prompts.csv   (XSTest; Roettger et al., 2024; CC-BY-4.0)
#   data/do_not_answer.jsonl  (Do-Not-Answer; Wang et al., 2024; Apache-2.0)

python 01_build_prompt_set.py    # -> data/prompts.jsonl
python 02_generate_responses.py  # -> data/responses/
python 03_grade.py               # -> data/graded/
python 04_analyze.py             # -> results/metrics.json, results/escalation_cases.json
```

Cross-model inference uses prompt-matched exact McNemar tests with Holm
correction. Pooled uncertainty resamples unique prompts together with all
available model outputs, preserving the repeated-prompt dependence.

## Additional analyses

- `analysis/statistical_uncertainty/compute_ci.py` — Wilson 95% CIs and
  bootstrap intervals for the transition and category findings; see
  `ci_results.json` and `results.md`.
- `analysis/relevance_significance/compute_relevance_tests.py` — permutation
  chi-square / Fisher tests with effect sizes for the relevance analysis; see
  `relevance_test_results.json` and `results.md`.
- `analysis/grader_error_analysis/` — mechanical enumeration and LLM-assisted
  coding of response-side grader errors; see `error_taxonomy.md` and
  `llm_error_analysis.json`.

These three operate on the internal labeled corpus (not released); point them at
your own labeled data via the `PAIREDSAFETY_LABELS` environment variable.

## Data

The `public_mini_eval/data/` and `results/` directories include the public
evaluation artifacts:

- `data/prompts.jsonl` — the sampled public prompt set with public labels.
- `data/responses/*.jsonl` — raw model responses for each generator.
- `data/graded/*.jsonl` — grader **severity labels only** (`{H,S,V,SH}` + ids);
  no response text.
- `results/metrics.json`, `results/escalation_cases.json` — aggregate metrics
  and escalation records (severity dictionaries only).

**Content warning:** `data/responses/*.jsonl` contains freshly generated model
outputs to public harmful prompts, including unsafe text (191 severity-2 and 21
severity-3 responses under the released grader labels). These files are included
to make the public evaluation auditable end-to-end; use them only for research
and safety evaluation.

## Citation

Please cite the paper (anonymized during review). Public benchmarks used:
XSTest (Roettger et al., 2024) and Do-Not-Answer (Wang et al., 2024).