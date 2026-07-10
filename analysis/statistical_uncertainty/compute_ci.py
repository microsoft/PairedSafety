#!/usr/bin/env python3
"""Quantify uncertainty in the paper's transition and category findings.

The analysis uses the final April labelled snapshot that reproduces the paper's
reported counts. It reports:

* Wilson score 95% confidence intervals for binomial proportions;
* a percentile bootstrap cross-check for the 32/40 escalation-origin split;
* interaction-level paired bootstrap intervals for differences between Sexual
    persistence and each other category, including Bonferroni simultaneous
    intervals; and
* an exact two-sided binomial test of the descriptive 32-versus-8 split.

Definitions follow Section 3 of the paper. For category c, s^c is the highest
active severity in {0, 1, 2, 3}, and max(s) is the maximum across categories.
Aggregate escalation, preservation, and reduction respectively mean
max(s_r) >, =, and < max(s_p). Category persistence is
Pr(s^c_r >= 1 | s^c_p >= 1), and category reduction is its complement.

The 32/8 analysis is labelled an ``origin split`` because max(s_p) identifies
whether escalation starts from an overall-safe or harmful prompt. A substantive
mechanism label (for example, unsolicited elaboration) additionally requires
the paper's manual audit of whether the added content was requested/on-task.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


# Labelled paired corpus (not publicly released); supply your own via --data or
# the PAIREDSAFETY_LABELS environment variable.
DEFAULT_DATA_PATH = Path(
    os.environ.get("PAIREDSAFETY_LABELS", "data/paired_response_labels.jsonl")
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("ci_results.json")

CONFIDENCE_LEVEL = 0.95
Z_95 = 1.959963984540054
N_BOOTSTRAP = 100_000
RNG_SEED = 20260709

CATEGORIES = {"Hate": "H", "Self-harm": "SH", "Sexual": "S", "Violence": "V"}


def category_severity(label_dict, prefix):
    """Return the highest active severity (0--3) for one harm category."""
    level = 0
    for severity in (1, 2, 3):
        value = label_dict.get(f"{prefix}{severity}")
        if value is True or (isinstance(value, (int, float)) and value > 0):
            level = severity
    return level


def max_severity(label_dict):
    """Return maximum severity across the four paper categories."""
    return max(category_severity(label_dict, prefix) for prefix in CATEGORIES.values())


def wilson_ci(successes, trials, z=Z_95):
    """Return estimate and two-sided Wilson score interval for a proportion.

    The interval is obtained by inverting the score test. Unlike the Wald
    interval, its bounds remain in [0, 1] and it behaves well for small counts
    and proportions near zero or one.
    """
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    estimate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (estimate + z**2 / (2 * trials)) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(estimate * (1 - estimate) / trials + z**2 / (4 * trials**2))
    )
    return estimate, max(0.0, center - half_width), min(1.0, center + half_width)


def proportion_result(successes, trials):
    """Create a JSON-ready proportion estimate with a Wilson 95% CI."""
    successes, trials = int(successes), int(trials)
    estimate, lower, upper = wilson_ci(successes, trials)
    return {
        "numerator": successes,
        "denominator": trials,
        "estimate": estimate,
        "ci_95_wilson": [lower, upper],
        "ci_width": upper - lower,
    }


def percentile_interval(values, confidence_level=CONFIDENCE_LEVEL):
    """Return an equal-tailed percentile interval from bootstrap estimates."""
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return [float(lower), float(upper)]


def load_rows(path):
    with path.open(encoding="utf-8") as data_file:
        rows = [json.loads(line) for line in data_file if line.strip()]
    if not rows:
        raise ValueError(f"no records found in {path}")
    if "prompt_label" not in rows[0].get("metadata", {}):
        raise ValueError(f"dataset does not contain metadata.prompt_label: {path}")
    return rows


def paired_category_bootstrap(prompt_severity, response_severity):
    """Bootstrap Sexual-minus-other category persistence differences.

    Whole interactions are resampled, preserving overlap/dependence among the
    four category indicators. In addition to pointwise 95% intervals, a
    Bonferroni interval with per-contrast confidence 1 - .05/3 is reported so
    that the three-contrast family has at least 95% simultaneous coverage.
    """
    harmful_prompt = prompt_severity >= 1
    persistent = harmful_prompt & (response_severity >= 1)
    observed_rates = persistent.sum(axis=0) / harmful_prompt.sum(axis=0)
    sexual_index = list(CATEGORIES).index("Sexual")
    comparison_indices = [index for index in range(len(CATEGORIES)) if index != sexual_index]

    rng = np.random.default_rng(RNG_SEED)
    bootstrap_differences = np.empty((N_BOOTSTRAP, len(comparison_indices)))
    batch_size = 1_000
    n = len(prompt_severity)

    for start in range(0, N_BOOTSTRAP, batch_size):
        size = min(batch_size, N_BOOTSTRAP - start)
        sampled_indices = rng.integers(0, n, size=(size, n))
        sampled_harmful = np.take(harmful_prompt, sampled_indices, axis=0).sum(axis=1)
        sampled_persistent = np.take(persistent, sampled_indices, axis=0).sum(axis=1)
        sampled_rates = sampled_persistent / sampled_harmful
        bootstrap_differences[start : start + size] = (
            sampled_rates[:, sexual_index, None] - sampled_rates[:, comparison_indices]
        )

    results = []
    category_names = list(CATEGORIES)
    bonferroni_confidence = 1 - (1 - CONFIDENCE_LEVEL) / len(comparison_indices)
    for column, comparison_index in enumerate(comparison_indices):
        distribution = bootstrap_differences[:, column]
        results.append(
            {
                "contrast": f"Sexual - {category_names[comparison_index]}",
                "difference": float(observed_rates[sexual_index] - observed_rates[comparison_index]),
                "ci_95_paired_percentile_bootstrap": percentile_interval(distribution),
                "ci_98_33_bonferroni_paired_percentile_bootstrap": percentile_interval(
                    distribution, bonferroni_confidence
                ),
            }
        )
    return results


def analyze(rows, data_path):
    n = len(rows)
    category_names = list(CATEGORIES)
    prompt_severity = np.empty((n, len(CATEGORIES)), dtype=np.int8)
    response_severity = np.empty_like(prompt_severity)

    for row_index, row in enumerate(rows):
        prompt_labels = row["metadata"]["prompt_label"]
        response_labels = row["labels"]
        for category_index, prefix in enumerate(CATEGORIES.values()):
            prompt_severity[row_index, category_index] = category_severity(prompt_labels, prefix)
            response_severity[row_index, category_index] = category_severity(response_labels, prefix)

    prompt_max = prompt_severity.max(axis=1)
    response_max = response_severity.max(axis=1)
    escalation = response_max > prompt_max
    preservation = response_max == prompt_max
    reduction = response_max < prompt_max
    harmful_prompt = prompt_max >= 1
    harmful_response = response_max >= 1

    overall = {
        "escalation": proportion_result(escalation.sum(), n),
        "preservation": proportion_result(preservation.sum(), n),
        "reduction": proportion_result(reduction.sum(), n),
        "conditional_reduction_given_harmful_prompt": proportion_result(
            np.count_nonzero(reduction & harmful_prompt), harmful_prompt.sum()
        ),
        "harmful_response_prevalence": proportion_result(harmful_response.sum(), n),
    }

    category_results = {}
    for category_index, category in enumerate(category_names):
        category_harmful_prompt = prompt_severity[:, category_index] >= 1
        category_persistence = category_harmful_prompt & (response_severity[:, category_index] >= 1)
        persistence_count = category_persistence.sum()
        harmful_prompt_count = category_harmful_prompt.sum()
        category_results[category] = {
            "harmful_prompts": int(harmful_prompt_count),
            "persistence": proportion_result(persistence_count, harmful_prompt_count),
            "reduction_to_safe": proportion_result(
                harmful_prompt_count - persistence_count, harmful_prompt_count
            ),
        }

    escalation_indices = np.flatnonzero(escalation)
    safe_origin = prompt_max[escalation_indices] == 0
    harmful_origin = ~safe_origin
    origin_bootstrap_rng = np.random.default_rng(RNG_SEED)
    origin_bootstrap = origin_bootstrap_rng.choice(
        safe_origin.astype(float), size=(N_BOOTSTRAP, len(escalation_indices)), replace=True
    ).mean(axis=1)
    safe_origin_bootstrap_ci = percentile_interval(origin_bootstrap)
    exact_split_test = binomtest(
        int(safe_origin.sum()), len(escalation_indices), p=0.5, alternative="two-sided"
    )
    origin_split = {
        "safe_prompt_origin": {
            "among_escalations": proportion_result(safe_origin.sum(), len(escalation_indices)),
            "among_all_pairs": proportion_result(safe_origin.sum(), n),
            "ci_95_percentile_bootstrap_among_escalations": safe_origin_bootstrap_ci,
        },
        "harmful_prompt_origin": {
            "among_escalations": proportion_result(harmful_origin.sum(), len(escalation_indices)),
            "among_all_pairs": proportion_result(harmful_origin.sum(), n),
            "ci_95_percentile_bootstrap_among_escalations": [
                1 - safe_origin_bootstrap_ci[1], 1 - safe_origin_bootstrap_ci[0]
            ],
        },
        "exploratory_exact_binomial_test_equal_split": {
            "null_safe_origin_proportion": 0.5,
            "alternative": "two-sided",
            "p_value": float(exact_split_test.pvalue),
            "interpretation_limit": (
                "Tests imbalance relative to an equal 50/50 split; it does not validate mechanism labels."
            ),
        },
    }

    category_escalation = {}
    for category_index, category in enumerate(category_names):
        escalated = response_severity[:, category_index] > prompt_severity[:, category_index]
        room_to_escalate = prompt_severity[:, category_index] < 3
        category_escalation[category] = {
            "all_pairs": proportion_result(escalated.sum(), n),
            "prompts_with_room_to_escalate": proportion_result(
                escalated.sum(), room_to_escalate.sum()
            ),
        }

    return {
        "metadata": {
            "data_source": "author-supplied restricted internal corpus",
            "input_filename": data_path.name,
            "n_pairs": n,
            "confidence_level": CONFIDENCE_LEVEL,
            "wilson_z": Z_95,
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": RNG_SEED,
            "analysis_unit": "prompt-response interaction",
        },
        "definitions": {
            "escalation": "max(s_response) > max(s_prompt)",
            "preservation": "max(s_response) = max(s_prompt)",
            "reduction": "max(s_response) < max(s_prompt)",
            "conditional_reduction": "Pr(reduction | max(s_prompt) >= 1)",
            "category_persistence": "Pr(s_response^c >= 1 | s_prompt^c >= 1)",
            "category_reduction": "Pr(s_response^c = 0 | s_prompt^c >= 1)",
            "category_escalation": "s_response^c > s_prompt^c",
        },
        "methods": {
            "proportion_intervals": "two-sided Wilson score 95% confidence intervals",
            "origin_cross_check": "equal-tailed percentile bootstrap over the 40 escalation cases",
            "category_differences": (
                "equal-tailed paired percentile bootstrap resampling whole interactions; "
                "pointwise 95% and Bonferroni 98.33% per-contrast intervals for at least "
                "95% family-wise coverage across three Sexual-versus-other contrasts"
            ),
        },
        "overall_transitions": overall,
        "category_persistence_reduction": category_results,
        "sexual_persistence_pairwise_differences": paired_category_bootstrap(
            prompt_severity, response_severity
        ),
        "escalation_origin_split": origin_split,
        "category_escalation": category_escalation,
    }


def pct(value):
    return f"{100 * value:.2f}%"


def print_summary(results):
    print(f"Dataset: {results['metadata']['data_source']} "
          f"({results['metadata']['input_filename']})")
    print(f"Paired interactions: {results['metadata']['n_pairs']}")
    print("\nOverall transitions")
    for name, result in results["overall_transitions"].items():
        lower, upper = result["ci_95_wilson"]
        print(
            f"  {name:46s} {result['numerator']:4d}/{result['denominator']:<4d} "
            f"{pct(result['estimate'])} [{pct(lower)}, {pct(upper)}]"
        )

    print("\nCategory persistence")
    for category, result in results["category_persistence_reduction"].items():
        persistence = result["persistence"]
        lower, upper = persistence["ci_95_wilson"]
        print(
            f"  {category:10s} {persistence['numerator']:3d}/{persistence['denominator']:<3d} "
            f"{pct(persistence['estimate'])} [{pct(lower)}, {pct(upper)}]"
        )

    print("\nSexual persistence pairwise differences")
    for result in results["sexual_persistence_pairwise_differences"]:
        lower, upper = result["ci_95_paired_percentile_bootstrap"]
        family_lower, family_upper = result[
            "ci_98_33_bonferroni_paired_percentile_bootstrap"
        ]
        print(
            f"  {result['contrast']:24s} {pct(result['difference'])} "
            f"95% [{pct(lower)}, {pct(upper)}], "
            f"simultaneous [{pct(family_lower)}, {pct(family_upper)}]"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="April labelled JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.data)
    results = analyze(rows, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_summary(results)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
