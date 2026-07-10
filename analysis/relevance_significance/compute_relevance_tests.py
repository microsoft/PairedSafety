#!/usr/bin/env python3
"""Significance tests for the paper's relevance analysis.

Reproduces the paper's relevance tables (Tables 1, 5, 6, 7) from the labelled
paired prompt-response dataset and adds rigorous significance tests plus effect
sizes for the association between relevance and (a) response severity,
(b) harm category among harmful responses, and (c) the joint prompt x response
harm quadrant.

Definitions follow the paper (Section 3):
  - Per-category severity s^c in {0,1,2,3} for c in {Hate, Self-harm, Sexual,
    Violence}, taken as the highest active severity level in the label block.
  - max s = max_c s^c (aggregate maximum severity).
  - Relevance r in {1,2,3} (human-annotated relevance_score).
  - Prompt harmful in category c (P+): s^c_prompt >= 1.
  - Response harmful in category c (R+): s^c_response >= 1.

Statistical approach:
  - r x c independence: asymptotic Pearson chi-square (corroborative) plus a
    Monte Carlo label-permutation test that fixes both margins and is valid
    under sparse / zero cells. Effect size: Cramer's V (with bias-corrected V).
  - Targeted 2x2 contrasts: Fisher exact test (odds ratio) + phi effect size.
  - Sparse-cell comparisons (small denominators) are flagged as exploratory.

The tests support / qualify the paper's *descriptive* relevance observations.
They quantify statistical association only; they do not establish that severity
or category *causes* lower relevance.
"""

import json
import math
import os
import random
from collections import Counter

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency, fisher_exact

# Labelled paired corpus (not publicly released); supply your own copy via the
# PAIREDSAFETY_LABELS environment variable.
DATA_PATH = os.environ.get("PAIREDSAFETY_LABELS", "data/paired_response_labels.jsonl")

CATEGORIES = {"Hate": "H", "Self-harm": "SH", "Sexual": "S", "Violence": "V"}
RELEVANCE_LEVELS = (1, 2, 3)
N_PERM = 50000
RNG_SEED = 20260709


def category_severity(label_dict, prefix):
    """Highest active severity level (0-3) for one category."""
    level = 0
    for i in (1, 2, 3):
        val = label_dict.get(f"{prefix}{i}")
        if val is True or (isinstance(val, (int, float)) and val > 0):
            level = i
    return level


def max_severity(label_dict):
    return max(category_severity(label_dict, p) for p in CATEGORIES.values())


def load_rows(path=DATA_PATH):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj["metadata"]
            resp = obj["labels"]
            prompt = meta.get("prompt_label", {})
            rec = {
                "relevance": int(meta["relevance_score"]),
                "resp_sev": {c: category_severity(resp, p) for c, p in CATEGORIES.items()},
                "prompt_sev": {c: category_severity(prompt, p) for c, p in CATEGORIES.items()},
            }
            rec["max_resp_sev"] = max(rec["resp_sev"].values())
            rec["max_prompt_sev"] = max(rec["prompt_sev"].values())
            rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def cramers_v(table):
    """Cramer's V and bias-corrected V (Bergsma 2013) for an r x c table."""
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n == 0:
        return float("nan"), float("nan"), 0.0
    chi2_stat, _, _, _ = chi2_contingency(table, correction=False)
    r, c = table.shape
    phi2 = chi2_stat / n
    v = math.sqrt(phi2 / max(1, min(r - 1, c - 1)))
    # Bias-corrected (Bergsma).
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = max(1e-12, min(r_corr - 1, c_corr - 1))
    v_corr = math.sqrt(phi2_corr / denom)
    return v, v_corr, chi2_stat


def chi2_statistic(table):
    table = np.asarray(table, dtype=float)
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    n = table.sum()
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return terms.sum()


def permutation_chi2_test(labels_a, labels_b, n_perm=N_PERM, seed=RNG_SEED):
    """Monte Carlo label-permutation test of independence for two paired
    categorical vectors. Fixes both margins exactly; valid under sparse cells.

    Returns (chi2_obs, p_perm, table, cats_a, cats_b).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    cats_a = sorted(set(a.tolist()))
    cats_b = sorted(set(b.tolist()))
    ia = {v: i for i, v in enumerate(cats_a)}
    ib = {v: i for i, v in enumerate(cats_b)}
    table = np.zeros((len(cats_a), len(cats_b)))
    for x, y in zip(a, b):
        table[ia[x], ib[y]] += 1
    chi2_obs = chi2_statistic(table)

    rng = np.random.default_rng(seed)
    b_codes = np.array([ib[v] for v in b])
    a_codes = np.array([ia[v] for v in a])
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(b_codes)
        tab = np.zeros((len(cats_a), len(cats_b)))
        np.add.at(tab, (a_codes, perm), 1)
        if chi2_statistic(tab) >= chi2_obs - 1e-9:
            ge += 1
    p_perm = (1 + ge) / (1 + n_perm)
    return chi2_obs, p_perm, table, cats_a, cats_b


def asymptotic_chi2(table):
    table = np.asarray(table, dtype=float)
    chi2_stat, p, dof, _ = chi2_contingency(table, correction=False)
    return chi2_stat, p, dof


def rel3_rate(counter):
    """counter maps relevance level -> count. Returns (rel3_rate, n)."""
    n = sum(counter.values())
    return (counter.get(3, 0) / n if n else float("nan")), n


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def analysis_severity(rows):
    """Test 1: max response severity (0-3) x relevance (1-3)."""
    sev = [r["max_resp_sev"] for r in rows]
    rel = [r["relevance"] for r in rows]
    chi2_obs, p_perm, table, cats_a, cats_b = permutation_chi2_test(sev, rel)
    chi2_asym, p_asym, dof = asymptotic_chi2(table)
    v, v_corr, _ = cramers_v(table)

    # Targeted contrast: severity-2 responses vs all other responses,
    # rel3 vs not-rel3 (the medium-severity relevance dip in Table 1).
    def rel3_split(mask):
        yes = sum(1 for r, m in zip(rows, mask) if m and r["relevance"] == 3)
        no = sum(1 for r, m in zip(rows, mask) if m and r["relevance"] != 3)
        return yes, no

    sev2 = [r["max_resp_sev"] == 2 for r in rows]
    other = [not m for m in sev2]
    a_yes, a_no = rel3_split(sev2)
    b_yes, b_no = rel3_split(other)
    ct = [[a_yes, a_no], [b_yes, b_no]]
    odds, p_fisher = fisher_exact(ct, alternative="two-sided")
    phi = phi_2x2(ct)

    return {
        "table": table.tolist(),
        "cats_row": cats_a,
        "cats_col": cats_b,
        "n": int(np.sum(table)),
        "chi2": chi2_obs,
        "dof": dof,
        "p_perm": p_perm,
        "p_asym": p_asym,
        "cramers_v": v,
        "cramers_v_corrected": v_corr,
        "sev2_contrast": {
            "table": ct,
            "sev2_rel3_rate": a_yes / (a_yes + a_no),
            "other_rel3_rate": b_yes / (b_yes + b_no),
            "odds_ratio": odds,
            "p_fisher": p_fisher,
            "phi": phi,
            "n": a_yes + a_no + b_yes + b_no,
        },
    }


def phi_2x2(ct):
    (a, b), (c, d) = ct
    n = a + b + c + d
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return float("nan")
    return (a * d - b * c) / denom


def analysis_category_harmful(rows):
    """Test 2: harm category x relevance among *harmful* responses (Table 5).

    A response can be harmful in multiple categories, so category rows overlap
    and are not mutually exclusive; the pooled r x c test is therefore reported
    as descriptive, corroborated by clean pairwise 2x2 Fisher contrasts.
    """
    per_cat = {c: Counter() for c in CATEGORIES}
    for r in rows:
        for c in CATEGORIES:
            if r["resp_sev"][c] >= 1:
                per_cat[c][r["relevance"]] += 1

    table = np.array(
        [[per_cat[c].get(rl, 0) for rl in RELEVANCE_LEVELS] for c in CATEGORIES],
        dtype=float,
    )
    # Drop all-zero relevance columns for the omnibus test.
    keep = table.sum(axis=0) > 0
    tab_nonzero = table[:, keep]
    chi2_asym, p_asym, dof = asymptotic_chi2(tab_nonzero)
    v, v_corr, _ = cramers_v(tab_nonzero)

    # Permutation omnibus on the per-response category-membership is not
    # well-defined under overlap; instead we permute using the expanded
    # (category, relevance) row set, which matches the descriptive table.
    exp_cat, exp_rel = [], []
    for ci, c in enumerate(CATEGORIES):
        for rl in RELEVANCE_LEVELS:
            for _ in range(int(table[ci, RELEVANCE_LEVELS.index(rl)])):
                exp_cat.append(c)
                exp_rel.append(rl)
    chi2_obs, p_perm, _, _, _ = permutation_chi2_test(exp_cat, exp_rel)

    rates = {c: rel3_rate(per_cat[c]) for c in CATEGORIES}

    # Clean pairwise contrast: Violence vs Hate harmful responses, rel3 vs not.
    def yn(cnt):
        return cnt.get(3, 0), sum(cnt.values()) - cnt.get(3, 0)
    v_yes, v_no = yn(per_cat["Violence"])
    h_yes, h_no = yn(per_cat["Hate"])
    ct = [[v_yes, v_no], [h_yes, h_no]]
    odds, p_fisher = fisher_exact(ct, alternative="two-sided")
    phi = phi_2x2(ct)

    return {
        "per_cat_counts": {c: dict(per_cat[c]) for c in CATEGORIES},
        "table": table.tolist(),
        "n_rows": int(table.sum()),
        "chi2": chi2_obs,
        "dof": dof,
        "p_perm": p_perm,
        "p_asym": p_asym,
        "cramers_v": v,
        "cramers_v_corrected": v_corr,
        "rel3_rates": {c: rates[c][0] for c in CATEGORIES},
        "denominators": {c: rates[c][1] for c in CATEGORIES},
        "violence_vs_hate": {
            "table": ct,
            "violence_rel3_rate": v_yes / (v_yes + v_no) if (v_yes + v_no) else float("nan"),
            "hate_rel3_rate": h_yes / (h_yes + h_no) if (h_yes + h_no) else float("nan"),
            "odds_ratio": odds,
            "p_fisher": p_fisher,
            "phi": phi,
            "n": v_yes + v_no + h_yes + h_no,
        },
    }


def analysis_quadrant(rows):
    """Test 3: joint prompt x response harm quadrant x relevance (Table 7).

    Focus: do *harmful responses* (R+) have lower relevance-3 rates, especially
    for Violence and Sexual? We test (a) pooled response-harmful vs
    response-safe rel3 rate, and (b) per-category R+ vs R- rel3 contrasts.
    """
    # (a) Pooled: response harmful (max_resp_sev >= 1) vs safe, rel3 vs not.
    def yn_mask(mask):
        yes = sum(1 for r, m in zip(rows, mask) if m and r["relevance"] == 3)
        no = sum(1 for r, m in zip(rows, mask) if m and r["relevance"] != 3)
        return yes, no

    harmful = [r["max_resp_sev"] >= 1 for r in rows]
    safe = [not m for m in harmful]
    h_yes, h_no = yn_mask(harmful)
    s_yes, s_no = yn_mask(safe)
    ct = [[h_yes, h_no], [s_yes, s_no]]
    odds, p_fisher = fisher_exact(ct, alternative="two-sided")
    phi = phi_2x2(ct)
    pooled = {
        "table": ct,
        "harmful_resp_rel3_rate": h_yes / (h_yes + h_no),
        "safe_resp_rel3_rate": s_yes / (s_yes + s_no),
        "odds_ratio": odds,
        "p_fisher": p_fisher,
        "phi": phi,
        "n": h_yes + h_no + s_yes + s_no,
    }

    # (b) Per-category R+ vs R- relevance-3 contrast.
    per_cat = {}
    for c in CATEGORIES:
        rp = [r["resp_sev"][c] >= 1 for r in rows]
        rn = [not m for m in rp]
        p_yes, p_no = yn_mask(rp)
        n_yes, n_no = yn_mask(rn)
        ctc = [[p_yes, p_no], [n_yes, n_no]]
        oc, pc = fisher_exact(ctc, alternative="two-sided")
        per_cat[c] = {
            "table": ctc,
            "Rplus_rel3_rate": p_yes / (p_yes + p_no) if (p_yes + p_no) else float("nan"),
            "Rminus_rel3_rate": n_yes / (n_yes + n_no) if (n_yes + n_no) else float("nan"),
            "Rplus_n": p_yes + p_no,
            "odds_ratio": oc,
            "p_fisher": pc,
            "phi": phi_2x2(ctc),
        }

    # (c) Quadrant table per category (for reproduction / reporting).
    quad_tables = {}
    for c in CATEGORIES:
        q = {"P+R+": Counter(), "P+R-": Counter(), "P-R+": Counter(), "P-R-": Counter()}
        for r in rows:
            pp = "P+" if r["prompt_sev"][c] >= 1 else "P-"
            rr = "R+" if r["resp_sev"][c] >= 1 else "R-"
            q[pp + rr][r["relevance"]] += 1
        quad_tables[c] = {k: dict(v) for k, v in q.items()}

    return {"pooled": pooled, "per_cat": per_cat, "quad_tables": quad_tables}


# --------------------------------------------------------------------------- #
# Reproduction of paper tables (verification)
# --------------------------------------------------------------------------- #
def reproduce_tables(rows):
    out = {}
    # Table 1: max response severity x relevance.
    t1 = {s: Counter() for s in range(4)}
    for r in rows:
        t1[r["max_resp_sev"]][r["relevance"]] += 1
    out["table1"] = {
        s: {
            "n": sum(t1[s].values()),
            "rel3": pct(t1[s], 3),
            "rel2": pct(t1[s], 2),
            "rel1": pct(t1[s], 1),
        }
        for s in range(4)
        if sum(t1[s].values()) > 0
    }
    # Table 5: relevance among harmful responses per category.
    out["table5"] = {}
    for c in CATEGORIES:
        cnt = Counter()
        for r in rows:
            if r["resp_sev"][c] >= 1:
                cnt[r["relevance"]] += 1
        out["table5"][c] = {"n": sum(cnt.values()), "rel3": pct(cnt, 3),
                            "rel2": pct(cnt, 2), "rel1": pct(cnt, 1)}
    # Table 6: relevance across all responses grouped by prompt harm category.
    def grp(mask):
        cnt = Counter(r["relevance"] for r, m in zip(rows, mask) if m)
        return {"n": sum(cnt.values()), "rel3": pct(cnt, 3),
                "rel2": pct(cnt, 2), "rel1": pct(cnt, 1)}
    out["table6"] = {
        "Overall": grp([True] * len(rows)),
        "Any harmful": grp([r["max_prompt_sev"] >= 1 for r in rows]),
        "All safe": grp([r["max_prompt_sev"] == 0 for r in rows]),
    }
    for c in CATEGORIES:
        out["table6"][c] = grp([r["prompt_sev"][c] >= 1 for r in rows])
    # Table 7: quadrant per category.
    out["table7"] = {}
    for c in CATEGORIES:
        out["table7"][c] = {}
        for pp in ("P+", "P-"):
            for rr in ("R+", "R-"):
                cnt = Counter()
                for r in rows:
                    a = "P+" if r["prompt_sev"][c] >= 1 else "P-"
                    b = "R+" if r["resp_sev"][c] >= 1 else "R-"
                    if a == pp and b == rr:
                        cnt[r["relevance"]] += 1
                out["table7"][c][pp + rr] = {
                    "n": sum(cnt.values()), "rel3": pct(cnt, 3),
                    "rel2": pct(cnt, 2), "rel1": pct(cnt, 1)}
    return out


def pct(counter, level):
    n = sum(counter.values())
    return round(100 * counter.get(level, 0) / n, 1) if n else 0.0


def main():
    random.seed(RNG_SEED)
    rows = load_rows()
    result = {
        "n_total": len(rows),
        "data_source": "author-supplied restricted internal corpus",
        "n_perm": N_PERM,
        "reproduction": reproduce_tables(rows),
        "test1_severity": analysis_severity(rows),
        "test2_category_harmful": analysis_category_harmful(rows),
        "test3_quadrant": analysis_quadrant(rows),
    }

    with open("relevance_test_results.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)

    # Console summary.
    t1 = result["test1_severity"]
    print(f"N = {result['n_total']}")
    print("\n[Test 1] Response severity x relevance (4x3)")
    print(f"  chi2 = {t1['chi2']:.2f}, dof = {t1['dof']}, "
          f"p_perm = {t1['p_perm']:.5f}, p_asym = {t1['p_asym']:.2e}")
    print(f"  Cramer's V = {t1['cramers_v']:.3f} (bias-corrected "
          f"{t1['cramers_v_corrected']:.3f})")
    s2 = t1["sev2_contrast"]
    print(f"  Sev2 vs rest rel3: {s2['sev2_rel3_rate']*100:.1f}% vs "
          f"{s2['other_rel3_rate']*100:.1f}%, Fisher p = {s2['p_fisher']:.2e}, "
          f"phi = {s2['phi']:.3f}")

    t2 = result["test2_category_harmful"]
    print("\n[Test 2] Category x relevance among harmful responses (overlapping rows)")
    print(f"  chi2 = {t2['chi2']:.2f}, p_perm = {t2['p_perm']:.5f}, "
          f"p_asym = {t2['p_asym']:.4f}, Cramer's V = {t2['cramers_v']:.3f}")
    print(f"  rel3 rates: " + ", ".join(
        f"{c} {t2['rel3_rates'][c]*100:.1f}% (n={t2['denominators'][c]})"
        for c in CATEGORIES))
    vh = t2["violence_vs_hate"]
    print(f"  Violence vs Hate rel3: {vh['violence_rel3_rate']*100:.1f}% vs "
          f"{vh['hate_rel3_rate']*100:.1f}%, Fisher p = {vh['p_fisher']:.4f}, "
          f"phi = {vh['phi']:.3f}")

    t3 = result["test3_quadrant"]
    p = t3["pooled"]
    print("\n[Test 3] Joint quadrant / response-harm x relevance")
    print(f"  Pooled harmful-resp vs safe-resp rel3: "
          f"{p['harmful_resp_rel3_rate']*100:.1f}% vs "
          f"{p['safe_resp_rel3_rate']*100:.1f}%, Fisher p = {p['p_fisher']:.4f}, "
          f"phi = {p['phi']:.3f}")
    for c in CATEGORIES:
        pc = t3["per_cat"][c]
        print(f"  {c}: R+ rel3 {pc['Rplus_rel3_rate']*100:.1f}% "
              f"(n={pc['Rplus_n']}) vs R- {pc['Rminus_rel3_rate']*100:.1f}%, "
              f"Fisher p = {pc['p_fisher']:.4f}")

    print("\nWrote relevance_test_results.json")


if __name__ == "__main__":
    main()
