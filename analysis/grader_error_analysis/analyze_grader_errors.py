#!/usr/bin/env python3
"""Response-side LLM grader error analysis (EMNLP rebuttal, R2).

Loads the response-target grader predictions used in the paper, joins them to
the labelled paired prompt-response dataset, enumerates every grader error, and
classifies each error into an interpretable taxonomy.

Outputs (written next to this script):
  - error_analysis.json      : machine-readable counts + per-error records.
  - example_candidates.json  : sanitizable example pool (rationale + snippets).
  - error_taxonomy.md        : the rebuttal-ready markdown section.

Definitions (follow the paper, Section 3):
  - Per-category severity s^c in {0,1,2,3} for c in {Hate (H), Sexual (S),
    Violence (V), Self-harm (SH)}, taken as the highest active level X1..X3.
  - The grader predicts severity_0_3 per category on RESPONSE text only.

Error types (per category-label, i.e. one (record, category) mismatch):
  - false_negative : human >=1, grader = 0   (misses residual harm)
  - false_positive : human  =0, grader >=1   (over-flags safe text)
  - under_severity : human>=1, grader>=1, grader < human (boundary, too low)
  - over_severity  : human>=1, grader>=1, grader > human (boundary, too high)

The interpretable taxonomy assigns each error one primary label (priority
order below) and also records all applicable labels (multi-label), because a
single error can have several causes.
"""

import json
import os
import re
import statistics
from collections import Counter, defaultdict

# --- Paths -------------------------------------------------------------------
# The labelled paired corpus is not publicly released (compliance). Point these
# at your own copy via environment variables to reproduce the analysis.
PRED_DIR = os.environ.get("PAIREDSAFETY_PRED_DIR", "data/grader_preds")
SRC_PATH = os.environ.get("PAIREDSAFETY_LABELS", "data/paired_response_labels.jsonl")
GRADER = "response_gpt-5.jsonl"  # primary paper grader, response target

CATS = ["H", "S", "V", "SH"]
CATNAME = {"H": "Hate", "S": "Sexual", "V": "Violence", "SH": "Self-harm"}

# --- Text signal detectors ---------------------------------------------------
# Explicit refusal / safety-framing language (grader may treat as "safe").
REFUSAL = re.compile(
    r"\b(i can(?:no|')t|i cannot|i'?m sorry|i am sorry|i won'?t|i'?m unable|"
    r"i am unable|i'?m not able|cannot help|can'?t help|i must decline|"
    r"i do not (?:provide|condone|support)|i don'?t (?:provide|condone)|"
    r"against .{0,20}polic|not able to (?:help|assist|provide))",
    re.I,
)
# Educational / explanatory / historical framing around a harm topic.
EDU = re.compile(
    r"\b(histor|educat|it'?s important|important to (?:understand|note|be aware)|"
    r"for context|in general|awareness|prevention|disclaimer|please note|"
    r"note that|understanding|academic|research purposes|hypothetical|"
    r"for informational|from a .{0,15}perspective|it is worth noting)",
    re.I,
)


def category_severity(label_dict, prefix):
    """Highest active severity level (0-3) for one category (paper definition)."""
    level = 0
    for i in (1, 2, 3):
        val = label_dict.get(f"{prefix}{i}")
        if val is True or (isinstance(val, (int, float)) and val > 0):
            level = i
    return level


def load():
    src = {}
    for line in open(SRC_PATH):
        r = json.loads(line)
        src[str(r["metadata"]["hit_id"])] = r
    preds = [json.loads(l) for l in open(f"{PRED_DIR}/{GRADER}")]
    return src, preds


def collect_errors(src, preds):
    """Return list of per-(record, category) error dicts with text signals."""
    all_chars = []
    for p in preds:
        s = src[str(p["hit_id"])]
        all_chars.append(len(s["input"][0]["content"]))
    median_chars = statistics.median(all_chars)

    errors = []
    for p in preds:
        s = src[str(p["hit_id"])]
        resp = s["input"][0]["content"]
        prompt = s["metadata"].get("prompt", "")
        nchars = len(resp)
        has_refusal = bool(REFUSAL.search(resp))
        has_edu = bool(EDU.search(resp))
        is_long = nchars > median_chars
        true_active = [c for c in CATS if p["true_label"][c] >= 1]
        pred_active = [c for c in CATS if p["prediction"][c]["severity_0_3"] >= 1]

        for c in CATS:
            t = p["true_label"][c]
            pr = p["prediction"][c]["severity_0_3"]
            if t == pr:
                continue
            if t >= 1 and pr == 0:
                etype = "false_negative"
            elif t == 0 and pr >= 1:
                etype = "false_positive"
            elif pr < t:
                etype = "under_severity"
            else:
                etype = "over_severity"

            # category confusion: grader put harm in a *different* category
            # than the (co-located) true harm within the same record.
            if etype == "false_negative":
                cat_conf = any(pc != c for pc in pred_active)
            elif etype == "false_positive":
                cat_conf = any(tc != c for tc in true_active)
            else:
                cat_conf = False

            errors.append(
                {
                    "hit_id": str(p["hit_id"]),
                    "record_id": p.get("record_id"),
                    "category": c,
                    "category_name": CATNAME[c],
                    "error_type": etype,
                    "true_sev": t,
                    "pred_sev": pr,
                    "n_chars": nchars,
                    "is_long": is_long,
                    "has_refusal": has_refusal,
                    "has_edu": has_edu,
                    "multi_true_cats": len(true_active) >= 2,
                    "category_confusion": cat_conf,
                    "rationale": p["prediction"].get("rationale", ""),
                    "resp_gen_model": p.get("response_generation_model"),
                    "prompt": prompt,
                    "response": resp,
                }
            )
    return errors, median_chars


# --- Taxonomy ----------------------------------------------------------------
TAX = {
    "boundary": "Boundary ambiguity",
    "safe_refusal": "Safe-refusal prior",
    "educational": "Educational framing",
    "dilution": "Long-response dilution",
    "over_trigger": "Over-triggering lexical cues",
    "category_overlap": "Category overlap / confusion",
    "other": "Other / unclear",
}

TAX_DESC = {
    "boundary": "Correct category, severity off by one level (Low<->Medium).",
    "safe_refusal": "Response reads as a refusal / safety statement, so grader "
    "underlabels residual harmful content it still carries.",
    "educational": "Harmful or policy-relevant content sits inside explanatory, "
    "historical, or informational framing; grader treats framing as neutral.",
    "dilution": "A short harmful span is embedded in a long, mostly-safe "
    "response; grader misses it.",
    "over_trigger": "Safe response mentions harm-relevant terms/quotes and is "
    "flagged as harmful with no true harm.",
    "category_overlap": "Multiple harm signals present or grader assigns harm to "
    "the wrong category within the same response.",
    "other": "Error with no dominant identifiable cause from text signals.",
}


def classify(e):
    """Return (primary_label, [all_applicable_labels])."""
    labels = []
    et = e["error_type"]

    if et in ("under_severity", "over_severity"):
        labels.append("boundary")
    if et == "false_negative" and e["has_refusal"]:
        labels.append("safe_refusal")
    if e["has_edu"] and et in ("false_negative", "false_positive"):
        labels.append("educational")
    if et == "false_negative" and e["is_long"]:
        labels.append("dilution")
    if et == "false_positive":
        labels.append("over_trigger")
    if e["multi_true_cats"] or e["category_confusion"]:
        labels.append("category_overlap")

    # Primary label by priority.
    priority = [
        "boundary",
        "safe_refusal",
        "educational",
        "dilution",
        "over_trigger",
        "category_overlap",
    ]
    primary = next((p for p in priority if p in labels), "other")
    if not labels:
        labels = ["other"]
    return primary, labels


def main():
    src, preds = load()
    unjoined = sum(1 for p in preds if str(p["hit_id"]) not in src)
    assert unjoined == 0, f"{unjoined} predictions did not join to source"

    errors, median_chars = collect_errors(src, preds)

    # Assign taxonomy.
    primary_counts = Counter()
    multi_counts = Counter()
    for e in errors:
        primary, labels = classify(e)
        e["primary"] = primary
        e["labels"] = labels
        primary_counts[primary] += 1
        for l in labels:
            multi_counts[l] += 1

    n_err = len(errors)
    type_counts = Counter(e["error_type"] for e in errors)

    # Table 2: per category, FN / FP counts + dominant primary taxonomy.
    cat_fn = Counter()
    cat_fp = Counter()
    cat_primary = defaultdict(Counter)
    for e in errors:
        c = e["category_name"]
        if e["error_type"] == "false_negative":
            cat_fn[c] += 1
        elif e["error_type"] == "false_positive":
            cat_fp[c] += 1
        cat_primary[c][e["primary"]] += 1

    # ---- Write JSON ---------------------------------------------------------
    out = {
        "grader": GRADER,
        "n_records": len(preds),
        "n_error_instances": n_err,
        "median_response_chars": median_chars,
        "error_type_counts": dict(type_counts),
        "taxonomy_primary_counts": {TAX[k]: v for k, v in primary_counts.items()},
        "taxonomy_multilabel_counts": {TAX[k]: v for k, v in multi_counts.items()},
        "per_category": {
            c: {
                "false_negatives": cat_fn.get(c, 0),
                "false_positives": cat_fp.get(c, 0),
                "dominant_primary": TAX[cat_primary[c].most_common(1)[0][0]],
            }
            for c in CATNAME.values()
        },
    }
    with open("error_analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- Example candidate pool (for hand-sanitized write-up) ---------------
    def snippet(txt, n=220):
        txt = " ".join(txt.split())
        return txt[:n] + ("..." if len(txt) > n else "")

    pool = defaultdict(list)
    for e in errors:
        pool[e["primary"]].append(
            {
                "hit_id": e["hit_id"],
                "category": e["category_name"],
                "error_type": e["error_type"],
                "true_sev": e["true_sev"],
                "pred_sev": e["pred_sev"],
                "n_chars": e["n_chars"],
                "labels": [TAX[l] for l in e["labels"]],
                "grader_rationale": e["rationale"],
                "prompt_snippet": snippet(e["prompt"], 160),
                "response_snippet": snippet(e["response"], 300),
                "resp_gen_model": e["resp_gen_model"],
            }
        )
    with open("example_candidates.json", "w") as f:
        json.dump(pool, f, indent=2)

    # ---- Console summary ----------------------------------------------------
    print(f"grader={GRADER}  records={len(preds)}  error instances={n_err}")
    print("error types:", dict(type_counts))
    print("\nprimary taxonomy (each error counted once):")
    for k, _ in primary_counts.most_common():
        v = primary_counts[k]
        print(f"  {TAX[k]:32s} {v:4d}  {100*v/n_err:5.1f}%")
    print("\nmulti-label taxonomy (an error may count in >1 row):")
    for k, v in multi_counts.most_common():
        print(f"  {TAX[k]:32s} {v:4d}  {100*v/n_err:5.1f}%")
    print("\nper category (FN / FP / dominant):")
    for c in CATNAME.values():
        print(
            f"  {c:10s} FN={cat_fn.get(c,0):3d}  FP={cat_fp.get(c,0):3d}  "
            f"dominant={TAX[cat_primary[c].most_common(1)[0][0]]}"
        )


if __name__ == "__main__":
    main()
