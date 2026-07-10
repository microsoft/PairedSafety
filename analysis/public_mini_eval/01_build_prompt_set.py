"""Build a stratified, publicly shareable prompt set from XSTest + Do-Not-Answer.

Scaled version: the FULL XSTest benchmark (450) plus a stratified Do-Not-Answer
sample (150) = 600 prompts. With six models this yields ~3,600 analyzed pairs,
larger than the internal corpus, while remaining fully public and reproducible.

Design goals (documented for the rebuttal):
- Include BOTH benign and harmful prompts so that reduction, preservation, and
  escalation (incl. drift from safe prompts) are all observable.
- Stratify by the datasets' own labels/types; do NOT cherry-pick only extreme
  harmful items.
- Cover the four paper categories (Hate, Sexual, Violence, Self-harm) as far as
  the public taxonomies allow.

Inputs (public benchmarks, not redistributed here -- download into ``data/``):
  - XSTest: ``data/xstest_prompts.csv``      (Roettger et al., 2024; CC-BY-4.0).
  - Do-Not-Answer: ``data/do_not_answer.jsonl`` (Wang et al., 2024; Apache-2.0).
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict

from common import DATA

SEED = 20260709
random.seed(SEED)

# Use the ENTIRE XSTest benchmark (all 25 prompts per type, 450 total).
XSTEST_PER_TYPE = 25

# Do-Not-Answer strata mapped to the paper's four harm categories, scaled to 150
# harmful prompts. Chosen to boost Sexual / Self-harm / Hate coverage that XSTest
# is thin on, plus clearly-harmful Violence-adjacent requests.
DNA_STRATA = {
    "Toxic language (hate speech)": 25,
    "Social stereotypes and unfair discrimination": 20,
    "Adult Content": 25,
    "Mental Health or Overreliance Crisis": 25,
    "Assisting illegal activities": 30,
    "Nudging or advising users to perform unethical or unsafe actions": 25,
}


def build_xstest() -> list[dict]:
    rows = list(csv.DictReader((DATA / "xstest_prompts.csv").open(encoding="utf-8")))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    out = []
    for t in sorted(by_type):
        pool = by_type[t]
        pick = random.sample(pool, min(XSTEST_PER_TYPE, len(pool)))
        for r in pick:
            out.append({
                "uid": f"xstest-{r['id']}",
                "source": "XSTest",
                "prompt": r["prompt"],
                "public_label": r["label"],           # safe / unsafe
                "public_category": r["note"],           # focus harm area
                "stratum": r["type"],
            })
    return out


def build_dna() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "do_not_answer.jsonl").open(encoding="utf-8")]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["types_of_harm"]].append(r)
    out = []
    for stratum, n in DNA_STRATA.items():
        pool = by_type.get(stratum, [])
        pick = random.sample(pool, min(n, len(pool)))
        for r in pick:
            out.append({
                "uid": f"dna-{r['id']}",
                "source": "Do-Not-Answer",
                "prompt": r["question"],
                "public_label": "unsafe",              # DNA items are all sensitive
                "public_category": r["types_of_harm"],
                "stratum": r["types_of_harm"],
            })
    return out


def main() -> None:
    prompts = build_xstest() + build_dna()
    # Stable ordering by uid so idx is deterministic; resume/join keys are uid.
    prompts.sort(key=lambda p: p["uid"])
    out_path = DATA / "prompts.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            p["idx"] = i
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n = len(prompts)
    n_safe = sum(1 for p in prompts if p["public_label"] == "safe")
    print(f"wrote {out_path} n={n} (safe={n_safe} unsafe={n - n_safe})")
    by_src: dict[str, int] = defaultdict(int)
    for p in prompts:
        by_src[p["source"]] += 1
    print("by source:", dict(by_src))


if __name__ == "__main__":
    main()
