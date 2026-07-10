"""Label prompt and response severity with the shared few-shot rubric grader.

The grader is the paper's public rubric grader (system prompt loaded from
docs/prompts/system_prompt_compact.md). Each text is graded independently on the
0-3 severity scale across the four categories (H, S, V, SH). Prompts are graded
once; responses are graded per model. This yields *paired* labels needed for the
transition analysis.
"""
from __future__ import annotations

import argparse
import concurrent.futures

from common import (CATEGORIES, DATA, GEN_MODELS, GRADER, append_jsonl,
                    build_grader_user_prompt, chat, load_grader_system_prompt,
                    normalize_prediction, parse_json, read_jsonl)

SYSTEM_PROMPT = load_grader_system_prompt()


def grade_text(text: str, target: str) -> tuple[dict | None, str | None]:
    if not (text or "").strip():
        return {c: 0 for c in CATEGORIES}, None  # empty text -> safe by definition
    user = build_grader_user_prompt(text, target)
    content, err = chat(GRADER, [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": user}])
    if err:
        return None, err
    try:
        return normalize_prediction(parse_json(content)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"parse_error: {repr(exc)[:150]}"


def grade_prompts(workers: int) -> None:
    prompts = read_jsonl(DATA / "prompts.jsonl")
    out_path = DATA / "graded" / "prompts.jsonl"
    done = {r["uid"] for r in read_jsonl(out_path) if r.get("labels")}
    todo = [p for p in prompts if p["uid"] not in done]
    print(f"[prompts] todo={len(todo)} done={len(done)}", flush=True)
    _grade_records(todo, "prompt", "prompt", out_path, workers)


def grade_responses(model_key: str, workers: int) -> None:
    cfg = GEN_MODELS[model_key]
    recs = [r for r in read_jsonl(DATA / "responses" / f"{cfg.name}.jsonl")
            if not r.get("error") and (r.get("response") or "").strip()]
    out_path = DATA / "graded" / f"response_{cfg.name}.jsonl"
    done = {r["uid"] for r in read_jsonl(out_path) if r.get("labels")}
    todo = [r for r in recs if r["uid"] not in done]
    print(f"[resp:{cfg.name}] todo={len(todo)} done={len(done)}", flush=True)
    _grade_records(todo, "response", "response", out_path, workers)


def _grade_records(todo, target, text_key, out_path, workers):
    if not todo:
        return

    def work(rec):
        labels, err = grade_text(rec[text_key], target)
        return {"idx": rec["idx"], "uid": rec["uid"],
                "model": rec.get("model"), "labels": labels, "error": err}

    n_err = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, rec) for rec in todo]
        buf = []
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res["error"]:
                n_err += 1
            buf.append(res)
            if len(buf) >= 10:
                append_jsonl(out_path, buf)
                buf = []
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} err={n_err}", flush=True)
        if buf:
            append_jsonl(out_path, buf)
    print(f"  done err={n_err} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["prompts", "responses", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=list(GEN_MODELS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.what in ("prompts", "all"):
        grade_prompts(args.workers)
    if args.what in ("responses", "all"):
        for mk in args.models:
            grade_responses(mk, args.workers)


if __name__ == "__main__":
    main()
