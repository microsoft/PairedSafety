"""Generate responses for every prompt from all six models.

No system prompt is used (matching the paper's generation setup). Decoding:
- standard models: temperature=0.7, top_p=1.0, max_tokens=1024
- reasoning models (gpt-5, grok-4, gpt-oss-120b): API defaults, max_completion/
  max_tokens budget = 4000 (reasoning tokens are consumed internally)
Outputs one JSONL per model under data/responses/<model>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import concurrent.futures

from common import DATA, GEN_MODELS, chat, read_jsonl, append_jsonl


def gen_one(cfg, rec):
    content, err = chat(cfg, [{"role": "user", "content": rec["prompt"]}])
    return {
        "idx": rec["idx"],
        "uid": rec["uid"],
        "source": rec["source"],
        "public_label": rec["public_label"],
        "public_category": rec["public_category"],
        "stratum": rec["stratum"],
        "prompt": rec["prompt"],
        "model": cfg.name,
        "family": cfg.family,
        "response": content,
        "error": err,
    }


def run_model(model_key: str, workers: int) -> None:
    cfg = GEN_MODELS[model_key]
    prompts = read_jsonl(DATA / "prompts.jsonl")
    out_path = DATA / "responses" / f"{cfg.name}.jsonl"
    done = {r["uid"] for r in read_jsonl(out_path) if not r.get("error") and (r.get("response") or "").strip()}
    todo = [p for p in prompts if p["uid"] not in done]
    print(f"[{cfg.name}] todo={len(todo)} done={len(done)} -> {out_path}", flush=True)
    if not todo:
        return
    n_err = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(gen_one, cfg, rec) for rec in todo]
        buf = []
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res["error"] or not (res["response"] or "").strip():
                n_err += 1
            buf.append(res)
            if len(buf) >= 10:
                append_jsonl(out_path, buf)
                buf = []
            if i % 25 == 0:
                print(f"[{cfg.name}] {i}/{len(todo)} err={n_err}", flush=True)
        if buf:
            append_jsonl(out_path, buf)
    print(f"[{cfg.name}] done err={n_err}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(GEN_MODELS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for mk in args.models:
        run_model(mk, args.workers)


if __name__ == "__main__":
    main()
