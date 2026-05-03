"""
Benchmark runner. Loads dev set (eval JSONL produced by prepare_dataset.py),
runs predictions, computes execution accuracy.

Supports two modes:
  --no-agent  : single-shot generation (baseline number)
  default     : with self-correction loop (your differentiator number)

Outputs:
  results/<run_name>.jsonl   per-example predictions and traces
  results/<run_name>.summary.json   accuracy + breakdowns
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.agent.executor import execute_sql
from src.agent.self_correction import _clean_generated_sql, run_agent
from src.data.prepare_dataset import find_db_path
from src.data.schema_formatter import build_prompt, format_schema
from src.eval.execution_accuracy import categorize_failure, execution_accuracy


def load_eval_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def make_generate_fn(model_name: str, adapter_path: str | None = None):
    """
    Build a generate_fn closure.

    Loads model once, returns a callable. Uses 4-bit quant via Unsloth-compatible loader.
    """
    from src.model.inference import InferenceEngine
    engine = InferenceEngine(model_name=model_name, adapter_path=adapter_path)

    def generate(prompt: str) -> str:
        return engine.generate(prompt, max_new_tokens=256)

    return generate


def run_benchmark(
    eval_path: Path,
    spider_root: Path,
    bird_root: Path,
    model_name: str,
    adapter_path: str | None,
    use_agent: bool,
    output_dir: Path,
    run_name: str,
    limit: int | None = None,
):
    examples = load_eval_jsonl(eval_path)
    if limit:
        examples = examples[:limit]

    generate_fn = make_generate_fn(model_name, adapter_path)
    search_roots = [spider_root, bird_root]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{run_name}.jsonl"
    summary_path = output_dir / f"{run_name}.summary.json"

    correct = 0
    total = 0
    by_source = Counter()
    correct_by_source = Counter()
    failure_buckets = Counter()
    retries = []

    t0 = time.time()
    with open(results_path, "w") as out_f:
        for ex in tqdm(examples, desc=run_name):
            db_path = find_db_path(ex["db_id"], search_roots)
            if db_path is None:
                continue

            schema = format_schema(str(db_path))
            question = ex["question"]
            gold_sql = ex["sql"]
            evidence = ex.get("evidence", "")

            if use_agent:
                trace = run_agent(
                    question=question,
                    schema=schema,
                    db_path=str(db_path),
                    generate_fn=generate_fn,
                    evidence=evidence,
                    max_retries=2,
                )
                pred_sql = trace.final_sql
                retries_used = trace.retries_used
                attempts = trace.attempts
            else:
                prompt = build_prompt(schema, question, evidence)
                raw = generate_fn(prompt)
                pred_sql = _clean_generated_sql(raw)
                retries_used = 0
                attempts = [{"attempt": 0, "sql": pred_sql}]

            is_correct = execution_accuracy(pred_sql, gold_sql, str(db_path))
            bucket = None if is_correct else categorize_failure(pred_sql, gold_sql, str(db_path))

            total += 1
            by_source[ex["source"]] += 1
            if is_correct:
                correct += 1
                correct_by_source[ex["source"]] += 1
            else:
                failure_buckets[bucket] += 1
            retries.append(retries_used)

            out_f.write(json.dumps({
                "db_id": ex["db_id"],
                "source": ex["source"],
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "correct": is_correct,
                "retries_used": retries_used,
                "failure_bucket": bucket,
                "attempts": attempts,
            }, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    summary = {
        "run_name": run_name,
        "model": model_name,
        "adapter": adapter_path,
        "use_agent": use_agent,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_source": {
            src: {
                "total": by_source[src],
                "correct": correct_by_source[src],
                "accuracy": correct_by_source[src] / by_source[src] if by_source[src] else 0.0,
            } for src in by_source
        },
        "failure_buckets": dict(failure_buckets),
        "retry_distribution": dict(Counter(retries)),
        "avg_retries": sum(retries) / len(retries) if retries else 0.0,
        "elapsed_sec": elapsed,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {run_name} ===")
    print(f"accuracy: {summary['accuracy']:.4f} ({correct}/{total})")
    for src, stats in summary["by_source"].items():
        print(f"  {src}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    print(f"avg retries: {summary['avg_retries']:.2f}")
    print(f"elapsed: {elapsed:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", type=Path, required=True)
    ap.add_argument("--spider_root", type=Path, default=Path("data/spider"))
    ap.add_argument("--bird_root", type=Path, default=Path("data/bird"))
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--no_agent", action="store_true",
                    help="run single-shot baseline (no self-correction)")
    ap.add_argument("--output_dir", type=Path, default=Path("results"))
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_benchmark(
        eval_path=args.eval_path,
        spider_root=args.spider_root,
        bird_root=args.bird_root,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_agent=not args.no_agent,
        output_dir=args.output_dir,
        run_name=args.run_name,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
