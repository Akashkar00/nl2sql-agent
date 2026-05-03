"""
Convert Spider + BIRD datasets to instruction-format JSONL for training.

Spider format: {"db_id": "...", "question": "...", "query": "..."}
BIRD format:   {"db_id": "...", "question": "...", "SQL": "...", "evidence": "..."}

Output line format:
    {"text": "<full prompt + completion>", "db_id": "...", "split": "spider"|"bird"}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.schema_formatter import build_prompt, format_schema


def load_spider(spider_root: Path, split: str = "train") -> list[dict]:
    """Spider train_spider.json + train_others.json combined for full train set."""
    files = []
    if split == "train":
        files = ["train_spider.json", "train_others.json"]
    elif split == "dev":
        files = ["dev.json"]
    else:
        raise ValueError(f"unknown split: {split}")

    examples = []
    for fname in files:
        path = spider_root / fname
        if not path.exists():
            print(f"warn: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for ex in data:
            examples.append({
                "db_id": ex["db_id"],
                "question": ex["question"],
                "sql": ex["query"],
                "evidence": "",
                "source": "spider",
            })
    return examples


def load_bird(bird_root: Path, split: str = "train") -> list[dict]:
    """BIRD has train.json and dev.json with 'SQL' (capital) and 'evidence' fields."""
    fname = "train.json" if split == "train" else "dev.json"
    path = bird_root / fname
    if not path.exists():
        print(f"warn: {path} not found, skipping")
        return []
    with open(path) as f:
        data = json.load(f)
    return [
        {
            "db_id": ex["db_id"],
            "question": ex["question"],
            "sql": ex["SQL"],
            "evidence": ex.get("evidence", ""),
            "source": "bird",
        }
        for ex in data
    ]


def find_db_path(db_id: str, search_roots: list[Path]) -> Path | None:
    """Spider: database/<db_id>/<db_id>.sqlite. BIRD: train_databases/<db_id>/<db_id>.sqlite."""
    for root in search_roots:
        for candidate in [
            root / "database" / db_id / f"{db_id}.sqlite",
            root / "train_databases" / db_id / f"{db_id}.sqlite",
            root / "dev_databases" / db_id / f"{db_id}.sqlite",
            root / db_id / f"{db_id}.sqlite",
        ]:
            if candidate.exists():
                return candidate
    return None


def format_example(ex: dict, db_path: Path, include_samples: bool = True) -> dict | None:
    """Build full training text: prompt + SQL + EOS marker."""
    try:
        schema = format_schema(str(db_path), include_samples=include_samples)
    except Exception as e:
        print(f"skip {ex['db_id']}: {e}")
        return None

    prompt = build_prompt(schema, ex["question"], ex.get("evidence", ""))
    completion = ex["sql"].strip()
    if not completion.endswith(";"):
        completion += ";"

    # End with explicit token; Unsloth handles EOS via tokenizer
    text = prompt + completion

    return {
        "text": text,
        "db_id": ex["db_id"],
        "source": ex["source"],
        "question": ex["question"],   # kept for eval, not used in training
        "sql": ex["sql"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_root", type=Path, default=Path("data/spider"))
    ap.add_argument("--bird_root", type=Path, default=Path("data/bird"))
    ap.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--include_samples", action="store_true", default=True)
    ap.add_argument("--holdout_size", type=int, default=500,
                    help="size of fast eval holdout pulled from train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load both train splits
    examples: list[dict] = []
    examples.extend(load_spider(args.spider_root, "train"))
    examples.extend(load_bird(args.bird_root, "train"))
    print(f"loaded {len(examples)} raw examples")

    # Search roots for DB files
    search_roots = [args.spider_root, args.bird_root]

    formatted: list[dict] = []
    for ex in examples:
        db_path = find_db_path(ex["db_id"], search_roots)
        if db_path is None:
            print(f"db not found for {ex['db_id']}, skip")
            continue
        out = format_example(ex, db_path, args.include_samples)
        if out is not None:
            formatted.append(out)

    print(f"formatted {len(formatted)} examples")

    # Shuffle deterministically, split holdout
    import random
    random.seed(args.seed)
    random.shuffle(formatted)
    holdout = formatted[: args.holdout_size]
    train = formatted[args.holdout_size :]

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval_holdout.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(eval_path, "w") as f:
        for ex in holdout:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also dump dev sets for full benchmarking
    for name, loader, root in [
        ("spider_dev", load_spider, args.spider_root),
        ("bird_dev", load_bird, args.bird_root),
    ]:
        dev_examples = loader(root, "dev")
        dev_formatted = []
        for ex in dev_examples:
            db_path = find_db_path(ex["db_id"], search_roots)
            if db_path is None:
                continue
            out = format_example(ex, db_path, args.include_samples)
            if out is not None:
                dev_formatted.append(out)
        out_path = args.output_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for ex in dev_formatted:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {out_path} ({len(dev_formatted)} examples)")

    print(f"\ntrain: {train_path} ({len(train)})")
    print(f"holdout: {eval_path} ({len(holdout)})")


if __name__ == "__main__":
    main()
