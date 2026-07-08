# ============================================================
#  EMBER - Download Fine-tuning Data
#  Run once before fine-tuning:
#    python scripts/download_finetune_data.py
#
#  Downloads and merges:
#    - Alpaca (30k general instruction pairs)
#    - CodeAlpaca (15k coding pairs)
#    - Math QA (10k math reasoning)
# ============================================================

import sys
import json
from pathlib import Path
sys.path.insert(0, ".")


OUTPUT_DIR = Path("datasets/finetune")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_alpaca(max_samples: int = 30_000):
    """General instruction following data."""
    print("Downloading Alpaca dataset...")
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=False)
    data = []

    for item in ds:
        instruction = item.get("instruction", "").strip()
        output      = item.get("output", "").strip()
        input_text  = item.get("input", "").strip()

        if instruction and output and len(output) > 10:
            data.append({
                "instruction": instruction,
                "input":       input_text,
                "output":      output,
            })

        if len(data) >= max_samples:
            break

    out_path = OUTPUT_DIR / "alpaca.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   Saved {len(data):,} examples → {out_path}")
    return len(data)


def download_code_alpaca(max_samples: int = 15_000):
    """Coding instruction data."""
    print("Downloading CodeAlpaca dataset...")
    from datasets import load_dataset

    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", trust_remote_code=False)
    data = []

    for item in ds:
        instruction = item.get("instruction", "").strip()
        output      = item.get("output", "").strip()
        input_text  = item.get("input", "").strip()

        if instruction and output and len(output) > 10:
            data.append({
                "instruction": instruction,
                "input":       input_text,
                "output":      output,
            })

        if len(data) >= max_samples:
            break

    out_path = OUTPUT_DIR / "code_alpaca.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   Saved {len(data):,} examples → {out_path}")
    return len(data)


def download_math(max_samples: int = 10_000):
    """
    Math reasoning data.
    Tries multiple dataset sources in order until one works.
    """
    print("Downloading math dataset...")
    from datasets import load_dataset

    # List of datasets to try in order
    candidates = [
        ("qwedsacf/grade-school-math", None,        "train", "question", "answer"),
        ("gsm8k",                      "main",       "train", "question", "answer"),
        ("math_qa",                    None,         "train", "Problem",  "Rationale"),
        ("aqua_rat",                   "raw",        "train", "question", "rationale"),
    ]

    for dataset_name, config, split, q_field, a_field in candidates:
        try:
            print(f"   Trying {dataset_name}...")
            if config:
                ds = load_dataset(dataset_name, config, split=split, trust_remote_code=False)
            else:
                ds = load_dataset(dataset_name, split=split, trust_remote_code=False)

            data = []
            for item in ds:
                question = item.get(q_field, "").strip()
                answer   = item.get(a_field, "").strip()

                if question and answer and len(answer) > 5:
                    data.append({
                        "instruction": question,
                        "input":       "",
                        "output":      answer,
                    })

                if len(data) >= max_samples:
                    break

            if data:
                out_path = OUTPUT_DIR / "math.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                print(f"   Saved {len(data):,} examples → {out_path}")
                return len(data)

        except Exception as e:
            print(f"   Failed: {e}")
            continue

    # If all datasets fail, generate basic math examples
    print("   All math datasets failed — generating basic arithmetic examples...")
    import random
    data = []
    ops = [
        ("+",  lambda a, b: a + b),
        ("-",  lambda a, b: a - b),
        ("*",  lambda a, b: a * b),
    ]
    for _ in range(max_samples):
        a, b     = random.randint(1, 1000), random.randint(1, 1000)
        op, func = random.choice(ops)
        result   = func(a, b)
        data.append({
            "instruction": f"What is {a} {op} {b}?",
            "input":       "",
            "output":      f"{a} {op} {b} = {result}",
        })

    out_path = OUTPUT_DIR / "math.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   Saved {len(data):,} basic math examples → {out_path}")
    return len(data)


def main():
    print("=" * 52)
    print("  EMBER — Downloading Fine-tune Data")
    print("=" * 52)
    print()

    # Skip already downloaded files
    total = 0

    alpaca_path = OUTPUT_DIR / "alpaca.jsonl"
    if alpaca_path.exists():
        count = sum(1 for _ in open(alpaca_path))
        print(f"Alpaca already downloaded ({count:,} examples) — skipping")
        total += count
    else:
        total += download_alpaca(max_samples=30_000)

    print()

    code_path = OUTPUT_DIR / "code_alpaca.jsonl"
    if code_path.exists():
        count = sum(1 for _ in open(code_path))
        print(f"CodeAlpaca already downloaded ({count:,} examples) — skipping")
        total += count
    else:
        total += download_code_alpaca(max_samples=15_000)

    print()

    math_path = OUTPUT_DIR / "math.jsonl"
    if math_path.exists():
        count = sum(1 for _ in open(math_path))
        print(f"Math already downloaded ({count:,} examples) — skipping")
        total += count
    else:
        total += download_math(max_samples=10_000)

    print()
    print("=" * 52)
    print(f"  Total examples : {total:,}")
    print(f"  Saved to       : {OUTPUT_DIR}/")
    print("=" * 52)
    print()
    print("  Now run: python scripts/finetune.py")


if __name__ == "__main__":
    main()
