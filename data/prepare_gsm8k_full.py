#!/usr/bin/env python3
"""下载并导出 GSM8K 全量 test 集到 jsonl（用于正式预实验）"""

import argparse
import json
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gsm8k_test_full.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds = load_dataset("gsm8k", "main", split="test")

    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            f.write(
                json.dumps(
                    {
                        "id": f"gsm8k-test-{i}",
                        "question": row["question"],
                        "reference_solution": row["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"saved {len(ds)} samples to {args.out}")


if __name__ == "__main__":
    main()
