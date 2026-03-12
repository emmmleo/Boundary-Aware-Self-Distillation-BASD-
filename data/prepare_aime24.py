#!/usr/bin/env python3
"""下载并导出 AIME 2024 数据集到 jsonl。

默认使用 Hugging Face 数据集（可通过参数覆盖数据集名 / 配置 / split / 字段名）。
输出格式与仓库现有数据保持一致：
{"id": ..., "question": ..., "reference_solution": ...}
"""

import argparse
import json
import os
from typing import Any, Optional

from datasets import load_dataset


def _pick_first(row: dict, candidates: list[str]) -> Optional[Any]:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceH4/aime_2024", help="Hugging Face 数据集名")
    parser.add_argument("--config", default=None, help="数据集配置名（可选）")
    parser.add_argument("--split", default="train", help="数据切分，默认 train")
    parser.add_argument("--out", default="data/aime24_full.jsonl", help="输出 jsonl 路径")
    parser.add_argument(
        "--question-field",
        default=None,
        help="题目字段名（默认会自动在 problem/question 等字段中探测）",
    )
    parser.add_argument(
        "--answer-field",
        default=None,
        help="答案字段名（默认会自动在 answer/final_answer 等字段中探测）",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    question_candidates = [
        args.question_field,
        "problem",
        "question",
        "prompt",
        "input",
    ]
    answer_candidates = [
        args.answer_field,
        "answer",
        "final_answer",
        "solution",
        "target",
    ]
    question_candidates = [x for x in question_candidates if x]
    answer_candidates = [x for x in answer_candidates if x]

    with open(args.out, "w", encoding="utf-8") as f:
        kept = 0
        for i, row in enumerate(ds):
            question = _pick_first(row, question_candidates)
            answer = _pick_first(row, answer_candidates)
            if question is None or answer is None:
                continue

            sample_id = row.get("id") or row.get("uid") or f"aime24-{i}"
            f.write(
                json.dumps(
                    {
                        "id": str(sample_id),
                        "question": str(question),
                        "reference_solution": str(answer),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    print(f"saved {kept} samples to {args.out} (from total {len(ds)} rows)")


if __name__ == "__main__":
    main()
