#!/usr/bin/env python3
"""下载 OpenThoughts 数据集并自动采样 30k 条数学推理样本。"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Iterable, Optional

from datasets import load_dataset


MATH_KEYWORDS = {
    "math",
    "mathematics",
    "algebra",
    "geometry",
    "number theory",
    "combinatorics",
    "calculus",
    "arithmetic",
    "gsm8k",
    "aime",
    "amc",
    "olympiad",
}


def _pick_first(row: dict[str, Any], candidates: list[str]) -> Optional[Any]:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_textify(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {_textify(v)}" for k, v in value.items())
    return str(value)


def _contains_math_keyword(text: str) -> bool:
    text = text.lower()
    for kw in MATH_KEYWORDS:
        if kw in text:
            return True
    return False


def _is_math_reasoning_row(row: dict[str, Any]) -> bool:
    # 优先使用明显的元数据字段。
    meta_fields = [
        "domain",
        "category",
        "task",
        "source",
        "dataset",
        "subset",
        "tags",
        "subject",
    ]
    for field in meta_fields:
        if field in row and _contains_math_keyword(_textify(row.get(field))):
            return True

    # 回退：在全部字段做轻量关键词检测。
    merged = " ".join(f"{k} {_textify(v)}" for k, v in row.items())
    return _contains_math_keyword(merged)


def _sample_without_replacement(rows: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= target_size:
        return rows
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), target_size)
    return [rows[i] for i in indices]


def _iter_records(ds: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    question_candidates = [
        "question",
        "problem",
        "prompt",
        "input",
        "instruction",
    ]
    reasoning_candidates = [
        "reference_solution",
        "solution",
        "rationale",
        "response",
        "output",
        "answer",
    ]
    answer_candidates = [
        "gold_answer",
        "final_answer",
        "answer",
        "target",
    ]

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if not _is_math_reasoning_row(row):
            continue

        question = _pick_first(row, question_candidates)
        reference_solution = _pick_first(row, reasoning_candidates)
        if question is None or reference_solution is None:
            continue

        sample_id = row.get("sample_id") or row.get("id") or row.get("uid") or f"openthoughts-math-{i}"
        record = {
            "id": str(sample_id),
            "question": str(question),
            "reference_solution": str(reference_solution),
        }

        gold_answer = _pick_first(row, answer_candidates)
        if gold_answer is not None:
            record["gold_answer"] = str(gold_answer)

        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="open-thoughts/OpenThoughts-114k", help="Hugging Face 数据集名")
    parser.add_argument("--config", default=None, help="数据集配置（可选）")
    parser.add_argument("--split", default="train", help="数据切分")
    parser.add_argument("--sample-size", type=int, default=30000, help="采样条数（默认 30000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", default="data/openthoughts_math_30k.jsonl", help="输出 jsonl 路径")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    all_math_rows = _iter_records(ds)
    sampled_rows = _sample_without_replacement(all_math_rows, args.sample_size, args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"saved {len(sampled_rows)} samples to {args.out}; "
        f"math_candidates={len(all_math_rows)}, requested={args.sample_size}"
    )
    if len(all_math_rows) < args.sample_size:
        print("warning: math 子集数量小于目标采样数，已输出全部可用样本。")


if __name__ == "__main__":
    main()
