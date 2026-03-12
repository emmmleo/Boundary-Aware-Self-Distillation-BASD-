#!/usr/bin/env python3
"""Convert OPSD train data into the BASD JSONL format."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable, Iterator, Optional


def _pick_first(row: dict[str, Any], candidates: list[str]) -> Optional[Any]:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _iter_input_rows(path: str) -> Iterator[dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at line {line_no}, got {type(row).__name__}.")
                yield row
        return

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        for idx, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at index {idx}, got {type(row).__name__}.")
            yield row
        return

    if isinstance(payload, dict):
        # Common wrappers: {"data": [...]}, {"train": [...]}, {"rows": [...]}
        for key in ("data", "train", "rows", "samples", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                for idx, row in enumerate(rows):
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected JSON object at {key}[{idx}], got {type(row).__name__}.")
                    yield row
                return

    raise ValueError("Unsupported input format. Expected .jsonl, a JSON list, or a wrapper object with a list field.")


def _convert_rows(rows: Iterable[dict[str, Any]], id_prefix: str) -> list[dict[str, str]]:
    question_candidates = [
        "problem",
        "question",
        "Question",
        "prompt",
        "input",
    ]
    solution_candidates = [
        "solution",
        "COT_Reason",
        "reference_solution",
        "output",
        "response",
        "Answer_with_COT",
    ]
    answer_candidates = [
        "Answer",
        "answer",
        "final_answer",
        "gold_answer",
        "target",
    ]

    converted: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        question = _pick_first(row, question_candidates)
        reference_solution = _pick_first(row, solution_candidates)
        if question is None or reference_solution is None:
            continue

        sample_id = (
            row.get("sample_id")
            or row.get("id")
            or row.get("uid")
            or row.get("idx")
            or f"{id_prefix}-{idx}"
        )
        record = {
            "id": str(sample_id),
            "question": _stringify(question),
            "reference_solution": _stringify(reference_solution),
        }

        gold_answer = _pick_first(row, answer_candidates)
        if gold_answer is not None and _stringify(gold_answer).strip():
            record["gold_answer"] = _stringify(gold_answer).strip()

        converted.append(record)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the OPSD train json/jsonl file.")
    parser.add_argument("--out", default="data/opsdtrain_basd.jsonl", help="Output BASD-format jsonl path.")
    parser.add_argument("--id-prefix", default="opsdtrain", help="Fallback prefix for generated sample ids.")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    converted = _convert_rows(_iter_input_rows(args.input), args.id_prefix)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved {len(converted)} samples to {args.out}")


if __name__ == "__main__":
    main()
