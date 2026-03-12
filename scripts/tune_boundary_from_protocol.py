#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_records", type=str, default="outputs/basd_protocol_run100/records.jsonl")
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.protocol_records).read_text(encoding="utf-8").splitlines() if x.strip()]
    correct_max, wrong_jump = [], []
    for r in rows:
        gaps = np.asarray(r.get("token_gap", []), dtype=float)
        if gaps.size < 2:
            continue
        if r.get("is_correct", False):
            correct_max.append(float(gaps.max()))
        else:
            wrong_jump.append(float(np.diff(gaps).max()))

    result = {
        "abs_threshold": float(np.percentile(correct_max, 95)) if correct_max else 0.8,
        "jump_threshold": float(np.percentile(wrong_jump, 50)) if wrong_jump else 0.5,
        "persist_window": 8,
        "smooth_alpha": 0.3,
        "num_rows": len(rows),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
