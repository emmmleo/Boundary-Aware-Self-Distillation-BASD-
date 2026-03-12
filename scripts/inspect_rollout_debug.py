#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_jsonl", type=str, required=True)
    parser.add_argument("--sample_id", type=str, required=False)
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.debug_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.sample_id:
        rows = [r for r in rows if r.get("sample_id") == args.sample_id]
    if not rows:
        print("No matching rows")
        return
    row = rows[0]
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
