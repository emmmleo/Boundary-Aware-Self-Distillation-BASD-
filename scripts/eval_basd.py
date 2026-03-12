#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basd.eval.evaluator import evaluate_rollout_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=str, required=True)
    args = parser.parse_args()
    print(evaluate_rollout_records(args.records))


if __name__ == "__main__":
    main()
