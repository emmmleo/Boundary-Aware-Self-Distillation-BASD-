#!/usr/bin/env python
import argparse

from basd.eval.evaluator import evaluate_rollout_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=str, required=True)
    args = parser.parse_args()
    print(evaluate_rollout_records(args.records))


if __name__ == "__main__":
    main()
