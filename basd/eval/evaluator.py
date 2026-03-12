import json
from pathlib import Path

from basd.eval.metrics import accuracy


def evaluate_rollout_records(records_path: str) -> dict:
    rows = [json.loads(x) for x in Path(records_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return {
        "n": len(rows),
        "acc": accuracy([bool(r.get("is_correct", False)) for r in rows]),
    }
