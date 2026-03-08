#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

STEP_RE = re.compile(r"Step\s*(\d+)\s*:\s*(.*?)(?=(?:\nStep\s*\d+\s*:)|(?:\nFinal Answer\s*:)|\Z)", re.S | re.I)
TOKEN_RE = re.compile(r"\d+\.\d+|\d+|[A-Za-z_]+|[^\sA-Za-z_\d]")


@dataclass
class Sample:
    sample_id: str
    question: str
    reference_solution: str
    student_solution: str
    is_correct: Optional[bool] = None
    manual_first_error_step: Optional[int] = None


def parse_steps(solution: str) -> List[Tuple[int, str]]:
    steps = []
    for m in STEP_RE.finditer(solution):
        steps.append((int(m.group(1)), m.group(2).strip()))
    return steps


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    vocab = set(p) | set(q)
    kl = 0.0
    for tok in vocab:
        pv = max(p.get(tok, 0.0), eps)
        qv = max(q.get(tok, 0.0), eps)
        kl += pv * math.log(pv / qv)
    return kl


def peaked_distribution(target_token: str, vocab: List[str], peak: float = 0.7) -> Dict[str, float]:
    if not vocab:
        return {}
    n = len(vocab)
    base = (1.0 - peak) / max(n - 1, 1)
    dist = {t: base for t in vocab}
    dist[target_token] = peak if n > 1 else 1.0
    return dist


def first_mismatch_step(reference_steps: List[Tuple[int, str]], student_steps: List[Tuple[int, str]]) -> Optional[int]:
    for (rid, rtxt), (sid, stxt) in zip(reference_steps, student_steps):
        if rid != sid:
            return sid
        if tokenize(rtxt) != tokenize(stxt):
            return sid
    if len(student_steps) > len(reference_steps):
        return student_steps[len(reference_steps)][0]
    return None


def compute_step_kls(sample: Sample) -> Dict:
    reference_steps = parse_steps(sample.reference_solution)
    student_steps = parse_steps(sample.student_solution)
    if not student_steps:
        raise ValueError(f"Sample {sample.sample_id} has unparsable student steps.")

    inferred_first_error = first_mismatch_step(reference_steps, student_steps)
    step_kls = []
    for idx, (sid, stext) in enumerate(student_steps):
        rtokens = tokenize(reference_steps[idx][1]) if idx < len(reference_steps) else []
        stokens = tokenize(stext)
        vocab = sorted(set(rtokens) | set(stokens) | {"<unk>"})
        token_kls = []
        for t_pos, stok in enumerate(stokens):
            rtok = rtokens[t_pos] if t_pos < len(rtokens) else "<unk>"
            p_teacher = peaked_distribution(rtok, vocab)
            p_student = peaked_distribution(stok, vocab)
            token_kls.append(kl_divergence(p_teacher, p_student))
        step_kls.append({
            "step": sid,
            "kl": sum(token_kls) / len(token_kls) if token_kls else 0.0,
            "token_count": len(stokens),
        })

    jumps = []
    for i in range(1, len(step_kls)):
        jumps.append((step_kls[i]["step"], step_kls[i]["kl"] - step_kls[i - 1]["kl"]))
    boundary_step = max(jumps, key=lambda x: x[1])[0] if jumps else step_kls[0]["step"]

    pre = [s["kl"] for s in step_kls if s["step"] < boundary_step]
    post = [s["kl"] for s in step_kls if s["step"] >= boundary_step]

    max_abs_step = max(step_kls, key=lambda x: x["kl"])["step"]

    return {
        "sample_id": sample.sample_id,
        "is_correct": sample.is_correct,
        "manual_first_error_step": sample.manual_first_error_step,
        "inferred_first_error_step": inferred_first_error,
        "boundary_step_jump": boundary_step,
        "boundary_step_max_abs_kl": max_abs_step,
        "pre_boundary_kl_mean": sum(pre) / len(pre) if pre else 0.0,
        "post_boundary_kl_mean": sum(post) / len(post) if post else 0.0,
        "step_kls": step_kls,
    }


def load_samples(path: str) -> List[Sample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            samples.append(
                Sample(
                    sample_id=obj["id"],
                    question=obj["question"],
                    reference_solution=obj["reference_solution"],
                    student_solution=obj["student_solution"],
                    is_correct=obj.get("is_correct"),
                    manual_first_error_step=obj.get("manual_first_error_step"),
                )
            )
    return samples


def save_annotation_template(path: str, samples: List[Sample], records: List[Dict]) -> None:
    by_id = {r["sample_id"]: r for r in records}
    fields = [
        "id",
        "question",
        "student_solution",
        "reference_solution",
        "auto_boundary_step",
        "manual_first_error_step",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "id": s.sample_id,
                    "question": s.question,
                    "student_solution": s.student_solution,
                    "reference_solution": s.reference_solution,
                    "auto_boundary_step": by_id[s.sample_id]["boundary_step_jump"],
                    "manual_first_error_step": s.manual_first_error_step or "",
                }
            )


def compute_eval(records: List[Dict], label_field: str) -> Dict:
    usable = [r for r in records if r.get(label_field) is not None]
    if not usable:
        return {}

    exact = sum(r["boundary_step_jump"] == r[label_field] for r in usable) / len(usable)
    hit1 = sum(abs(r["boundary_step_jump"] - r[label_field]) <= 1 for r in usable) / len(usable)
    mae = sum(abs(r["boundary_step_jump"] - r[label_field]) for r in usable) / len(usable)

    jump_vs_abs = sum(
        abs(r["boundary_step_jump"] - r[label_field]) <= abs(r["boundary_step_max_abs_kl"] - r[label_field])
        for r in usable
    ) / len(usable)

    return {
        "n": len(usable),
        "exact_match": exact,
        "hit_at_1": hit1,
        "mae": mae,
        "jump_better_or_equal_than_max_abs_ratio": jump_vs_abs,
    }


def maybe_plot(records: List[Dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not records:
        return

    # single case
    case = records[0]
    xs = [s["step"] for s in case["step_kls"]]
    ys = [s["kl"] for s in case["step_kls"]]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label="step KL")
    plt.axvline(case["boundary_step_jump"], color="r", linestyle="--", label="auto boundary")
    if case.get("manual_first_error_step"):
        plt.axvline(case["manual_first_error_step"], color="g", linestyle=":", label="manual first error")
    plt.xlabel("Step")
    plt.ylabel("KL(teacher || student)")
    plt.title(f"Single-case KL curve: {case['sample_id']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "single_case_kl.png"), dpi=140)
    plt.close()

    # average correct/incorrect
    def resample_to_bins(vals: List[float], bins: int = 10) -> List[float]:
        if not vals:
            return [0.0] * bins
        if len(vals) == 1:
            return vals * bins
        out = []
        for i in range(bins):
            pos = i * (len(vals) - 1) / (bins - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            w = pos - lo
            out.append(vals[lo] * (1 - w) + vals[hi] * w)
        return out

    correct = [r for r in records if r.get("is_correct") is True]
    incorrect = [r for r in records if r.get("is_correct") is False]

    def avg_curve(rs: List[Dict]) -> List[float]:
        if not rs:
            return [0.0] * 10
        curves = [resample_to_bins([s["kl"] for s in r["step_kls"]], 10) for r in rs]
        return [sum(x[i] for x in curves) / len(curves) for i in range(10)]

    cavg = avg_curve(correct)
    iavg = avg_curve(incorrect)
    bins = list(range(1, 11))
    plt.figure(figsize=(6, 4))
    plt.plot(bins, cavg, marker="o", label="correct")
    plt.plot(bins, iavg, marker="o", label="incorrect")
    plt.xlabel("Normalized step bin (1-10)")
    plt.ylabel("Average KL")
    plt.title("Average step-KL curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_kl_correct_vs_incorrect.png"), dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal runnable pilot for boundary-aware KL analysis.")
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--out_dir", default="outputs/pilot", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.input)
    records = [compute_step_kls(s) for s in samples]

    with open(os.path.join(args.out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    eval_manual = compute_eval(records, "manual_first_error_step")
    eval_inferred = compute_eval(records, "inferred_first_error_step")

    summary = {
        "n_samples": len(records),
        "manual_eval": eval_manual,
        "inferred_eval": eval_inferred,
        "mean_pre_boundary_kl": sum(r["pre_boundary_kl_mean"] for r in records) / len(records),
        "mean_post_boundary_kl": sum(r["post_boundary_kl_mean"] for r in records) / len(records),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_annotation_template(os.path.join(args.out_dir, "annotation_template.csv"), samples, records)
    maybe_plot(records, args.out_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
