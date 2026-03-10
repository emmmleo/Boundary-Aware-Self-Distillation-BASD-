#!/usr/bin/env python3
"""Offline analysis and plotting for standalone BASD protocol outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

from basd_protocol.common import ensure_dir, read_annotation_csv, read_jsonl, write_csv


def merge_annotations(records: List[Dict], annotation_path: Optional[str]) -> List[Dict]:
    if not annotation_path:
        return records
    by_id = read_annotation_csv(annotation_path)
    merged: List[Dict] = []
    for record in records:
        updated = dict(record)
        annotation = by_id.get(record["sample_id"], {})
        updated["first_error_step"] = parse_optional_int(annotation.get("first_error_step"))
        updated["error_type"] = annotation.get("error_type") or None
        updated["boundary_pattern"] = annotation.get("boundary_pattern") or None
        updated["annotation_comments"] = annotation.get("comments") or None
        merged.append(updated)
    return merged


def parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def compute_boundary_eval(records: Sequence[Dict], prediction_field: str) -> Dict:
    usable = [record for record in records if record.get("first_error_step") is not None and record.get(prediction_field) is not None]
    if not usable:
        return {}
    exact = sum(record[prediction_field] == record["first_error_step"] for record in usable) / len(usable)
    within_one = sum(abs(record[prediction_field] - record["first_error_step"]) <= 1 for record in usable) / len(usable)
    mae = sum(abs(record[prediction_field] - record["first_error_step"]) for record in usable) / len(usable)
    return {
        "n": len(usable),
        "exact_match_rate": exact,
        "near_match_rate": within_one,
        "mean_absolute_error": mae,
    }


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return mean(values) if values else None


def safe_median(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return median(values) if values else None


def max_metric(record: Dict, field: str) -> Optional[float]:
    values = [metric[field] for metric in record.get("step_metrics", []) if metric.get(field) is not None]
    return max(values) if values else None


def compare_correct_wrong(records: Sequence[Dict], field: str) -> Dict:
    correct = [max_metric(record, field) for record in records if record.get("is_correct") is True]
    wrong = [max_metric(record, field) for record in records if record.get("is_correct") is False]
    correct = [value for value in correct if value is not None]
    wrong = [value for value in wrong if value is not None]
    return {
        "field": field,
        "correct_mean": safe_mean(correct),
        "correct_median": safe_median(correct),
        "wrong_mean": safe_mean(wrong),
        "wrong_median": safe_median(wrong),
        "wrong_gt_correct_ratio": (
            sum(w > c for c in correct for w in wrong) / (len(correct) * len(wrong)) if correct and wrong else None
        ),
    }


def step_curve_by_group(records: Sequence[Dict], field: str, predicate) -> List[Dict]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for record in records:
        if not predicate(record):
            continue
        for metric in record.get("step_metrics", []):
            value = metric.get(field)
            if value is not None:
                grouped[metric["step_id"]].append(value)
    curve: List[Dict] = []
    for step_id in sorted(grouped):
        curve.append({"step_id": step_id, "mean_value": mean(grouped[step_id]), "count": len(grouped[step_id])})
    return curve


def compare_prediction(record: Dict, field: str, tolerance: int = 0) -> Optional[bool]:
    if record.get("first_error_step") is None or record.get(field) is None:
        return None
    return abs(record[field] - record["first_error_step"]) <= tolerance


def build_sample_summary(records: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for record in records:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "is_correct": record.get("is_correct"),
                "format_fail": record.get("format_fail"),
                "num_steps": record.get("num_steps"),
                "predicted_boundary_by_gap": record.get("predicted_boundary_by_gap"),
                "predicted_boundary_by_gap_jump": record.get("predicted_boundary_by_gap_jump"),
                "predicted_boundary_by_kl": record.get("predicted_boundary_by_kl"),
                "human_first_error_step": record.get("first_error_step"),
                "error_type": record.get("error_type"),
                "boundary_pattern": record.get("boundary_pattern"),
                "exact_match_gap": compare_prediction(record, "predicted_boundary_by_gap"),
                "within_one_gap": compare_prediction(record, "predicted_boundary_by_gap", tolerance=1),
                "exact_match_gap_jump": compare_prediction(record, "predicted_boundary_by_gap_jump"),
                "within_one_gap_jump": compare_prediction(record, "predicted_boundary_by_gap_jump", tolerance=1),
                "exact_match_kl": compare_prediction(record, "predicted_boundary_by_kl"),
                "within_one_kl": compare_prediction(record, "predicted_boundary_by_kl", tolerance=1),
                "max_gap": max_metric(record, "avg_gap"),
                "max_kl": max_metric(record, "avg_kl"),
            }
        )
    return rows


def write_markdown_report(path: str, summary: Dict) -> None:
    lines = [
        "# BASD Protocol Analysis",
        "",
        "## Format Stats",
        f"- n_records: {summary['n_records']}",
        f"- format_success_rate: {summary['format_success_rate']:.4f}",
        f"- avg_num_steps: {summary['avg_num_steps']}",
        f"- avg_num_tokens: {summary['avg_num_tokens']}",
        "",
        "## Boundary Metrics",
        f"- gap_argmax: {json.dumps(summary['boundary_eval_by_gap'], ensure_ascii=False)}",
        f"- gap_jump: {json.dumps(summary['boundary_eval_by_gap_jump'], ensure_ascii=False)}",
        f"- kl_argmax: {json.dumps(summary['boundary_eval_by_kl'], ensure_ascii=False)}",
        "",
        "## Correct vs Wrong",
        f"- max_gap: {json.dumps(summary['correct_wrong_gap'], ensure_ascii=False)}",
        f"- max_kl: {json.dumps(summary['correct_wrong_kl'], ensure_ascii=False)}",
        "",
        "## Boundary Coverage",
        f"- boundary_pattern_yes_among_wrong: {summary['boundary_pattern_yes_among_wrong']}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def select_case_records(records: Sequence[Dict], limit: int) -> List[Dict]:
    ranked = sorted(
        (record for record in records if not record.get("format_fail")),
        key=lambda record: (
            record.get("is_correct") is True,
            -(max_metric(record, "avg_gap") or -math.inf),
        ),
    )
    return ranked[:limit]


def maybe_plot(records: Sequence[Dict], out_dir: str, case_limit: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    token_dir = os.path.join(out_dir, "token_level_cases")
    step_dir = os.path.join(out_dir, "step_level_cases")
    aggregate_dir = os.path.join(out_dir, "aggregate_plots")
    heatmap_dir = os.path.join(out_dir, "heatmaps")
    for directory in [token_dir, step_dir, aggregate_dir, heatmap_dir]:
        ensure_dir(directory)

    for record in select_case_records(records, case_limit):
        plot_token_case(record, os.path.join(token_dir, f"{record['sample_id']}.png"), plt)
        plot_step_case(record, os.path.join(step_dir, f"{record['sample_id']}.png"), plt)

    plot_group_curve(records, "avg_gap", os.path.join(aggregate_dir, "avg_gap_correct_vs_wrong.png"), plt)
    plot_group_curve(records, "avg_kl", os.path.join(aggregate_dir, "avg_kl_correct_vs_wrong.png"), plt)
    plot_boundary_groups(records, os.path.join(aggregate_dir, "avg_gap_by_boundary_pattern.png"), plt)
    plot_heatmap(records, "avg_gap", os.path.join(heatmap_dir, "step_gap_heatmap.png"), plt)
    plot_heatmap(records, "avg_kl", os.path.join(heatmap_dir, "step_kl_heatmap.png"), plt)


def plot_token_case(record: Dict, path: str, plt) -> None:
    xs = [token["token_id"] for token in record.get("tokens", [])]
    student = [token["student_logprob"] for token in record.get("tokens", [])]
    teacher = [token["teacher_logprob"] for token in record.get("tokens", [])]
    gap = [token["gap"] for token in record.get("tokens", [])]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, student, label="student_logprob", linewidth=1.2)
    plt.plot(xs, teacher, label="teacher_logprob", linewidth=1.2)
    plt.plot(xs, gap, label="gap", linewidth=1.2)
    for step in record.get("steps", []):
        first_token = next((token for token in record["tokens"] if token.get("step_id") == step["step_id"]), None)
        if first_token is not None:
            plt.axvline(first_token["token_id"], color="gray", linestyle="--", alpha=0.3)
    plt.xlabel("Token Index")
    plt.ylabel("Value")
    plt.title(record["sample_id"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_step_case(record: Dict, path: str, plt) -> None:
    xs = [metric["step_id"] for metric in record.get("step_metrics", [])]
    gaps = [metric.get("avg_gap") for metric in record.get("step_metrics", [])]
    kls = [metric.get("avg_kl") for metric in record.get("step_metrics", [])]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, gaps, marker="o", label="avg_gap")
    if any(value is not None for value in kls):
        plt.plot(xs, kls, marker="s", label="avg_kl")
    if record.get("first_error_step") is not None:
        plt.axvline(record["first_error_step"], color="red", linestyle="--", label="human_first_error")
    plt.xlabel("Step")
    plt.ylabel("Value")
    title = record["sample_id"]
    if record.get("is_correct") is not None:
        title += f" | correct={record['is_correct']}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_group_curve(records: Sequence[Dict], field: str, path: str, plt) -> None:
    correct_curve = step_curve_by_group(records, field, lambda record: record.get("is_correct") is True)
    wrong_curve = step_curve_by_group(records, field, lambda record: record.get("is_correct") is False)
    if not correct_curve and not wrong_curve:
        return
    plt.figure(figsize=(8, 4.5))
    if correct_curve:
        plt.plot([point["step_id"] for point in correct_curve], [point["mean_value"] for point in correct_curve], marker="o", label="correct")
    if wrong_curve:
        plt.plot([point["step_id"] for point in wrong_curve], [point["mean_value"] for point in wrong_curve], marker="o", label="wrong")
    plt.xlabel("Step")
    plt.ylabel(field)
    plt.title(f"{field}: correct vs wrong")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_boundary_groups(records: Sequence[Dict], path: str, plt) -> None:
    groups = [
        ("boundary_yes", lambda record: record.get("boundary_pattern") == "yes"),
        ("boundary_no", lambda record: record.get("boundary_pattern") == "no"),
        ("boundary_unclear", lambda record: record.get("boundary_pattern") == "unclear"),
    ]
    curves = [(name, step_curve_by_group(records, "avg_gap", predicate)) for name, predicate in groups]
    if not any(curve for _, curve in curves):
        return
    plt.figure(figsize=(8, 4.5))
    for name, curve in curves:
        if curve:
            plt.plot([point["step_id"] for point in curve], [point["mean_value"] for point in curve], marker="o", label=name)
    plt.xlabel("Step")
    plt.ylabel("avg_gap")
    plt.title("avg_gap by boundary pattern")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_heatmap(records: Sequence[Dict], field: str, path: str, plt) -> None:
    filtered = [record for record in records if record.get("step_metrics")]
    if not filtered:
        return
    filtered.sort(key=lambda record: (record.get("first_error_step") is None, record.get("first_error_step") or record.get("predicted_boundary_by_gap") or 999))
    max_steps = max(len(record["step_metrics"]) for record in filtered)
    matrix: List[List[float]] = []
    for record in filtered:
        row = []
        for metric in record["step_metrics"]:
            value = metric.get(field)
            row.append(value if value is not None else float("nan"))
        while len(row) < max_steps:
            row.append(float("nan"))
        matrix.append(row)
    plt.figure(figsize=(max(8, max_steps * 0.5), max(5, len(matrix) * 0.2)))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label=field)
    plt.xlabel("Step")
    plt.ylabel("Sample")
    plt.title(f"{field} heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def boundary_ratio(records: Sequence[Dict], value: str) -> Optional[float]:
    wrong = [record for record in records if record.get("is_correct") is False and record.get("boundary_pattern") is not None]
    if not wrong:
        return None
    return sum(record.get("boundary_pattern") == value for record in wrong) / len(wrong)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BASD protocol records.")
    parser.add_argument("--records", required=True, help="Path to records.jsonl.")
    parser.add_argument("--out_dir", required=True, help="Directory for analysis artifacts.")
    parser.add_argument("--annotations", help="Optional annotation csv with first_error_step/error_type/boundary_pattern.")
    parser.add_argument("--plot_cases", type=int, default=6, help="How many representative case plots to export.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    records = read_jsonl(args.records)
    records = merge_annotations(records, args.annotations)

    sample_summary_rows = build_sample_summary(records)
    write_csv(
        os.path.join(args.out_dir, "sample_summary.csv"),
        sample_summary_rows,
        [
            "sample_id",
            "is_correct",
            "format_fail",
            "num_steps",
            "predicted_boundary_by_gap",
            "predicted_boundary_by_gap_jump",
            "predicted_boundary_by_kl",
            "human_first_error_step",
            "error_type",
            "boundary_pattern",
            "exact_match_gap",
            "within_one_gap",
            "exact_match_gap_jump",
            "within_one_gap_jump",
            "exact_match_kl",
            "within_one_kl",
            "max_gap",
            "max_kl",
        ],
    )

    summary = {
        "n_records": len(records),
        "format_success_rate": sum(not record.get("format_fail", False) for record in records) / len(records) if records else 0.0,
        "avg_num_steps": safe_mean(record.get("num_steps", 0) for record in records if not record.get("format_fail")),
        "avg_num_tokens": safe_mean(record.get("num_tokens", 0) for record in records),
        "boundary_eval_by_gap": compute_boundary_eval(records, "predicted_boundary_by_gap"),
        "boundary_eval_by_gap_jump": compute_boundary_eval(records, "predicted_boundary_by_gap_jump"),
        "boundary_eval_by_kl": compute_boundary_eval(records, "predicted_boundary_by_kl"),
        "correct_wrong_gap": compare_correct_wrong(records, "avg_gap"),
        "correct_wrong_kl": compare_correct_wrong(records, "avg_kl"),
        "boundary_pattern_yes_among_wrong": boundary_ratio(records, "yes"),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown_report(os.path.join(args.out_dir, "report.md"), summary)
    maybe_plot(records, args.out_dir, args.plot_cases)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
