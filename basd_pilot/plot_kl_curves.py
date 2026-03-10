#!/usr/bin/env python3
"""Plot KL curves for cleaned wrong cases.

Usage:
  python basd_pilot/plot_kl_curves.py ^
    --input outputs/formal_gsm8k_qwen3_8b_1/cleaned_analysis/wrong_for_jump_analysis.jsonl ^
    --out_dir outputs/formal_gsm8k_qwen3_8b_1/cleaned_analysis/kl_plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_text(text: str, max_len: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def plot_single_case(row: Dict, out_path: str) -> None:
    step_kls = row["step_kls"]
    xs = [item["step"] for item in step_kls]
    ys = [item["kl"] for item in step_kls]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xs, ys, color="#1f77b4", marker="o", linewidth=2, markersize=5)
    ax.axvline(
        row["boundary_step_jump"],
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"jump={row['boundary_step_jump']}",
    )
    ax.axvline(
        row["boundary_step_max_abs_kl"],
        color="#2ca02c",
        linestyle=":",
        linewidth=1.5,
        label=f"max_abs={row['boundary_step_max_abs_kl']}",
    )

    ax.set_title(f"{row['id']} | ans={safe_text(str(row.get('clean_student_final_answer', '')))}")
    ax.set_xlabel("Step")
    ax.set_ylabel("KL")
    ax.set_xticks(xs)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(fontsize=8)

    note = safe_text(row.get("question", ""), max_len=120)
    fig.text(0.02, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all_cases(rows: List[Dict], out_path: str, cols: int = 5) -> None:
    if not rows:
        return

    n = len(rows)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 3.2), squeeze=False)
    axes_flat = axes.flatten()

    for ax, row in zip(axes_flat, rows):
        step_kls = row["step_kls"]
        xs = [item["step"] for item in step_kls]
        ys = [item["kl"] for item in step_kls]

        ax.plot(xs, ys, color="#1f77b4", marker="o", linewidth=1.8, markersize=4)
        ax.axvline(row["boundary_step_jump"], color="#d62728", linestyle="--", linewidth=1.2)
        ax.axvline(row["boundary_step_max_abs_kl"], color="#2ca02c", linestyle=":", linewidth=1.2)
        ax.set_title(f"{row['id']} | j={row['boundary_step_jump']} a={row['boundary_step_max_abs_kl']}", fontsize=9)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("KL Curves for 35 Clean Wrong Cases", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_index(rows: List[Dict], out_path: str, image_dir: str) -> None:
    lines = [
        "# KL Curves",
        "",
        f"Total cases: {len(rows)}",
        "",
        "Legend:",
        "- Blue line: step KL",
        "- Red dashed line: boundary_step_jump",
        "- Green dotted line: boundary_step_max_abs_kl",
        "",
    ]
    for row in rows:
        img_name = f"{row['id']}.png"
        lines.append(f"## {row['id']}")
        lines.append(f"- Question: {safe_text(row.get('question', ''), max_len=140)}")
        lines.append(f"- jump={row['boundary_step_jump']}, max_abs={row['boundary_step_max_abs_kl']}")
        lines.append(f"![{row['id']}]({os.path.join(image_dir, img_name).replace(os.sep, '/')})")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KL curves for cleaned wrong cases.")
    parser.add_argument(
        "--input",
        default="outputs/formal_gsm8k_qwen3_8b_1/cleaned_analysis/wrong_for_jump_analysis.jsonl",
        help="Input jsonl file.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/formal_gsm8k_qwen3_8b_1/cleaned_analysis/kl_plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    os.makedirs(args.out_dir, exist_ok=True)
    per_case_dir = os.path.join(args.out_dir, "per_case")
    os.makedirs(per_case_dir, exist_ok=True)

    for row in rows:
        out_path = os.path.join(per_case_dir, f"{row['id']}.png")
        plot_single_case(row, out_path)

    plot_all_cases(rows, os.path.join(args.out_dir, "all_kl_curves.png"))
    write_index(rows, os.path.join(args.out_dir, "index.md"), "per_case")

    print(f"Saved {len(rows)} case plots to: {per_case_dir}")
    print(f"Saved combined figure to: {os.path.join(args.out_dir, 'all_kl_curves.png')}")
    print(f"Saved index to: {os.path.join(args.out_dir, 'index.md')}")


if __name__ == "__main__":
    main()
