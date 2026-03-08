#!/usr/bin/env python3
"""正式预实验：Qwen3-8B teacher-student 边界分析（inference-only）"""

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEP_RE = re.compile(r"Step\s*(\d+)\s*:\s*(.*?)(?=(?:\nStep\s*\d+\s*:)|(?:\nFinal Answer\s*:)|\Z)", re.S | re.I)

STUDENT_FORMAT_PROMPT = (
    "请严格按照以下格式解题：\n"
    "Step 1: ...\nStep 2: ...\nStep 3: ...\n...\nFinal Answer: ...\n"
)


@dataclass
class Problem:
    sample_id: str
    question: str
    reference_solution: str


def parse_steps(solution: str) -> List[Tuple[int, str]]:
    return [(int(m.group(1)), m.group(2).strip()) for m in STEP_RE.finditer(solution)]


def extract_final_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()
    m = re.search(r"Final\s*Answer\s*:\s*(.+)$", text, flags=re.I | re.M)
    return m.group(1).strip() if m else text.strip().splitlines()[-1]


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace(",", "")
    ans = ans.replace("$", "")
    return ans


def build_student_prompt(question: str) -> str:
    return f"{STUDENT_FORMAT_PROMPT}\n题目：{question}\n请开始解题。"


def build_teacher_prefix(question: str, reference_solution: str) -> str:
    return (
        "你是教师模型。你知道这道题的参考解，请据此对学生解答进行下一token概率评估。\n"
        f"题目：{question}\n"
        f"参考解：\n{reference_solution}\n"
        "以下是学生解答前缀：\n"
    )


def build_student_prefix(question: str) -> str:
    return (
        "你是学生模型，仅根据题目作答。请对你自己的下一token分布打分。\n"
        f"题目：{question}\n"
        "以下是你的解答前缀：\n"
    )


def generate_student_solution(model, tokenizer, question: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = build_student_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def next_token_distribution(model, tokenizer, prefix: str) -> torch.Tensor:
    inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
    return probs.squeeze(0)


def kl_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return float(torch.sum(p * torch.log(p / q)).item())


def compute_step_kl_curve(
    model,
    tokenizer,
    question: str,
    reference_solution: str,
    student_solution: str,
) -> List[Dict]:
    steps = parse_steps(student_solution)
    if not steps:
        return []

    curve = []
    accumulated = ""
    for sid, stext in steps:
        step_tokens = tokenizer(stext, add_special_tokens=False)["input_ids"]
        token_kls = []
        step_prefix = f"Step {sid}: "
        for tok_id in step_tokens:
            teacher_prefix = build_teacher_prefix(question, reference_solution) + accumulated + step_prefix
            student_prefix = build_student_prefix(question) + accumulated + step_prefix

            p_teacher = next_token_distribution(model, tokenizer, teacher_prefix)
            p_student = next_token_distribution(model, tokenizer, student_prefix)
            token_kls.append(kl_from_probs(p_teacher, p_student))

            token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
            step_prefix += token_text

        accumulated += f"Step {sid}: {stext}\n"
        curve.append(
            {
                "step": sid,
                "kl": float(sum(token_kls) / len(token_kls)) if token_kls else 0.0,
                "token_count": len(step_tokens),
            }
        )
    return curve


def detect_boundary(step_kls: List[Dict]) -> Tuple[int, int]:
    if not step_kls:
        return -1, -1
    if len(step_kls) == 1:
        s = step_kls[0]["step"]
        return s, s
    jumps = [(step_kls[i]["step"], step_kls[i]["kl"] - step_kls[i - 1]["kl"]) for i in range(1, len(step_kls))]
    jump_step = max(jumps, key=lambda x: x[1])[0]
    abs_step = max(step_kls, key=lambda x: x["kl"])["step"]
    return jump_step, abs_step


def infer_first_error(reference_solution: str, student_solution: str) -> Optional[int]:
    rs = parse_steps(reference_solution)
    ss = parse_steps(student_solution)
    for (ri, rt), (si, st) in zip(rs, ss):
        if ri != si or rt.strip() != st.strip():
            return si
    if len(ss) > len(rs):
        return ss[len(rs)][0]
    return None


def load_jsonl(path: str) -> List[Problem]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append(
                Problem(
                    sample_id=str(obj["id"]),
                    question=obj["question"],
                    reference_solution=obj["reference_solution"],
                )
            )
    return data


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def evaluate(rows: List[Dict]) -> Dict:
    labeled = [r for r in rows if r.get("manual_first_error_step") is not None]
    if not labeled:
        return {}
    exact = sum(r["boundary_step_jump"] == r["manual_first_error_step"] for r in labeled) / len(labeled)
    hit1 = sum(abs(r["boundary_step_jump"] - r["manual_first_error_step"]) <= 1 for r in labeled) / len(labeled)
    mae = sum(abs(r["boundary_step_jump"] - r["manual_first_error_step"]) for r in labeled) / len(labeled)
    jump_better = sum(
        abs(r["boundary_step_jump"] - r["manual_first_error_step"]) <=
        abs(r["boundary_step_max_abs_kl"] - r["manual_first_error_step"])
        for r in labeled
    ) / len(labeled)
    return {
        "n": len(labeled),
        "exact_match": exact,
        "hit_at_1": hit1,
        "mae": mae,
        "jump_better_or_equal_ratio": jump_better,
    }


def save_annotation_csv(path: str, rows: List[Dict]) -> None:
    fields = [
        "id", "question", "student_solution", "reference_solution", "auto_boundary_step", "manual_first_error_step"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "id": r["id"],
                    "question": r["question"],
                    "student_solution": r["student_solution"],
                    "reference_solution": r["reference_solution"],
                    "auto_boundary_step": r["boundary_step_jump"],
                    "manual_first_error_step": r.get("manual_first_error_step", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal BASD pilot with Qwen3-8B + full dataset.")
    parser.add_argument("--input", required=True, help="jsonl dataset path (full set)")
    parser.add_argument("--out_dir", default="outputs/formal")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0, help="0表示跑完整数据集")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    problems = load_jsonl(args.input)
    if args.limit > 0:
        problems = problems[: args.limit]

    rows = []
    for p in problems:
        student_solution = generate_student_solution(
            model, tokenizer, p.question, args.max_new_tokens, args.temperature, args.top_p
        )
        step_kls = compute_step_kl_curve(model, tokenizer, p.question, p.reference_solution, student_solution)
        if not step_kls:
            continue

        boundary_jump, boundary_abs = detect_boundary(step_kls)
        gt_final = normalize_answer(extract_final_answer(p.reference_solution))
        pred_final = normalize_answer(extract_final_answer(student_solution))
        is_correct = gt_final == pred_final

        rows.append(
            {
                "id": p.sample_id,
                "question": p.question,
                "reference_solution": p.reference_solution,
                "student_solution": student_solution,
                "is_correct": is_correct,
                "manual_first_error_step": None,
                "inferred_first_error_step": infer_first_error(p.reference_solution, student_solution),
                "boundary_step_jump": boundary_jump,
                "boundary_step_max_abs_kl": boundary_abs,
                "step_kls": step_kls,
            }
        )

    write_jsonl(os.path.join(args.out_dir, "records.jsonl"), rows)
    save_annotation_csv(os.path.join(args.out_dir, "annotation_template.csv"), rows)

    # 基于自动推断 first error 的弱监督统计，人工标注后可替换
    weak_rows = [dict(r, manual_first_error_step=r["inferred_first_error_step"]) for r in rows if r["inferred_first_error_step"] is not None]
    summary = {
        "model": args.model_name,
        "n_total": len(rows),
        "n_correct": sum(r["is_correct"] for r in rows),
        "n_wrong": sum(not r["is_correct"] for r in rows),
        "weak_eval_from_inferred_first_error": evaluate(weak_rows),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
