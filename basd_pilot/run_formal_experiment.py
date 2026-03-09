#!/usr/bin/env python3
"""Debugged formal BASD experiment pipeline.

Key fixes:
1. Use chat-template aligned prompts for both generation and scoring.
2. Stop generation on common prompt-leak markers.
3. Sanitize student output before parsing steps / final answer.
4. Use stronger final-answer normalization.
5. Replace exact-text first-error heuristic with a numeric heuristic.
6. Add progress logging.
7. Use KV cache for token-wise KL scoring to avoid apparent hangs.
"""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

STEP_RE = re.compile(
    r"(?:^|\n)Step\s*(\d+)\s*:\s*(.*?)(?=(?:\nStep\s*\d+\s*:)|(?:\nFinal\s*Answer\s*:)|\Z)",
    re.S | re.I,
)
FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+?)(?:\n|\Z)", re.I | re.S)
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")

STUDENT_SYSTEM = (
    "你是一个严谨的数学推理助手。"
    "严格按照用户要求的格式输出，并在给出 Final Answer 后立即停止，不要继续写新的题目、角色标签或解释。"
)
TEACHER_SYSTEM_TEMPLATE = (
    "你是教师模型。你知道题目的参考解，并需要对学生当前解答前缀的下一 token 概率分布进行评估。"
    "你必须延续学生当前的解答风格与格式，不要开启新对话，不要输出角色标签。\n"
    "参考解如下：\n{reference_solution}"
)

STUDENT_FORMAT_PROMPT = (
    "请严格按照以下格式解题：\n"
    "Step 1: ...\n"
    "Step 2: ...\n"
    "Step 3: ...\n"
    "...\n"
    "Final Answer: ..."
)

STOP_STRINGS = [
    "\nHuman:",
    "\nAssistant:",
    "\nUser:",
    "<|im_end|>",
    "<|endoftext|>",
]


@dataclass
class Problem:
    sample_id: str
    question: str
    reference_solution: str


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str], prompt_len: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0][self.prompt_len :], skip_special_tokens=False)
        return any(s in text for s in self.stop_strings)


def render_chat_prompt(tokenizer, system_text: str, user_text: str, assistant_prefix: str = "") -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System: {system_text}\nUser: {user_text}\nAssistant: "
    return prompt + assistant_prefix


def build_user_text(question: str) -> str:
    return f"{STUDENT_FORMAT_PROMPT}\n\n题目：{question}\n请开始解题。"


def sanitize_student_solution(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    for marker in STOP_STRINGS:
        if marker in text:
            text = text.split(marker)[0]

    m = re.search(r"(Step\s*1\s*:.*)", text, flags=re.S | re.I)
    if m:
        text = m.group(1)

    fm = FINAL_RE.search(text)
    if fm:
        end = fm.end()
        text = text[:end]

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def parse_steps(solution: str) -> List[Tuple[int, str]]:
    solution = sanitize_student_solution(solution)
    return [(int(m.group(1)), m.group(2).strip()) for m in STEP_RE.finditer(solution)]


def extract_final_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()
    text = sanitize_student_solution(text)
    matches = FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace("\\boxed{", "").replace("}", "")
    ans = ans.replace(",", "").replace("$", "")
    nums = NUM_RE.findall(ans)
    if nums:
        return nums[-1]
    return re.sub(r"\s+", " ", ans).strip().lower()


def build_generation_prompt(tokenizer, question: str) -> str:
    return render_chat_prompt(tokenizer, STUDENT_SYSTEM, build_user_text(question), assistant_prefix="")


def build_teacher_prefix(tokenizer, question: str, reference_solution: str, student_prefix: str) -> str:
    system_text = TEACHER_SYSTEM_TEMPLATE.format(reference_solution=reference_solution)
    return render_chat_prompt(tokenizer, system_text, build_user_text(question), assistant_prefix=student_prefix)


def build_student_prefix(tokenizer, question: str, student_prefix: str) -> str:
    return render_chat_prompt(tokenizer, STUDENT_SYSTEM, build_user_text(question), assistant_prefix=student_prefix)


def get_prompt_device(model):
    return next(model.parameters()).device


def generate_student_solution(model, tokenizer, question: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = build_generation_prompt(tokenizer, question)
    device = get_prompt_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for tok in ["<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)
    eos_ids = list(dict.fromkeys([x for x in eos_ids if x is not None]))

    stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, STOP_STRINGS, inputs["input_ids"].shape[1])])
    do_sample = temperature is not None and temperature > 0

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            no_repeat_ngram_size=6,
            stopping_criteria=stopping,
            use_cache=True,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return sanitize_student_solution(text)


def init_cached_state(model, tokenizer, prompt: str):
    device = get_prompt_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def advance_one_token(model, token_id: int, past_key_values):
    device = get_prompt_device(model)
    token_tensor = torch.tensor([[token_id]], device=device)
    with torch.inference_mode():
        out = model(input_ids=token_tensor, past_key_values=past_key_values, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def kl_from_logits(teacher_logits: torch.Tensor, student_logits: torch.Tensor, eps: float = 1e-12) -> float:
    p = torch.softmax(teacher_logits, dim=-1).squeeze(0)
    q = torch.softmax(student_logits, dim=-1).squeeze(0)
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return float(torch.sum(p * torch.log(p / q)).item())


def compute_step_kl_curve(model, tokenizer, question: str, reference_solution: str, student_solution: str) -> List[Dict]:
    steps = parse_steps(student_solution)
    if not steps:
        return []

    curve = []
    accumulated = ""
    for sid, stext in steps:
        step_prefix = accumulated + f"Step {sid}: "
        step_token_ids = tokenizer(stext, add_special_tokens=False)["input_ids"]
        if not step_token_ids:
            curve.append({"step": sid, "kl": 0.0, "token_count": 0})
            accumulated += f"Step {sid}: {stext}\n"
            continue

        teacher_prompt = build_teacher_prefix(tokenizer, question, reference_solution, step_prefix)
        student_prompt = build_student_prefix(tokenizer, question, step_prefix)
        teacher_logits, teacher_past = init_cached_state(model, tokenizer, teacher_prompt)
        student_logits, student_past = init_cached_state(model, tokenizer, student_prompt)

        token_kls = []
        for tok_id in step_token_ids:
            token_kls.append(kl_from_logits(teacher_logits, student_logits))
            teacher_logits, teacher_past = advance_one_token(model, tok_id, teacher_past)
            student_logits, student_past = advance_one_token(model, tok_id, student_past)

        curve.append(
            {
                "step": sid,
                "kl": float(sum(token_kls) / len(token_kls)),
                "token_count": len(step_token_ids),
            }
        )
        accumulated += f"Step {sid}: {stext}\n"
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


def last_numeric_token(text: str) -> Optional[str]:
    nums = NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def infer_first_error(reference_solution: str, student_solution: str) -> Optional[int]:
    rs = parse_steps(reference_solution)
    ss = parse_steps(student_solution)
    if not ss:
        return None

    for (ri, rt), (si, st) in zip(rs, ss):
        if ri != si:
            return si
        rnum = last_numeric_token(rt)
        snum = last_numeric_token(st)
        if rnum is not None and snum is not None and normalize_answer(rnum) != normalize_answer(snum):
            return si

    ref_final = normalize_answer(extract_final_answer(reference_solution))
    stu_final = normalize_answer(extract_final_answer(student_solution))
    if stu_final != ref_final:
        return ss[min(len(ss), max(1, len(rs))) - 1][0]
    return None


def load_jsonl(path: str) -> List[Problem]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append(Problem(sample_id=str(obj["id"]), question=obj["question"], reference_solution=obj["reference_solution"]))
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
        abs(r["boundary_step_jump"] - r["manual_first_error_step"]) <= abs(r["boundary_step_max_abs_kl"] - r["manual_first_error_step"])
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
    fields = ["id", "question", "student_solution", "reference_solution", "auto_boundary_step", "manual_first_error_step"]
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
    parser = argparse.ArgumentParser(description="Debugged formal BASD pilot with Qwen3-8B + full dataset.")
    parser.add_argument("--input", required=True, help="jsonl dataset path (full set)")
    parser.add_argument("--out_dir", default="outputs/formal")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="建议调试时用 0.0 做 greedy")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0, help="0 表示跑完整数据集")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("[3/4] loading dataset", flush=True)
    problems = load_jsonl(args.input)
    if args.limit > 0:
        problems = problems[: args.limit]

    print(f"[4/4] running {len(problems)} samples", flush=True)
    rows = []
    for idx, p in enumerate(problems, start=1):
        print(f"[{idx}/{len(problems)}] generating student solution: {p.sample_id}", flush=True)
        student_solution = generate_student_solution(model, tokenizer, p.question, args.max_new_tokens, args.temperature, args.top_p)
        print(f"[{idx}/{len(problems)}] scoring KL by step", flush=True)
        step_kls = compute_step_kl_curve(model, tokenizer, p.question, p.reference_solution, student_solution)
        if not step_kls:
            print(f"[{idx}/{len(problems)}] skipped (no valid steps parsed)", flush=True)
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

        if idx % 10 == 0 or idx == len(problems):
            write_jsonl(os.path.join(args.out_dir, "records.partial.jsonl"), rows)

    write_jsonl(os.path.join(args.out_dir, "records.jsonl"), rows)
    save_annotation_csv(os.path.join(args.out_dir, "annotation_template.csv"), rows)

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

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
