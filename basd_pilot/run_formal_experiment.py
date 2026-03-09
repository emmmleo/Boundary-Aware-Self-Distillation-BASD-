#!/usr/bin/env python3
# Fast BASD pipeline for 8x A100
# Main idea: full-sequence KL in 2 forwards/sample + torchrun data sharding.

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

STEP_RE = re.compile(
    r"(?:^|\n)(?:Step|步骤)\s*(\d+)\s*[:：]?\s*(.*?)(?=(?:\n(?:Step|步骤)\s*\d+\s*[:：]?)|(?:\n(?:Final\s*Answer|最终答案)\s*[:：])|\Z)",
    re.S | re.I,
)
FINAL_RE = re.compile(r"(?:Final\s*Answer|最终答案)\s*[:：]\s*(.+?)(?:\n|$)", re.I | re.S)
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
THINK_OPEN_RE = re.compile(r"<think>.*", re.S | re.I)
NUMBERED_LINE_RE = re.compile(
    r"(?:^|\n)\s*(\d+)\s*[\.)、)]\s*(.+?)(?=(?:\n\s*\d+\s*[\.)、)])|(?:\n(?:Final\s*Answer|最终答案)\s*[:：])|\Z)",
    re.S,
)

STUDENT_SYSTEM = (
    "You are a rigorous mathematical reasoning assistant. "
    "Use English only. "
    "Use exactly the format `Step 1:`, `Step 2:`, ... and end with exactly one line "
    "`Final Answer: <answer>`. "
    "Do not output role labels, XML tags, or start a new problem."
)
TEACHER_SYSTEM_TEMPLATE = (
    "You are the teacher model. You know the reference solution and must score the "
    "next-token probability distribution for the student's solution continuation. "
    "Do not start a new conversation or output role labels.\n"
    "Reference solution:\n{reference_solution}"
)
STUDENT_FORMAT_PROMPT = (
    "Solve the problem strictly using this format:\n"
    "Step 1: ...\n"
    "Step 2: ...\n"
    "Step 3: ...\n"
    "...\n"
    "Final Answer: ...\n"
)
STOP_STRINGS = ["\nHuman:", "\nAssistant:", "\nUser:", "<|im_end|>", "<|endoftext|>", "\nProblem:"]

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
        text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=False)
        return any(s in text for s in self.stop_strings)

def maybe_init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def log(msg: str):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[rank{rank}] {msg}", flush=True)

def render_chat_prompt(tokenizer, system_text: str, user_text: str, assistant_prefix: str = "") -> str:
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System: {system_text}\nUser: {user_text}\nAssistant: "
    return prompt + assistant_prefix

def build_user_text(question: str) -> str:
    return f"{STUDENT_FORMAT_PROMPT}\nProblem: {question}\nBegin."

def build_generation_prompt(tokenizer, question: str) -> str:
    return render_chat_prompt(tokenizer, STUDENT_SYSTEM, build_user_text(question), assistant_prefix="")

def build_teacher_prompt(tokenizer, question: str, reference_solution: str, student_prefix: str) -> str:
    return render_chat_prompt(
        tokenizer,
        TEACHER_SYSTEM_TEMPLATE.format(reference_solution=reference_solution),
        build_user_text(question),
        assistant_prefix=student_prefix,
    )

def build_student_prompt(tokenizer, question: str, student_prefix: str) -> str:
    return render_chat_prompt(tokenizer, STUDENT_SYSTEM, build_user_text(question), assistant_prefix=student_prefix)

def strip_think(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_OPEN_RE.sub("", text)
    return text

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("：", ":")
    text = text.replace("步骤", "Step ")
    return strip_think(text)

def convert_numbered_list_to_steps(text: str) -> str:
    if re.search(r"(?:^|\n)\s*Step\s*\d+\s*:", text, flags=re.I):
        return text
    matches = list(NUMBERED_LINE_RE.finditer(text))
    if not matches:
        return text
    parts = [f"Step {int(m.group(1))}: {m.group(2).strip()}" for m in matches]
    fm = FINAL_RE.search(text)
    if fm:
        parts.append(f"Final Answer: {fm.group(1).strip()}")
    return "\n".join(parts)

def sanitize_student_solution(text: str) -> str:
    text = normalize_text(text).strip()
    for marker in STOP_STRINGS:
        if marker in text:
            text = text.split(marker)[0]
    m = re.search(r"((?:Step\s*1\s*:)|(?:1\s*[\.)、)]))", text, flags=re.I)
    if m:
        text = text[m.start():]
    text = convert_numbered_list_to_steps(text)
    fm = FINAL_RE.search(text)
    if fm:
        text = text[:fm.end()]
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
    ans = normalize_text(ans).strip().replace("\\boxed{", "").replace("}", "").replace(",", "").replace("$", "")
    nums = NUM_RE.findall(ans)
    if nums:
        return nums[-1]
    return re.sub(r"\s+", " ", ans).strip().lower()

def load_jsonl(path: str) -> List[Problem]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rows.append(Problem(sample_id=str(obj["id"]), question=obj["question"], reference_solution=obj["reference_solution"]))
    return rows

def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_debug_jsonl(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_annotation_csv(path: str, rows: List[Dict]) -> None:
    fields = ["id", "question", "student_solution", "reference_solution", "auto_boundary_step", "manual_first_error_step"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "id": r["id"], "question": r["question"], "student_solution": r["student_solution"],
                "reference_solution": r["reference_solution"], "auto_boundary_step": r["boundary_step_jump"],
                "manual_first_error_step": r.get("manual_first_error_step", ""),
            })

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
    return {"n": len(labeled), "exact_match": exact, "hit_at_1": hit1, "mae": mae, "jump_better_or_equal_ratio": jump_better}

def detect_boundary(step_kls: List[Dict]) -> Tuple[int, int]:
    if not step_kls:
        return -1, -1
    if len(step_kls) == 1:
        s = step_kls[0]["step"]
        return s, s
    jumps = [(step_kls[i]["step"], step_kls[i]["kl"] - step_kls[i-1]["kl"]) for i in range(1, len(step_kls))]
    return max(jumps, key=lambda x: x[1])[0], max(step_kls, key=lambda x: x["kl"])["step"]

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

def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    return model, tokenizer

def generate_student_solution(model, tokenizer, question: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = build_generation_prompt(tokenizer, question)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is not None and tid != tokenizer.unk_token_id:
        eos_ids.append(tid)
    eos_ids = list(dict.fromkeys([x for x in eos_ids if x is not None]))
    stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, STOP_STRINGS, inputs["input_ids"].shape[1])])
    do_sample = temperature is not None and temperature > 0
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.02,
        no_repeat_ngram_size=4,
        stopping_criteria=stopping,
        use_cache=True,
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    with torch.inference_mode():
        output = model.generate(**kwargs)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return sanitize_student_solution(text)

def encode_full_sequence(tokenizer, prompt: str, continuation: str, device: torch.device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    return len(prompt_ids), full_ids

def compute_step_kl_curve_fast(model, tokenizer, question: str, reference_solution: str, student_solution: str) -> List[Dict]:
    steps = parse_steps(student_solution)
    if not steps:
        return []

    normalized_solution = sanitize_student_solution(student_solution)
    teacher_prompt = build_teacher_prompt(tokenizer, question, reference_solution, "")
    student_prompt = build_student_prompt(tokenizer, question, "")

    device = next(model.parameters()).device
    teacher_prompt_len, teacher_ids = encode_full_sequence(tokenizer, teacher_prompt, normalized_solution, device)
    student_prompt_len, student_ids = encode_full_sequence(tokenizer, student_prompt, normalized_solution, device)

    teacher_cont_len = teacher_ids.shape[1] - teacher_prompt_len
    student_cont_len = student_ids.shape[1] - student_prompt_len
    if teacher_cont_len <= 0 or teacher_cont_len != student_cont_len:
        return []

    with torch.inference_mode():
        t_logits = model(teacher_ids, use_cache=False).logits[0]
        s_logits = model(student_ids, use_cache=False).logits[0]

    t_pred = t_logits[teacher_prompt_len - 1 : teacher_prompt_len - 1 + teacher_cont_len]
    s_pred = s_logits[student_prompt_len - 1 : student_prompt_len - 1 + student_cont_len]

    t_logp = torch.log_softmax(t_pred.float(), dim=-1)
    s_logp = torch.log_softmax(s_pred.float(), dim=-1)
    t_prob = t_logp.exp()
    token_kls = torch.sum(t_prob * (t_logp - s_logp), dim=-1)

    curve = []
    offset = 0
    for sid, stext in steps:
        chunk = f"Step {sid}: {stext}\n"
        chunk_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        if offset + len(chunk_ids) > teacher_cont_len:
            chunk = f"Step {sid}: {stext}"
            chunk_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        n = len(chunk_ids)
        if n == 0:
            curve.append({"step": sid, "kl": 0.0, "token_count": 0})
            continue
        curve.append({"step": sid, "kl": float(token_kls[offset:offset+n].mean().item()), "token_count": n})
        offset += n
    return curve

def shard_problems(problems: List[Problem], rank: int, world_size: int) -> List[Problem]:
    return [p for i, p in enumerate(problems) if i % world_size == rank]

def merge_shards(out_dir: str, world_size: int):
    all_rows = []
    for r in range(world_size):
        path = os.path.join(out_dir, f"records.rank{r}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_rows.append(json.loads(line))
    all_rows.sort(key=lambda x: x["id"])
    write_jsonl(os.path.join(out_dir, "records.jsonl"), all_rows)
    save_annotation_csv(os.path.join(out_dir, "annotation_template.csv"), all_rows)
    weak_rows = [dict(r, manual_first_error_step=r["inferred_first_error_step"]) for r in all_rows if r.get("inferred_first_error_step") is not None]
    summary = {
        "n_total": len(all_rows),
        "n_correct": sum(r["is_correct"] for r in all_rows),
        "n_wrong": sum(not r["is_correct"] for r in all_rows),
        "weak_eval_from_inferred_first_error": evaluate(weak_rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Fast BASD formal experiment pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="outputs/formal_fast")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rank, world = maybe_init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    os.makedirs(args.out_dir, exist_ok=True)
    debug_skipped_path = os.path.join(args.out_dir, f"skipped_debug.rank{rank}.jsonl")
    if os.path.exists(debug_skipped_path):
        os.remove(debug_skipped_path)

    log("loading model/tokenizer")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    problems = load_jsonl(args.input)
    if args.limit > 0:
        problems = problems[:args.limit]
    my_problems = shard_problems(problems, rank, world)
    log(f"processing {len(my_problems)} / {len(problems)} samples")

    rows = []
    for idx, p in enumerate(my_problems, start=1):
        if idx <= 3 or idx % 20 == 0:
            log(f"[{idx}/{len(my_problems)}] {p.sample_id}")
        raw_student_solution = generate_student_solution(model, tokenizer, p.question, args.max_new_tokens, args.temperature, args.top_p)
        student_solution = sanitize_student_solution(raw_student_solution)
        step_kls = compute_step_kl_curve_fast(model, tokenizer, p.question, p.reference_solution, student_solution)

        if not step_kls:
            append_debug_jsonl(debug_skipped_path, {
                "id": p.sample_id,
                "question": p.question,
                "raw_student_solution": raw_student_solution,
                "sanitized_student_solution": student_solution,
            })
            continue

        boundary_jump, boundary_abs = detect_boundary(step_kls)
        gt_final = normalize_answer(extract_final_answer(p.reference_solution))
        pred_final = normalize_answer(extract_final_answer(student_solution))
        is_correct = gt_final == pred_final

        rows.append({
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
        })

        if idx % 50 == 0:
            write_jsonl(os.path.join(args.out_dir, f"records.rank{rank}.jsonl"), rows)

    write_jsonl(os.path.join(args.out_dir, f"records.rank{rank}.jsonl"), rows)
    barrier()

    if rank == 0:
        summary = merge_shards(args.out_dir, world)
        log(json.dumps(summary, ensure_ascii=False, indent=2))

    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
