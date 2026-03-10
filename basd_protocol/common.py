#!/usr/bin/env python3
"""Shared utilities for the standalone BASD protocol pipeline."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

STEP_BLOCK_RE = re.compile(
    r"(?P<label>(?:^|\n)Step\s*(?P<step>\d+)\s*:)(?P<body>.*?)(?=(?:\nStep\s*\d+\s*:)|(?:\nFinal\s*Answer\s*:)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
FINAL_BLOCK_RE = re.compile(
    r"(?P<label>(?:^|\n)Final\s*Answer\s*:)(?P<body>.*?)(?=\Z)",
    re.IGNORECASE | re.DOTALL,
)
FIRST_MARKER_RE = re.compile(r"(?:^|\n)(?:Step\s*\d+\s*:|Final\s*Answer\s*:)", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
NUMERIC_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")

STUDENT_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Reply only in the exact format `Step 1: ...`, `Step 2: ...`, ..., `Final Answer: ...`. "
    "Do not use XML tags, role names, or extra sections."
)
STUDENT_USER_TEMPLATE = (
    "Solve the problem step by step.\n"
    "Follow this format exactly:\n"
    "Step 1: ...\n"
    "Step 2: ...\n"
    "...\n"
    "Final Answer: ...\n\n"
    "Keep the reasoning concise, use at most {max_steps} reasoning steps, and stop immediately after `Final Answer:`.\n\n"
    "Problem:\n{question}"
)
TEACHER_SYSTEM_TEMPLATE = (
    "You are a privileged teacher model. "
    "You know the gold solution and answer. "
    "Your job is to score the student's next token under the same answer prefix, not to rewrite the solution.\n\n"
    "Privileged information:\n{privileged_info}"
)


@dataclass
class SampleInput:
    sample_id: str
    question: str
    reference_solution: str
    gold_answer: str


@dataclass
class StepSpan:
    step_id: int
    original_step_id: int
    step_text: str
    char_start: int
    char_end: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_annotation_csv(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["sample_id"]: row for row in reader if row.get("sample_id")}


def sanitize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = THINK_TAG_RE.sub("", text)
    if "assistant\n" in text.lower():
        lowered = text.lower()
        idx = lowered.find("assistant\n")
        text = text[idx + len("assistant\n") :]
    marker = FIRST_MARKER_RE.search(text)
    if marker:
        text = text[marker.start() :].lstrip("\n")
    return text.strip()


def extract_answer(text: str) -> str:
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    cleaned = sanitize_text(text)
    final_match = FINAL_BLOCK_RE.search(cleaned)
    if final_match:
        return final_match.group("body").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(text: str) -> str:
    text = extract_answer(text)
    text = text.replace("\\boxed{", "").replace("}", "")
    text = text.replace(",", "").replace("$", "").strip()
    numeric_matches = NUMERIC_RE.findall(text)
    if numeric_matches:
        return numeric_matches[-1]
    return re.sub(r"\s+", " ", text).lower()


def load_samples(path: str) -> List[SampleInput]:
    samples: List[SampleInput] = []
    for raw in read_jsonl(path):
        sample_id = str(raw.get("sample_id", raw.get("id", "")))
        question = raw["question"]
        reference_solution = raw.get("reference_solution", "")
        gold_answer = raw.get("gold_answer") or raw.get("final_answer") or extract_answer(reference_solution)
        samples.append(
            SampleInput(
                sample_id=sample_id,
                question=question,
                reference_solution=reference_solution,
                gold_answer=gold_answer,
            )
        )
    return samples


def render_student_user_prompt(question: str, max_steps: int) -> str:
    return STUDENT_USER_TEMPLATE.format(question=question, max_steps=max_steps)


def render_teacher_privileged_info(sample: SampleInput) -> str:
    return (
        f"Question:\n{sample.question}\n\n"
        f"Reference solution:\n{sample.reference_solution}\n\n"
        f"Gold final answer:\n{sample.gold_answer}"
    )


def render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: "


def parse_structured_solution(text: str, max_step_count: int, overflow_mode: str) -> Dict:
    cleaned = sanitize_text(text)
    steps: List[StepSpan] = []
    notes: List[str] = []
    numbering_repaired = False
    step_matches = list(STEP_BLOCK_RE.finditer(cleaned))
    final_match = FINAL_BLOCK_RE.search(cleaned)

    if not step_matches:
        return {
            "cleaned_text": cleaned,
            "format_fail": True,
            "format_error": "missing_steps",
            "format_notes": ["No `Step N:` blocks found."],
            "steps": [],
            "final_answer_text": extract_answer(cleaned),
            "final_answer_span": None,
            "step_overflow": False,
            "numbering_repaired": False,
            "analyzed_step_count": 0,
        }

    if final_match is None:
        notes.append("Missing `Final Answer:` block.")

    for expected_id, match in enumerate(step_matches, start=1):
        original_id = int(match.group("step"))
        if original_id != expected_id:
            numbering_repaired = True
            notes.append(f"Repaired step numbering: observed {original_id}, normalized to {expected_id}.")
        steps.append(
            StepSpan(
                step_id=expected_id,
                original_step_id=original_id,
                step_text=match.group("body").strip(),
                char_start=match.start("label"),
                char_end=match.end("body"),
            )
        )

    step_overflow = len(steps) > max_step_count
    if step_overflow:
        if overflow_mode == "discard":
            return {
                "cleaned_text": cleaned,
                "format_fail": True,
                "format_error": "step_overflow",
                "format_notes": notes + [f"Step count {len(steps)} exceeds max_step_count={max_step_count}."],
                "steps": [],
                "final_answer_text": extract_answer(cleaned),
                "final_answer_span": None,
                "step_overflow": True,
                "numbering_repaired": numbering_repaired,
                "analyzed_step_count": 0,
            }
        notes.append(f"Truncated analysis to the first {max_step_count} steps.")
        steps = steps[:max_step_count]

    final_span = None
    if final_match is not None:
        final_span = {
            "char_start": final_match.start("label"),
            "char_end": final_match.end("body"),
        }

    format_fail = final_match is None
    format_error = "missing_final_answer" if format_fail else None
    return {
        "cleaned_text": cleaned,
        "format_fail": format_fail,
        "format_error": format_error,
        "format_notes": notes,
        "steps": steps,
        "final_answer_text": final_match.group("body").strip() if final_match else extract_answer(cleaned),
        "final_answer_span": final_span,
        "step_overflow": step_overflow,
        "numbering_repaired": numbering_repaired,
        "analyzed_step_count": len(steps),
    }


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        encoded = tokenizer(text, add_special_tokens=False)
    token_ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        offsets = fallback_offsets(tokenizer, token_ids)
    offsets = [tuple(item) for item in offsets]
    token_texts = [tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token_id in token_ids]
    return token_ids, token_texts, offsets


def fallback_offsets(tokenizer, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    previous = ""
    for index in range(len(token_ids)):
        current = tokenizer.decode(
            token_ids[: index + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        offsets.append((len(previous), len(current)))
        previous = current
    return offsets


def locate_step_id(char_start: int, step_spans: Sequence[StepSpan], final_answer_span: Optional[Dict]) -> Tuple[Optional[int], str]:
    for step in step_spans:
        if step.char_start <= char_start < step.char_end:
            return step.step_id, "reasoning"
    if final_answer_span and final_answer_span["char_start"] <= char_start < final_answer_span["char_end"]:
        return None, "final_answer"
    return None, "other"


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


def predicted_boundary(step_metrics: Sequence[Dict], key: str) -> Optional[int]:
    valid = [metric for metric in step_metrics if metric.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda metric: metric[key])["step_id"]


def predicted_jump_boundary(step_metrics: Sequence[Dict], key: str) -> Optional[int]:
    previous_value: Optional[float] = None
    best_step: Optional[int] = None
    best_jump = -math.inf
    for metric in step_metrics:
        value = metric.get(key)
        if value is None:
            continue
        if previous_value is None:
            previous_value = value
            continue
        jump = value - previous_value
        if jump > best_jump:
            best_jump = jump
            best_step = metric["step_id"]
        previous_value = value
    return best_step if best_step is not None else predicted_boundary(step_metrics, key)


def build_annotation_template_rows(records: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for record in records:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "is_correct": record["is_correct"],
                "first_error_step": "",
                "error_type": "",
                "boundary_pattern": "",
                "comments": "",
                "predicted_boundary_by_gap": record.get("predicted_boundary_by_gap") or "",
                "predicted_boundary_by_gap_jump": record.get("predicted_boundary_by_gap_jump") or "",
                "predicted_boundary_by_kl": record.get("predicted_boundary_by_kl") or "",
                "num_steps": record.get("num_steps", 0),
            }
        )
    return rows


def write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
