#!/usr/bin/env python3
"""Run the standalone BASD protocol end to end."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basd_protocol.common import (
    STUDENT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_TEMPLATE,
    build_annotation_template_rows,
    ensure_dir,
    extract_answer,
    load_samples,
    locate_step_id,
    mean_or_none,
    normalize_answer,
    parse_structured_solution,
    predicted_boundary,
    predicted_jump_boundary,
    render_chat_prompt,
    render_student_user_prompt,
    render_teacher_privileged_info,
    tokenize_with_offsets,
    write_csv,
    write_jsonl,
)


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def build_student_prompt(tokenizer, question: str, max_steps: int) -> str:
    return render_chat_prompt(tokenizer, STUDENT_SYSTEM_PROMPT, render_student_user_prompt(question, max_steps))


def build_teacher_prompt(tokenizer, sample, max_steps: int) -> str:
    teacher_system = TEACHER_SYSTEM_TEMPLATE.format(privileged_info=render_teacher_privileged_info(sample))
    return render_chat_prompt(tokenizer, teacher_system, render_student_user_prompt(sample.question, max_steps))


def generate_student_text(model, tokenizer, sample, max_steps: int, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = build_student_prompt(tokenizer, sample.question, max_steps)
    device = device_of(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    generated_ids = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def init_next_token_state(model, tokenizer, prompt: str, max_context_tokens: int) -> Tuple[torch.Tensor, object]:
    device = device_of(model)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_tokens,
    ).to(device)
    with torch.inference_mode():
        output = model(**encoded, use_cache=True)
    return output.logits[:, -1, :], output.past_key_values


def advance_state(model, token_id: int, past_key_values) -> Tuple[torch.Tensor, object]:
    device = device_of(model)
    input_ids = torch.tensor([[token_id]], device=device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
    return output.logits[:, -1, :], output.past_key_values


def score_teacher_student(
    model,
    tokenizer,
    sample,
    continuation_ids: List[int],
    max_steps: int,
    max_context_tokens: int,
    compute_token_kl: bool,
) -> Tuple[List[float], List[float], List[float | None]]:
    student_prompt = build_student_prompt(tokenizer, sample.question, max_steps)
    teacher_prompt = build_teacher_prompt(tokenizer, sample, max_steps)

    student_logits, student_past = init_next_token_state(model, tokenizer, student_prompt, max_context_tokens)
    teacher_logits, teacher_past = init_next_token_state(model, tokenizer, teacher_prompt, max_context_tokens)

    student_scores: List[float] = []
    teacher_scores: List[float] = []
    token_kls: List[float | None] = []

    for token_id in continuation_ids:
        student_logprobs = torch.log_softmax(student_logits, dim=-1)
        teacher_logprobs = torch.log_softmax(teacher_logits, dim=-1)

        student_scores.append(float(student_logprobs[0, token_id].item()))
        teacher_scores.append(float(teacher_logprobs[0, token_id].item()))

        if compute_token_kl:
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            student_probs = torch.softmax(student_logits, dim=-1)
            kl_value = torch.sum(
                teacher_probs * (torch.log(teacher_probs.clamp_min(1e-12)) - torch.log(student_probs.clamp_min(1e-12)))
            )
            token_kls.append(float(kl_value.item()))
        else:
            token_kls.append(None)

        student_logits, student_past = advance_state(model, token_id, student_past)
        teacher_logits, teacher_past = advance_state(model, token_id, teacher_past)

    return student_scores, teacher_scores, token_kls


def build_token_records(
    tokenizer,
    solution_text: str,
    step_parse: Dict,
    student_logprobs: List[float],
    teacher_logprobs: List[float],
    token_kls: List[float | None],
) -> List[Dict]:
    token_ids, token_strings, offsets = tokenize_with_offsets(tokenizer, solution_text)
    tokens: List[Dict] = []
    for token_id, token_string, offset, student_lp, teacher_lp, token_kl in zip(
        token_ids,
        token_strings,
        offsets,
        student_logprobs,
        teacher_logprobs,
        token_kls,
    ):
        char_start, char_end = int(offset[0]), int(offset[1])
        step_id, section = locate_step_id(char_start, step_parse["steps"], step_parse["final_answer_span"])
        tokens.append(
            {
                "token_id": len(tokens),
                "token_int_id": token_id,
                "token_str": token_string,
                "char_start": char_start,
                "char_end": char_end,
                "step_id": step_id,
                "section": section,
                "student_logprob": student_lp,
                "teacher_logprob": teacher_lp,
                "gap": teacher_lp - student_lp,
                "token_kl": token_kl,
            }
        )
    return tokens


def compute_step_metrics(step_parse: Dict, tokens: List[Dict]) -> List[Dict]:
    bucketed: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for token in tokens:
        if token["section"] != "reasoning" or token["step_id"] is None:
            continue
        step_bucket = bucketed[token["step_id"]]
        step_bucket["student_logprob"].append(token["student_logprob"])
        step_bucket["teacher_logprob"].append(token["teacher_logprob"])
        step_bucket["gap"].append(token["gap"])
        if token["token_kl"] is not None:
            step_bucket["token_kl"].append(token["token_kl"])

    metrics: List[Dict] = []
    for step in step_parse["steps"]:
        values = bucketed.get(step.step_id, {})
        metrics.append(
            {
                "step_id": step.step_id,
                "step_text": step.step_text,
                "token_count": len(values.get("gap", [])),
                "avg_student_logprob": mean_or_none(values.get("student_logprob", [])),
                "avg_teacher_logprob": mean_or_none(values.get("teacher_logprob", [])),
                "avg_gap": mean_or_none(values.get("gap", [])),
                "avg_kl": mean_or_none(values.get("token_kl", [])),
            }
        )
    return metrics


def with_step_token_spans(step_parse: Dict, token_records: List[Dict]) -> List[Dict]:
    serialized_steps: List[Dict] = []
    for step in step_parse["steps"]:
        step_tokens = [token["token_id"] for token in token_records if token.get("step_id") == step.step_id]
        serialized_steps.append(
            {
                "step_id": step.step_id,
                "original_step_id": step.original_step_id,
                "step_text": step.step_text,
                "char_start": step.char_start,
                "char_end": step.char_end,
                "token_start": step_tokens[0] if step_tokens else None,
                "token_end": step_tokens[-1] if step_tokens else None,
            }
        )
    return serialized_steps


def summarize_run(records: List[Dict], args) -> Dict:
    format_success = [record for record in records if not record["format_fail"]]
    wrong = [record for record in records if record["is_correct"] is False]
    correct = [record for record in records if record["is_correct"] is True]
    return {
        "input_path": args.input,
        "model_name": args.model_name,
        "n_records": len(records),
        "format_success_rate": len(format_success) / len(records) if records else 0.0,
        "avg_num_steps": mean_or_none([record["num_steps"] for record in format_success]),
        "avg_num_tokens": mean_or_none([record["num_tokens"] for record in records]),
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_step_count": args.max_step_count,
        "overflow_mode": args.overflow_mode,
        "compute_token_kl": args.compute_token_kl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone BASD protocol runner.")
    parser.add_argument("--input", required=True, help="Input jsonl with question/reference_solution/final_answer fields.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--model_name", required=True, help="HF model name or local path.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N samples. 0 means all.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_step_count", type=int, default=16)
    parser.add_argument("--overflow_mode", choices=["truncate", "discard"], default="truncate")
    parser.add_argument("--max_context_tokens", type=int, default=4096)
    parser.add_argument("--compute_token_kl", action="store_true", help="Also compute token-level KL.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    samples = load_samples(args.input)
    if args.limit > 0:
        samples = samples[: args.limit]

    records: List[Dict] = []
    for index, sample in enumerate(samples, start=1):
        started = time.time()
        raw_text = generate_student_text(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            max_steps=args.max_step_count,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        step_parse = parse_structured_solution(
            text=raw_text,
            max_step_count=args.max_step_count,
            overflow_mode=args.overflow_mode,
        )
        solution_text = step_parse["cleaned_text"]
        continuation_ids, _, _ = tokenize_with_offsets(tokenizer, solution_text)
        student_logprobs, teacher_logprobs, token_kls = score_teacher_student(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            continuation_ids=continuation_ids,
            max_steps=args.max_step_count,
            max_context_tokens=args.max_context_tokens,
            compute_token_kl=args.compute_token_kl,
        )
        token_records = build_token_records(
            tokenizer=tokenizer,
            solution_text=solution_text,
            step_parse=step_parse,
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            token_kls=token_kls,
        )
        step_metrics = compute_step_metrics(step_parse, token_records)

        student_final_answer = step_parse["final_answer_text"] or extract_answer(solution_text)
        is_correct = normalize_answer(student_final_answer) == normalize_answer(sample.gold_answer)
        predicted_by_gap = predicted_boundary(step_metrics, "avg_gap")
        predicted_by_gap_jump = predicted_jump_boundary(step_metrics, "avg_gap")
        predicted_by_kl = predicted_boundary(step_metrics, "avg_kl")

        record = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "reference_solution": sample.reference_solution,
            "gold_answer": sample.gold_answer,
            "student_full_text": solution_text,
            "student_final_answer": student_final_answer,
            "is_correct": is_correct,
            "format_fail": step_parse["format_fail"],
            "format_error": step_parse["format_error"],
            "format_notes": step_parse["format_notes"],
            "step_overflow": step_parse["step_overflow"],
            "numbering_repaired": step_parse["numbering_repaired"],
            "num_steps": step_parse["analyzed_step_count"],
            "num_tokens": len(token_records),
            "predicted_boundary_by_gap": predicted_by_gap,
            "predicted_boundary_by_gap_jump": predicted_by_gap_jump,
            "predicted_boundary_by_kl": predicted_by_kl,
            "steps": with_step_token_spans(step_parse, token_records),
            "tokens": token_records,
            "step_metrics": step_metrics,
            "runtime_seconds": round(time.time() - started, 4),
        }
        records.append(record)

        if index % 10 == 0 or index == len(samples):
            write_jsonl(os.path.join(args.out_dir, "records.partial.jsonl"), records)

    write_jsonl(os.path.join(args.out_dir, "records.jsonl"), records)
    write_csv(
        os.path.join(args.out_dir, "annotation_template.csv"),
        build_annotation_template_rows(records),
        [
            "sample_id",
            "is_correct",
            "first_error_step",
            "error_type",
            "boundary_pattern",
            "comments",
            "predicted_boundary_by_gap",
            "predicted_boundary_by_gap_jump",
            "predicted_boundary_by_kl",
            "num_steps",
        ],
    )

    summary = summarize_run(records, args)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
