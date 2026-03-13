import torch

from basd.data.answer_extractor import extract_final_answer, is_correct_answer
from basd.data.prompt_builder import PromptConfig, build_student_prompt, build_teacher_prompt
from basd.loss.distill import compute_distill_loss
from basd.loss.masks import build_distill_mask
from basd.model.loader import disable_student_adapter, enable_student_adapter
from basd.model.scoring import get_completion_logits
from basd.rollout.aligner import align_tokens_to_steps
from basd.rollout.generator import generate_student_rollout
from basd.rollout.parser import parse_steps_from_text
from basd.signal.boundary_detector import detect_boundary
from basd.signal.token_metrics import build_token_metrics
from basd.signal.weighting import build_token_weights
from basd.types import BoundaryResult, DistillBatchOutput


def run_train_step(batch_examples, model, tokenizer, accelerator, cfg):
    total_loss = None
    aux_rows = []

    for ex in batch_examples:
        student_prompt = build_student_prompt(ex.question, PromptConfig(**cfg["prompt"]))
        completion_ids, completion_text = generate_student_rollout(model, tokenizer, student_prompt, cfg["generation"])
        step_spans = parse_steps_from_text(completion_text)

        if not step_spans:
            continue

        token_step_ids, reasoning_mask, final_answer_mask = align_tokens_to_steps(
            tokenizer=tokenizer,
            completion_ids=completion_ids,
            completion_text=completion_text,
            step_spans=step_spans,
        )

        pred_answer = extract_final_answer(completion_text)
        correct = is_correct_answer(pred_answer, ex.gold_answer)

        prompt_ids = tokenizer(student_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(completion_ids.device)
        enable_student_adapter(model)
        student_logits = get_completion_logits(model, prompt_ids, completion_ids)

        teacher_prompt = build_teacher_prompt(ex.question, ex.reference_solution)
        teacher_prompt_ids = tokenizer(teacher_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(completion_ids.device)
        disable_student_adapter(model)
        teacher_logits = get_completion_logits(model, teacher_prompt_ids, completion_ids)
        enable_student_adapter(model)

        token_metrics = build_token_metrics(
            student_logits,
            teacher_logits,
            completion_ids,
            compute_kl=(cfg["boundary"].get("signal_type") == "token_kl" or cfg["distill"].get("vocab_mode") == "full"),
        )

        if not correct:
            signal = token_metrics.sampled_token_gap if cfg["boundary"].get("signal_type") == "sampled_gap" else token_metrics.token_kl
            boundary = detect_boundary(signal, token_step_ids, reasoning_mask, cfg["boundary"])
        else:
            zeros = torch.zeros_like(token_metrics.sampled_token_gap)
            boundary = BoundaryResult(False, -1, -1, zeros, zeros, zeros)

        distill_region = reasoning_mask | final_answer_mask
        weights = build_token_weights(token_step_ids, distill_region, correct, boundary, cfg["weighting"])
        loss_mask = build_distill_mask(reasoning_mask, final_answer_mask)

        loss = compute_distill_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            completion_ids=completion_ids,
            token_weights=weights,
            loss_mask=loss_mask,
            cfg=cfg["distill"],
        )

        total_loss = loss if total_loss is None else total_loss + loss
        aux_rows.append(
            {
                "sample_id": ex.sample_id,
                "is_correct": correct,
                "boundary_found": boundary.found,
                "boundary_step": boundary.boundary_step_id,
                "completion_len": int(completion_ids.numel()),
                "weight_mass": float(weights.sum().item()),
            }
        )

    if aux_rows:
        total_loss = total_loss / len(aux_rows)
    else:
        total_loss = torch.zeros((), device=accelerator.device)
    return DistillBatchOutput(loss=total_loss, aux={"rows": aux_rows, "empty_batch": not aux_rows})
