import torch


def _full_kl(student_logits, teacher_logits):
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)


def _teacher_topk_kl(student_logits, teacher_logits, k: int):
    _, topk_idx = torch.topk(teacher_logits, k=k, dim=-1)
    teacher_sel = torch.gather(teacher_logits, dim=-1, index=topk_idx)
    student_sel = torch.gather(student_logits, dim=-1, index=topk_idx)

    teacher_sel_log_probs = torch.log_softmax(teacher_sel, dim=-1)
    student_sel_log_probs = torch.log_softmax(student_sel, dim=-1)
    teacher_sel_probs = teacher_sel_log_probs.exp()
    return (teacher_sel_probs * (teacher_sel_log_probs - student_sel_log_probs)).sum(dim=-1)


def compute_distill_loss(student_logits, teacher_logits, completion_ids, token_weights, loss_mask, cfg):
    del completion_ids
    mode = cfg.get("vocab_mode", "teacher_topk")
    if mode == "full":
        per_token_loss = _full_kl(student_logits, teacher_logits)
    elif mode == "teacher_topk":
        per_token_loss = _teacher_topk_kl(student_logits, teacher_logits, int(cfg.get("topk", 64)))
    else:
        raise ValueError(f"Unknown distill vocab mode: {mode}")

    weighted = per_token_loss * token_weights * loss_mask
    denom = (token_weights * loss_mask).sum().clamp_min(1.0)
    return weighted.sum() / denom
