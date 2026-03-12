import torch


def _select_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mode: str, topk: int):
    if mode == "full":
        return student_logits, teacher_logits
    if mode == "teacher_topk":
        _, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
        student_sel = torch.gather(student_logits, dim=-1, index=topk_idx)
        teacher_sel = torch.gather(teacher_logits, dim=-1, index=topk_idx)
        return student_sel, teacher_sel
    raise ValueError(f"Unknown distill vocab mode: {mode}")


def _safe_weighted_mean(per_token: torch.Tensor, token_weights: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    weighted = per_token * token_weights * loss_mask
    denom = (token_weights * loss_mask).sum().clamp_min(1.0)
    return weighted.sum() / denom


def _kl(p_log_probs: torch.Tensor, q_log_probs: torch.Tensor) -> torch.Tensor:
    p_probs = p_log_probs.exp()
    return (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)


def _jsd_and_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    mix_log_probs = torch.log((student_log_probs.exp() + teacher_log_probs.exp()) * 0.5 + 1e-12)

    jsd = 0.5 * (_kl(student_log_probs, mix_log_probs) + _kl(teacher_log_probs, mix_log_probs))
    reverse_kl = _kl(student_log_probs, teacher_log_probs)
    return jsd, reverse_kl, student_log_probs, teacher_log_probs


def _pg_loss_from_sampled(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    gather_idx = completion_ids.unsqueeze(-1)
    student_sampled_lp = torch.gather(student_log_probs, dim=-1, index=gather_idx).squeeze(-1)
    teacher_sampled_lp = torch.gather(teacher_log_probs, dim=-1, index=gather_idx).squeeze(-1)
    advantage = (teacher_sampled_lp - student_sampled_lp).detach()
    return -(advantage * student_sampled_lp)


def compute_distill_loss(student_logits, teacher_logits, completion_ids, token_weights, loss_mask, cfg):
    mode = cfg.get("vocab_mode", "teacher_topk")
    topk = int(cfg.get("topk", 64))
    objective = cfg.get("objective", "opsd_jsd_reverse_kl_pg")

    student_sel, teacher_sel = _select_logits(student_logits, teacher_logits, mode=mode, topk=topk)
    jsd, reverse_kl, student_log_probs, teacher_log_probs = _jsd_and_reverse_kl(student_sel, teacher_sel)

    if mode == "teacher_topk":
        _, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
        completion_sel = torch.zeros_like(completion_ids)
        for t in range(completion_ids.size(0)):
            match = (topk_idx[t] == completion_ids[t]).nonzero(as_tuple=False)
            completion_sel[t] = match[0, 0] if match.numel() > 0 else 0
        pg_per_token = _pg_loss_from_sampled(student_log_probs, teacher_log_probs, completion_sel)
    else:
        pg_per_token = _pg_loss_from_sampled(student_log_probs, teacher_log_probs, completion_ids)

    if objective == "jsd":
        return _safe_weighted_mean(jsd, token_weights, loss_mask)
    if objective == "reverse_kl":
        return _safe_weighted_mean(reverse_kl, token_weights, loss_mask)
    if objective == "pg":
        return _safe_weighted_mean(pg_per_token, token_weights, loss_mask)
    if objective == "opsd_jsd_reverse_kl_pg":
        w_jsd = float(cfg.get("w_jsd", 1.0))
        w_reverse_kl = float(cfg.get("w_reverse_kl", 1.0))
        w_pg = float(cfg.get("w_pg", 1.0))
        total = w_jsd * jsd + w_reverse_kl * reverse_kl + w_pg * pg_per_token
        return _safe_weighted_mean(total, token_weights, loss_mask)

    raise ValueError(f"Unknown objective: {objective}")
