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


def _to_2d(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if t.dim() == 1:
        return t.unsqueeze(0), True
    return t, False


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


def _build_teacher_topk_with_sampled(teacher_logits: torch.Tensor, completion_ids: torch.Tensor, topk: int) -> torch.Tensor:
    vocab_size = teacher_logits.size(-1)
    k = min(max(1, topk), vocab_size)
    _, topk_idx = torch.topk(teacher_logits, k=k, dim=-1)
    sampled_idx = completion_ids.unsqueeze(-1)
    contains_sampled = (topk_idx == sampled_idx).any(dim=-1, keepdim=True)
    replacement = torch.where(contains_sampled, topk_idx[..., -1:], sampled_idx)
    topk_idx[..., -1:] = replacement
    return topk_idx


def compute_distill_loss(student_logits, teacher_logits, completion_ids, token_weights, loss_mask, cfg):
    mode = cfg.get("vocab_mode", "teacher_topk")
    topk = int(cfg.get("topk", 64))
    objective = cfg.get("objective", "opsd_jsd_reverse_kl_pg")

    student_logits_2d, logits_was_1d = _to_2d(student_logits)
    teacher_logits_2d, _ = _to_2d(teacher_logits)
    completion_ids_2d, _ = _to_2d(completion_ids)
    token_weights_2d, weights_was_1d = _to_2d(token_weights)
    loss_mask_2d, mask_was_1d = _to_2d(loss_mask)

    if mode == "teacher_topk":
        topk_idx = _build_teacher_topk_with_sampled(teacher_logits_2d, completion_ids_2d, topk=topk)
        student_sel = torch.gather(student_logits_2d, dim=-1, index=topk_idx)
        teacher_sel = torch.gather(teacher_logits_2d, dim=-1, index=topk_idx)
        completion_sel = (topk_idx == completion_ids_2d.unsqueeze(-1)).to(torch.long).argmax(dim=-1)
    else:
        student_sel, teacher_sel = _select_logits(student_logits_2d, teacher_logits_2d, mode=mode, topk=topk)
        completion_sel = completion_ids_2d

    jsd, reverse_kl, student_log_probs, teacher_log_probs = _jsd_and_reverse_kl(student_sel, teacher_sel)
    pg_per_token = _pg_loss_from_sampled(student_log_probs, teacher_log_probs, completion_sel)

    if logits_was_1d:
        jsd = jsd.squeeze(0)
        reverse_kl = reverse_kl.squeeze(0)
        pg_per_token = pg_per_token.squeeze(0)
    if weights_was_1d:
        token_weights = token_weights_2d.squeeze(0)
    if mask_was_1d:
        loss_mask = loss_mask_2d.squeeze(0)

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
