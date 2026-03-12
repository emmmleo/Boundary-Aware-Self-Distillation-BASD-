import torch

from basd.model.scoring import gather_sampled_token_logprobs
from basd.types import TokenMetrics


def build_token_metrics(student_logits, teacher_logits, completion_ids, compute_kl: bool = True) -> TokenMetrics:
    student_lp = gather_sampled_token_logprobs(student_logits, completion_ids)
    teacher_lp = gather_sampled_token_logprobs(teacher_logits, completion_ids)
    gap = teacher_lp - student_lp

    token_kl = None
    if compute_kl:
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    return TokenMetrics(
        student_logprob=student_lp,
        teacher_logprob=teacher_lp,
        sampled_token_gap=gap,
        token_kl=token_kl,
    )
