import torch

from basd.loss.distill import compute_distill_loss


def test_distill_modes_are_finite():
    torch.manual_seed(0)
    student = torch.randn(8, 32)
    teacher = torch.randn(8, 32)
    ids = torch.randint(0, 32, (8,))
    weights = torch.ones(8)
    mask = torch.ones(8)

    l1 = compute_distill_loss(student, teacher, ids, weights, mask, {"vocab_mode": "full", "objective": "opsd_jsd_reverse_kl_pg"})
    l2 = compute_distill_loss(student, teacher, ids, weights, mask, {"vocab_mode": "teacher_topk", "topk": 8, "objective": "opsd_jsd_reverse_kl_pg"})
    l3 = compute_distill_loss(student, teacher, ids, weights, mask, {"vocab_mode": "full", "objective": "jsd"})

    assert torch.isfinite(l1)
    assert torch.isfinite(l2)
    assert torch.isfinite(l3)
