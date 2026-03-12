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


def test_teacher_topk_pg_includes_sampled_token_when_outside_topk():
    # sampled token id=4 is outside teacher top-2 initially
    student = torch.tensor([[0.2, -0.1, 0.3, 0.1, 2.0]], dtype=torch.float32)
    teacher = torch.tensor([[5.0, 4.0, 0.2, 0.1, -3.0]], dtype=torch.float32)
    ids = torch.tensor([4], dtype=torch.long)
    weights = torch.ones(1)
    mask = torch.ones(1)

    loss = compute_distill_loss(
        student,
        teacher,
        ids,
        weights,
        mask,
        {"vocab_mode": "teacher_topk", "topk": 2, "objective": "pg"},
    )

    assert torch.isfinite(loss)
    # inclusion of sampled token should keep a non-trivial pg gradient signal
    assert loss.abs().item() > 1e-8


def test_batched_sequence_inputs_are_supported():
    torch.manual_seed(1)
    student = torch.randn(2, 4, 16)
    teacher = torch.randn(2, 4, 16)
    ids = torch.randint(0, 16, (2, 4))
    weights = torch.ones(2, 4)
    mask = torch.ones(2, 4)

    loss = compute_distill_loss(
        student,
        teacher,
        ids,
        weights,
        mask,
        {"vocab_mode": "teacher_topk", "topk": 4, "objective": "opsd_jsd_reverse_kl_pg"},
    )

    assert torch.isfinite(loss)
