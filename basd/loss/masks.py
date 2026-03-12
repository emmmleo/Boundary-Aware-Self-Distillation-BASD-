import torch


def build_distill_mask(reasoning_mask: torch.Tensor, final_answer_mask: torch.Tensor) -> torch.Tensor:
    return (reasoning_mask | final_answer_mask).float()
