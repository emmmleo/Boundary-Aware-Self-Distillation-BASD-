from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class Example:
    sample_id: str
    question: str
    gold_answer: str
    reference_solution: str


@dataclass
class StepSpan:
    step_id: int
    char_start: int
    char_end: int
    text: str
    is_final: bool = False


@dataclass
class RolloutRecord:
    sample_id: str
    prompt_text: str
    completion_text: str
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    step_spans: list[StepSpan]
    token_step_ids: torch.Tensor
    reasoning_token_mask: torch.Tensor
    final_answer_mask: torch.Tensor
    final_answer: str
    is_correct: bool


@dataclass
class TokenMetrics:
    student_logprob: torch.Tensor
    teacher_logprob: torch.Tensor
    sampled_token_gap: torch.Tensor
    token_kl: Optional[torch.Tensor]


@dataclass
class BoundaryResult:
    found: bool
    boundary_token_idx: int
    boundary_step_id: int
    signal: torch.Tensor
    signal_smooth: torch.Tensor
    delta: torch.Tensor


@dataclass
class DistillBatchOutput:
    loss: torch.Tensor
    aux: dict[str, Any]
