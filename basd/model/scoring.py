import torch


def forward_completion_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def get_completion_logits(model, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
    logits = forward_completion_logits(model, full_ids)[0]
    prompt_len = prompt_ids.size(0)
    comp_len = completion_ids.size(0)
    start = prompt_len - 1
    end = start + comp_len
    return logits[start:end]


def gather_sampled_token_logprobs(logits: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)
