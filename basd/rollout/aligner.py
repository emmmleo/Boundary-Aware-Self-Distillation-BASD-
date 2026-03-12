import torch

from basd.types import StepSpan


def align_tokens_to_steps(tokenizer, completion_ids: torch.Tensor, completion_text: str, step_spans: list[StepSpan]):
    enc = tokenizer(
        completion_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    rt_ids = enc["input_ids"][0]
    if rt_ids.size(0) != completion_ids.size(0) or not torch.equal(rt_ids.cpu(), completion_ids.cpu()):
        raise ValueError("Retokenized ids mismatch generated completion ids")

    offsets = enc["offset_mapping"][0].tolist()
    token_step_ids = torch.full((completion_ids.size(0),), -1, dtype=torch.long)
    reasoning_mask = torch.zeros_like(token_step_ids, dtype=torch.bool)
    final_answer_mask = torch.zeros_like(token_step_ids, dtype=torch.bool)

    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        c = (start + end) // 2
        for span in step_spans:
            if span.char_start <= c < span.char_end:
                token_step_ids[idx] = span.step_id
                if span.is_final:
                    final_answer_mask[idx] = True
                else:
                    reasoning_mask[idx] = True
                break

    return token_step_ids, reasoning_mask, final_answer_mask
