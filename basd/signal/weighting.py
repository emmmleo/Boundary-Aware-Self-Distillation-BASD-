import torch


def build_token_weights(token_step_ids, reasoning_mask, is_correct, boundary, cfg):
    base_weight = cfg.get("incorrect_fallback_uniform_weight", 1.0)
    if is_correct:
        return reasoning_mask.float() * cfg.get("correct_uniform_weight", 1.0)

    weights = reasoning_mask.float() * base_weight
    if not boundary.found:
        return weights

    table = {int(k): float(v) for k, v in cfg.get("step_weight_table", {}).items()}
    for i in range(token_step_ids.numel()):
        if not reasoning_mask[i]:
            continue
        dist = int(token_step_ids[i].item()) - boundary.boundary_step_id
        if dist in table:
            weights[i] = weights[i] * table[dist]
    return weights
