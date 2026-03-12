import torch

from basd.types import BoundaryResult


def _ema(signal: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.zeros_like(signal)
    if signal.numel() == 0:
        return out
    out[0] = signal[0]
    for i in range(1, signal.numel()):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
    return out


def detect_boundary(signal, token_step_ids, valid_mask, cfg) -> BoundaryResult:
    alpha = cfg.get("smooth_alpha", 0.3)
    abs_threshold = cfg.get("abs_threshold", 0.8)
    jump_threshold = cfg.get("jump_threshold", 0.5)
    persist_window = cfg.get("persist_window", 8)
    min_prefix = cfg.get("min_valid_tokens_before_boundary", 12)

    smooth = _ema(signal, alpha)
    delta = torch.zeros_like(signal)
    if signal.numel() > 1:
        delta[1:] = smooth[1:] - smooth[:-1]

    valid_idx = valid_mask.nonzero(as_tuple=False).flatten()
    for t in valid_idx.tolist():
        if (valid_mask[:t].sum().item() < min_prefix) or t == 0:
            continue
        if smooth[t] < abs_threshold or delta[t] < jump_threshold:
            continue
        right = valid_idx[valid_idx >= t][:persist_window]
        if right.numel() == 0:
            continue
        if smooth[right].mean().item() < abs_threshold:
            continue
        return BoundaryResult(
            found=True,
            boundary_token_idx=t,
            boundary_step_id=int(token_step_ids[t].item()),
            signal=signal,
            signal_smooth=smooth,
            delta=delta,
        )

    return BoundaryResult(
        found=False,
        boundary_token_idx=-1,
        boundary_step_id=-1,
        signal=signal,
        signal_smooth=smooth,
        delta=delta,
    )
