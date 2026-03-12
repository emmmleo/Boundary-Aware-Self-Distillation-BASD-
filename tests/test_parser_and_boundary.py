import torch

from basd.rollout.parser import extract_final_region, parse_steps_from_text
from basd.signal.boundary_detector import detect_boundary


def test_parse_steps_and_final_region():
    text = "<<STEP_1>>a\n<<STEP_2>>b\n<<FINAL>>42"
    spans = parse_steps_from_text(text)
    assert len(spans) == 3
    s, e = extract_final_region(text)
    assert text[s:e] == "42"


def test_boundary_detects_jump():
    signal = torch.tensor([0.1, 0.2, 0.15, 0.2, 1.2, 1.3, 1.1], dtype=torch.float)
    token_step_ids = torch.tensor([1, 1, 1, 2, 2, 2, 3])
    valid_mask = torch.ones_like(token_step_ids, dtype=torch.bool)
    cfg = {
        "smooth_alpha": 0.6,
        "abs_threshold": 0.5,
        "jump_threshold": 0.3,
        "persist_window": 2,
        "min_valid_tokens_before_boundary": 2,
    }
    br = detect_boundary(signal, token_step_ids, valid_mask, cfg)
    assert br.found
    assert br.boundary_token_idx >= 4
