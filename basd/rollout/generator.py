import torch

from basd.model.loader import enable_student_adapter


def generate_student_rollout(model, tokenizer, prompt_text: str, cfg: dict):
    enable_student_adapter(model)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=cfg.get("do_sample", True),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.95),
            repetition_penalty=cfg.get("repetition_penalty", 1.0),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion_ids = output[0, inputs["input_ids"].shape[1] :]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return completion_ids, completion_text
