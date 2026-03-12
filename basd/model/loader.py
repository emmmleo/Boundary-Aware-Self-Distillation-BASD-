from contextlib import contextmanager

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_student_teacher_model(cfg):
    model_cfg = cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name_or_path"],
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
    )
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if model_cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_cfg["lora_r"],
            lora_alpha=model_cfg["lora_alpha"],
            lora_dropout=model_cfg.get("lora_dropout", 0.0),
            target_modules=model_cfg["target_modules"],
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg)
    return model


def enable_student_adapter(model):
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()


def disable_student_adapter(model):
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()


@contextmanager
def teacher_mode(model):
    disable_student_adapter(model)
    yield
    enable_student_adapter(model)
