#!/usr/bin/env python
import argparse

from basd.model.loader import load_student_teacher_model, load_tokenizer
from basd.trainer.engine import BASDTrainer
from basd.utils.config import load_yaml
from basd.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/basd_qwen3_8b.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["train"]["seed"])
    tokenizer = load_tokenizer(cfg)
    model = load_student_teacher_model(cfg)

    trainer = BASDTrainer(model=model, tokenizer=tokenizer, cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
