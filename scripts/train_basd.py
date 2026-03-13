#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from basd.model.loader import load_student_teacher_model, load_tokenizer
from basd.trainer.engine import BASDTrainer
from basd.utils.config import load_yaml
from basd.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/basd_qwen3_8b.yaml")
    args = parser.parse_args()

    print(f"[main] loading config from {args.config}", flush=True)
    cfg = load_yaml(args.config)
    set_seed(cfg["train"]["seed"])
    print(f"[main] seed set to {cfg['train']['seed']}", flush=True)
    print(f"[main] loading tokenizer from {cfg['model']['base_model_name_or_path']}", flush=True)
    tokenizer = load_tokenizer(cfg)
    print(f"[main] loading model from {cfg['model']['base_model_name_or_path']}", flush=True)
    model = load_student_teacher_model(cfg)

    print(f"[main] starting trainer, outputs -> {cfg.get('output_dir', 'output/train_runs/default')}", flush=True)
    trainer = BASDTrainer(model=model, tokenizer=tokenizer, cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
