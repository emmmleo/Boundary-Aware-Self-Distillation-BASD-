from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from basd.data.collator import collate_examples
from basd.data.dataset import JsonlMathDataset
from basd.trainer.logger import JsonlLogger
from basd.trainer.step_fn import run_train_step


class BASDTrainer:
    def __init__(self, model, tokenizer, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.accelerator = Accelerator(gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"])
        self.optimizer = AdamW(self.model.parameters(), lr=cfg["train"]["learning_rate"])
        out_dir = Path(cfg.get("output_dir", "output/train_runs/default"))
        self.debug_logger = JsonlLogger(str(out_dir / "train_debug.jsonl"))

    def _build_loader(self):
        dataset = JsonlMathDataset(self.cfg["data"]["train_file"], self.cfg["data"].get("max_samples"))
        return DataLoader(
            dataset,
            batch_size=self.cfg["train"]["per_device_batch_size"],
            shuffle=True,
            collate_fn=collate_examples,
        )

    def train(self):
        loader = self._build_loader()
        self.model, self.optimizer, loader = self.accelerator.prepare(self.model, self.optimizer, loader)
        self.model.train()

        global_step = 0
        for batch in loader:
            with self.accelerator.accumulate(self.model):
                result = run_train_step(batch, self.model, self.tokenizer, self.accelerator, self.cfg)
                if result.aux.get("empty_batch", False):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.accelerator.backward(result.loss)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % self.cfg["train"]["log_steps"] == 0:
                rows = result.aux.get("rows", [])
                boundary_rate = sum(1 for r in rows if r["boundary_found"]) / max(1, len(rows))
                self.debug_logger.log(
                    {
                        "step": global_step,
                        "train/loss": float(result.loss.detach().item()),
                        "train/boundary_found_rate": boundary_rate,
                        "train/distill_vocab_mode": self.cfg["distill"]["vocab_mode"],
                        "rows": rows,
                    }
                )

            if global_step >= self.cfg["train"]["num_train_steps"]:
                break

        if self.accelerator.is_main_process:
            out_dir = Path(self.cfg.get("output_dir", "output/train_runs/default"))
            out_dir.mkdir(parents=True, exist_ok=True)
            unwrap = self.accelerator.unwrap_model(self.model)
            unwrap.save_pretrained(str(out_dir / "final_ckpt"))
